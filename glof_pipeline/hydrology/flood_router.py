"""Flood routing: FNO surrogate rollout with the finite-volume solver as reference.

Both paths take the same breach hydrograph and the same bed, so the comparison
reported by :mod:`glof_pipeline.evaluate.benchmark` is like for like. The surrogate
is applied at the terrain resolution even though it was trained coarser, which is
the resolution-invariance property that motivates using a neural operator here
rather than a plain convolutional network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glof_pipeline.physics.breach import BreachResult
from glof_pipeline.physics.swe_solver import ShallowWaterSolver, SWEResult, hydrograph_source
from glof_pipeline.surrogates.fno_swe import FloodOperator
from glof_pipeline.terrain.synthetic_dem import ValleyTerrain
from glof_pipeline.utils.runtime import Timer, get_logger

LOGGER = get_logger("glof.routing")


@dataclass
class RoutingOutcome:
    """Inundation fields, arrival times and the wall-clock cost of producing them."""

    method: str                       # solver | fno
    time_s: np.ndarray
    max_depth_m: np.ndarray
    final_depth_m: np.ndarray
    arrival_times_s: dict[str, float]
    peak_depth_m: float
    inundated_area_km2: float
    wall_time_s: float
    mass_conservation_error: float | None = None
    # Full (nt, ny, nx) depth history, kept for the animated Omniverse scene.
    depth_sequence: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "peak_depth_m": self.peak_depth_m,
            "inundated_area_km2": self.inundated_area_km2,
            "arrival_times_s": self.arrival_times_s,
            "wall_time_s": self.wall_time_s,
            "mass_conservation_error": self.mass_conservation_error,
        }


def breach_cells(terrain: ValleyTerrain, width_m: float) -> np.ndarray:
    """Grid cells across the crest through which the hydrograph is injected."""
    row = terrain.moraine_row
    centre = int(np.argmin(terrain.z[row]))
    half = max(1, int(round(0.5 * width_m / terrain.dx)))
    cols = np.arange(max(centre - half, 0), min(centre + half + 1, terrain.shape[1]))
    return np.stack([np.full(cols.size, row), cols], axis=1)


def route_with_solver(
    terrain: ValleyTerrain,
    breach: BreachResult,
    routing_cfg: dict[str, Any],
    receptors: list[tuple[str, int]],
) -> tuple[RoutingOutcome, SWEResult]:
    """Route the hydrograph with the reference finite-volume solver."""
    physics = routing_cfg["physics"]
    solver_cfg = routing_cfg["solver"]
    solver = ShallowWaterSolver(
        terrain.z,
        terrain.dx,
        gravity=float(physics["gravity"]),
        manning_n=float(physics["manning_n"]),
        dry_depth=float(physics["dry_depth_m"]),
        cfl=float(physics["cfl"]),
        boundary=str(solver_cfg["boundary"]),
    )
    cells = breach_cells(terrain, breach.final_breach_width_m)
    source = hydrograph_source(
        terrain.shape, cells, breach.time_s, breach.discharge_m3_per_s, solver.cell_area
    )
    duration = float(solver_cfg["horizon_hours"]) * 3600.0
    with Timer("swe_solver", logger=LOGGER) as timer:
        result = solver.run(
            np.zeros(terrain.shape),
            duration_s=duration,
            output_interval_s=float(solver_cfg["output_interval_s"]),
            source=source,
            max_steps=int(solver_cfg["max_steps"]),
        )
    threshold = float(routing_cfg["inundation_threshold_m"])
    max_depth = result.max_depth
    outcome = RoutingOutcome(
        method="solver",
        time_s=result.time_s,
        max_depth_m=max_depth,
        final_depth_m=result.depth[-1],
        arrival_times_s=result.arrival_times(receptors, threshold),
        peak_depth_m=float(max_depth.max()),
        inundated_area_km2=float((max_depth >= threshold).sum() * terrain.cell_area / 1.0e6),
        wall_time_s=timer.elapsed,
        mass_conservation_error=result.mass_conservation_error(),
        depth_sequence=result.depth,
        meta={"steps": result.steps, "frames": int(result.time_s.size)},
    )
    return outcome, result


def route_with_surrogate(
    terrain: ValleyTerrain,
    operator: FloodOperator,
    initial_depth: np.ndarray,
    routing_cfg: dict[str, Any],
    receptors: list[tuple[str, int]],
    n_steps: int,
) -> RoutingOutcome:
    """Roll the FNO forward from a seeded initial state."""
    solver_cfg = routing_cfg["solver"]
    interval = float(solver_cfg["output_interval_s"])
    with Timer("fno_rollout", logger=LOGGER) as timer:
        rollout = operator.rollout(
            initial_depth,
            np.zeros_like(initial_depth),
            np.zeros_like(initial_depth),
            terrain.z,
            n_steps=n_steps,
        )
    depth = rollout["depth"]
    times = np.arange(depth.shape[0]) * interval
    threshold = float(routing_cfg["inundation_threshold_m"])
    max_depth = depth.max(axis=0)

    arrival: dict[str, float] = {}
    for name, row in receptors:
        exceeded = depth[:, row, :].max(axis=1) >= threshold
        arrival[name] = float(times[int(np.argmax(exceeded))]) if exceeded.any() else float("nan")

    return RoutingOutcome(
        method="fno",
        time_s=times,
        max_depth_m=max_depth,
        final_depth_m=depth[-1],
        arrival_times_s=arrival,
        peak_depth_m=float(max_depth.max()),
        inundated_area_km2=float((max_depth >= threshold).sum() * terrain.cell_area / 1.0e6),
        wall_time_s=timer.elapsed,
        depth_sequence=depth,
        meta={"backend": operator.backend, "steps": int(n_steps)},
    )


def route_flood(
    terrain: ValleyTerrain,
    breach: BreachResult | None,
    routing_cfg: dict[str, Any],
    receptors: list[tuple[str, int]],
    operator: FloodOperator | None = None,
) -> dict[str, RoutingOutcome]:
    """Run the reference solver and, when available, the surrogate on the same event.

    Returns an empty mapping when no breach is predicted, which is the correct
    operational answer rather than a routed flood of zero depth.
    """
    if breach is None:
        LOGGER.info("No breach predicted; routing skipped.")
        return {}

    outcomes: dict[str, RoutingOutcome] = {}
    solver_outcome, solver_result = route_with_solver(terrain, breach, routing_cfg, receptors)
    outcomes["solver"] = solver_outcome

    if operator is not None:
        # Seed the surrogate from the first solver frame that is wet, so both
        # methods start from the same physical state.
        wet_frames = np.argmax(solver_result.depth.max(axis=(1, 2)) > 0.0)
        seed_index = int(wet_frames)
        n_steps = int(solver_result.time_s.size - seed_index - 1)
        if n_steps > 0:
            outcomes["fno"] = route_with_surrogate(
                terrain, operator, solver_result.depth[seed_index], routing_cfg, receptors, n_steps
            )
    return outcomes
