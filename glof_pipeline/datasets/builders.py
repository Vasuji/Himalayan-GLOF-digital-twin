"""Dataset builders: sample scenarios, run the reference model, store arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from glof_pipeline.atmospheric.forecaster import coarsen
from glof_pipeline.atmospheric.downscaling import normalise_orography
from glof_pipeline.physics.breach import simulate_breach
from glof_pipeline.physics.moraine_stability import evaluate_stability
from glof_pipeline.physics.swe_solver import ShallowWaterSolver, hydrograph_source
from glof_pipeline.surrogates.mgn_moraine import MoraineNodeState, assemble_node_features
from glof_pipeline.terrain.mesh_builder import MoraineGraph, distance_to_lake, slope_and_aspect
from glof_pipeline.terrain.synthetic_dem import ValleyTerrain
from glof_pipeline.utils.io_helpers import save_npz
from glof_pipeline.utils.runtime import get_logger

LOGGER = get_logger("glof.datasets")


# ---------------------------------------------------------------------------
# Moraine stability
# ---------------------------------------------------------------------------
def build_moraine_dataset(
    terrain: ValleyTerrain,
    graph: MoraineGraph,
    moraine_cfg: dict[str, Any],
    n_scenarios: int,
    rng: np.random.Generator,
    path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Sample hydrological and geotechnical states and label them with the solver.

    The sampled ranges deliberately straddle failure: lake stage runs from well
    below the crest to overtopping, and cohesion from a weakly cemented till to a
    strongly ice-bonded one. A training set drawn only from stable states teaches a
    surrogate nothing about the decision boundary that matters.
    """
    rows, cols = graph.node_rc[:, 0], graph.node_rc[:, 1]
    beta_grid, aspect_grid = slope_and_aspect(terrain.z, terrain.dx)
    cached = {
        "slope": beta_grid[rows, cols],
        "aspect": aspect_grid[rows, cols],
        "distance": distance_to_lake(terrain, graph),
    }

    reference_depth = float(moraine_cfg["till_depth_m"])
    reference_cohesion = float(moraine_cfg["cohesion_kpa"])
    crest = terrain.crest_elevation
    lake_bed = float(terrain.meta.get("lake_bed_min_m", terrain.z.min()))

    node_features = np.zeros((n_scenarios, graph.num_nodes, 9))
    targets = np.zeros((n_scenarios, graph.num_nodes, 2))
    scalars = np.zeros((n_scenarios, 4))

    for i in range(n_scenarios):
        lake_level = float(rng.uniform(max(lake_bed + 5.0, crest - 40.0), crest + 2.0))
        pdd = float(rng.uniform(0.0, 400.0))
        cohesion = float(np.clip(reference_cohesion * rng.uniform(0.35, 1.9), 0.5, None))
        # Spatially correlated till-depth field: thickness varies along the dam.
        noise = ndimage.gaussian_filter(rng.standard_normal(graph.num_nodes), sigma=3.0)
        noise = noise / max(float(np.abs(noise).max()), 1e-6)
        till_depth = reference_depth * np.clip(1.0 + 0.30 * noise, 0.4, 1.8)

        state = MoraineNodeState(
            lake_level_m=lake_level,
            cumulative_pdd_c_day=pdd,
            till_depth_m=till_depth,
            cohesion_kpa=cohesion,
        )
        node_features[i] = assemble_node_features(terrain, graph, state, moraine_cfg, cached)
        field = evaluate_stability(
            terrain, graph, lake_level, pdd, moraine_cfg,
            till_depth_m=till_depth, cohesion_kpa=cohesion,
        )
        targets[i, :, 0] = np.log(np.clip(field.factor_of_safety, 1e-3, 1e3))
        targets[i, :, 1] = field.pore_pressure_ratio
        scalars[i] = [lake_level, pdd, cohesion, field.min_factor_of_safety]

    dataset = {
        "node_features": node_features,
        "targets": targets,
        "edge_features": graph.edge_attr_static,
        "edge_index": graph.edge_index,
        "scenario_scalars": scalars,
    }
    LOGGER.info(
        "Moraine dataset: %d scenarios x %d nodes; %.0f%% contain a failing node",
        n_scenarios, graph.num_nodes, 100.0 * float(np.mean(scalars[:, 3] < 1.0)),
    )
    if path is not None:
        save_npz(path, **dataset)
    return dataset


# ---------------------------------------------------------------------------
# Shallow-water routing
# ---------------------------------------------------------------------------
def build_swe_dataset(
    terrain: ValleyTerrain,
    routing_cfg: dict[str, Any],
    breach_cfg: dict[str, Any],
    n_scenarios: int,
    frames_per_scenario: int,
    resolution: int,
    rng: np.random.Generator,
    path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Run the reference solver on randomised breach scenarios at ``resolution``.

    The bed is block-averaged to the training resolution and the solver is run
    there, so the FNO learns the time-advance operator on a self-consistent grid.
    Because the operator is resolution-invariant it is later applied at the full
    terrain resolution without retraining.
    """
    physics = routing_cfg["physics"]
    solver_cfg = routing_cfg["solver"]

    coarse_bed = coarsen(terrain.z, (resolution, resolution))
    scale = terrain.z.shape[0] / resolution
    dx_coarse = terrain.dx * scale
    breach_row = int(round(terrain.moraine_row / scale))
    breach_row = int(np.clip(breach_row, 1, resolution - 2))
    breach_col = int(np.clip(round(int(np.argmin(coarse_bed[breach_row])) ), 1, resolution - 2))
    cells = np.array([[breach_row, breach_col]], dtype=int)

    output_interval = float(solver_cfg["output_interval_s"])
    duration = output_interval * frames_per_scenario

    depth = np.zeros((n_scenarios, frames_per_scenario + 1, resolution, resolution), dtype=np.float32)
    mx = np.zeros_like(depth)
    my = np.zeros_like(depth)
    scalars = np.zeros((n_scenarios, 3))

    # A breach only initiates with the lake at or near the crest, and the outflow is
    # identically zero once the stage falls below the breach invert
    # (crest - breach.min_breach_depth_m). Sampling uniformly over an 8 m drawdown
    # therefore produced scenarios with no discharge at all, which teach the operator
    # nothing. Sample near the crest instead, allowing a little overtopping.
    min_breach_depth = float(breach_cfg["min_breach_depth_m"])
    stage_low = terrain.crest_elevation - 0.5 * min_breach_depth
    stage_high = terrain.crest_elevation + 1.0

    for i in range(n_scenarios):
        lake_level = float(rng.uniform(stage_low, stage_high))
        mechanism = "overtopping" if rng.random() < 0.6 else "piping"
        breach = simulate_breach(terrain, lake_level, mechanism, breach_cfg)
        manning = float(np.clip(physics["manning_n"] * rng.uniform(0.6, 1.6), 0.015, 0.12))

        solver = ShallowWaterSolver(
            coarse_bed,
            dx_coarse,
            gravity=float(physics["gravity"]),
            manning_n=manning,
            dry_depth=float(physics["dry_depth_m"]),
            cfl=float(physics["cfl"]),
            boundary=str(solver_cfg["boundary"]),
        )
        # The Froehlich formation time for a lake this size is of order an hour, so a
        # short training window starting at t = 0 sees only the toe of the rising limb
        # and yields dry frames. Shift the hydrograph so the window straddles the peak,
        # which is the part of the flow the operator has to represent.
        time_offset = max(0.0, breach.time_to_peak_s - 0.25 * duration)
        source = hydrograph_source(
            (resolution, resolution), cells, breach.time_s - time_offset,
            breach.discharge_m3_per_s, solver.cell_area,
        )
        result = solver.run(
            np.zeros((resolution, resolution)),
            duration_s=duration,
            output_interval_s=output_interval,
            source=source,
            max_steps=int(solver_cfg["max_steps"]),
        )
        n = min(frames_per_scenario + 1, result.depth.shape[0])
        depth[i, :n] = result.depth[:n]
        mx[i, :n] = result.momentum_x[:n]
        my[i, :n] = result.momentum_y[:n]
        scalars[i] = [breach.peak_discharge_m3_per_s, manning, lake_level]
        if float(depth[i].max()) <= 0.0:
            LOGGER.warning(
                "SWE scenario %d produced no water: peak Q = %.1f m3/s, offset = %.0f s. "
                "Check that the sampled stage exceeds the breach invert.",
                i + 1, breach.peak_discharge_m3_per_s, time_offset,
            )
        LOGGER.info(
            "SWE scenario %d/%d: peak Q = %.0f m3/s, n = %.3f, max depth = %.2f m",
            i + 1, n_scenarios, breach.peak_discharge_m3_per_s, manning, float(depth[i].max()),
        )

    dataset = {
        "depth": depth,
        "momentum_x": mx,
        "momentum_y": my,
        "bed": coarse_bed.astype(np.float32),
        "scenario_scalars": scalars,
        "dx": np.array(dx_coarse),
        "output_interval_s": np.array(output_interval),
    }
    if path is not None:
        save_npz(path, **dataset)
    return dataset


# ---------------------------------------------------------------------------
# Generative downscaling
# ---------------------------------------------------------------------------
def build_downscaling_dataset(
    terrain: ValleyTerrain,
    atmospheric_cfg: dict[str, Any],
    n_scenarios: int,
    rng: np.random.Generator,
    path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Paired coarse/fine temperature and precipitation fields over the terrain.

    Fine fields are generated first, with orographic structure the coarse grid
    cannot represent, and the coarse field is then produced by block-averaging.
    Building the pair in that order guarantees the learning problem is genuine
    super-resolution rather than an inversion of an arbitrary interpolation.
    """
    generator = atmospheric_cfg["toy_generator"]
    coarse_shape = tuple(int(v) for v in generator["coarse_shape"])
    reference_elevation = float(atmospheric_cfg["target_region"]["reference_elevation_m"])
    lapse = float(generator["lapse_rate_c_per_km"])

    elevation = terrain.z
    relief = normalise_orography(elevation)
    ny, nx = elevation.shape

    fine = np.zeros((n_scenarios, 2, ny, nx), dtype=np.float32)
    coarse = np.zeros((n_scenarios, 2, *coarse_shape), dtype=np.float32)

    for i in range(n_scenarios):
        base_temperature = float(generator["base_temperature_c"]) + rng.normal(0.0, 3.0)
        burst = float(generator["precip_burst_mm_per_h"]) * rng.uniform(0.0, 1.2)
        enhancement = rng.uniform(0.3, 1.4)

        temperature_noise = ndimage.gaussian_filter(rng.standard_normal((ny, nx)), sigma=3.0)
        temperature_noise /= max(float(temperature_noise.std()), 1e-6)
        precip_noise = ndimage.gaussian_filter(rng.standard_normal((ny, nx)), sigma=2.0)
        precip_noise /= max(float(precip_noise.std()), 1e-6)

        temperature = (
            base_temperature
            - lapse * (elevation - reference_elevation) / 1000.0
            + float(generator["temperature_noise_c"]) * temperature_noise
        )
        precipitation = np.clip(
            float(generator["precip_base_mm_per_h"])
            + burst * (1.0 + enhancement * relief)
            + float(generator["precip_noise_mm_per_h"]) * precip_noise,
            0.0,
            None,
        )

        fine[i, 0] = temperature
        fine[i, 1] = precipitation
        coarse[i, 0] = coarsen(temperature, coarse_shape)
        coarse[i, 1] = coarsen(precipitation, coarse_shape)

    dataset = {
        "fine": fine,
        "coarse": coarse,
        "orography": relief.astype(np.float32),
        "elevation_m": elevation.astype(np.float32),
    }
    LOGGER.info(
        "Downscaling dataset: %d pairs, coarse %s -> fine %s", n_scenarios, coarse_shape, (ny, nx)
    )
    if path is not None:
        save_npz(path, **dataset)
    return dataset
