"""Surrogate-versus-solver benchmark.

The draft manuscript carried a table of speedups (20,000x to 55,000x) with the
caption "illustrative, requiring validation". This module replaces those numbers
with measured ones: the same event is routed by the finite-volume solver and by
the FNO on the same machine, and the table records both wall-clock times and the
accuracy that was traded for them. A speedup without the paired error is not a
result.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glof_pipeline.evaluate.metrics import arrival_time_error, summarise_field_error
from glof_pipeline.hydrology.flood_router import RoutingOutcome
from glof_pipeline.utils.runtime import environment_report


def benchmark_routing(outcomes: dict[str, RoutingOutcome]) -> dict[str, Any]:
    """Compare a surrogate routing outcome against the solver reference."""
    if "solver" not in outcomes:
        raise KeyError("Benchmark requires a 'solver' outcome as the reference.")
    reference = outcomes["solver"]
    report: dict[str, Any] = {
        "reference": {
            "method": "solver",
            "wall_time_s": reference.wall_time_s,
            "peak_depth_m": reference.peak_depth_m,
            "inundated_area_km2": reference.inundated_area_km2,
            "mass_conservation_error": reference.mass_conservation_error,
            "steps": reference.meta.get("steps"),
        },
        "environment": environment_report(),
        "comparisons": {},
    }

    for name, outcome in outcomes.items():
        if name == "solver":
            continue
        # Compare the depth envelopes on the overlapping frames.
        field_errors = summarise_field_error(outcome.max_depth_m, reference.max_depth_m)
        speedup = reference.wall_time_s / max(outcome.wall_time_s, 1e-9)
        report["comparisons"][name] = {
            "wall_time_s": outcome.wall_time_s,
            "speedup_vs_solver": float(speedup),
            "max_depth_relative_l2": field_errors["relative_l2"],
            "max_depth_rmse_m": field_errors["rmse"],
            "peak_depth_m": outcome.peak_depth_m,
            "peak_depth_error_m": outcome.peak_depth_m - reference.peak_depth_m,
            "inundated_area_km2": outcome.inundated_area_km2,
            "inundated_area_error_km2": outcome.inundated_area_km2 - reference.inundated_area_km2,
            "arrival_time_error_s": arrival_time_error(
                outcome.arrival_times_s, reference.arrival_times_s
            ),
            "backend": outcome.meta.get("backend"),
        }
    return report


def benchmark_table(report: dict[str, Any]) -> str:
    """Render the benchmark as a fixed-width table for the run log and README."""
    lines = [
        f"{'method':<10} {'wall time [s]':>14} {'speedup':>10} "
        f"{'rel L2 (max depth)':>20} {'peak depth [m]':>16}",
        "-" * 74,
    ]
    reference = report["reference"]
    lines.append(
        f"{'solver':<10} {reference['wall_time_s']:>14.3f} {'1.0x':>10} "
        f"{'-':>20} {reference['peak_depth_m']:>16.3f}"
    )
    for name, entry in report["comparisons"].items():
        lines.append(
            f"{name:<10} {entry['wall_time_s']:>14.3f} "
            f"{entry['speedup_vs_solver']:>9.1f}x {entry['max_depth_relative_l2']:>20.4f} "
            f"{entry['peak_depth_m']:>16.3f}"
        )
    return "\n".join(lines)


def timing_breakdown(timings: dict[str, float]) -> str:
    """Stage-by-stage wall-clock table for the run manifest."""
    total = sum(timings.values())
    lines = [f"{'stage':<28} {'seconds':>10} {'share':>8}", "-" * 48]
    for stage, seconds in sorted(timings.items(), key=lambda item: -item[1]):
        share = 100.0 * seconds / max(total, 1e-9)
        lines.append(f"{stage:<28} {seconds:>10.3f} {share:>7.1f}%")
    lines.append("-" * 48)
    lines.append(f"{'total':<28} {total:>10.3f} {100.0:>7.1f}%")
    return "\n".join(lines)


def ensemble_statistics(values: np.ndarray) -> dict[str, float]:
    """Mean, spread and 10th/90th percentiles of an ensemble quantity."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
    }
