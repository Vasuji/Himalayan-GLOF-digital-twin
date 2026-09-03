"""Diagnostic figures for a pipeline run."""

from __future__ import annotations

from typing import Any

import numpy as np


def _figure(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] = (7.0, 5.0)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)


def plot_terrain(terrain, sensors=None):
    """Bed elevation with the lake, moraine footprint and instrument positions."""
    fig, ax = _figure(figsize=(7.0, 6.0))
    extent = [terrain.x[0], terrain.x[-1], terrain.y[-1], terrain.y[0]]
    image = ax.imshow(terrain.z, cmap="terrain", extent=extent, aspect="equal")
    fig.colorbar(image, ax=ax, label="bed elevation (m a.s.l.)")
    ax.contour(terrain.x, terrain.y, terrain.lake_mask.astype(float), levels=[0.5], colors="tab:blue")
    ax.contour(terrain.x, terrain.y, terrain.moraine_mask.astype(float), levels=[0.5], colors="tab:red")
    if sensors is not None:
        for label, cells, marker in (
            ("piezometers", sensors.piezometers, "^"),
            ("stage gauges", sensors.stage_gauges, "o"),
            ("stream gauges", sensors.stream_gauges, "s"),
        ):
            if cells.size:
                ax.scatter(
                    terrain.x[cells[:, 1]], terrain.y[cells[:, 0]],
                    marker=marker, s=22, edgecolor="k", linewidth=0.4, label=label,
                )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("downvalley distance (m)")
    ax.set_title("Synthetic moraine-dammed valley\nblue: lake, red: moraine mesh footprint")
    return fig


def plot_forcing(forcings, time_h):
    """Ensemble catchment temperature, inflow and lake filling."""
    fig, axes = _figure(3, 1, figsize=(7.5, 8.0))
    for forcing in forcings:
        axes[0].plot(time_h, forcing.temperature_c, color="tab:red", alpha=0.35, linewidth=1.0)
        axes[1].plot(time_h, forcing.inflow_m3_per_s, color="tab:blue", alpha=0.35, linewidth=1.0)
        axes[2].plot(time_h, forcing.cumulative_pdd_c_day, color="tab:green", alpha=0.35, linewidth=1.0)
    axes[0].set_ylabel("catchment T2M (degC)")
    axes[1].set_ylabel("net lake inflow (m3/s)")
    axes[2].set_ylabel("cumulative PDD (degC d)")
    axes[2].set_xlabel("lead time (h)")
    axes[0].set_title(f"Downscaled forcing, {len(forcings)} ensemble members")
    for ax in axes:
        ax.grid(alpha=0.3)
    return fig


def plot_stability(assessment, terrain, graph):
    """Factor-of-safety trajectory per member and the final field on the mesh."""
    fig, axes = _figure(1, 2, figsize=(11.0, 4.6))
    for outcome in assessment.members:
        axes[0].plot(outcome.min_factor_of_safety, alpha=0.5, linewidth=1.0,
                     color="tab:red" if outcome.breached else "tab:grey")
    axes[0].axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="FS = 1")
    axes[0].set_xlabel("forecast step")
    axes[0].set_ylabel("minimum factor of safety")
    axes[0].set_title(f"Breach probability {assessment.breach_probability:.2f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    representative = assessment.members[assessment.representative_member].final_state
    scatter = axes[1].scatter(
        graph.node_xyz[:, 0], graph.node_xyz[:, 1],
        c=representative.factor_of_safety, cmap="RdYlGn", vmin=0.5, vmax=2.0, s=18,
    )
    fig.colorbar(scatter, ax=axes[1], label="factor of safety")
    axes[1].set_xlabel("easting (m)")
    axes[1].set_ylabel("downvalley distance (m)")
    axes[1].set_title(f"Moraine mesh, mechanism: {representative.mechanism}")
    return fig


def plot_breach_hydrograph(breach):
    """Outflow hydrograph, lake stage and breach growth."""
    fig, axes = _figure(2, 1, figsize=(7.5, 6.0))
    hours = breach.time_s / 3600.0
    axes[0].plot(hours, breach.discharge_m3_per_s, color="tab:blue")
    axes[0].axhline(
        breach.froehlich_peak_estimate_m3_per_s, color="tab:orange", linestyle="--",
        label=f"Froehlich (1995) peak estimate {breach.froehlich_peak_estimate_m3_per_s:.0f} m3/s",
    )
    axes[0].set_ylabel("breach outflow (m3/s)")
    axes[0].set_title(
        f"{breach.mechanism} breach: peak {breach.peak_discharge_m3_per_s:.0f} m3/s, "
        f"{breach.released_volume_m3 / 1e6:.1f} Mm3 released"
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(hours, breach.lake_level_m, color="tab:green", label="lake stage")
    axes[1].plot(hours, breach.breach_invert_m, color="tab:red", label="breach invert")
    axes[1].set_xlabel("time since breach initiation (h)")
    axes[1].set_ylabel("elevation (m a.s.l.)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    return fig


def plot_flood_snapshots(terrain, outcome, n_snapshots: int = 4, depths: np.ndarray | None = None):
    """Depth envelope plus a few snapshots down the valley."""
    fig, axes = _figure(1, n_snapshots + 1, figsize=(3.2 * (n_snapshots + 1), 3.8))
    extent = [terrain.x[0], terrain.x[-1], terrain.y[-1], terrain.y[0]]
    envelope = np.where(outcome.max_depth_m > 0.05, outcome.max_depth_m, np.nan)
    image = axes[0].imshow(envelope, cmap="Blues", extent=extent, aspect="equal", vmin=0.0)
    fig.colorbar(image, ax=axes[0], label="max depth (m)")
    axes[0].set_title(f"envelope ({outcome.method})")

    if depths is not None:
        indices = np.linspace(0, depths.shape[0] - 1, n_snapshots).astype(int)
        for panel, index in enumerate(indices, start=1):
            frame = np.where(depths[index] > 0.05, depths[index], np.nan)
            axes[panel].imshow(frame, cmap="Blues", extent=extent, aspect="equal",
                               vmin=0.0, vmax=float(np.nanmax(envelope)))
            axes[panel].set_title(f"t = {outcome.time_s[index] / 60.0:.0f} min")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def plot_benchmark(report: dict[str, Any]):
    """Wall time against accuracy for every routing method."""
    fig, ax = _figure(figsize=(6.5, 4.5))
    reference = report["reference"]
    ax.scatter([reference["wall_time_s"]], [0.0], marker="*", s=180, color="k", label="solver (reference)")
    for name, entry in report["comparisons"].items():
        ax.scatter([entry["wall_time_s"]], [entry["max_depth_relative_l2"]], s=90,
                   label=f"{name} ({entry['speedup_vs_solver']:.0f}x)")
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock time (s, log scale)")
    ax.set_ylabel("relative L2 of the depth envelope")
    ax.set_title("Measured cost and accuracy of flood routing")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig
