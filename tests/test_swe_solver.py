"""Verification of the shallow-water solver against analytic benchmarks.

Three properties are checked because each catches a different class of bug:
well-balancedness (bed-slope source term), mass conservation (flux bookkeeping)
and the Ritter dam break (wave speeds and the wet/dry front).
"""

from __future__ import annotations

import numpy as np
import pytest

from glof_pipeline.physics.swe_solver import ShallowWaterSolver, hydrograph_source, ritter_dam_break


def test_lake_at_rest_stays_at_rest(rng) -> None:
    """Well-balanced property: a flat water surface over rough bed generates no flow."""
    bed = (
        2.0 * np.sin(np.arange(40) / 5.0)[None, :]
        + 1.5 * np.cos(np.arange(40) / 7.0)[:, None]
        + rng.normal(0.0, 0.3, (40, 40))
    )
    depth = (bed.max() + 3.0) - bed
    solver = ShallowWaterSolver(bed, dx=10.0, manning_n=0.0, boundary="wall", cfl=0.4)
    h, hu, hv = depth.copy(), np.zeros_like(depth), np.zeros_like(depth)
    for _ in range(30):
        h, hu, hv, _ = solver.step(h, hu, hv, solver.time_step(h, hu, hv))
    assert np.max(np.abs(hu)) < 1e-10
    assert np.max(np.abs(hv)) < 1e-10
    assert np.allclose(h, depth, atol=1e-12)


def test_closed_basin_conserves_mass(rng) -> None:
    bed = rng.normal(0.0, 0.5, (32, 32))
    depth = np.clip(3.0 - bed, 0.0, None)
    solver = ShallowWaterSolver(bed, dx=20.0, manning_n=0.03, boundary="wall", cfl=0.4)
    result = solver.run(depth, duration_s=120.0, output_interval_s=30.0)
    relative_change = abs(result.volume_m3[-1] - result.volume_m3[0]) / result.volume_m3[0]
    assert relative_change < 1e-10
    assert result.mass_conservation_error() < 1e-10


def test_ritter_dam_break_matches_the_analytic_solution() -> None:
    n, length = 400, 2000.0
    dx = length / n
    bed = np.zeros((3, n))
    depth = np.zeros((3, n))
    depth[:, : n // 2] = 10.0
    solver = ShallowWaterSolver(bed, dx, manning_n=0.0, boundary="outflow", cfl=0.4, dry_depth=1e-4)
    result = solver.run(depth, duration_s=20.0, output_interval_s=20.0)

    x = (np.arange(n) + 0.5) * dx - length / 2.0
    exact = ritter_dam_break(x, 20.0, 10.0)
    numerical = result.depth[-1, 1, :]
    # First-order Godunov: a few per cent of the upstream depth is the expected error.
    assert np.abs(numerical - exact).mean() / 10.0 < 0.02
    # The rarefaction fan reaches the correct upstream limit.
    assert numerical[0] == pytest.approx(10.0, abs=1e-6)


def test_wet_dry_front_stays_physical() -> None:
    bed = np.tile(np.linspace(10.0, 0.0, 48), (48, 1))
    depth = np.zeros((48, 48))
    depth[:, :6] = 4.0
    solver = ShallowWaterSolver(bed, dx=25.0, manning_n=0.05, boundary="outflow", cfl=0.4)
    result = solver.run(depth, duration_s=300.0, output_interval_s=60.0)
    assert np.all(np.isfinite(result.depth))
    assert np.all(result.depth >= 0.0)


def test_hydrograph_source_injects_the_prescribed_volume() -> None:
    shape = (24, 24)
    cells = np.array([[5, 12]])
    time_s = np.array([0.0, 100.0, 200.0])
    discharge = np.array([0.0, 50.0, 0.0])
    source = hydrograph_source(shape, cells, time_s, discharge, cell_area=100.0)
    total = sum(source(t, 1.0).sum() * 100.0 for t in np.arange(0.0, 200.0, 1.0))
    expected = float(np.trapezoid(discharge, time_s))
    assert total == pytest.approx(expected, rel=0.02)
