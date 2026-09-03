"""Moraine stability, breach hydraulics and catchment mass balance."""

from __future__ import annotations

import numpy as np
import pytest

from glof_pipeline.physics.breach import (
    froehlich_geometry,
    froehlich_peak_discharge,
    simulate_breach,
    weir_discharge,
)
from glof_pipeline.physics.mass_balance import (
    degree_day_melt,
    integrate_catchment,
    partition_precipitation,
)
from glof_pipeline.physics.moraine_stability import (
    critical_hydraulic_gradient,
    evaluate_stability,
    infinite_slope_fos,
    phreatic_surface,
)


# --- moraine stability -----------------------------------------------------
def test_cohesionless_dry_slope_reduces_to_the_closed_form() -> None:
    """With c' = 0 and no water table, FS must equal tan(phi)/tan(beta) exactly."""
    beta = np.deg2rad(np.array([10.0, 20.0, 30.0, 40.0]))
    phi = 35.0
    fos = infinite_slope_fos(beta, 20.0, 0.0, 0.0, phi, 20.0, 9.81)
    assert np.allclose(fos, np.tan(np.deg2rad(phi)) / np.tan(beta), rtol=1e-12)


def test_pore_pressure_reduces_stability_monotonically() -> None:
    beta = np.deg2rad(28.0)
    heads = np.linspace(0.0, 20.0, 6)
    fos = infinite_slope_fos(beta, 20.0, heads, 5.0, 35.0, 20.0, 9.81)
    assert np.all(np.diff(fos) < 0.0)


def test_critical_gradient_matches_terzaghi() -> None:
    assert critical_hydraulic_gradient(2.68, 0.55) == pytest.approx((2.68 - 1.0) / 1.55)


def test_phreatic_surface_decays_from_face_to_toe() -> None:
    """Head is measured above the slip plane, 25 m below a ground surface at 110 m."""
    head = phreatic_surface(
        lake_level_m=100.0,
        node_elevation_m=np.array([110.0, 110.0, 110.0]),
        distance_to_lake_m=np.array([0.0, 150.0, 300.0]),
        seepage_length_m=300.0,
        slab_depth_m=25.0,
    )
    assert head[0] == pytest.approx(15.0)
    assert head[1] == pytest.approx(7.5)
    assert head[2] == pytest.approx(0.0)


def test_phreatic_head_is_capped_by_the_slab_depth() -> None:
    """A fully submerged slab saturates; it cannot hold more water than its own depth."""
    head = phreatic_surface(200.0, np.array([120.0]), np.array([0.0]), 300.0, 25.0)
    assert head[0] == pytest.approx(25.0)


def test_rising_lake_lowers_the_factor_of_safety(terrain, graph, config) -> None:
    moraine_cfg = config.get("moraine")
    low = evaluate_stability(terrain, graph, terrain.initial_lake_level - 12.0, 0.0, moraine_cfg)
    high = evaluate_stability(terrain, graph, terrain.initial_lake_level, 0.0, moraine_cfg)
    assert high.min_factor_of_safety < low.min_factor_of_safety
    assert high.piping_ratio > low.piping_ratio


def test_overtopping_is_detected_above_the_crest(terrain, graph, config) -> None:
    field = evaluate_stability(terrain, graph, terrain.crest_elevation + 0.5, 0.0, config.get("moraine"))
    assert field.breached
    assert field.mechanism == "overtopping"
    assert field.freeboard_m < 0.0


def test_thermal_degradation_weakens_the_dam(terrain, graph, config) -> None:
    moraine_cfg = config.get("moraine")
    cold = evaluate_stability(terrain, graph, terrain.initial_lake_level, 0.0, moraine_cfg)
    warm = evaluate_stability(terrain, graph, terrain.initial_lake_level, 400.0, moraine_cfg)
    assert warm.min_factor_of_safety < cold.min_factor_of_safety


# --- breach ----------------------------------------------------------------
def test_froehlich_geometry_scales_with_volume() -> None:
    small_width, small_time = froehlich_geometry(5.0e6, 40.0, 1.3)
    large_width, large_time = froehlich_geometry(50.0e6, 40.0, 1.3)
    assert large_width > small_width
    assert large_time > small_time
    # Published example magnitudes: tens to a few hundred metres, minutes to hours.
    assert 10.0 < small_width < 500.0
    assert 60.0 < small_time < 20000.0


def test_weir_discharge_increases_with_head_and_width() -> None:
    assert weir_discharge(2.0, 30.0, 1.0, 1.7, 1.35) > weir_discharge(1.0, 30.0, 1.0, 1.7, 1.35)
    assert weir_discharge(2.0, 60.0, 1.0, 1.7, 1.35) > weir_discharge(2.0, 30.0, 1.0, 1.7, 1.35)
    assert weir_discharge(0.0, 30.0, 1.0, 1.7, 1.35) == 0.0


def test_breach_hydrograph_conserves_the_released_volume(terrain, config) -> None:
    breach = simulate_breach(terrain, terrain.crest_elevation, "overtopping", config.get("breach"))
    assert breach.mass_balance_error() < 0.02
    assert breach.peak_discharge_m3_per_s > 0.0
    assert breach.released_volume_m3 > 0.0
    assert 0.0 <= breach.time_to_peak_s <= breach.time_s[-1]


def test_breach_peak_is_within_an_order_of_magnitude_of_the_regression(terrain, config) -> None:
    breach = simulate_breach(terrain, terrain.crest_elevation, "overtopping", config.get("breach"))
    estimate = froehlich_peak_discharge(
        terrain.lake_volume(terrain.crest_elevation), breach.breach_height_m
    )
    ratio = breach.peak_discharge_m3_per_s / estimate
    assert 0.2 < ratio < 5.0, f"peak {breach.peak_discharge_m3_per_s:.0f} vs regression {estimate:.0f}"


# --- mass balance ----------------------------------------------------------
def test_degree_day_melt_is_zero_below_threshold_and_linear_above() -> None:
    assert degree_day_melt(-3.0, 7.0, 0.0) == 0.0
    assert degree_day_melt(0.0, 7.0, 0.0) == 0.0
    assert degree_day_melt(2.0, 7.0, 0.0) == pytest.approx(14.0)
    assert degree_day_melt(4.0, 7.0, 0.0) == pytest.approx(2.0 * degree_day_melt(2.0, 7.0, 0.0))


def test_precipitation_partition_uses_the_threshold() -> None:
    rain, snow = partition_precipitation(
        np.array([5.0, 5.0]), np.array([3.0, -3.0]), rain_snow_threshold_c=1.0
    )
    assert rain.tolist() == [5.0, 0.0]
    assert snow.tolist() == [0.0, 5.0]


def test_warm_forcing_fills_the_lake(config) -> None:
    time_h = np.arange(0.0, 168.0, 6.0)
    warm = integrate_catchment(time_h, np.full_like(time_h, 6.0), np.full_like(time_h, 0.5), config.get("glaciology"))
    cold = integrate_catchment(time_h, np.full_like(time_h, -6.0), np.full_like(time_h, 0.5), config.get("glaciology"))
    assert warm.volume_gain_m3 > cold.volume_gain_m3
    assert warm.cumulative_pdd_c_day[-1] > 0.0
    assert cold.cumulative_pdd_c_day[-1] == pytest.approx(0.0)
    # The snowpack is exhausted under sustained melt.
    assert warm.snow_water_equivalent_mm[-1] < warm.snow_water_equivalent_mm[0]
