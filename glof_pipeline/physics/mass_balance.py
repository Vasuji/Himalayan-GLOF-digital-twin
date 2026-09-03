"""Catchment mass balance: degree-day melt, rain/snow partition, lake filling.

The degree-day (temperature-index) approach is the standard operational choice
for Himalayan catchments where the energy-balance terms needed for a full
surface-energy model are unobserved (see Hock, *J. Hydrol.* 282, 2003 for the
review of temperature-index melt modelling). The pipeline uses it to convert the
downscaled 1 km temperature and precipitation fields into a lake inflow series
and a cumulative positive-degree-day total, which is what drives both lake
filling and thermal degradation of the moraine's ice core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SECONDS_PER_DAY = 86400.0
SECONDS_PER_HOUR = 3600.0


@dataclass
class CatchmentForcing:
    """Result of integrating the downscaled forcing over the catchment."""

    time_h: np.ndarray            # (nt,) hours from forecast initialisation
    temperature_c: np.ndarray     # (nt,) catchment-mean 2 m temperature
    precipitation_mm_per_h: np.ndarray
    melt_mm_per_h: np.ndarray
    rain_mm_per_h: np.ndarray
    snow_water_equivalent_mm: np.ndarray
    inflow_m3_per_s: np.ndarray   # (nt,) net inflow to the lake
    cumulative_pdd_c_day: np.ndarray  # (nt,) positive degree-day sum
    volume_gain_m3: float

    def as_summary(self) -> dict[str, float]:
        return {
            "peak_inflow_m3_per_s": float(self.inflow_m3_per_s.max()),
            "mean_inflow_m3_per_s": float(self.inflow_m3_per_s.mean()),
            "total_melt_mm": float(np.trapezoid(self.melt_mm_per_h, self.time_h)),
            "total_rain_mm": float(np.trapezoid(self.rain_mm_per_h, self.time_h)),
            "cumulative_pdd_c_day": float(self.cumulative_pdd_c_day[-1]),
            "volume_gain_m3": float(self.volume_gain_m3),
        }


def degree_day_melt(
    temperature_c: np.ndarray | float,
    degree_day_factor: float,
    melt_threshold_c: float = 0.0,
) -> np.ndarray:
    """Melt rate in mm water equivalent per day.

    ``melt = DDF * max(T - T_melt, 0)``: zero at or below the threshold, linear above.
    """
    excess = np.clip(np.asarray(temperature_c, dtype=float) - melt_threshold_c, 0.0, None)
    return degree_day_factor * excess


def partition_precipitation(
    precipitation: np.ndarray, temperature_c: np.ndarray, rain_snow_threshold_c: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split precipitation into rain and snow with a single-threshold rule."""
    precipitation = np.asarray(precipitation, dtype=float)
    is_rain = np.asarray(temperature_c, dtype=float) > rain_snow_threshold_c
    return np.where(is_rain, precipitation, 0.0), np.where(is_rain, 0.0, precipitation)


def integrate_catchment(
    time_h: np.ndarray,
    temperature_c: np.ndarray,
    precipitation_mm_per_h: np.ndarray,
    cfg: dict[str, Any],
) -> CatchmentForcing:
    """Convert a forcing time series into lake inflow and a PDD total.

    Snow-covered ground melts with the snow degree-day factor until the pack is
    exhausted, after which the (larger) ice factor applies over the glacierised
    fraction of the catchment. Rain runs off directly. Both are routed to the lake
    with a lumped runoff coefficient, and the configured baseflow is subtracted as
    the outflow through the moraine.
    """
    time_h = np.asarray(time_h, dtype=float)
    temperature_c = np.asarray(temperature_c, dtype=float)
    precipitation_mm_per_h = np.asarray(precipitation_mm_per_h, dtype=float)
    if not (time_h.shape == temperature_c.shape == precipitation_mm_per_h.shape):
        raise ValueError("Forcing arrays must share one shape.")
    if time_h.size < 2:
        raise ValueError("Need at least two forcing samples to integrate.")

    ddf_ice = float(cfg["ddf_ice_mm_per_c_per_day"])
    ddf_snow = float(cfg["ddf_snow_mm_per_c_per_day"])
    melt_threshold = float(cfg["melt_threshold_c"])
    rain_threshold = float(cfg["rain_snow_threshold_c"])
    area_m2 = float(cfg["catchment_area_km2"]) * 1.0e6
    glacier_fraction = float(cfg["glacierised_fraction"])
    runoff_coefficient = float(cfg["runoff_coefficient"])
    baseflow = float(cfg["baseflow_m3_per_s"])

    swe = float(cfg["initial_snow_water_equivalent_mm"])
    rain, snow = partition_precipitation(precipitation_mm_per_h, temperature_c, rain_threshold)

    n = time_h.size
    melt_series = np.zeros(n)
    swe_series = np.zeros(n)
    dt_h = np.gradient(time_h)

    for i in range(n):
        swe += snow[i] * dt_h[i]
        # Snow melts first, at the snow factor, over the whole catchment.
        potential_snow_melt = degree_day_melt(temperature_c[i], ddf_snow, melt_threshold)
        potential_snow_melt = potential_snow_melt / 24.0 * dt_h[i]
        actual_snow_melt = min(potential_snow_melt, swe)
        swe -= actual_snow_melt
        # Exposed ice melts only where the pack has gone, over glacierised ground.
        exposure = 0.0 if swe > 1.0e-6 else 1.0
        ice_melt = degree_day_melt(temperature_c[i], ddf_ice, melt_threshold)
        ice_melt = ice_melt / 24.0 * dt_h[i] * glacier_fraction * exposure
        melt_series[i] = (actual_snow_melt + ice_melt) / max(dt_h[i], 1e-9)
        swe_series[i] = swe

    water_input_mm_per_h = melt_series + rain
    inflow_m3_per_s = (
        water_input_mm_per_h * 1.0e-3 * area_m2 * runoff_coefficient / SECONDS_PER_HOUR
    )
    net_inflow = inflow_m3_per_s - baseflow

    pdd_increment = np.clip(temperature_c - melt_threshold, 0.0, None) * dt_h / 24.0
    cumulative_pdd = np.cumsum(pdd_increment)

    volume_gain = float(np.trapezoid(net_inflow, time_h) * SECONDS_PER_HOUR)

    return CatchmentForcing(
        time_h=time_h,
        temperature_c=temperature_c,
        precipitation_mm_per_h=precipitation_mm_per_h,
        melt_mm_per_h=melt_series,
        rain_mm_per_h=rain,
        snow_water_equivalent_mm=swe_series,
        inflow_m3_per_s=net_inflow,
        cumulative_pdd_c_day=cumulative_pdd,
        volume_gain_m3=volume_gain,
    )
