"""Moraine breach formation and the outflow hydrograph.

Breach geometry follows Froehlich's (2008) regressions on 74 documented
embankment failures, which are the standard empirical basis for dam-breach
parameter estimation:

* average breach width  ``B = 0.27 k_o V_w^0.32 h_b^0.04``
* formation time        ``t_f = 63.2 sqrt(V_w / (g h_b^2))``

with ``k_o = 1.3`` for overtopping and 1.0 otherwise, reservoir volume ``V_w``
[m^3] and breach height ``h_b`` [m]. Outflow through the widening trapezoidal
breach uses the broad-crested weir relation of the DAMBRK/HEC family,

``Q = C_r b H^{3/2} + C_t s H^{5/2}``,

and the lake is depleted against its own stage-storage curve, so the integral of
the hydrograph equals the released volume by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from glof_pipeline.terrain.synthetic_dem import ValleyTerrain, lake_hypsometry, level_from_volume

GRAVITY = 9.81


@dataclass
class BreachResult:
    """Outflow hydrograph and breach geometry evolution."""

    time_s: np.ndarray
    discharge_m3_per_s: np.ndarray
    lake_level_m: np.ndarray
    breach_width_m: np.ndarray
    breach_invert_m: np.ndarray
    peak_discharge_m3_per_s: float
    time_to_peak_s: float
    released_volume_m3: float
    formation_time_s: float
    final_breach_width_m: float
    breach_height_m: float
    mechanism: str
    froehlich_peak_estimate_m3_per_s: float

    def as_summary(self) -> dict[str, Any]:
        return {
            "peak_discharge_m3_per_s": self.peak_discharge_m3_per_s,
            "time_to_peak_s": self.time_to_peak_s,
            "released_volume_m3": self.released_volume_m3,
            "formation_time_s": self.formation_time_s,
            "final_breach_width_m": self.final_breach_width_m,
            "breach_height_m": self.breach_height_m,
            "mechanism": self.mechanism,
            "froehlich_peak_estimate_m3_per_s": self.froehlich_peak_estimate_m3_per_s,
        }

    def mass_balance_error(self) -> float:
        """Relative difference between the integrated hydrograph and released volume."""
        integrated = float(np.trapezoid(self.discharge_m3_per_s, self.time_s))
        if self.released_volume_m3 <= 0.0:
            return 0.0
        return abs(integrated - self.released_volume_m3) / self.released_volume_m3


def froehlich_geometry(
    volume_m3: float, breach_height_m: float, overtopping_coefficient: float
) -> tuple[float, float]:
    """Return ``(average_breach_width_m, formation_time_s)`` (Froehlich 2008)."""
    volume_m3 = max(float(volume_m3), 1.0)
    breach_height_m = max(float(breach_height_m), 1.0)
    width = 0.27 * overtopping_coefficient * volume_m3**0.32 * breach_height_m**0.04
    formation_time = 63.2 * np.sqrt(volume_m3 / (GRAVITY * breach_height_m**2))
    return float(width), float(formation_time)


def froehlich_peak_discharge(volume_m3: float, water_depth_m: float) -> float:
    """Independent peak-discharge regression, used only as a sanity check.

    ``Q_p = 0.607 V_w^0.295 h_w^1.24`` (Froehlich 1995, metric units).
    """
    return float(0.607 * max(volume_m3, 1.0) ** 0.295 * max(water_depth_m, 1.0) ** 1.24)


def weir_discharge(
    head_m: float,
    bottom_width_m: float,
    side_slope: float,
    coefficient_rectangular: float,
    coefficient_triangular: float,
) -> float:
    """Trapezoidal broad-crested weir discharge [m^3 s^-1]."""
    head = max(float(head_m), 0.0)
    if head <= 0.0 or bottom_width_m <= 0.0:
        return 0.0
    rectangular = coefficient_rectangular * bottom_width_m * head**1.5
    triangular = coefficient_triangular * side_slope * head**2.5
    return float(rectangular + triangular)


def simulate_breach(
    terrain: ValleyTerrain,
    lake_level_m: float,
    mechanism: str,
    cfg: dict[str, Any],
    inflow_m3_per_s: float = 0.0,
) -> BreachResult:
    """Integrate the breach outflow until the lake drains to the breach invert.

    The breach deepens and widens linearly over the Froehlich formation time; at
    each step the weir head is taken from the current lake stage, and the withdrawn
    volume is converted back to a stage through the DEM-derived stage-storage curve.
    """
    levels, _, volumes = lake_hypsometry(terrain)

    invert_target = max(
        terrain.crest_elevation - float(cfg["min_breach_depth_m"]),
        float(terrain.meta.get("lake_bed_min_m", levels[0])),
    )
    breach_height = float(terrain.crest_elevation - invert_target)
    if breach_height <= 0.0:
        raise ValueError("Non-positive breach height; check the crest and lake-bed elevations.")

    initial_volume = float(np.interp(lake_level_m, levels, volumes))
    k_o = float(cfg["froehlich_overtopping_coefficient"]) if mechanism == "overtopping" else 1.0
    final_width, formation_time = froehlich_geometry(initial_volume, breach_height, k_o)

    side_slope = float(cfg["side_slope_h_per_v"])
    c_rect = float(cfg["weir_coefficient_rectangular"])
    c_tri = float(cfg["weir_coefficient_triangular"])
    dt = float(cfg["time_step_s"])
    max_duration = float(cfg["max_duration_s"])

    times: list[float] = [0.0]
    discharges: list[float] = [0.0]
    stages: list[float] = [float(lake_level_m)]
    widths: list[float] = [0.0]
    inverts: list[float] = [float(terrain.crest_elevation)]

    volume = initial_volume
    level = float(lake_level_m)
    t = 0.0
    residual_volume = float(np.interp(invert_target, levels, volumes))

    while t < max_duration:
        progress = min(1.0, (t + dt) / max(formation_time, dt))
        invert = terrain.crest_elevation - breach_height * progress
        width = final_width * progress
        head = level - invert
        discharge = weir_discharge(head, width, side_slope, c_rect, c_tri)

        volume = max(residual_volume, volume - (discharge - inflow_m3_per_s) * dt)
        level = level_from_volume(levels, volumes, volume)
        t += dt

        times.append(t)
        discharges.append(discharge)
        stages.append(level)
        widths.append(width)
        inverts.append(invert)

        drained = (volume - residual_volume) <= 1.0e-6 * max(initial_volume, 1.0)
        if progress >= 1.0 and (head <= 0.02 or drained):
            break

    time_s = np.asarray(times)
    discharge_series = np.asarray(discharges)
    peak_index = int(np.argmax(discharge_series))

    return BreachResult(
        time_s=time_s,
        discharge_m3_per_s=discharge_series,
        lake_level_m=np.asarray(stages),
        breach_width_m=np.asarray(widths),
        breach_invert_m=np.asarray(inverts),
        peak_discharge_m3_per_s=float(discharge_series[peak_index]),
        time_to_peak_s=float(time_s[peak_index]),
        released_volume_m3=float(initial_volume - volume),
        formation_time_s=float(formation_time),
        final_breach_width_m=float(final_width),
        breach_height_m=breach_height,
        mechanism=mechanism,
        froehlich_peak_estimate_m3_per_s=froehlich_peak_discharge(
            initial_volume, float(lake_level_m - invert_target)
        ),
    )
