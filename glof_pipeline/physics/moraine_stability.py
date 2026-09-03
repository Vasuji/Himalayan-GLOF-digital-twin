r"""Moraine-dam stability by effective-stress limit equilibrium.

The draft architecture triggered a breach when a MeshGraphNet's scalar output
crossed a fixed "yield stress in kPa". That is not a failure criterion: a moraine
fails when the mobilised shear stress on a potential slip surface exceeds the
Mohr-Coulomb shear strength *at the prevailing pore pressure*, or when seepage
gradients exceed the critical gradient for internal erosion, or when the lake
overtops the crest. This module implements all three, and it is these fields that
the MeshGraphNet is trained to reproduce.

Infinite-slope limit equilibrium with a phreatic surface (the standard form, e.g.
Duncan, Wright & Brandon, *Soil Strength and Slope Stability*):

.. math::

    FS = \frac{c' + (\gamma_{sat} z - \gamma_w h_w)\cos^2\beta \tan\phi'}
              {\gamma_{sat} z \sin\beta \cos\beta}

with slab depth :math:`z`, slope angle :math:`\beta`, phreatic height above the
slip plane :math:`h_w`, effective cohesion :math:`c'` and friction angle
:math:`\phi'`. Stresses are in kPa throughout (kN m^-3 times m).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from glof_pipeline.terrain.mesh_builder import MoraineGraph, distance_to_lake, slope_and_aspect
from glof_pipeline.terrain.synthetic_dem import ValleyTerrain

MIN_SLOPE_RAD = np.deg2rad(1.0)
MAX_SLOPE_RAD = np.deg2rad(60.0)


@dataclass
class StabilityField:
    """Per-node stability state of the moraine plus the dam-scale triggers."""

    factor_of_safety: np.ndarray      # (N,)
    shear_stress_kpa: np.ndarray      # (N,) mobilised
    shear_strength_kpa: np.ndarray    # (N,)
    pore_pressure_ratio: np.ndarray   # (N,) r_u
    phreatic_height_m: np.ndarray     # (N,)
    slope_rad: np.ndarray             # (N,)
    min_factor_of_safety: float
    failing_fraction: float           # fraction of nodes with FS < 1
    piping_ratio: float               # backward-erosion gradient / critical gradient
    heave_ratio: float                # Terzaghi heave gradient ratio
    seepage_gradient: float           # average gradient across the concentrated path
    freeboard_m: float
    breached: bool
    mechanism: str                    # overtopping | piping | slope_failure | stable
    lake_level_m: float

    def as_summary(self) -> dict[str, Any]:
        return {
            "min_factor_of_safety": self.min_factor_of_safety,
            "failing_node_fraction": self.failing_fraction,
            "piping_ratio": self.piping_ratio,
            "heave_ratio": self.heave_ratio,
            "seepage_gradient": self.seepage_gradient,
            "freeboard_m": self.freeboard_m,
            "breached": self.breached,
            "mechanism": self.mechanism,
            "lake_level_m": self.lake_level_m,
        }


def infinite_slope_fos(
    slope_rad: np.ndarray | float,
    slab_depth_m: np.ndarray | float,
    phreatic_height_m: np.ndarray | float,
    cohesion_kpa: np.ndarray | float,
    friction_angle_deg: float,
    unit_weight_saturated: float,
    unit_weight_water: float,
) -> np.ndarray:
    """Factor of safety of an infinite slope with a phreatic surface.

    With ``cohesion_kpa = 0`` and ``phreatic_height_m = 0`` this reduces exactly to
    ``tan(phi) / tan(beta)``, which is the analytic check in the test suite.
    """
    beta = np.clip(np.asarray(slope_rad, dtype=float), MIN_SLOPE_RAD, MAX_SLOPE_RAD)
    z = np.asarray(slab_depth_m, dtype=float)
    h_w = np.clip(np.asarray(phreatic_height_m, dtype=float), 0.0, None)
    h_w = np.minimum(h_w, z)

    tan_phi = np.tan(np.deg2rad(friction_angle_deg))
    normal_effective = (unit_weight_saturated * z - unit_weight_water * h_w) * np.cos(beta) ** 2
    normal_effective = np.clip(normal_effective, 0.0, None)
    strength = np.asarray(cohesion_kpa, dtype=float) + normal_effective * tan_phi
    mobilised = unit_weight_saturated * z * np.sin(beta) * np.cos(beta)
    return strength / np.clip(mobilised, 1e-9, None)


def critical_hydraulic_gradient(specific_gravity: float, void_ratio: float) -> float:
    """Terzaghi critical gradient ``i_c = (G_s - 1) / (1 + e)`` for upward-seepage heave."""
    return (specific_gravity - 1.0) / (1.0 + void_ratio)


def phreatic_surface(
    lake_level_m: float,
    node_elevation_m: np.ndarray,
    distance_to_lake_m: np.ndarray,
    seepage_length_m: float,
    slab_depth_m: np.ndarray | float,
) -> np.ndarray:
    """Linear (Dupuit) phreatic surface decaying from the upstream face to the toe.

    The head is measured **above the slip plane**, which sits ``slab_depth_m`` below
    the ground surface, not above the ground surface itself. That distinction is what
    lets the lake pressurise a slab whose surface stands a few metres proud of the
    water line -- exactly the near-crest cells that fail first as a lake fills.

    A linear drawdown is the classical Dupuit approximation for steady seepage through
    an embankment; replacing it with a Richards-equation solution is the obvious
    refinement once piezometer records exist for the site.
    """
    slab = np.asarray(slab_depth_m, dtype=float)
    slip_plane = np.asarray(node_elevation_m, dtype=float) - slab
    head_at_face = np.clip(lake_level_m - slip_plane, 0.0, None)
    decay = np.clip(
        1.0 - np.asarray(distance_to_lake_m, dtype=float) / max(seepage_length_m, 1e-9), 0.0, 1.0
    )
    return np.minimum(head_at_face * decay, slab)


def degraded_cohesion(
    reference_cohesion_kpa: float,
    cumulative_pdd_c_day: float,
    ice_core_fraction: float,
    degradation_per_pdd: float,
) -> float:
    """Cohesion after thermal degradation of the moraine's ice core.

    Buried ice supplies part of the apparent cohesion of an ice-cored moraine; as
    positive degree-days accumulate it melts out and that contribution is lost.
    The linear law here is a placeholder for a site-calibrated relation.
    """
    loss = ice_core_fraction * degradation_per_pdd * max(cumulative_pdd_c_day, 0.0)
    return float(np.clip(reference_cohesion_kpa * (1.0 - loss), 0.1, None))


def evaluate_stability(
    terrain: ValleyTerrain,
    graph: MoraineGraph,
    lake_level_m: float,
    cumulative_pdd_c_day: float,
    cfg: dict[str, Any],
    till_depth_m: np.ndarray | None = None,
    cohesion_kpa: float | None = None,
) -> StabilityField:
    """Evaluate the three failure mechanisms on the moraine mesh."""
    beta_grid, _ = slope_and_aspect(terrain.z, terrain.dx)
    rows, cols = graph.node_rc[:, 0], graph.node_rc[:, 1]
    beta = beta_grid[rows, cols]
    node_z = graph.node_xyz[:, 2]

    reference_depth = float(cfg["till_depth_m"])
    depth = np.full(beta.shape, reference_depth) if till_depth_m is None else np.asarray(till_depth_m, float)

    seepage_length = float(cfg["seepage_path_length_m"])
    distance = distance_to_lake(terrain, graph)
    h_w = phreatic_surface(lake_level_m, node_z, distance, seepage_length, depth)

    c_eff = (
        degraded_cohesion(
            float(cfg["cohesion_kpa"]),
            cumulative_pdd_c_day,
            float(cfg["ice_core_fraction"]),
            float(cfg["thermal_degradation_per_pdd"]),
        )
        if cohesion_kpa is None
        else float(cohesion_kpa)
    )

    gamma_sat = float(cfg["unit_weight_saturated_kn_per_m3"])
    gamma_w = float(cfg["unit_weight_water_kn_per_m3"])
    phi = float(cfg["friction_angle_deg"])

    fos = infinite_slope_fos(beta, depth, h_w, c_eff, phi, gamma_sat, gamma_w)

    beta_clipped = np.clip(beta, MIN_SLOPE_RAD, MAX_SLOPE_RAD)
    mobilised = gamma_sat * depth * np.sin(beta_clipped) * np.cos(beta_clipped)
    normal_effective = np.clip(
        (gamma_sat * depth - gamma_w * h_w) * np.cos(beta_clipped) ** 2, 0.0, None
    )
    strength = c_eff + normal_effective * np.tan(np.deg2rad(phi))
    r_u = gamma_w * h_w * np.cos(beta_clipped) ** 2 / np.clip(gamma_sat * depth, 1e-9, None)

    # Dam-scale checks. The driving head is measured from the lake surface to the
    # downstream toe; the concentrated leak path is a fraction of the dam base width.
    toe_elevation = float(np.percentile(node_z, 5.0))
    head = max(lake_level_m - toe_elevation, 0.0)
    heave_gradient = head / max(seepage_length, 1e-9)
    path_fraction = float(cfg.get("piping_path_fraction", 1.0))
    erosion_gradient = head / max(seepage_length * path_fraction, 1e-9)

    i_heave = critical_hydraulic_gradient(
        float(cfg["specific_gravity_solids"]), float(cfg["void_ratio"])
    )
    i_erosion = float(cfg.get("backward_erosion_critical_gradient", i_heave))
    heave_ratio = float(heave_gradient / i_heave)
    piping_ratio = float(erosion_gradient / i_erosion)
    freeboard = float(terrain.crest_elevation - lake_level_m)

    triggers = cfg["triggers"]
    min_fos = float(np.min(fos))
    mechanism = "stable"
    if freeboard <= float(triggers["overtopping_freeboard_m"]):
        mechanism = "overtopping"
    elif piping_ratio >= float(triggers["piping_gradient_ratio"]):
        mechanism = "piping"
    elif min_fos < float(triggers["min_factor_of_safety"]):
        mechanism = "slope_failure"

    return StabilityField(
        factor_of_safety=fos,
        shear_stress_kpa=mobilised,
        shear_strength_kpa=strength,
        pore_pressure_ratio=r_u,
        phreatic_height_m=h_w,
        slope_rad=beta,
        min_factor_of_safety=min_fos,
        failing_fraction=float(np.mean(fos < 1.0)),
        piping_ratio=piping_ratio,
        heave_ratio=heave_ratio,
        seepage_gradient=float(erosion_gradient),
        freeboard_m=freeboard,
        breached=mechanism != "stable",
        mechanism=mechanism,
        lake_level_m=float(lake_level_m),
    )
