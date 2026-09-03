"""Moraine assessment across the forecast ensemble.

The output that matters operationally is not a single factor of safety but a
*breach probability*: each ensemble member and each generative downscaling sample
gives a different lake filling history, and therefore a different stability state.
This stage evaluates the whole set and reports the fraction that fail, together
with the mechanism mix and the lead time to the first predicted failure.

When a trained MeshGraphNet is supplied it is used for the per-node stability
field and the reference limit-equilibrium model is retained as the label for
verification, so the two are always reported side by side rather than one
silently replacing the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glof_pipeline.physics.mass_balance import CatchmentForcing
from glof_pipeline.physics.moraine_stability import StabilityField, evaluate_stability
from glof_pipeline.surrogates.mgn_moraine import MoraineNodeState, MoraineOperator, assemble_node_features
from glof_pipeline.terrain.mesh_builder import MoraineGraph
from glof_pipeline.terrain.synthetic_dem import ValleyTerrain, lake_hypsometry, level_from_volume
from glof_pipeline.utils.runtime import get_logger

LOGGER = get_logger("glof.moraine")


@dataclass
class MemberOutcome:
    """Stability trajectory of a single ensemble member."""

    member: int
    lake_level_m: np.ndarray        # (nt,)
    min_factor_of_safety: np.ndarray
    breached: bool
    mechanism: str
    breach_time_h: float
    final_state: StabilityField


@dataclass
class MoraineAssessment:
    """Ensemble summary handed to the routing stage."""

    members: list[MemberOutcome]
    breach_probability: float
    mechanism_counts: dict[str, int]
    earliest_breach_h: float
    representative_member: int
    surrogate_comparison: dict[str, float] = field(default_factory=dict)

    def as_summary(self) -> dict[str, Any]:
        return {
            "breach_probability": self.breach_probability,
            "mechanism_counts": self.mechanism_counts,
            "earliest_breach_h": self.earliest_breach_h,
            "n_members": len(self.members),
            "representative_member": self.representative_member,
            "surrogate_comparison": self.surrogate_comparison,
        }


def _lake_level_trajectory(
    terrain: ValleyTerrain, forcing: CatchmentForcing, initial_level: float
) -> np.ndarray:
    """Integrate lake stage from the net inflow through the stage-storage curve."""
    levels, _, volumes = lake_hypsometry(terrain)
    volume = float(np.interp(initial_level, levels, volumes))
    trajectory = np.zeros(forcing.time_h.size)
    dt_s = np.gradient(forcing.time_h) * 3600.0
    for i in range(forcing.time_h.size):
        volume = max(volumes[0], volume + forcing.inflow_m3_per_s[i] * dt_s[i])
        trajectory[i] = level_from_volume(levels, volumes, volume)
    return trajectory


def assess_moraine(
    terrain: ValleyTerrain,
    graph: MoraineGraph,
    forcings: list[CatchmentForcing],
    moraine_cfg: dict[str, Any],
    operator: MoraineOperator | None = None,
) -> MoraineAssessment:
    """Evaluate stability along every member's filling trajectory."""
    outcomes: list[MemberOutcome] = []
    surrogate_errors: list[float] = []

    for member, forcing in enumerate(forcings):
        levels = _lake_level_trajectory(terrain, forcing, terrain.initial_lake_level)
        min_fos = np.zeros(levels.size)
        breach_index = -1
        mechanism = "stable"
        final: StabilityField | None = None

        for k, level in enumerate(levels):
            field_k = evaluate_stability(
                terrain, graph, float(level), float(forcing.cumulative_pdd_c_day[k]), moraine_cfg
            )
            min_fos[k] = field_k.min_factor_of_safety
            final = field_k
            if field_k.breached and breach_index < 0:
                breach_index = k
                mechanism = field_k.mechanism

        assert final is not None
        outcomes.append(
            MemberOutcome(
                member=member,
                lake_level_m=levels,
                min_factor_of_safety=min_fos,
                breached=breach_index >= 0,
                mechanism=mechanism,
                breach_time_h=float(forcing.time_h[breach_index]) if breach_index >= 0 else float("nan"),
                final_state=final,
            )
        )

        # Surrogate verification against the reference field at the final state.
        if operator is not None:
            state = MoraineNodeState(
                lake_level_m=float(levels[-1]),
                cumulative_pdd_c_day=float(forcing.cumulative_pdd_c_day[-1]),
                till_depth_m=np.full(graph.num_nodes, float(moraine_cfg["till_depth_m"])),
                cohesion_kpa=float(moraine_cfg["cohesion_kpa"]),
            )
            features = assemble_node_features(terrain, graph, state, moraine_cfg)
            predicted = operator.predict(features, graph.edge_attr_static, graph)
            reference = final.factor_of_safety
            surrogate_errors.append(
                float(
                    np.linalg.norm(predicted["factor_of_safety"] - reference)
                    / max(np.linalg.norm(reference), 1e-9)
                )
            )

    breached = [o for o in outcomes if o.breached]
    mechanism_counts: dict[str, int] = {}
    for outcome in outcomes:
        mechanism_counts[outcome.mechanism] = mechanism_counts.get(outcome.mechanism, 0) + 1

    if breached:
        representative = int(min(breached, key=lambda o: o.breach_time_h).member)
        earliest = float(min(o.breach_time_h for o in breached))
    else:
        representative = int(min(outcomes, key=lambda o: float(o.min_factor_of_safety.min())).member)
        earliest = float("nan")

    comparison: dict[str, float] = {}
    if surrogate_errors:
        comparison = {
            "mgn_relative_l2_vs_reference": float(np.mean(surrogate_errors)),
            "mgn_relative_l2_max": float(np.max(surrogate_errors)),
        }

    assessment = MoraineAssessment(
        members=outcomes,
        breach_probability=float(len(breached) / max(len(outcomes), 1)),
        mechanism_counts=mechanism_counts,
        earliest_breach_h=earliest,
        representative_member=representative,
        surrogate_comparison=comparison,
    )
    LOGGER.info(
        "Breach probability %.2f over %d members; mechanisms %s",
        assessment.breach_probability, len(outcomes), mechanism_counts,
    )
    return assessment
