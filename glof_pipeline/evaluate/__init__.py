"""Verification metrics and the surrogate-versus-solver benchmark."""

from .benchmark import benchmark_routing, benchmark_table
from .metrics import (
    arrival_time_error,
    brier_score,
    crps_ensemble,
    probabilistic_report,
    rank_probability,
    relative_l2,
    rmse,
    summarise_field_error,
)

__all__ = [
    "arrival_time_error",
    "benchmark_routing",
    "benchmark_table",
    "brier_score",
    "crps_ensemble",
    "probabilistic_report",
    "rank_probability",
    "relative_l2",
    "rmse",
    "summarise_field_error",
]
