"""Verification metrics, delegated to PhysicsNeMo and Earth-2.

Every general-purpose metric is the library implementation. This matters for
ensemble scores in particular: CRPS has several inequivalent finite-ensemble
estimators, and using the library's keeps reported values comparable with
published verification numbers.

* deterministic error -- ``physicsnemo.metrics.general.mse.rmse`` and
  ``relative_error.relative_l2``;
* ensemble CRPS -- ``physicsnemo.metrics.general.crps.kcrps`` (unbiased kernel
  estimator) and ``earth2studio.statistics.crps``;
* reliability -- ``physicsnemo.metrics.general.calibration.find_rank`` and
  ``rank_probability_score``, plus ``earth2studio.statistics.rank_histogram``;
* Brier score and spread-skill -- ``earth2studio.statistics``.

Only :func:`arrival_time_error` is local, because "when does the wave reach this
village" is not a metric any weather library defines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from physicsnemo.metrics.general.calibration import find_rank, rank_probability_score
from physicsnemo.metrics.general.crps import crps as pn_crps, kcrps as pn_kcrps
from physicsnemo.metrics.general.histogram import histogram
from physicsnemo.metrics.general.mse import rmse as pn_rmse
from physicsnemo.metrics.general.relative_error import relative_l2 as pn_relative_l2

__all__ = [
    "arrival_time_error",
    "brier_score",
    "crps",
    "crps_ensemble",
    "probabilistic_report",
    "rank_probability",
    "relative_l2",
    "rmse",
    "summarise_field_error",
]


def _tensor(values: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float64))


def rmse(prediction: Any, target: Any) -> float:
    """Root-mean-square error -- ``physicsnemo.metrics.general.mse.rmse``."""
    return float(pn_rmse(_tensor(prediction), _tensor(target)).mean())


def relative_l2(prediction: Any, target: Any) -> float:
    """Relative L2 error -- ``physicsnemo.metrics.general.relative_error.relative_l2``."""
    return float(pn_relative_l2(_tensor(prediction), _tensor(target)).mean())


def crps_ensemble(ensemble: Any, observation: float | Any, kernel: bool = True) -> float:
    """CRPS of a ``(members, ...)`` ensemble -- ``physicsnemo.metrics.general.crps``.

    ``kernel=True`` selects the unbiased kernel estimator ``kcrps``; otherwise the
    empirical-CDF estimator ``crps``.
    """
    predictions = _tensor(ensemble)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        truth = _tensor(observation).reshape(1)
    else:
        truth = _tensor(observation).reshape(predictions.shape[1:])
    values = pn_kcrps(predictions, truth, dim=0) if kernel else pn_crps(predictions, truth, dim=0)
    return float(values.mean())


def crps(ensemble: Any, observation: float | Any, prefer_earth2: bool = True) -> float:
    """CRPS, preferring Earth-2's estimator so numbers match its verification tooling."""
    if prefer_earth2:
        try:
            from glof_pipeline.nvidia.statistics import crps_earth2

            return crps_earth2(np.atleast_2d(ensemble), np.atleast_1d(observation))
        except (ImportError, RuntimeError, ValueError):
            pass
    return crps_ensemble(ensemble, observation)


def rank_probability(ensemble: Any, observation: Any, bins: int = 10) -> dict[str, Any]:
    """Reliability via PhysicsNeMo's rank statistics.

    ``find_rank`` places each observation within the ensemble's own distribution and
    ``rank_probability_score`` scores the resulting rank histogram; a flat histogram
    (score near zero) is a reliable ensemble.
    """
    predictions = _tensor(ensemble)
    truth = _tensor(observation).reshape(-1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    flattened = predictions.reshape(predictions.shape[0], -1)
    bin_edges, counts = histogram(flattened, bins=bins)
    ranks = find_rank(bin_edges, counts, truth)
    return {
        "rank_probability_score": float(rank_probability_score(ranks).mean()),
        "mean_rank": float(ranks.double().mean()),
        "bins": int(bins),
    }


def brier_score(probabilities: Any, outcomes: Any) -> float:
    """Brier score of forecast probabilities against binary outcomes.

    Delegates to ``earth2studio.statistics.brier_score`` when the ensemble form is
    available; the direct mean-squared form is used for pre-computed probabilities,
    which is a definition rather than an estimator choice.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if probabilities.shape != outcomes.shape:
        raise ValueError("Probabilities and outcomes must share one shape.")
    return float(np.mean((probabilities - outcomes) ** 2))


def arrival_time_error(predicted_s: Any, reference_s: Any) -> dict[str, float]:
    """Signed arrival-time error at each receptor, in seconds.

    Local by necessity: arrival time at a named downstream settlement is the
    decision-relevant quantity for a warning system and is not a metric any
    weather or physics-ML library defines. Receptors the wave never reaches are
    excluded rather than counted as zero error.
    """
    predicted = np.asarray(predicted_s, dtype=float)
    reference = np.asarray(reference_s, dtype=float)
    valid = np.isfinite(predicted) & np.isfinite(reference)
    if not valid.any():
        return {"mean_error_s": float("nan"), "mean_abs_error_s": float("nan"), "n": 0}
    difference = predicted[valid] - reference[valid]
    return {
        "mean_error_s": float(difference.mean()),
        "mean_abs_error_s": float(np.abs(difference).mean()),
        "max_abs_error_s": float(np.abs(difference).max()),
        "n": int(valid.sum()),
    }


def summarise_field_error(prediction: Any, target: Any) -> dict[str, float]:
    """Error summary for a 2-D field, from PhysicsNeMo's metric implementations.

    Used by the surrogate-versus-solver benchmark. Reports both the absolute scale
    of the error (RMSE, in metres of depth) and its relative size (relative L2),
    because a surrogate can look accurate on one and poor on the other depending on
    how much of the domain is wet.
    """
    predicted = np.asarray(prediction, dtype=float)
    reference = np.asarray(target, dtype=float)
    if predicted.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {reference.shape}.")
    return {
        "rmse": rmse(predicted, reference),
        "relative_l2": relative_l2(predicted, reference),
        "max_abs_error": float(np.abs(predicted - reference).max()),
        "bias": float((predicted - reference).mean()),
    }


def probabilistic_report(
    ensemble: Any, observation: Any, thresholds: list[float] | None = None
) -> dict[str, Any]:
    """Full probabilistic verification, labelled with the library that produced it."""
    ensemble = np.atleast_2d(np.asarray(ensemble, dtype=float))
    observation = np.atleast_1d(np.asarray(observation, dtype=float))
    report: dict[str, Any] = {"members": int(ensemble.shape[0]), "backend": "physicsnemo"}
    report["crps_kernel"] = crps_ensemble(ensemble, observation)
    report.update(rank_probability(ensemble, observation))
    try:
        from glof_pipeline.nvidia.statistics import (
            crps_earth2,
            rank_histogram_earth2,
            spread_skill_ratio_earth2,
        )

        report["backend"] = "physicsnemo+earth2studio"
        report["crps"] = crps_earth2(ensemble, observation)
        report["spread_skill_ratio"] = spread_skill_ratio_earth2(ensemble, observation)
        report["rank_histogram"] = rank_histogram_earth2(ensemble, observation)
        if thresholds:
            from glof_pipeline.nvidia.statistics import brier_score_earth2

            report["brier_score"] = brier_score_earth2(ensemble, observation, thresholds)
    except (ImportError, RuntimeError, ValueError) as error:
        report["earth2_note"] = f"{type(error).__name__}: {error}"
        report["crps"] = report["crps_kernel"]
    return report
