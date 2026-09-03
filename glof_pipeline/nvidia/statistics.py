"""Verification statistics from Earth2Studio and PhysicsNeMo.

Ensemble verification is a solved problem with a canonical implementation, and
estimator choice materially affects reported ensemble scores in a
forecast system. This module routes every probabilistic metric through
``earth2studio.statistics`` (which operates on tensors plus an explicit coordinate
system) and ``physicsnemo.metrics.general`` for the deterministic ones.

Verified signatures for the installed releases:

* ``earth2studio.statistics.crps(ensemble_dimension, reduction_dimensions=None,
  weights=None, fair=False)``, called as ``(x, x_coords, y, y_coords)``;
* ``earth2studio.statistics.rmse(reduction_dimensions, weights=None,
  batch_update=False, ensemble_dimension=None)``;
* ``earth2studio.statistics.brier_score(reduction_dimensions, thresholds,
  ensemble_dimension=None, batch_update=False)``;
* ``earth2studio.statistics.spread_skill_ratio(ensemble_dimension,
  reduction_dimensions, ...)``;
* ``physicsnemo.metrics.general.crps.crps(pred, obs, dim=0, method='kernel')`` and
  ``kcrps(pred, obs, dim=0, biased=True)``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from glof_pipeline.nvidia import earth2studio_available, physicsnemo_available, require

require("physicsnemo")

from physicsnemo.metrics.general.crps import crps as pn_crps, kcrps as pn_kcrps  # noqa: E402

__all__ = [
    "brier_score_earth2",
    "crps_earth2",
    "crps_physicsnemo",
    "rank_histogram_earth2",
    "spread_skill_ratio_earth2",
    "statistics_backend",
]


def statistics_backend() -> dict[str, bool]:
    return {"earth2studio": earth2studio_available(), "physicsnemo": physicsnemo_available()}


# ---------------------------------------------------------------------------
# PhysicsNeMo (deterministic and kernel CRPS)
# ---------------------------------------------------------------------------
def crps_physicsnemo(ensemble: np.ndarray, observation: np.ndarray, kernel: bool = True) -> float:
    """CRPS of an ``(members, ...)`` ensemble against an observation.

    ``kernel=True`` uses the unbiased kernel estimator ``kcrps``; otherwise the
    empirical-CDF estimator ``crps``.
    """
    predictions = torch.as_tensor(np.asarray(ensemble, dtype=np.float64))
    truth = torch.as_tensor(np.asarray(observation, dtype=np.float64))
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        truth = truth.reshape(1)
    values = pn_kcrps(predictions, truth, dim=0) if kernel else pn_crps(predictions, truth, dim=0)
    return float(torch.as_tensor(values).mean())


# ---------------------------------------------------------------------------
# Earth2Studio (probabilistic)
# ---------------------------------------------------------------------------
def _coords(names: tuple[str, ...], shape: tuple[int, ...]) -> OrderedDict:
    return OrderedDict((name, np.arange(size)) for name, size in zip(names, shape, strict=True))


def _as_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32))


def _ensemble_pair(
    ensemble: np.ndarray, observation: np.ndarray
) -> tuple[torch.Tensor, OrderedDict, torch.Tensor, OrderedDict]:
    """Shape an ``(members, n)`` ensemble and ``(n,)`` observation for Earth2Studio."""
    forecast = np.asarray(ensemble, dtype=np.float32).reshape(np.shape(ensemble)[0], -1)
    truth = np.asarray(observation, dtype=np.float32).reshape(-1)
    if truth.size != forecast.shape[1]:
        raise ValueError(
            f"Observation has {truth.size} values but the ensemble has {forecast.shape[1]} per member."
        )
    x_coords = _coords(("ensemble", "point"), forecast.shape)
    y_coords = _coords(("point",), truth.shape)
    return _as_tensor(forecast), x_coords, _as_tensor(truth), y_coords


def crps_earth2(ensemble: np.ndarray, observation: np.ndarray, fair: bool = False) -> float:
    """Earth2Studio CRPS, reduced over all points."""
    require("earth2studio")
    from earth2studio.statistics import crps

    x, x_coords, y, y_coords = _ensemble_pair(ensemble, observation)
    metric = crps(ensemble_dimension="ensemble", reduction_dimensions=["point"], fair=fair)
    value, _ = metric(x, x_coords, y, y_coords)
    return float(torch.as_tensor(value).mean())


def spread_skill_ratio_earth2(ensemble: np.ndarray, observation: np.ndarray) -> float:
    """Ensemble spread divided by ensemble-mean error; 1.0 is a calibrated ensemble."""
    require("earth2studio")
    from earth2studio.statistics import spread_skill_ratio

    x, x_coords, y, y_coords = _ensemble_pair(ensemble, observation)
    metric = spread_skill_ratio(ensemble_dimension="ensemble", reduction_dimensions=["point"])
    value, _ = metric(x, x_coords, y, y_coords)
    return float(torch.as_tensor(value).mean())


def brier_score_earth2(
    ensemble: np.ndarray, observation: np.ndarray, thresholds: list[float]
) -> float:
    """Brier score of exceedance probabilities at the given thresholds."""
    require("earth2studio")
    from earth2studio.statistics import brier_score

    x, x_coords, y, y_coords = _ensemble_pair(ensemble, observation)
    metric = brier_score(
        reduction_dimensions=["point"],
        thresholds=[float(t) for t in thresholds],
        ensemble_dimension="ensemble",
    )
    value, _ = metric(x, x_coords, y, y_coords)
    return float(torch.as_tensor(value).mean())


def rank_histogram_earth2(ensemble: np.ndarray, observation: np.ndarray) -> dict[str, Any]:
    """Rank histogram: a flat histogram indicates a reliable ensemble."""
    require("earth2studio")
    from earth2studio.statistics import rank_histogram

    x, x_coords, y, y_coords = _ensemble_pair(ensemble, observation)
    metric = rank_histogram(ensemble_dimension="ensemble", reduction_dimensions=["point"])
    value, coords = metric(x, x_coords, y, y_coords)
    array = torch.as_tensor(value).cpu().numpy()
    return {"histogram": array.tolist(), "dimensions": [str(k) for k in coords]}
