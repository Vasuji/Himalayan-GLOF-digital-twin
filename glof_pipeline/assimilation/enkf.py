r"""Stochastic ensemble Kalman filter for the lake-and-moraine state.

Implements the perturbed-observation (stochastic) EnKF of Burgers, van Leeuwen &
Evensen (1998):

.. math::

    \mathbf{K} = \mathbf{P}^f\mathbf{H}^\top
                 (\mathbf{H}\mathbf{P}^f\mathbf{H}^\top + \mathbf{R})^{-1}, \qquad
    \mathbf{x}^a_i = \mathbf{x}^f_i
        + \mathbf{K}\left(\mathbf{y} + \boldsymbol{\epsilon}_i - \mathbf{H}\mathbf{x}^f_i\right).

Multiplicative covariance inflation counteracts the ensemble collapse that a small
ensemble suffers over repeated cycles. The state is low dimensional (three
variables), so no localisation is applied; the configuration exposes a
localisation radius for the spatially-distributed extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .sensors import Observation


@dataclass
class StateSpec:
    """Names, prior spread and physical bounds of the estimated state."""

    names: list[str]
    prior_std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> StateSpec:
        names = list(cfg["state_variables"])
        bounds = {
            "lake_level_m": (-np.inf, np.inf),
            "pore_pressure_ratio": (0.0, 1.0),
            "ddf_ice_mm_per_c_per_day": (1.0, 20.0),
        }
        lower = np.array([bounds.get(name, (-np.inf, np.inf))[0] for name in names])
        upper = np.array([bounds.get(name, (-np.inf, np.inf))[1] for name in names])
        return cls(names=names, prior_std=np.asarray(cfg["state_prior_std"], float),
                   lower=lower, upper=upper)

    @property
    def size(self) -> int:
        return len(self.names)


# Which state variable each observation kind measures, and its scaling.
_OBSERVATION_MAP = {
    "lake_stage": ("lake_level_m", 1.0),
    "piezometric_head": ("pore_pressure_ratio", 1.0),
    "discharge": ("ddf_ice_mm_per_c_per_day", 1.0),
}


class EnsembleKalmanFilter:
    """Small-state stochastic EnKF with multiplicative inflation."""

    def __init__(self, spec: StateSpec, cfg: dict[str, Any], rng: np.random.Generator):
        self.spec = spec
        self.ensemble_size = int(cfg["ensemble_size"])
        self.inflation = float(cfg.get("inflation", 1.0))
        self.rng = rng
        self.ensemble: np.ndarray | None = None

    def initialise(self, background: np.ndarray) -> np.ndarray:
        """Draw the prior ensemble around a background state."""
        background = np.asarray(background, dtype=float)
        perturbation = self.rng.normal(0.0, 1.0, size=(self.ensemble_size, self.spec.size))
        self.ensemble = np.clip(
            background + perturbation * self.spec.prior_std, self.spec.lower, self.spec.upper
        )
        return self.ensemble

    def observation_operator(self, observations: list[Observation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assemble ``(H, y, R_diagonal)`` from a list of observations."""
        rows, values, variances = [], [], []
        for observation in observations:
            if observation.kind not in _OBSERVATION_MAP:
                continue
            name, scale = _OBSERVATION_MAP[observation.kind]
            if name not in self.spec.names:
                continue
            row = np.zeros(self.spec.size)
            row[self.spec.names.index(name)] = scale
            rows.append(row)
            values.append(observation.value)
            variances.append(observation.error_std**2)
        if not rows:
            return np.empty((0, self.spec.size)), np.empty(0), np.empty(0)
        return np.stack(rows), np.asarray(values), np.asarray(variances)

    def analysis(self, observations: list[Observation]) -> np.ndarray:
        """Update the ensemble with one batch of observations."""
        if self.ensemble is None:
            raise RuntimeError("Call initialise() before analysis().")
        H, y, r_diagonal = self.observation_operator(observations)
        if y.size == 0:
            return self.ensemble

        mean = self.ensemble.mean(axis=0, keepdims=True)
        anomalies = (self.ensemble - mean) * self.inflation
        self.ensemble = mean + anomalies

        n = self.ensemble_size
        covariance = anomalies.T @ anomalies / max(n - 1, 1)
        R = np.diag(r_diagonal)
        innovation_covariance = H @ covariance @ H.T + R
        gain = covariance @ H.T @ np.linalg.inv(innovation_covariance)

        perturbed = y[None, :] + self.rng.normal(0.0, np.sqrt(r_diagonal), size=(n, y.size))
        innovation = perturbed - self.ensemble @ H.T
        self.ensemble = np.clip(self.ensemble + innovation @ gain.T, self.spec.lower, self.spec.upper)
        return self.ensemble

    def mean(self) -> np.ndarray:
        if self.ensemble is None:
            raise RuntimeError("Filter has no ensemble.")
        return self.ensemble.mean(axis=0)

    def spread(self) -> np.ndarray:
        if self.ensemble is None:
            raise RuntimeError("Filter has no ensemble.")
        return self.ensemble.std(axis=0)

    def rmse_against(self, truth: np.ndarray) -> float:
        """Root-mean-square error of the ensemble mean, normalised by the prior spread."""
        error = (self.mean() - np.asarray(truth, dtype=float)) / self.spec.prior_std
        return float(np.sqrt(np.mean(error**2)))
