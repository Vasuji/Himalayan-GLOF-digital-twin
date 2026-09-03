"""Feature standardisation shared by the datasets, training loops and inference.

Normalisation statistics belong to the checkpoint, not the training script: a
surrogate loaded months later must apply exactly the transform it was fitted with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    """Per-channel zero-mean unit-variance transform."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axis: tuple[int, ...] = (0,), eps: float = 1e-8) -> Standardizer:
        mean = np.asarray(values, dtype=np.float64).mean(axis=axis)
        std = np.asarray(values, dtype=np.float64).std(axis=axis)
        return cls(mean=mean, std=np.clip(std, eps, None))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": np.asarray(self.mean).tolist(), "std": np.asarray(self.std).tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> Standardizer:
        return cls(mean=np.asarray(payload["mean"], dtype=np.float64),
                   std=np.asarray(payload["std"], dtype=np.float64))
