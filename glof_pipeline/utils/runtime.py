"""Determinism, timing, logging, and environment provenance."""

from __future__ import annotations

import logging
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def get_logger(name: str = "glof", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def get_training_logger(name: str = "glof.train") -> Any:
    """PhysicsNeMo's console logger for training loops, or the stdlib logger.

    ``physicsnemo.utils.logging.PythonLogger`` is what PhysicsNeMo's own recipes use,
    so surrogate training output is formatted the same way as a PhysicsNeMo job.
    It takes pre-formatted messages, so callers must use f-strings rather than
    printf-style arguments.
    """
    try:
        from glof_pipeline.nvidia.launch import get_logger as physicsnemo_logger

        return physicsnemo_logger(name)
    except (ImportError, RuntimeError):
        return get_logger(name)


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if present) PyTorch.

    ``deterministic_torch`` also disables cuDNN autotuning so that repeated GPU runs
    of the production tier reproduce bit-for-bit where the kernels allow it.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


def _version_or_none(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment_report() -> dict[str, Any]:
    """Package versions and hardware facts recorded in every run manifest."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _version_or_none(name)
            for name in (
                "numpy",
                "scipy",
                "torch",
                "nvidia-physicsnemo",
                "earth2studio",
                "torch-geometric",
            )
        },
    }
    try:
        import torch

        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        report["torch_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            report["torch_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover
        report["torch_cuda_available"] = False
    return report


@dataclass
class Timer:
    """Context manager that accumulates wall-clock timings by label.

    Timings feed the surrogate-versus-solver benchmark table, so they are measured
    rather than quoted.
    """

    label: str = "block"
    logger: logging.Logger | None = None
    records: dict[str, float] = field(default_factory=dict)
    _start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        elapsed = time.perf_counter() - self._start
        self.records[self.label] = self.records.get(self.label, 0.0) + elapsed
        if self.logger is not None:
            # PhysicsNeMo's PythonLogger.info() accepts a single pre-formatted
            # message, unlike the stdlib logger, so format here rather than
            # relying on printf-style substitution.
            self.logger.info(f"{self.label} took {elapsed:.3f} s")

    @property
    def elapsed(self) -> float:
        return self.records.get(self.label, 0.0)
