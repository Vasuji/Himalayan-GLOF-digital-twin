"""Training-loop utilities from PhysicsNeMo: checkpointing and logging.

``physicsnemo.utils.checkpoint`` handles the cases a bare ``torch.save`` does not:
distributed and compiled models are unwrapped, optimiser and scheduler state ride
along, and the archive format is the one PhysicsNeMo's own recipes read. Using it
means a checkpoint written on this laptop is loadable by a PhysicsNeMo training
job on a GPU host without translation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from glof_pipeline.nvidia import require

require("physicsnemo")

from physicsnemo.utils.checkpoint import (  # noqa: E402
    load_checkpoint as _pn_load_checkpoint,
    save_checkpoint as _pn_save_checkpoint,
)
from physicsnemo.utils.logging import PythonLogger  # noqa: E402

__all__ = ["get_logger", "load_checkpoint", "save_checkpoint"]


def get_logger(name: str = "glof") -> Any:
    """PhysicsNeMo's console logger, so surrogate training logs look like its recipes."""
    return PythonLogger(name)


def save_checkpoint(
    path: str | Path,
    models: torch.nn.Module | list[torch.nn.Module],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a PhysicsNeMo checkpoint directory and return it.

    PhysicsNeMo treats ``path`` as a directory and manages the filenames inside it.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    _pn_save_checkpoint(
        path=directory,
        models=models,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metadata=metadata,
    )
    return directory


def load_checkpoint(
    path: str | Path,
    models: torch.nn.Module | list[torch.nn.Module],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
    metadata_dict: dict[str, Any] | None = None,
) -> int:
    """Restore a PhysicsNeMo checkpoint; returns the stored epoch."""
    return int(
        _pn_load_checkpoint(
            path=Path(path),
            models=models,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            metadata_dict=metadata_dict,
        )
    )
