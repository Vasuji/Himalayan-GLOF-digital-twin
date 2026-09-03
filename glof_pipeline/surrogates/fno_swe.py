"""Flood-routing surrogate: an FNO that advances the shallow-water state.

The operator learns the *time-advance map* over one output interval,

``(h, hu, hv, z_bed)_t  ->  (h, hu, hv)_{t+dt}``,

and is applied autoregressively at inference. Training on the increment rather
than the absolute next state is what makes long rollouts stable, following the
standard practice for autoregressive neural surrogates.

Fields are non-dimensionalised before the network sees them (depth by a reference
depth, momentum by ``h_ref * sqrt(g h_ref)``) so the three channels are of
comparable magnitude, which matters because the loss is a relative L2 over the
stacked tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from glof_pipeline.backends import build_fno


@dataclass
class FloodOperatorScales:
    """Non-dimensionalisation constants; stored with the checkpoint."""

    depth_ref_m: float = 5.0
    bed_ref_m: float = 100.0
    bed_offset_m: float = 0.0
    gravity: float = 9.81

    @property
    def momentum_ref(self) -> float:
        return self.depth_ref_m * float(np.sqrt(self.gravity * self.depth_ref_m))

    def to_dict(self) -> dict[str, float]:
        return {
            "depth_ref_m": self.depth_ref_m,
            "bed_ref_m": self.bed_ref_m,
            "bed_offset_m": self.bed_offset_m,
            "gravity": self.gravity,
        }


class FloodOperator:
    """Wraps an FNO backend with the packing, scaling and rollout logic."""

    def __init__(self, cfg: dict[str, Any], device: str = "cpu", ):
        self.cfg = dict(cfg)
        self.device = torch.device(device)
        self.model, self.backend = build_fno(cfg)
        self.model.to(self.device)
        self.scales = FloodOperatorScales()
        self.metadata: dict[str, Any] = {"backend": self.backend}

    # -- tensor plumbing ----------------------------------------------------
    def pack(self, depth: np.ndarray, mx: np.ndarray, my: np.ndarray, bed: np.ndarray) -> torch.Tensor:
        """Stack and non-dimensionalise into the network's input tensor."""
        s = self.scales
        channels = np.stack(
            [
                np.asarray(depth) / s.depth_ref_m,
                np.asarray(mx) / s.momentum_ref,
                np.asarray(my) / s.momentum_ref,
                (np.asarray(bed) - s.bed_offset_m) / s.bed_ref_m,
            ],
            axis=-3,
        )
        tensor = torch.as_tensor(channels, dtype=torch.float32, device=self.device)
        return tensor if tensor.ndim == 4 else tensor.unsqueeze(0)

    def unpack(self, tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inverse of :meth:`pack` for the three predicted channels."""
        s = self.scales
        array = tensor.detach().cpu().numpy()
        depth = np.clip(array[..., 0, :, :] * s.depth_ref_m, 0.0, None)
        return depth, array[..., 1, :, :] * s.momentum_ref, array[..., 2, :, :] * s.momentum_ref

    def forward_increment(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the next state as ``state + network(state)`` (residual form)."""
        increment = self.model(inputs)
        next_state = inputs[:, :3] + increment
        # Depth cannot be negative; clamping inside the rollout keeps long
        # autoregressive sequences from drifting into unphysical states.
        next_state = torch.cat(
            [torch.clamp(next_state[:, :1], min=0.0), next_state[:, 1:]], dim=1
        )
        return next_state

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        depth0: np.ndarray,
        mx0: np.ndarray,
        my0: np.ndarray,
        bed: np.ndarray,
        n_steps: int,
    ) -> dict[str, np.ndarray]:
        """Autoregressive rollout of ``n_steps`` output intervals."""
        self.model.eval()
        state = self.pack(depth0, mx0, my0, bed)
        bed_channel = state[:, 3:4]
        depths, mxs, mys = [np.asarray(depth0)], [np.asarray(mx0)], [np.asarray(my0)]
        for _ in range(n_steps):
            state = self.forward_increment(state)
            state = torch.cat([state, bed_channel], dim=1)
            depth, mx, my = self.unpack(state[:, :3])
            depths.append(depth[0])
            mxs.append(mx[0])
            mys.append(my[0])
        return {
            "depth": np.stack(depths),
            "momentum_x": np.stack(mxs),
            "momentum_y": np.stack(mys),
        }

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.cfg,
                "scales": self.scales.to_dict(),
                "backend": self.backend,
                "metadata": {**self.metadata, **(extra or {})},
            },
            destination,
        )
        self._save_physicsnemo_checkpoint(destination, extra)
        return destination

    def _save_physicsnemo_checkpoint(self, destination: Path, extra: dict[str, Any] | None) -> Path | None:
        """Mirror the checkpoint in PhysicsNeMo's own format when that backend is active.

        ``physicsnemo.utils.checkpoint.save_checkpoint`` unwraps distributed and
        compiled models and writes the archive layout PhysicsNeMo's recipes read, so
        a checkpoint written here is loadable by a PhysicsNeMo training job on a GPU
        host without translation.
        """
        if self.backend != "physicsnemo":
            return None
        try:
            from glof_pipeline.nvidia.launch import save_checkpoint
        except (ImportError, RuntimeError):
            return None
        directory = Path(str(destination).removesuffix(".pt") + "_physicsnemo")
        return save_checkpoint(
            directory, models=self.model, metadata={**self.metadata, **(extra or {})}
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", ) -> FloodOperator:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        operator = cls(payload["config"], device=device)
        operator.model.load_state_dict(payload["state_dict"])
        operator.scales = FloodOperatorScales(**payload["scales"])
        operator.metadata = payload.get("metadata", {})
        if payload.get("backend") != operator.backend:
            operator.metadata["backend_mismatch"] = (
                f"checkpoint trained with {payload.get('backend')}, loaded with {operator.backend}"
            )
        return operator
