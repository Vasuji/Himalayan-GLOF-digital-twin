"""Moraine-mechanics surrogate: a MeshGraphNet over the dam mesh.

The network predicts two per-node fields that together define the stability state
of the moraine: ``log(factor of safety)`` and the pore-pressure ratio ``r_u``. The
log transform matters -- the factor of safety is a positive ratio spanning more
than an order of magnitude across the mesh, and regressing it directly makes the
loss dominated by the stable interior rather than the near-critical toe where the
decision is actually made.

The breach decision is then taken from the predicted field with the same
Mohr-Coulomb threshold the reference model uses, so surrogate and solver are
compared on identical criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from glof_pipeline.backends import build_meshgraphnet
from glof_pipeline.surrogates.normalization import Standardizer
from glof_pipeline.terrain.mesh_builder import MoraineGraph, distance_to_lake, slope_and_aspect
from glof_pipeline.terrain.synthetic_dem import ValleyTerrain


@dataclass
class MoraineNodeState:
    """Per-scenario quantities that vary between MeshGraphNet evaluations."""

    lake_level_m: float
    cumulative_pdd_c_day: float
    till_depth_m: np.ndarray
    cohesion_kpa: float


def assemble_node_features(
    terrain: ValleyTerrain,
    graph: MoraineGraph,
    state: MoraineNodeState,
    moraine_cfg: dict[str, Any],
    cached: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the ``(N, 9)`` node feature matrix declared in ``NODE_FEATURES``.

    ``cached`` may carry the slope, aspect and lake-distance arrays, which depend
    only on the terrain and are therefore constant across a training set of
    thousands of hydrological scenarios.
    """
    rows, cols = graph.node_rc[:, 0], graph.node_rc[:, 1]
    if cached is None:
        beta_grid, aspect_grid = slope_and_aspect(terrain.z, terrain.dx)
        cached = {
            "slope": beta_grid[rows, cols],
            "aspect": aspect_grid[rows, cols],
            "distance": distance_to_lake(terrain, graph),
        }
    slope = cached["slope"]
    aspect = cached["aspect"]
    distance = cached["distance"]

    node_z = graph.node_xyz[:, 2]
    reference_depth = float(moraine_cfg["till_depth_m"])
    reference_cohesion = float(moraine_cfg["cohesion_kpa"])
    seepage_length = float(moraine_cfg["seepage_path_length_m"])

    head = np.clip(state.lake_level_m - node_z, 0.0, None) / max(reference_depth, 1e-6)
    features = np.stack(
        [
            (node_z - terrain.crest_elevation) / max(reference_depth, 1e-6),
            np.sin(slope),
            np.cos(slope),
            np.sin(aspect),
            np.cos(aspect),
            state.till_depth_m / max(reference_depth, 1e-6),
            head,
            np.full(node_z.shape, state.cohesion_kpa / max(reference_cohesion, 1e-6)),
            distance / max(seepage_length, 1e-6),
        ],
        axis=1,
    )
    return features.astype(np.float64)


class MoraineOperator:
    """Wraps a MeshGraphNet backend with feature assembly and persistence."""

    def __init__(self, cfg: dict[str, Any], device: str = "cpu", ):
        self.cfg = dict(cfg)
        self.device = torch.device(device)
        self.model, self.backend = build_meshgraphnet(cfg)
        self.model.to(self.device)
        self.node_scaler: Standardizer | None = None
        self.edge_scaler: Standardizer | None = None
        self.target_scaler: Standardizer | None = None
        self.metadata: dict[str, Any] = {"backend": self.backend}

    def graph_object(self, graph: MoraineGraph):
        """Return the graph carrier matching the active backend."""
        # PhysicsNeMo 2.x MeshGraphNet takes a PyTorch Geometric Data object.
        return graph.to_pyg(self.device)

    def _tensor(self, values: np.ndarray, scaler: Standardizer | None) -> torch.Tensor:
        array = scaler.transform(values) if scaler is not None else np.asarray(values, dtype=np.float64)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(
        self, node_features: np.ndarray, edge_features: np.ndarray, graph: MoraineGraph
    ) -> dict[str, np.ndarray]:
        """Predict ``factor_of_safety`` and ``pore_pressure_ratio`` per node."""
        self.model.eval()
        outputs = self.model(
            self._tensor(node_features, self.node_scaler),
            self._tensor(edge_features, self.edge_scaler),
            self.graph_object(graph),
        )
        array = outputs.detach().cpu().numpy().astype(np.float64)
        if self.target_scaler is not None:
            array = self.target_scaler.inverse(array)
        return {
            "factor_of_safety": np.exp(array[:, 0]),
            "pore_pressure_ratio": np.clip(array[:, 1], 0.0, 1.0),
        }

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.cfg,
                "backend": self.backend,
                "node_scaler": self.node_scaler.to_dict() if self.node_scaler else None,
                "edge_scaler": self.edge_scaler.to_dict() if self.edge_scaler else None,
                "target_scaler": self.target_scaler.to_dict() if self.target_scaler else None,
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
    def load(cls, path: str | Path, device: str = "cpu", ) -> MoraineOperator:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        operator = cls(payload["config"], device=device)
        operator.model.load_state_dict(payload["state_dict"])
        for attribute in ("node_scaler", "edge_scaler", "target_scaler"):
            stored = payload.get(attribute)
            if stored is not None:
                setattr(operator, attribute, Standardizer.from_dict(stored))
        operator.metadata = payload.get("metadata", {})
        return operator
