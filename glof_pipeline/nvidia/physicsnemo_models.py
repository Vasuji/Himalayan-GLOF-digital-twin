"""PhysicsNeMo model construction: the FNO and MeshGraphNet used by the twin.

Imports are explicit and at module scope. Importing this module on a machine
without PhysicsNeMo is an ``ImportError``, by design.
"""

from __future__ import annotations

from typing import Any

# Warp cache redirection happens in glof_pipeline.nvidia.__init__ and must precede
# the physicsnemo.models import below.
from glof_pipeline.nvidia import require
from glof_pipeline.nvidia._introspect import construct

require("physicsnemo")

import physicsnemo  # noqa: E402
from physicsnemo.models.fno import FNO  # noqa: E402
from physicsnemo.models.meshgraphnet import MeshGraphNet  # noqa: E402

__all__ = ["FNO", "MeshGraphNet", "build_fno", "build_meshgraphnet", "version"]


def version() -> str:
    return str(physicsnemo.__version__)


def build_fno(cfg: dict[str, Any]) -> tuple[FNO, list[str]]:
    """Build ``physicsnemo.models.fno.FNO`` from the ``surrogates.fno`` block.

    The FNO learns the shallow-water time-advance operator
    ``(h, hu, hv, z_bed) -> (h, hu, hv)`` over one output interval.
    """
    return construct(
        FNO,
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        dimension=int(cfg.get("dimension", 2)),
        num_fno_layers=int(cfg.get("num_fno_layers", 4)),
        num_fno_modes=cfg.get("num_fno_modes", 16),
        latent_channels=int(cfg.get("latent_channels", 32)),
        decoder_layers=int(cfg.get("decoder_layers", 1)),
        decoder_layer_size=int(cfg.get("decoder_layer_size", 32)),
        padding=int(cfg.get("padding", 8)),
        padding_type=str(cfg.get("padding_type", "constant")),
        activation_fn=str(cfg.get("activation_fn", "gelu")),
        coord_features=bool(cfg.get("coord_features", True)),
    )


def build_meshgraphnet(cfg: dict[str, Any]) -> tuple[MeshGraphNet, list[str]]:
    """Build ``physicsnemo.models.meshgraphnet.MeshGraphNet`` from ``surrogates.mgn``.

    Since PhysicsNeMo 2.x the ``graph`` argument of ``forward`` is a PyTorch
    Geometric ``Data`` object; DGL support was withdrawn upstream. The caller is
    responsible for supplying that object -- see
    :meth:`glof_pipeline.terrain.mesh_builder.MoraineGraph.to_pyg`.
    """
    return construct(
        MeshGraphNet,
        input_dim_nodes=int(cfg["node_features"]),
        input_dim_edges=int(cfg["edge_features"]),
        output_dim=int(cfg["output_dim"]),
        processor_size=int(cfg.get("processor_size", 15)),
        mlp_activation_fn=str(cfg.get("mlp_activation_fn", "relu")),
        num_layers_node_processor=int(cfg.get("num_layers_node_processor", 2)),
        num_layers_edge_processor=int(cfg.get("num_layers_edge_processor", 2)),
        hidden_dim_processor=int(cfg.get("hidden_dim_processor", 128)),
        hidden_dim_node_encoder=int(cfg.get("hidden_dim_node_encoder", 128)),
        num_layers_node_encoder=int(cfg.get("num_layers_node_encoder", 2)),
        hidden_dim_edge_encoder=int(cfg.get("hidden_dim_edge_encoder", 128)),
        num_layers_edge_encoder=int(cfg.get("num_layers_edge_encoder", 2)),
        hidden_dim_node_decoder=int(cfg.get("hidden_dim_node_decoder", 128)),
        num_layers_node_decoder=int(cfg.get("num_layers_node_decoder", 2)),
        aggregation=str(cfg.get("aggregation", "sum")),
    )
