"""Turn the moraine body into the graph a MeshGraphNet consumes.

PhysicsNeMo 2.2 removed DGL: ``MeshGraphNet.forward(node_features, edge_features,
graph)`` expects a PyTorch Geometric ``Data`` object. :class:`MoraineGraph` stores
the topology as a plain ``edge_index`` tensor and converts on demand, so the toy
tier needs neither PyG nor DGL installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .synthetic_dem import ValleyTerrain

# Node feature order. Kept as a module constant because the trained checkpoint,
# the dataset builder and the inference path must agree on it.
NODE_FEATURES: tuple[str, ...] = (
    "elevation_norm",       # (z - crest) / till_depth
    "slope_sin",            # sin(beta)
    "slope_cos",            # cos(beta)
    "aspect_sin",
    "aspect_cos",
    "till_depth_norm",      # till depth / reference depth
    "head_norm",            # (lake level - z) / till depth, clipped at 0
    "cohesion_norm",        # degraded cohesion / reference cohesion
    "distance_to_lake_norm",
)
EDGE_FEATURES: tuple[str, ...] = ("dx_norm", "dy_norm", "distance_norm", "dz_norm")


@dataclass
class MoraineGraph:
    """Static topology of the moraine mesh plus the cell indices it came from."""

    edge_index: np.ndarray        # (2, E) int64, source -> destination
    node_rc: np.ndarray           # (N, 2) row/col of each node in the DEM
    node_xyz: np.ndarray          # (N, 3) easting, downvalley distance, elevation
    edge_attr_static: np.ndarray  # (E, 4) geometric edge features
    dx: float

    @property
    def num_nodes(self) -> int:
        return int(self.node_rc.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_pyg(self, device: Any = "cpu"):
        """Build the ``torch_geometric.data.Data`` PhysicsNeMo expects."""
        import torch

        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise RuntimeError(
                "torch-geometric is required for the PhysicsNeMo MeshGraphNet path: "
                "pip install torch-geometric"
            ) from exc
        edge_index = torch.as_tensor(self.edge_index, dtype=torch.long, device=device)
        return Data(edge_index=edge_index, num_nodes=self.num_nodes)


def _terrain_gradients(z: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference surface gradients ``(dz/dx, dz/dy)``."""
    dz_dy, dz_dx = np.gradient(z, dx, dx)
    return dz_dx, dz_dy


def slope_and_aspect(z: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope angle ``beta`` [rad] and aspect [rad] of the bed surface."""
    dz_dx, dz_dy = _terrain_gradients(z, dx)
    beta = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dy, -dz_dx)
    return beta, aspect


def build_moraine_graph(terrain: ValleyTerrain, neighbours: int = 8) -> MoraineGraph:
    """Graph over the moraine footprint with 4- or 8-connected grid adjacency.

    Grid adjacency is used rather than k-nearest-neighbours because the mesh is
    derived from a raster: the neighbour set is then exact, symmetric and free of
    the spurious long edges a kd-tree produces across the crest.
    """
    if neighbours not in (4, 8):
        raise ValueError("neighbours must be 4 or 8 for raster-derived meshes.")
    mask = terrain.moraine_mask
    if not mask.any():
        raise ValueError("Empty moraine mask; check domain.synthetic settings.")

    rows, cols = np.nonzero(mask)
    node_rc = np.stack([rows, cols], axis=1).astype(np.int64)
    index_map = -np.ones(mask.shape, dtype=np.int64)
    index_map[rows, cols] = np.arange(node_rc.shape[0], dtype=np.int64)

    node_xyz = np.stack(
        [terrain.x[cols], terrain.y[rows], terrain.z[rows, cols]], axis=1
    ).astype(np.float64)

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighbours == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    src_list, dst_list = [], []
    ny, nx = mask.shape
    for d_row, d_col in offsets:
        r2, c2 = rows + d_row, cols + d_col
        valid = (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
        valid &= mask[np.clip(r2, 0, ny - 1), np.clip(c2, 0, nx - 1)]
        src_list.append(index_map[rows[valid], cols[valid]])
        dst_list.append(index_map[r2[valid], c2[valid]])

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    delta = node_xyz[dst] - node_xyz[src]
    horizontal = np.hypot(delta[:, 0], delta[:, 1])
    scale = terrain.dx * np.sqrt(2.0)
    edge_attr = np.stack(
        [
            delta[:, 0] / scale,
            delta[:, 1] / scale,
            horizontal / scale,
            delta[:, 2] / max(terrain.meta.get("moraine_height_m", 50.0), 1e-6),
        ],
        axis=1,
    ).astype(np.float64)

    return MoraineGraph(
        edge_index=edge_index,
        node_rc=node_rc,
        node_xyz=node_xyz,
        edge_attr_static=edge_attr,
        dx=terrain.dx,
    )


def distance_to_lake(terrain: ValleyTerrain, graph: MoraineGraph) -> np.ndarray:
    """Euclidean distance [m] from each moraine node to the nearest lake cell.

    This sets the seepage path length used by the phreatic-surface model, so the
    upstream face of the dam sees a full lake head and the downstream toe sees none.
    """
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(~terrain.lake_mask, sampling=terrain.dx)
    return np.asarray(distance)[graph.node_rc[:, 0], graph.node_rc[:, 1]]
