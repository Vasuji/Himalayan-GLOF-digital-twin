"""Terrain: synthetic valley construction, DEM ingest, and graph meshing."""

from .dem_io import load_dem, save_dem
from .mesh_builder import MoraineGraph, build_moraine_graph
from .synthetic_dem import (
    ValleyTerrain,
    build_synthetic_valley,
    delineate_from_dem,
    lake_hypsometry,
)

__all__ = [
    "MoraineGraph",
    "ValleyTerrain",
    "build_moraine_graph",
    "build_synthetic_valley",
    "delineate_from_dem",
    "lake_hypsometry",
    "load_dem",
    "save_dem",
]
