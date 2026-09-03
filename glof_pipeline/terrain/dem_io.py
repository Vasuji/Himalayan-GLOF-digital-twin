"""DEM ingest for the production tier and cached-terrain IO for the toy tier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .synthetic_dem import ValleyTerrain


def load_dem(path: str | Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Read a single-band elevation raster.

    Returns ``(z, dx, metadata)``. Requires rasterio, which is part of the ``io``
    extra rather than the toy tier's dependency set.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"DEM not found at {source}. Either point domain.dem_path at a real raster "
            "or run the toy tier, which builds a synthetic valley."
        )
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required to read DEM rasters: pip install rasterio") from exc

    with rasterio.open(source) as handle:
        z = handle.read(1).astype(np.float64)
        transform = handle.transform
        dx = float(abs(transform.a))
        dy = float(abs(transform.e))
        meta = {
            "source": str(source),
            "crs": str(handle.crs),
            "nodata": handle.nodata,
            "dx": dx,
            "dy": dy,
        }
    if not np.isclose(dx, dy, rtol=1e-3):
        raise ValueError(
            f"Anisotropic DEM cells ({dx} x {dy} m). Reproject to a square grid before use."
        )
    if meta["nodata"] is not None:
        z = np.where(z == meta["nodata"], np.nan, z)
    return z, dx, meta


def save_dem(path: str | Path, terrain: ValleyTerrain) -> Path:
    """Cache a terrain as ``.npz`` so later stages need not rebuild it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        z=terrain.z,
        dx=np.array(terrain.dx),
        x=terrain.x,
        y=terrain.y,
        lake_mask=terrain.lake_mask,
        moraine_mask=terrain.moraine_mask,
        thalweg_col=terrain.thalweg_col,
        crest_elevation=np.array(terrain.crest_elevation),
        initial_lake_level=np.array(terrain.initial_lake_level),
        moraine_row=np.array(terrain.moraine_row),
    )
    return destination


def load_cached_terrain(path: str | Path) -> ValleyTerrain:
    """Reload a terrain written by :func:`save_dem`."""
    with np.load(Path(path), allow_pickle=False) as handle:
        return ValleyTerrain(
            z=handle["z"],
            dx=float(handle["dx"]),
            x=handle["x"],
            y=handle["y"],
            lake_mask=handle["lake_mask"].astype(bool),
            moraine_mask=handle["moraine_mask"].astype(bool),
            thalweg_col=handle["thalweg_col"].astype(int),
            crest_elevation=float(handle["crest_elevation"]),
            initial_lake_level=float(handle["initial_lake_level"]),
            moraine_row=int(handle["moraine_row"]),
            meta={"source": str(path)},
        )
