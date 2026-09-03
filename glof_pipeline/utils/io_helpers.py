"""Array and metadata IO shared by the toy and production tiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable.")


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def read_zarr_variable(zarr_path: str | Path, variable: str) -> np.ndarray:
    """Read one variable from an Earth-2 Zarr store.

    Earth2Studio's ``ZarrBackend`` writes forecast fields into a Zarr group; this is the
    bridge from that store into the NumPy arrays the hydrology stages consume.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray and zarr are required: pip install xarray zarr") from exc

    dataset = xr.open_zarr(zarr_path)
    if variable not in dataset:
        available = ", ".join(sorted(dataset.data_vars))
        raise KeyError(f"Variable {variable!r} absent from {zarr_path}. Available: {available}")
    return np.asarray(dataset[variable].values)
