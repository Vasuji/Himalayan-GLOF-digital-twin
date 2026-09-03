"""USD export -- entry point onto the Omniverse scene author.

Scene authoring lives in :mod:`glof_pipeline.nvidia.omniverse`, which builds the
stage with ``pxr``: a layered composition carrying materials, lighting and a review
camera. This module exposes the single call the pipeline needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["export_flood_to_usd"]


def export_flood_to_usd(
    bed_elevation: np.ndarray,
    depth_sequence: np.ndarray,
    time_s: np.ndarray,
    dx: float,
    output_path: str | Path,
    threshold_m: float = 0.05,
    stride: int = 2,
    frames: int | None = None,
    **scene_kwargs: Any,
) -> Path:
    """Author the routed flood as a layered Omniverse USD scene.

    Raises if ``usd-core`` is absent rather than emitting a degraded asset, so a
    missing dependency is visible at export time instead of in a review session.
    """
    from glof_pipeline.nvidia.omniverse import author_flood_scene

    scene = author_flood_scene(
        bed_elevation=np.asarray(bed_elevation, dtype=float),
        depth_sequence=np.asarray(depth_sequence, dtype=float),
        time_s=np.asarray(time_s, dtype=float),
        dx=float(dx),
        output_path=output_path,
        threshold_m=threshold_m,
        stride=stride,
        frames=frames,
        **scene_kwargs,
    )
    return Path(scene["root"])
