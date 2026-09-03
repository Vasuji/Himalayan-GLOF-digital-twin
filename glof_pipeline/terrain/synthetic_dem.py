"""Synthetic moraine-dammed glacial valley.

The toy tier needs a terrain with the features that control a GLOF: an
over-deepened lake basin, a moraine ridge with a definable crest and seepage
path, and a downvalley channel for the flood wave. This module builds one
analytically so the whole pipeline is reproducible without a DEM download, and
exposes the same interface the production tier gets from a real raster.

Conventions
-----------
Arrays are ``(ny, nx)`` with row index increasing **downvalley** (+y) and column
index increasing across the valley (+x). Elevations are metres above sea level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover - dependency is declared
    raise RuntimeError("SciPy is required: pip install scipy") from exc


@dataclass
class ValleyTerrain:
    """Bed topography plus the masks the hydrology stages need."""

    z: np.ndarray                 # bed elevation (ny, nx) [m]
    dx: float                     # cell size [m]
    x: np.ndarray                 # cell-centre easting (nx,) [m]
    y: np.ndarray                 # cell-centre downvalley distance (ny,) [m]
    lake_mask: np.ndarray         # bool (ny, nx)
    moraine_mask: np.ndarray      # bool (ny, nx)
    thalweg_col: np.ndarray       # column index of the channel centreline (ny,)
    crest_elevation: float        # [m]
    initial_lake_level: float     # [m]
    moraine_row: int              # row index of the dam crest
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def cell_area(self) -> float:
        return float(self.dx * self.dx)

    def lake_area(self, level: float) -> float:
        """Water-surface area [m^2] of the lake at ``level``."""
        return float(np.count_nonzero(self.lake_mask & (self.z < level)) * self.cell_area)

    def lake_volume(self, level: float) -> float:
        """Storage [m^3] impounded behind the moraine at ``level``."""
        depth = np.where(self.lake_mask, level - self.z, 0.0)
        return float(np.clip(depth, 0.0, None).sum() * self.cell_area)

    def receptor_rows(self, receptors: list[dict[str, Any]]) -> list[tuple[str, int]]:
        """Map fractional downvalley positions onto row indices."""
        ny = self.shape[0]
        out = []
        for item in receptors:
            row = int(np.clip(round(float(item["position_frac"]) * (ny - 1)), 0, ny - 1))
            out.append((str(item["name"]), row))
        return out


def _smooth_ridge(distance: np.ndarray, half_width: float) -> np.ndarray:
    """Compactly supported bell used for the moraine cross-section."""
    scaled = np.clip(distance / max(half_width, 1e-9), -1.0, 1.0)
    return np.cos(0.5 * np.pi * scaled) ** 2


def build_synthetic_valley(cfg: dict[str, Any]) -> ValleyTerrain:
    """Construct the valley from the ``domain.synthetic`` configuration block.

    The bed is the sum of four analytic terms: a constant-gradient thalweg, a
    parabolic cross-section, a Gaussian over-deepening (the lake basin) and a
    transverse moraine ridge. The lake is then extracted as the connected set of
    cells below the still-water level upstream of the crest, which guarantees the
    hypsometry and the impounded volume are mutually consistent.
    """
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    dx = float(cfg["dx"])

    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    xx, yy = np.meshgrid(x, y)

    crest = float(cfg["crest_elevation_m"])
    slope = float(cfg["valley_slope"])
    half_width = float(cfg["valley_half_width_m"])
    wall_relief = float(cfg["valley_wall_relief_m"])
    meander_amp = float(cfg["thalweg_meander_amplitude_m"])
    meander_len = float(cfg["thalweg_meander_wavelength_m"])

    moraine_y = float(cfg["moraine_centre_frac"]) * y[-1]
    moraine_width = float(cfg["moraine_width_m"])
    moraine_height = float(cfg["moraine_height_m"])
    basin_depth = float(cfg["lake_basin_depth_m"])
    basin_length = float(cfg["lake_basin_length_m"])
    basin_width = float(cfg["lake_basin_width_m"])
    freeboard = float(cfg["freeboard_m"])

    # 1. Thalweg: constant gradient, gently meandering centreline. The datum is set
    #    at the dam so that the finished crest lands on the configured elevation.
    centre_x = 0.5 * x[-1] + meander_amp * np.sin(2.0 * np.pi * yy / max(meander_len, 1e-9))
    floor = crest - moraine_height - slope * (yy - moraine_y)

    # 2. Parabolic cross-section, capped at the valley wall relief.
    offset = xx - centre_x
    walls = np.minimum(wall_relief * (offset / half_width) ** 2, wall_relief)

    z = floor + walls

    # 3. Over-deepened basin upstream of the dam.
    basin = basin_depth * np.exp(
        -(((yy - (moraine_y - 0.55 * basin_length)) / (0.5 * basin_length)) ** 2)
        - ((offset / (0.5 * basin_width)) ** 2)
    )
    z = z - basin

    # 4. Transverse moraine ridge, tapering into the valley walls.
    ridge_profile = _smooth_ridge(yy - moraine_y, 0.5 * moraine_width)
    ridge_taper = np.clip(1.0 - (np.abs(offset) / (1.35 * half_width)) ** 2, 0.0, 1.0)
    z = z + moraine_height * ridge_profile * ridge_taper

    moraine_row = int(np.clip(round(moraine_y / dx - 0.5), 1, ny - 2))
    crest_elevation = float(z[moraine_row].min())  # lowest point of the crest line
    lake_level = crest_elevation - freeboard

    # Lake = connected below-level cells upstream of the crest that touch the basin.
    below = (z < lake_level)
    below[moraine_row:, :] = False
    labels, _ = ndimage.label(below)
    seed_row = int(np.clip(round((moraine_y - 0.55 * basin_length) / dx), 0, moraine_row - 1))
    seed_col = int(np.clip(round(centre_x[seed_row, 0] / dx), 0, nx - 1))
    seed_label = int(labels[seed_row, seed_col])
    lake_mask = (labels == seed_label) if seed_label > 0 else below

    # Moraine body: cells where the ridge raises the bed by more than 2% of its
    # height. Using the ridge contribution rather than a fixed distance keeps the
    # downstream face -- where the critical slip surface sits -- inside the mesh.
    #    Cells are additionally required to lie below the crest plus a margin, which
    #    excludes the abutments where the ridge climbs the valley walls: those are
    #    steep hillslope cells, not part of the dam body, and including them would let
    #    a dry hillslope rather than the water-loaded dam set the minimum safety factor.
    ridge_contribution = ridge_profile * ridge_taper
    abutment_margin = 1.00 * moraine_height
    moraine_mask = (
        (ridge_contribution > 0.02)
        & (z < crest_elevation + abutment_margin)
        & (~lake_mask)
    )

    thalweg_col = np.argmin(z, axis=1).astype(int)

    terrain = ValleyTerrain(
        z=z,
        dx=dx,
        x=x,
        y=y,
        lake_mask=lake_mask,
        moraine_mask=moraine_mask,
        thalweg_col=thalweg_col,
        crest_elevation=crest_elevation,
        initial_lake_level=float(lake_level),
        moraine_row=moraine_row,
        meta={
            "source": "synthetic",
            "moraine_width_m": moraine_width,
            "moraine_height_m": moraine_height,
            "freeboard_m": freeboard,
            "valley_slope": slope,
        },
    )
    terrain.meta["initial_lake_volume_m3"] = terrain.lake_volume(lake_level)
    terrain.meta["initial_lake_area_m2"] = terrain.lake_area(lake_level)
    terrain.meta["lake_bed_min_m"] = float(z[lake_mask].min()) if lake_mask.any() else float(z.min())
    return terrain


def lake_hypsometry(
    terrain: ValleyTerrain, n_levels: int = 96
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(levels, areas, volumes)`` for the impounded basin.

    Sampling the stage-storage curve once lets the breach integrator convert a
    withdrawn volume back into a lake stage without re-scanning the DEM at every
    sub-step.
    """
    if not terrain.lake_mask.any():
        raise ValueError("Terrain has no lake cells; check the synthetic domain settings.")
    bed = terrain.z[terrain.lake_mask]
    levels = np.linspace(bed.min(), terrain.crest_elevation, n_levels)
    areas = np.array([np.count_nonzero(bed < lvl) for lvl in levels], dtype=float)
    areas *= terrain.cell_area
    volumes = np.array(
        [np.clip(lvl - bed, 0.0, None).sum() for lvl in levels], dtype=float
    ) * terrain.cell_area
    return levels, areas, volumes


def level_from_volume(levels: np.ndarray, volumes: np.ndarray, volume: float) -> float:
    """Invert the stage-storage curve (monotone, so linear interpolation is safe)."""
    volume = float(np.clip(volume, volumes[0], volumes[-1]))
    return float(np.interp(volume, volumes, levels))


# ---------------------------------------------------------------------------
# Delineation from a real DEM
# ---------------------------------------------------------------------------
def _fill_depressions(z: np.ndarray, max_iterations: int = 200) -> np.ndarray:
    """Priority-flood style depression filling by morphological reconstruction.

    Returns the surface that would result if every closed depression were filled to
    its spill point. ``filled - z`` is then the depth of water each cell could hold,
    which is exactly the lake-basin field a GLOF assessment needs.

    Implemented here because no installed dependency provides it: SciPy ships
    greyscale morphology but not reconstruction-by-erosion, and the dedicated
    terrain-analysis packages (RichDEM, WhiteboxTools) are not part of this stack.
    The algorithm is the standard one (Soille & Ansoult 1990): initialise the
    marker at +inf inside the domain and at the DEM value on the boundary, then
    iterate pointwise minima against the neighbourhood until it stops changing.
    """
    marker = np.full_like(z, np.inf, dtype=float)
    marker[0, :] = z[0, :]
    marker[-1, :] = z[-1, :]
    marker[:, 0] = z[:, 0]
    marker[:, -1] = z[:, -1]

    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for _ in range(max_iterations):
        eroded = ndimage.grey_erosion(marker, footprint=footprint, mode="nearest")
        updated = np.maximum(z, eroded)
        if np.allclose(updated, marker, rtol=0.0, atol=1e-6):
            return updated
        marker = updated
    return marker


def delineate_from_dem(
    z: np.ndarray,
    dx: float,
    min_lake_depth_m: float = 2.0,
    min_lake_area_m2: float = 1.0e4,
    moraine_band_m: float = 500.0,
    freeboard_m: float | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> ValleyTerrain:
    """Build a :class:`ValleyTerrain` from a surveyed raster.

    The lake is the largest filled depression deeper than ``min_lake_depth_m``; its
    spill point is the crest, and the moraine is the band of cells within
    ``moraine_band_m`` downstream of the lake outlet that stand above the lake bed.

    Automatic delineation is a starting point, not a substitute for survey. Pass
    ``masks={'lake': ..., 'moraine': ...}`` to override either field with mapped
    polygons; the function then uses those and derives only the elevations.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {z.shape}.")
    ny, nx = z.shape
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx

    masks = masks or {}
    if "lake" in masks:
        lake_mask = np.asarray(masks["lake"], dtype=bool)
        if lake_mask.shape != z.shape:
            raise ValueError("Supplied lake mask does not match the DEM shape.")
    else:
        depth = _fill_depressions(z) - z
        candidate = depth > float(min_lake_depth_m)
        if not candidate.any():
            raise ValueError(
                f"No depression deeper than {min_lake_depth_m} m in this DEM. Lower "
                "min_lake_depth_m or supply domain.masks.lake."
            )
        labels, count = ndimage.label(candidate)
        areas = ndimage.sum(candidate, labels, index=np.arange(1, count + 1)) * dx * dx
        largest = int(np.argmax(areas)) + 1
        if areas.max() < float(min_lake_area_m2):
            raise ValueError(
                f"Largest depression is {areas.max():.0f} m2, below min_lake_area_m2="
                f"{min_lake_area_m2:.0f}. Supply domain.masks.lake if this is the lake."
            )
        lake_mask = labels == largest

    lake_bed_min = float(z[lake_mask].min())
    # Spill point: the lowest cell on the lake's outer boundary.
    dilated = ndimage.binary_dilation(lake_mask, iterations=1)
    rim = dilated & ~lake_mask
    crest_elevation = float(z[rim].min()) if rim.any() else float(z[lake_mask].max())
    lake_level = crest_elevation - float(freeboard_m or 0.0)

    # Outlet row: where the rim reaches its minimum, i.e. where the lake would spill.
    rim_rows, rim_cols = np.nonzero(rim)
    outlet = int(np.argmin(z[rim_rows, rim_cols]))
    moraine_row = int(rim_rows[outlet])

    if "moraine" in masks:
        moraine_mask = np.asarray(masks["moraine"], dtype=bool)
    else:
        distance = ndimage.distance_transform_edt(~lake_mask, sampling=dx)
        downstream = np.zeros_like(lake_mask)
        downstream[moraine_row:, :] = True
        moraine_mask = (
            (distance <= float(moraine_band_m))
            & downstream
            & (z > lake_bed_min)
            & (~lake_mask)
        )
    if not moraine_mask.any():
        raise ValueError("Empty moraine mask; widen moraine_band_m or supply domain.masks.moraine.")

    terrain = ValleyTerrain(
        z=z, dx=float(dx), x=x, y=y,
        lake_mask=lake_mask, moraine_mask=moraine_mask,
        thalweg_col=np.argmin(z, axis=1).astype(int),
        crest_elevation=crest_elevation,
        initial_lake_level=lake_level,
        moraine_row=moraine_row,
        meta={"source": "dem"},
    )
    terrain.meta.update(
        {
            "lake_bed_min_m": lake_bed_min,
            "initial_lake_area_m2": terrain.lake_area(lake_level),
            "initial_lake_volume_m3": terrain.lake_volume(lake_level),
            "moraine_height_m": float(z[moraine_mask].max() - lake_bed_min),
            "delineation": "auto" if not masks else "masks",
        }
    )
    return terrain
