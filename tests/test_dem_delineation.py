"""Delineation of a lake and moraine from a surveyed raster.

Exercised on a small synthetic raster rather than a real DEM: the point is that the
depression-filling delineation finds the basin, its spill point and a moraine band
downstream, and that supplied survey masks override the automatic result.
"""

from __future__ import annotations

import numpy as np
import pytest

from glof_pipeline.terrain.synthetic_dem import _fill_depressions, delineate_from_dem


def _raster_with_a_basin() -> tuple[np.ndarray, float]:
    """A tilted plane with a circular over-deepening behind a transverse ridge."""
    ny = nx = 64
    dx = 50.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    z = 5200.0 - 0.02 * dx * yy
    basin = 40.0 * np.exp(-(((yy - 20) / 6.0) ** 2) - (((xx - 32) / 8.0) ** 2))
    ridge = 30.0 * np.exp(-((yy - 30) / 2.5) ** 2)
    return z - basin + ridge, dx


def test_depression_filling_raises_the_basin_to_its_spill_point() -> None:
    z, _ = _raster_with_a_basin()
    filled = _fill_depressions(z)
    assert np.all(filled >= z - 1e-9)          # filling never lowers ground
    assert filled.max() == pytest.approx(z.max(), abs=1e-6)  # peaks are untouched
    assert (filled - z).max() > 5.0            # the basin actually filled


def test_delineation_finds_the_lake_and_its_crest() -> None:
    z, dx = _raster_with_a_basin()
    terrain = delineate_from_dem(z, dx, min_lake_depth_m=2.0, freeboard_m=3.0)
    assert terrain.lake_mask.any()
    assert terrain.moraine_mask.any()
    # The crest is the spill point: above the lake bed, below the ridge top.
    assert terrain.crest_elevation > z[terrain.lake_mask].min()
    assert terrain.crest_elevation <= z.max()
    assert terrain.initial_lake_level == pytest.approx(terrain.crest_elevation - 3.0)
    assert terrain.meta["initial_lake_volume_m3"] > 0.0
    assert terrain.meta["delineation"] == "auto"


def test_lake_and_moraine_do_not_overlap() -> None:
    z, dx = _raster_with_a_basin()
    terrain = delineate_from_dem(z, dx)
    assert not np.any(terrain.lake_mask & terrain.moraine_mask)


def test_survey_masks_override_the_automatic_delineation() -> None:
    """Mapped polygons must win: auto-delineation is a starting point, not truth."""
    z, dx = _raster_with_a_basin()
    lake = np.zeros_like(z, dtype=bool)
    lake[18:23, 28:36] = True
    moraine = np.zeros_like(z, dtype=bool)
    moraine[29:33, 20:44] = True
    terrain = delineate_from_dem(z, dx, masks={"lake": lake, "moraine": moraine})
    assert np.array_equal(terrain.lake_mask, lake)
    assert np.array_equal(terrain.moraine_mask, moraine)
    assert terrain.meta["delineation"] == "masks"


def test_a_raster_with_no_basin_is_reported_not_guessed() -> None:
    """A monotonic slope has no lake; the failure must say so."""
    z = np.tile(np.linspace(5200.0, 5000.0, 48), (48, 1))
    with pytest.raises(ValueError, match="No depression deeper"):
        delineate_from_dem(z, 50.0, min_lake_depth_m=2.0)
