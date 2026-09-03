"""Omniverse USD authoring, exercised against a real ``pxr`` stage.

These tests open the authored stage with ``Usd`` and inspect it, rather than
checking that a file exists: a USD file that opens but carries no materials,
lighting or time samples is not a usable deliverable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

from glof_pipeline.nvidia.omniverse import author_flood_scene  # noqa: E402


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """A four-frame flood over a tilted bed."""
    ny = nx = 24
    bed = np.tile(np.linspace(50.0, 0.0, ny)[:, None], (1, nx)).astype(float)
    depth = np.zeros((4, ny, nx))
    for k in range(4):
        depth[k, : 6 + 4 * k, :] = 1.0 + 0.5 * k
    time_s = np.arange(4) * 60.0
    path = tmp_path_factory.mktemp("usd") / "flood.usda"
    return author_flood_scene(
        bed_elevation=bed, depth_sequence=depth, time_s=time_s, dx=25.0,
        output_path=path, threshold_m=0.05, stride=1, frames=4,
    )


def test_root_layer_composes_terrain_and_water(scene) -> None:
    stage = Usd.Stage.Open(scene["root"])
    assert stage is not None
    sublayers = list(stage.GetRootLayer().subLayerPaths)
    assert len(sublayers) == 2, sublayers
    assert stage.GetDefaultPrim(), "root stage must declare a default prim"


def test_terrain_and_water_are_real_meshes(scene) -> None:
    stage = Usd.Stage.Open(scene["root"])
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) >= 2, [p.GetPath() for p in meshes]
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        attribute = mesh.GetPointsAttr()
        # The water surface authors points as time samples only, so a default-time
        # Get() correctly returns None; query the first sample instead.
        samples = attribute.GetTimeSamples()
        points = attribute.Get(samples[0]) if samples else attribute.Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        assert points, f"{prim.GetPath()} has no points"
        assert counts and indices
        # Quad topology: every face has four vertices and every index is in range.
        assert set(counts) == {4}
        assert max(indices) < len(points)


def test_water_surface_is_animated(scene) -> None:
    """Time samples are what make this a flood rather than a still."""
    stage = Usd.Stage.Open(scene["root"])
    water = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and "water" in str(p.GetPath()).lower()]
    assert water, [str(p.GetPath()) for p in stage.Traverse()]
    samples = UsdGeom.Mesh(water[0]).GetPointsAttr().GetTimeSamples()
    assert len(samples) >= 2, samples
    assert stage.GetStartTimeCode() < stage.GetEndTimeCode()


def test_materials_and_lighting_are_authored(scene) -> None:
    stage = Usd.Stage.Open(scene["root"])
    materials = [p for p in stage.Traverse() if p.IsA(UsdShade.Material)]
    lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
    cameras = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    assert materials, "no UsdShade.Material in the stage"
    assert lights, "no UsdLux light in the stage"
    assert cameras, "no review camera in the stage"


def test_simulated_time_is_recorded_in_the_layer(scene) -> None:
    """A reviewer must be able to read simulated seconds off the asset itself."""
    stage = Usd.Stage.Open(scene["root"])
    data = stage.GetRootLayer().customLayerData
    assert data, "no customLayerData on the root layer"
    assert any("time" in str(k).lower() or "second" in str(k).lower() for k in data), list(data)
