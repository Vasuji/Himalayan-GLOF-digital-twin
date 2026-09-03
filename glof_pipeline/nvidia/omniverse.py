"""Omniverse scene authoring: a USD stage an artist or reviewer can actually open.

A heightfield mesh with no material, no lighting and no camera technically opens
in usdview and tells a reviewer nothing. This module authors a scene structured
the way Omniverse expects:

* **layered composition** -- static terrain and the animated water surface are
  written as separate sublayers referenced by a thin root stage, so the terrain is
  loaded once and the (much larger) water animation can be reloaded or swapped
  without touching it;
* **materials** -- a ``UsdShade`` water material with depth-driven opacity, so
  shallow margins read as shallow rather than as a uniform blue sheet;
* **lighting and camera** -- a ``UsdLux`` distant light plus a dome light, and a
  camera framed on the valley, so the scene renders without manual setup;
* **physical time** -- time codes carry the simulation's seconds, recorded in
  stage metadata, so the animation is quantitative rather than decorative.

Publishing to a Nucleus server is optional and only attempted when ``omni.client``
is importable and a server URL is configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from glof_pipeline.nvidia import nucleus_available, require
from glof_pipeline.utils.runtime import get_logger

require("omniverse")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

LOGGER = get_logger("glof.omniverse")

__all__ = ["author_flood_scene", "publish_to_nucleus"]


def _grid_topology(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(ny * nx).reshape(ny, nx)
    quads = np.stack(
        [
            indices[:-1, :-1].ravel(),
            indices[:-1, 1:].ravel(),
            indices[1:, 1:].ravel(),
            indices[1:, :-1].ravel(),
        ],
        axis=1,
    )
    return np.full(quads.shape[0], 4, dtype=np.int32), quads.ravel().astype(np.int32)


def _points(bed: np.ndarray, surface: np.ndarray, spacing: float) -> np.ndarray:
    ny, nx = bed.shape
    grid_x, grid_y = np.meshgrid((np.arange(nx) + 0.5) * spacing, (np.arange(ny) + 0.5) * spacing)
    return np.stack([grid_x.ravel(), grid_y.ravel(), surface.ravel()], axis=1).astype(np.float32)


def _vec3f_array(points: np.ndarray):
    from pxr import Vt

    return Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])


def _int_array(values: np.ndarray):
    from pxr import Vt

    return Vt.IntArray([int(v) for v in values])


def _define_mesh(stage: Usd.Stage, path: str, points: np.ndarray, counts: np.ndarray, indices: np.ndarray):
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(path))
    mesh.CreatePointsAttr(_vec3f_array(points))
    mesh.CreateFaceVertexCountsAttr(_int_array(counts))
    mesh.CreateFaceVertexIndicesAttr(_int_array(indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def _water_material(stage: Usd.Stage, path: str, opacity: float, colour: tuple[float, float, float]):
    """A UsdPreviewSurface water material: physically plausible and renderer-agnostic."""
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.08)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)  # water
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _terrain_material(stage: Usd.Stage, path: str):
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.42, 0.40, 0.36))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _add_lighting(stage: Usd.Stage, parent: str) -> None:
    sun = UsdLux.DistantLight.Define(stage, Sdf.Path(f"{parent}/SunLight"))
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(0.53)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 25.0, 0.0))
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(f"{parent}/SkyLight"))
    dome.CreateIntensityAttr(600.0)


def _add_camera(stage: Usd.Stage, parent: str, extent_x: float, extent_y: float, elevation: float) -> None:
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(f"{parent}/ReviewCamera"))
    camera.CreateFocalLengthAttr(35.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(1.0, 10.0 * max(extent_x, extent_y)))
    translate = Gf.Vec3d(0.5 * extent_x, -0.85 * extent_y, elevation + 0.55 * extent_y)
    xform = UsdGeom.Xformable(camera)
    xform.AddTranslateOp().Set(translate)
    xform.AddRotateXYZOp().Set(Gf.Vec3f(58.0, 0.0, 0.0))


def author_flood_scene(
    bed_elevation: np.ndarray,
    depth_sequence: np.ndarray,
    time_s: np.ndarray,
    dx: float,
    output_path: str | Path,
    threshold_m: float = 0.05,
    stride: int = 2,
    frames: int | None = None,
    frames_per_second: float = 12.0,
    water_opacity: float = 0.72,
    water_colour: tuple[float, float, float] = (0.10, 0.32, 0.55),
) -> dict[str, str]:
    """Author a layered Omniverse-ready USD scene of the routed flood.

    Returns the paths of the root stage and the two sublayers. ``depth_sequence`` is
    ``(nt, ny, nx)``; ``stride`` decimates the grid so a production-resolution
    domain stays streamable.
    """
    bed = np.asarray(bed_elevation, dtype=float)[::stride, ::stride]
    depths = np.asarray(depth_sequence, dtype=float)[:, ::stride, ::stride]
    times = np.asarray(time_s, dtype=float)
    if frames is not None and depths.shape[0] > frames:
        selection = np.linspace(0, depths.shape[0] - 1, frames).astype(int)
        depths, times = depths[selection], times[selection]

    ny, nx = bed.shape
    spacing = dx * stride
    counts, indices = _grid_topology(ny, nx)

    root_path = Path(output_path)
    root_path.parent.mkdir(parents=True, exist_ok=True)
    stem = root_path.with_suffix("")
    terrain_path = Path(f"{stem}_terrain.usda")
    water_path = Path(f"{stem}_water.usda")

    # --- terrain sublayer (static) -----------------------------------------
    terrain_stage = Usd.Stage.CreateNew(str(terrain_path))
    UsdGeom.SetStageUpAxis(terrain_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(terrain_stage, 1.0)
    world = UsdGeom.Xform.Define(terrain_stage, Sdf.Path("/World"))
    terrain_stage.SetDefaultPrim(world.GetPrim())
    terrain_mesh = _define_mesh(
        terrain_stage, "/World/Terrain", _points(bed, bed, spacing), counts, indices
    )
    material = _terrain_material(terrain_stage, "/World/Looks/TerrainMaterial")
    UsdShade.MaterialBindingAPI(terrain_mesh).Bind(material)
    _add_lighting(terrain_stage, "/World")
    _add_camera(terrain_stage, "/World", nx * spacing, ny * spacing, float(bed.max()))
    terrain_stage.GetRootLayer().Save()

    # --- water sublayer (animated) -----------------------------------------
    water_stage = Usd.Stage.CreateNew(str(water_path))
    UsdGeom.SetStageUpAxis(water_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(water_stage, 1.0)
    water_world = UsdGeom.Xform.Define(water_stage, Sdf.Path("/World"))
    water_stage.SetDefaultPrim(water_world.GetPrim())
    water_mesh = UsdGeom.Mesh.Define(water_stage, Sdf.Path("/World/FloodWaterSurface"))
    water_mesh.CreateFaceVertexCountsAttr(_int_array(counts))
    water_mesh.CreateFaceVertexIndicesAttr(_int_array(indices))
    water_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    water_material = _water_material(water_stage, "/World/Looks/WaterMaterial", water_opacity, water_colour)
    UsdShade.MaterialBindingAPI(water_mesh).Bind(water_material)

    points_attr = water_mesh.CreatePointsAttr()
    # Dry cells are dropped onto the bed rather than floating a film of water.
    for frame_index, depth in enumerate(depths):
        surface = np.where(depth >= threshold_m, bed + depth, bed)
        points_attr.Set(_vec3f_array(_points(bed, surface, spacing)), Usd.TimeCode(float(frame_index)))

    water_stage.SetStartTimeCode(0.0)
    water_stage.SetEndTimeCode(float(max(len(depths) - 1, 0)))
    water_stage.SetTimeCodesPerSecond(frames_per_second)
    water_stage.GetRootLayer().Save()

    # --- root stage: composes the two sublayers ----------------------------
    root_stage = Usd.Stage.CreateNew(str(root_path))
    UsdGeom.SetStageUpAxis(root_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(root_stage, 1.0)
    root_layer = root_stage.GetRootLayer()
    root_layer.subLayerPaths = [terrain_path.name, water_path.name]
    # A USD asset intended to be referenced must declare a default prim, otherwise a
    # consumer referencing the file has no unambiguous entry point and Omniverse
    # reports the stage as having no default prim. Both sublayers define /World, so
    # the composed root resolves to the same path.
    root_world = root_stage.GetPrimAtPath(Sdf.Path("/World"))
    if not root_world:
        root_world = UsdGeom.Xform.Define(root_stage, Sdf.Path("/World")).GetPrim()
    root_stage.SetDefaultPrim(root_world)
    root_stage.SetStartTimeCode(0.0)
    root_stage.SetEndTimeCode(float(max(len(depths) - 1, 0)))
    root_stage.SetTimeCodesPerSecond(frames_per_second)
    root_layer.customLayerData = {
        "glof_simulated_seconds": float(times[-1]) if times.size else 0.0,
        "glof_frames": int(len(depths)),
        "glof_grid": f"{ny}x{nx}",
        "glof_cell_size_m": float(spacing),
        "glof_depth_threshold_m": float(threshold_m),
    }
    root_layer.Save()

    LOGGER.info("Omniverse scene: %s (+ %s, %s)", root_path.name, terrain_path.name, water_path.name)
    return {"root": str(root_path), "terrain": str(terrain_path), "water": str(water_path)}


def publish_to_nucleus(paths: dict[str, str], server_url: str) -> dict[str, str]:
    """Copy an authored scene to a Nucleus server via ``omni.client``.

    Returns the remote URLs. Raises if ``omni.client`` is unavailable, since a
    silent no-op here would leave a reviewer waiting for an asset that never
    appears.
    """
    if not nucleus_available():
        raise RuntimeError(
            "Publishing needs omni.client, which ships with an Omniverse Kit or Launcher "
            "install and is not on PyPI. Copy the .usda files manually, or run this from "
            "an Omniverse Kit environment."
        )
    import omni.client  # noqa: PLC0415 - optional Omniverse dependency

    published: dict[str, str] = {}
    base = server_url.rstrip("/")
    for role, local_path in paths.items():
        remote = f"{base}/{Path(local_path).name}"
        result = omni.client.copy(local_path, remote, omni.client.CopyBehavior.OVERWRITE)
        if result != omni.client.Result.OK:
            raise RuntimeError(f"Nucleus copy of {local_path} to {remote} failed: {result}")
        published[role] = remote
        LOGGER.info("Published %s -> %s", Path(local_path).name, remote)
    return published
