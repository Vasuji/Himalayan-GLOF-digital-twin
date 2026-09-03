"""Kilometre-scale downscaling and reduction to catchment-mean forcing.

Production tier
    An Earth2Studio diagnostic model (``earth2studio.models.dx``) applied through
    ``earth2studio.run.diagnostic``.

    A warning that belongs in the code and not only in a README: the published
    CorrDiff checkpoints are **region-specific**. ``CorrDiffTaiwan`` was trained on
    the Taiwan CWA domain and ``CorrDiffCosmoEra5`` over central Europe. Neither is
    valid over the Hindu Kush Himalaya. Using one here would produce fields that
    look plausible and are not, so the production path refuses to run with a
    foreign-domain checkpoint unless it is named explicitly in configuration.

Toy tier
    A narrow PhysicsNeMo CorrDiff (:mod:`glof_pipeline.nvidia.corrdiff`) trained on
    coarsened orographic fields by :mod:`glof_pipeline.training.train_downscaler`.
    Same architecture as the production tier at reduced width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from glof_pipeline.nvidia.corrdiff import CorrDiffDownscaler, build_condition
from glof_pipeline.utils.runtime import get_logger

LOGGER = get_logger("glof.downscaling")

# Checkpoints whose training domain is not the Hindu Kush Himalaya.
_FOREIGN_DOMAIN_MODELS = {"CorrDiffTaiwan", "CorrDiffCosmoEra5"}


def normalise_orography(elevation_m: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-scale orography used as the downscaler's conditioning field."""
    elevation = np.asarray(elevation_m, dtype=float)
    centred = elevation - elevation.mean()
    return centred / max(float(np.abs(centred).max()), 1e-6)


def lapse_correct(
    temperature_c: np.ndarray,
    elevation_m: np.ndarray,
    reference_elevation_m: float,
    lapse_rate_c_per_km: float,
) -> np.ndarray:
    """Move a temperature field from its own orography to a reference elevation."""
    delta = (np.asarray(elevation_m, dtype=float) - reference_elevation_m) / 1000.0
    return np.asarray(temperature_c, dtype=float) + lapse_rate_c_per_km * delta


@torch.no_grad()
def downscale_forecast(
    model: CorrDiffDownscaler,
    coarse_temperature: np.ndarray,
    coarse_precipitation: np.ndarray,
    fine_orography_m: np.ndarray,
    samples: int = 1,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Downscale a ``(nt, h, w)`` coarse sequence onto the terrain grid.

    Returns arrays shaped ``(samples, nt, ny, nx)``; the generative spread across
    samples is the downscaling uncertainty that propagates into the breach
    probability.
    """
    model.eval()
    target_shape = tuple(np.asarray(fine_orography_m).shape)
    orography = normalise_orography(fine_orography_m)
    coarse = np.stack([np.asarray(coarse_temperature), np.asarray(coarse_precipitation)], axis=1)

    condition = build_condition(coarse, orography, target_shape).to(device)
    generated = model(condition, samples=samples)  # (S, nt, 2, ny, nx)
    array = generated.cpu().numpy()
    return {
        "temperature_c": array[:, :, 0],
        "precipitation_mm_per_h": np.clip(array[:, :, 1], 0.0, None),
    }


def run_earth2_downscaling(
    cfg: dict[str, Any],
    atmospheric_cfg: dict[str, Any],
    initial_conditions: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Apply an Earth-2 diagnostic downscaler through ``earth2studio.run.diagnostic``.

    Delegates to :mod:`glof_pipeline.nvidia.earth2`, which also enforces the refusal
    to apply a region-foreign CorrDiff package to the Hindu Kush Himalaya.
    """
    from glof_pipeline.nvidia.earth2 import run_downscaling as _run

    return _run(cfg, atmospheric_cfg, initial_conditions["analysis_time"], output_dir)


def _legacy_run_earth2_downscaling(
    cfg: dict[str, Any],
    atmospheric_cfg: dict[str, Any],
    initial_conditions: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Retained inline implementation, superseded by glof_pipeline.nvidia.earth2."""
    try:
        import earth2studio.models.dx as dx
        import earth2studio.models.px as px
        import earth2studio.run as run
        from earth2studio.io import ZarrBackend
    except ImportError as exc:
        raise RuntimeError("earth2studio is required for the production tier.") from exc

    model_name = str(cfg["production_model"])
    checkpoint = cfg.get("production_checkpoint")
    if model_name in _FOREIGN_DOMAIN_MODELS and checkpoint is None:
        raise RuntimeError(
            f"{model_name} was trained outside the Hindu Kush Himalaya and its default "
            "package must not be used for this domain. Train a CorrDiff for this region "
            "with PhysicsNeMo and set downscaling.production_checkpoint, or select "
            "another downscaler."
        )

    model_class = getattr(dx, model_name, None)
    if model_class is None:
        raise RuntimeError(
            f"earth2studio.models.dx has no model {model_name!r}. Available downscalers "
            "include CorrDiff, CorrDiffTaiwan, CorrDiffCMIP6, CorrDiffCosmoEra5, CBottleSR."
        )

    package = model_class.load_default_package() if checkpoint is None else model_class.load_package(checkpoint)
    downscaler = model_class.load_model(package)

    prognostic_name = str(atmospheric_cfg["prognostic_model"])
    prognostic_class = getattr(px, prognostic_name)
    prognostic = prognostic_class.load_model(prognostic_class.load_default_package())

    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "downscaled_forecast.zarr"
    io = ZarrBackend(str(store_path))
    nsteps = int(atmospheric_cfg["lead_time_hours"] // atmospheric_cfg["time_step_hours"])

    run.diagnostic(
        [initial_conditions["analysis_time"]],
        nsteps,
        prognostic,
        downscaler,
        initial_conditions["source"],
        io,
    )
    LOGGER.info("Downscaled forecast written to %s", store_path)
    return {"store": str(store_path), "model": model_name, "nsteps": nsteps}


def catchment_mean_series(
    field: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Reduce a ``(..., nt, ny, nx)`` field to a ``(..., nt)`` catchment mean."""
    array = np.asarray(field, dtype=float)
    if mask is None:
        return array.mean(axis=(-2, -1))
    weights = np.asarray(mask, dtype=float)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("Catchment mask is empty.")
    return (array * weights).sum(axis=(-2, -1)) / total
