"""Earth-2 atmospheric forecasting and downscaling through Earth2Studio.

All Earth2Studio entry points the twin uses live here with explicit imports:
prognostic models (``models.px``), diagnostic downscalers (``models.dx``), data
sources (``data``), perturbation methods (``perturbation``), the Zarr writer
(``io.ZarrBackend``) and the workflow drivers (``run``).

Region safety
-------------
The published CorrDiff checkpoints are trained per region -- ``CorrDiffTaiwan`` on
the Taiwan CWA domain, ``CorrDiffCosmoEra5`` over central Europe. Neither is valid
over the Hindu Kush Himalaya, and applying one there fabricates orographic
structure from the wrong mountain range. :func:`load_downscaler` refuses those
checkpoints unless the configuration names an explicit override.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from glof_pipeline.nvidia import require
from glof_pipeline.utils.runtime import get_logger

require("earth2studio")

import earth2studio.data as e2data  # noqa: E402
import earth2studio.models.dx as dx  # noqa: E402
import earth2studio.models.px as px  # noqa: E402
import earth2studio.perturbation as e2perturbation  # noqa: E402
import earth2studio.run as e2run  # noqa: E402
from earth2studio.io import ZarrBackend  # noqa: E402

LOGGER = get_logger("glof.earth2")

__all__ = [
    "available_data_sources",
    "available_downscalers",
    "available_prognostics",
    "load_data_source",
    "load_downscaler",
    "load_prognostic",
    "most_recent_cycle",
    "run_downscaling",
    "run_forecast",
]

# CorrDiff packages whose training domain is not the Hindu Kush Himalaya.
FOREIGN_DOMAIN_DOWNSCALERS = frozenset({"CorrDiffTaiwan", "CorrDiffCosmoEra5"})


def _public_names(module: Any) -> list[str]:
    return sorted(n for n in dir(module) if not n.startswith("_") and n[0].isupper())


def available_prognostics() -> list[str]:
    """Prognostic models exposed by the installed ``earth2studio.models.px``."""
    return _public_names(px)


def available_downscalers() -> list[str]:
    return _public_names(dx)


def available_data_sources() -> list[str]:
    return _public_names(e2data)


def most_recent_cycle(now: datetime | None = None, cycle_hours: int = 6) -> datetime:
    """Latest synoptic cycle, allowing for provider publication lag."""
    now = now or datetime.now(UTC)
    lagged = now - timedelta(hours=cycle_hours)
    return lagged.replace(hour=(lagged.hour // cycle_hours) * cycle_hours, minute=0, second=0, microsecond=0)


def load_data_source(name: str) -> Any:
    """Open an Earth2Studio data source by name."""
    source_class = getattr(e2data, name, None)
    if source_class is None:
        raise RuntimeError(
            f"earth2studio.data has no source {name!r}. Available: {available_data_sources()}"
        )
    LOGGER.info("Earth-2 data source: earth2studio.data.%s", name)
    return source_class()


def load_prognostic(name: str) -> Any:
    """Load a prognostic model from its default package."""
    model_class = getattr(px, name, None)
    if model_class is None:
        raise RuntimeError(
            f"earth2studio.models.px has no model {name!r}. Available: {available_prognostics()}"
        )
    LOGGER.info("Earth-2 prognostic model: earth2studio.models.px.%s", name)
    return model_class.load_model(model_class.load_default_package())


def load_perturbation(name: str, noise_amplitude: float) -> Any:
    """Instantiate an ensemble perturbation method."""
    method_class = getattr(e2perturbation, name, None)
    if method_class is None:
        raise RuntimeError(
            f"earth2studio.perturbation has no method {name!r}. Available: "
            f"{_public_names(e2perturbation)}"
        )
    LOGGER.info("Earth-2 perturbation: earth2studio.perturbation.%s", name)
    try:
        return method_class(noise_amplitude=noise_amplitude)
    except TypeError:
        # Zero and some deterministic methods take no amplitude.
        return method_class()


def load_downscaler(name: str, checkpoint: str | None = None) -> Any:
    """Load a diagnostic downscaler, refusing region-foreign default packages."""
    if name in FOREIGN_DOMAIN_DOWNSCALERS and checkpoint is None:
        raise RuntimeError(
            f"{name} was trained outside the Hindu Kush Himalaya and its default package "
            "must not be applied to this domain. Train a CorrDiff for this region with "
            "PhysicsNeMo and set downscaling.production_checkpoint, or choose another "
            "downscaler. See docs/PRODUCTION.md."
        )
    model_class = getattr(dx, name, None)
    if model_class is None:
        raise RuntimeError(
            f"earth2studio.models.dx has no model {name!r}. Available: {available_downscalers()}"
        )
    package = model_class.load_default_package() if checkpoint is None else model_class.load_package(checkpoint)
    LOGGER.info("Earth-2 downscaler: earth2studio.models.dx.%s (checkpoint=%s)", name, checkpoint)
    return model_class.load_model(package)


def run_forecast(cfg: dict[str, Any], analysis_time: str, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic or ensemble Earth-2 forecast into a Zarr store."""
    model_name = str(cfg["prognostic_model"])
    model = load_prognostic(model_name)
    source = load_data_source(str(cfg["data_source"]))

    nsteps = int(cfg["lead_time_hours"] // cfg["time_step_hours"])
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / f"{model_name.lower()}_forecast.zarr"
    io = ZarrBackend(str(store_path))

    ensemble_cfg = cfg["ensemble"]
    if bool(ensemble_cfg["enabled"]):
        members = int(ensemble_cfg["members"])
        perturbation = load_perturbation(
            str(ensemble_cfg["perturbation"]), float(ensemble_cfg["noise_amplitude"])
        )
        e2run.ensemble(
            [analysis_time], nsteps, members, model, source, io, perturbation,
            batch_size=int(ensemble_cfg["batch_size"]),
        )
    else:
        members = 1
        e2run.deterministic([analysis_time], nsteps, model, source, io)

    LOGGER.info("Earth-2 forecast written to %s (%d steps, %d members)", store_path, nsteps, members)
    return {"store": str(store_path), "model": model_name, "nsteps": nsteps, "members": members}


def run_downscaling(
    downscaling_cfg: dict[str, Any],
    atmospheric_cfg: dict[str, Any],
    analysis_time: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Apply a diagnostic downscaler through ``earth2studio.run.diagnostic``."""
    downscaler = load_downscaler(
        str(downscaling_cfg["production_model"]), downscaling_cfg.get("production_checkpoint")
    )
    prognostic = load_prognostic(str(atmospheric_cfg["prognostic_model"]))
    source = load_data_source(str(atmospheric_cfg["data_source"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "downscaled_forecast.zarr"
    io = ZarrBackend(str(store_path))
    nsteps = int(atmospheric_cfg["lead_time_hours"] // atmospheric_cfg["time_step_hours"])

    e2run.diagnostic([analysis_time], nsteps, prognostic, downscaler, source, io)
    LOGGER.info("Downscaled forecast written to %s", store_path)
    return {
        "store": str(store_path),
        "model": str(downscaling_cfg["production_model"]),
        "nsteps": nsteps,
    }
