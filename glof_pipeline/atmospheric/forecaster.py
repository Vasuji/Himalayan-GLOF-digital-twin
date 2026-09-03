"""Global forecast: Earth-2 in production, a stochastic generator in the toy tier.

The toy generator is not a weather model and is not presented as one. It produces
an ensemble of coarse (~25 km) temperature and precipitation sequences with the
temporal persistence, spatial correlation, diurnal cycle, synoptic warm spell and
convective precipitation burst that determine whether a moraine-dammed lake fills
fast enough to fail. That is the signal the downstream stages consume, so the
pipeline can be exercised end to end without a GPU or an NGC key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass
class ForecastBundle:
    """Coarse-resolution ensemble forecast over the target region."""

    time_h: np.ndarray             # (nt,)
    temperature_c: np.ndarray      # (members, nt, h, w) at the coarse orography
    precipitation_mm_per_h: np.ndarray
    coarse_orography_m: np.ndarray  # (h, w)
    source: str
    members: int
    meta: dict[str, Any] = field(default_factory=dict)

    def ensemble_mean(self) -> tuple[np.ndarray, np.ndarray]:
        return self.temperature_c.mean(axis=0), self.precipitation_mm_per_h.mean(axis=0)


def _correlated_noise(
    shape: tuple[int, int], correlation_cells: float, rng: np.random.Generator
) -> np.ndarray:
    """Unit-variance Gaussian field with an imposed spatial correlation length."""
    white = rng.standard_normal(shape)
    if correlation_cells <= 0.0:
        return white
    smoothed = ndimage.gaussian_filter(white, sigma=correlation_cells, mode="wrap")
    std = smoothed.std()
    return smoothed / std if std > 1e-12 else smoothed


def coarsen(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Block-mean a fine field onto a coarse grid (the forward operator of downscaling)."""
    ny, nx = field.shape
    ty, tx = shape
    rows = np.array_split(np.arange(ny), ty)
    cols = np.array_split(np.arange(nx), tx)
    out = np.empty((ty, tx))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            out[i, j] = field[np.ix_(r, c)].mean()
    return out


def run_toy_forecast(
    cfg: dict[str, Any], fine_orography_m: np.ndarray, rng: np.random.Generator
) -> ForecastBundle:
    """Generate the coarse ensemble forecast used by the toy tier."""
    generator = cfg["toy_generator"]
    coarse_shape = tuple(int(v) for v in generator["coarse_shape"])
    lead_hours = float(cfg["lead_time_hours"])
    step_hours = float(cfg["time_step_hours"])
    time_h = np.arange(0.0, lead_hours + step_hours, step_hours)
    n_time = time_h.size

    ensemble_cfg = cfg["ensemble"]
    members = int(ensemble_cfg["members"]) if bool(ensemble_cfg["enabled"]) else 1

    coarse_orography = coarsen(np.asarray(fine_orography_m, dtype=float), coarse_shape)
    reference_elevation = float(cfg["target_region"]["reference_elevation_m"])
    lapse = float(generator["lapse_rate_c_per_km"])
    # Elevation-driven temperature offset relative to the lake surface.
    elevation_offset = -lapse * (coarse_orography - reference_elevation) / 1000.0
    # Normalised relief drives orographic precipitation enhancement.
    relief = coarse_orography - coarse_orography.mean()
    relief_norm = relief / max(float(np.abs(relief).max()), 1e-6)

    rho = float(generator["ar1_correlation"])
    correlation_cells = float(generator["spatial_correlation_km"]) / 25.0
    sigma_t = float(generator["temperature_noise_c"])
    sigma_p = float(generator["precip_noise_mm_per_h"])

    temperature = np.zeros((members, n_time, *coarse_shape))
    precipitation = np.zeros_like(temperature)

    for member in range(members):
        # Ensemble spread: perturb the amplitude and timing of the two forcing events.
        warm_amplitude = float(generator["warm_spell_amplitude_c"]) * (1.0 + 0.18 * rng.standard_normal())
        warm_centre = float(generator["warm_spell_centre_frac"]) + 0.05 * rng.standard_normal()
        burst_amplitude = float(generator["precip_burst_mm_per_h"]) * np.exp(0.28 * rng.standard_normal())
        burst_centre = float(generator["precip_burst_centre_frac"]) + 0.04 * rng.standard_normal()

        noise_t = _correlated_noise(coarse_shape, correlation_cells, rng)
        noise_p = _correlated_noise(coarse_shape, correlation_cells, rng)

        for k, t in enumerate(time_h):
            phase = t / max(lead_hours, 1e-9)
            diurnal = float(generator["diurnal_amplitude_c"]) * np.sin(2.0 * np.pi * t / 24.0 - np.pi / 2.0)
            warm_spell = warm_amplitude * np.exp(
                -0.5 * ((phase - warm_centre) / float(generator["warm_spell_width_frac"])) ** 2
            )
            burst = burst_amplitude * np.exp(
                -0.5 * ((phase - burst_centre) / float(generator["precip_burst_width_frac"])) ** 2
            )

            innovation_t = _correlated_noise(coarse_shape, correlation_cells, rng)
            innovation_p = _correlated_noise(coarse_shape, correlation_cells, rng)
            noise_t = rho * noise_t + np.sqrt(1.0 - rho**2) * innovation_t
            noise_p = rho * noise_p + np.sqrt(1.0 - rho**2) * innovation_p

            temperature[member, k] = (
                float(generator["base_temperature_c"])
                + diurnal
                + warm_spell
                + elevation_offset
                + sigma_t * noise_t
            )
            precipitation[member, k] = np.clip(
                float(generator["precip_base_mm_per_h"])
                + burst * (1.0 + 0.6 * relief_norm)
                + sigma_p * noise_p,
                0.0,
                None,
            )

    return ForecastBundle(
        time_h=time_h,
        temperature_c=temperature,
        precipitation_mm_per_h=precipitation,
        coarse_orography_m=coarse_orography,
        source="toy_stochastic_generator",
        members=members,
        meta={
            "coarse_shape": coarse_shape,
            "lead_time_hours": lead_hours,
            "time_step_hours": step_hours,
            "note": "synthetic forcing; not a weather forecast",
        },
    )


def run_earth2_forecast(
    cfg: dict[str, Any], initial_conditions: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Run the configured Earth-2 prognostic model and return the Zarr store path.

    Delegates to :mod:`glof_pipeline.nvidia.earth2`, where every Earth2Studio entry
    point is imported explicitly. Regional extraction from the store happens in
    :mod:`glof_pipeline.atmospheric.downscaling`.
    """
    from glof_pipeline.nvidia.earth2 import run_forecast as _run

    return _run(cfg, initial_conditions["analysis_time"], output_dir)


def _legacy_run_earth2_forecast(
    cfg: dict[str, Any], initial_conditions: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Retained inline implementation, superseded by glof_pipeline.nvidia.earth2."""
    try:
        import earth2studio.models.px as px
        import earth2studio.perturbation as perturbation_module
        import earth2studio.run as run
        from earth2studio.io import ZarrBackend
    except ImportError as exc:
        raise RuntimeError(
            "earth2studio is required for the production tier: "
            "pip install -r requirements-nvidia.txt"
        ) from exc

    model_name = str(cfg["prognostic_model"])
    model_class = getattr(px, model_name, None)
    if model_class is None:
        raise RuntimeError(
            f"earth2studio.models.px has no model {model_name!r}. Available models include "
            "FCN3, SFNO, GraphCastOperational, GraphCastSmall, AIFS, AIFSENS, Pangu24, "
            "Aurora, DLWP, DLESyM."
        )

    package = model_class.load_default_package()
    model = model_class.load_model(package)

    nsteps = int(cfg["lead_time_hours"] // cfg["time_step_hours"])
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / f"{model_name.lower()}_forecast.zarr"
    io = ZarrBackend(str(store_path))

    times = [initial_conditions["analysis_time"]]
    ensemble_cfg = cfg["ensemble"]

    if bool(ensemble_cfg["enabled"]):
        perturbation_name = str(ensemble_cfg["perturbation"])
        perturbation_class = getattr(perturbation_module, perturbation_name, None)
        if perturbation_class is None:
            raise RuntimeError(
                f"earth2studio.perturbation has no method {perturbation_name!r}. Available: "
                "Zero, Gaussian, SphericalGaussian, CorrelatedSphericalGaussian, BredVector, "
                "HemisphericCentredBredVector, Brown, LaggedEnsemble."
            )
        run.ensemble(
            times,
            nsteps,
            int(ensemble_cfg["members"]),
            model,
            initial_conditions["source"],
            io,
            perturbation_class(noise_amplitude=float(ensemble_cfg["noise_amplitude"])),
            batch_size=int(ensemble_cfg["batch_size"]),
        )
    else:
        run.deterministic(times, nsteps, model, initial_conditions["source"], io)

    return {
        "store": str(store_path),
        "model": model_name,
        "nsteps": nsteps,
        "members": int(ensemble_cfg["members"]) if bool(ensemble_cfg["enabled"]) else 1,
    }


def run_forecast(
    cfg: dict[str, Any],
    initial_conditions: dict[str, Any],
    fine_orography_m: np.ndarray,
    rng: np.random.Generator,
    output_dir: Path,
    tier: str = "toy",
) -> ForecastBundle | dict[str, Any]:
    """Tier dispatch for the atmospheric forecast stage."""
    if tier == "production":
        return run_earth2_forecast(cfg, initial_conditions, output_dir)
    return run_toy_forecast(cfg, fine_orography_m, rng)
