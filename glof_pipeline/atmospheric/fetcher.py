"""Initial conditions for the atmospheric forecast.

Production tier
    Returns a live Earth2Studio ``DataSource``. Earth2Studio 0.18 exposes GFS,
    GFS_FX, ARCO, IFS, HRRR, NCAR_ERA5, CDS, CMIP6 and GOES; the analysis time is
    snapped back to the most recent synoptic cycle the provider publishes.

Toy tier
    Returns a description of the same request without touching the network, so the
    run manifest records what *would* have been fetched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

_SUPPORTED_SOURCES = {
    "GFS", "GFS_FX", "ARCO", "IFS", "HRRR", "NCAR_ERA5", "CDS", "CMIP6", "GOES", "Random",
}


def most_recent_cycle(now: datetime | None = None, cycle_hours: int = 6) -> datetime:
    """Snap to the latest synoptic cycle, allowing for provider publication lag."""
    now = now or datetime.now(UTC)
    lagged = now - timedelta(hours=cycle_hours)
    hour = (lagged.hour // cycle_hours) * cycle_hours
    return lagged.replace(hour=hour, minute=0, second=0, microsecond=0)


def fetch_initial_conditions(
    cfg: dict[str, Any], tier: str = "toy", now: datetime | None = None
) -> dict[str, Any]:
    """Describe (toy) or open (production) the atmospheric initial-condition source."""
    source_name = str(cfg["data_source"])
    if source_name not in _SUPPORTED_SOURCES:
        raise ValueError(
            f"Unsupported Earth2Studio data source {source_name!r}. "
            f"Choose from {sorted(_SUPPORTED_SOURCES)}."
        )

    cycle = most_recent_cycle(now)
    request: dict[str, Any] = {
        "data_source": source_name,
        "analysis_time": cycle.isoformat(),
        "variables": list(cfg["variables"]),
        "region": dict(cfg["target_region"]),
        "tier": tier,
    }

    if tier != "production":
        return request

    try:
        import earth2studio.data as e2data
    except ImportError as exc:
        raise RuntimeError(
            "earth2studio is required for the production tier: "
            "pip install -r requirements-nvidia.txt"
        ) from exc

    source_class = getattr(e2data, source_name, None)
    if source_class is None:
        raise RuntimeError(
            f"earth2studio.data has no attribute {source_name!r}; "
            "check the installed earth2studio version."
        )
    request["source"] = source_class()
    return request
