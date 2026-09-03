"""Model construction.

PhysicsNeMo supplies the neural components of the twin and is a hard requirement.
This module is the only place models are constructed, and it raises with a message
naming the missing component when one is unavailable, so a run either uses the
NVIDIA stack throughout or does not proceed.

``physicsnemo.models.fno.FNO`` and ``physicsnemo.models.meshgraphnet.MeshGraphNet``
are built through :mod:`glof_pipeline.nvidia.physicsnemo_models`.
"""

from __future__ import annotations

from typing import Any

from glof_pipeline.nvidia import (
    earth2studio_available,
    nucleus_available,
    nvidia_versions,
    omniverse_available,
    physicsnemo_available,
)
from glof_pipeline.utils.runtime import get_logger

LOGGER = get_logger("glof.backends")

__all__ = [
    "build_fno",
    "build_meshgraphnet",
    "backend_report",
    "require_physicsnemo",
    "resolve_tier",
    "torch_geometric_available",
    "torch_scatter_available",
]

_INSTALL_HINT = (
    "Install with `pip install -r requirements-nvidia.txt`. PhysicsNeMo 2.2.1 requires "
    "torch>=2.10, and an older torch fails at import rather than at call time."
)


def _importable(module: str) -> bool:
    """Whether ``module`` actually imports.

    Deliberately an import rather than ``importlib.util.find_spec``: torch-scatter
    ships a compiled extension built against a specific torch build, and a mismatched
    wheel is present on disk (so ``find_spec`` succeeds) yet raises ``OSError`` when
    its ``.so`` is loaded. Checking the spec alone reports a dependency as satisfied
    and then fails deep inside PhysicsNeMo's message passing.
    """
    import importlib

    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - a broken extension raises OSError, not ImportError
        return False
    return True


def torch_geometric_available() -> bool:
    return _importable("torch_geometric")


def torch_scatter_available() -> bool:
    """PhysicsNeMo's MeshGraphNet needs ``torch_scatter`` for its message passing."""
    return _importable("torch_scatter")


def require_physicsnemo() -> None:
    if not physicsnemo_available():
        raise RuntimeError("nvidia-physicsnemo is required and not importable. " + _INSTALL_HINT)


def resolve_tier(requested: str) -> str:
    """Validate ``runtime.tier``; the production tier additionally requires Earth-2."""
    if requested not in ("toy", "production"):
        raise ValueError(f"runtime.tier must be 'toy' or 'production', got {requested!r}.")
    require_physicsnemo()
    if requested == "production" and not earth2studio_available():
        raise RuntimeError("runtime.tier=production requires earth2studio. " + _INSTALL_HINT)
    return requested


def build_fno(cfg: dict[str, Any]) -> tuple[Any, str]:
    """Build the flood-routing operator: ``physicsnemo.models.fno.FNO``."""
    require_physicsnemo()
    from glof_pipeline.nvidia.physicsnemo_models import build_fno as _build

    model, dropped = _build(cfg)
    LOGGER.info("FNO: physicsnemo.models.fno.FNO")
    if dropped:
        LOGGER.info("FNO: arguments not accepted by the installed release: %s", dropped)
    return model, "physicsnemo"


def build_meshgraphnet(cfg: dict[str, Any]) -> tuple[Any, str]:
    """Build the moraine-mechanics operator: PhysicsNeMo's ``MeshGraphNet``.

    Needs ``torch_geometric`` for the graph container and ``torch_scatter`` for the
    message passing itself. Construction succeeds without the latter and the forward
    pass then fails inside PhysicsNeMo, so both are checked up front.
    """
    require_physicsnemo()
    missing = [
        name
        for name, present in (
            ("torch-geometric", torch_geometric_available()),
            ("torch-scatter", torch_scatter_available()),
        )
        if not present
    ]
    if missing:
        raise RuntimeError(
            "PhysicsNeMo MeshGraphNet needs " + " and ".join(missing) + " for its message "
            "passing (DGL support was withdrawn upstream). torch-scatter compiles against "
            "the installed torch and needs a C++ toolchain, so on a machine without one "
            "the moraine surrogate must be trained on the GPU host. " + _INSTALL_HINT
        )
    from glof_pipeline.nvidia.physicsnemo_models import build_meshgraphnet as _build

    model, dropped = _build(cfg)
    LOGGER.info("MeshGraphNet: physicsnemo.models.meshgraphnet.MeshGraphNet")
    if dropped:
        LOGGER.info("MeshGraphNet: arguments not accepted by the installed release: %s", dropped)
    return model, "physicsnemo"


def backend_report() -> dict[str, Any]:
    """Component availability and versions, recorded in every run manifest."""
    return {
        "physicsnemo": physicsnemo_available(),
        "earth2studio": earth2studio_available(),
        "torch_geometric": torch_geometric_available(),
        "torch_scatter": torch_scatter_available(),
        "omniverse_usd": omniverse_available(),
        "omniverse_nucleus": nucleus_available(),
        "versions": nvidia_versions(),
    }
