"""NVIDIA stack adapters: PhysicsNeMo, Earth-2 (Earth2Studio) and Omniverse.

Every module in this package imports the NVIDIA libraries **explicitly at module
scope**. That is deliberate: importing them lazily inside functions hides which
NVIDIA APIs the project actually depends on, and makes it impossible to tell by
reading whether a run used PhysicsNeMo or a substitute. Import this package and
the dependency is either satisfied or it fails loudly.

The rest of ``glof_pipeline`` reaches this package through
:mod:`glof_pipeline.backends`, which is the single point of model construction.

Warp kernel cache
-----------------
``physicsnemo.models`` transitively imports ``physicsnemo.nn``, which calls
``warp.init()`` at import time. Warp then creates a kernel cache under the user
cache directory and raises ``PermissionError`` if that path is not writable --
which is the case in sandboxes, hardened CI runners and read-only containers.
:func:`configure_warp_cache` redirects it to a writable directory and must run
*before* the first ``physicsnemo.models`` import, so it is called here at package
import time.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "configure_warp_cache",
    "nvidia_versions",
    "physicsnemo_available",
    "earth2studio_available",
    "omniverse_available",
    "require",
]


def configure_warp_cache(directory: str | Path | None = None) -> str:
    """Point Warp's kernel cache at a writable directory.

    Honours an existing ``WARP_CACHE_PATH`` if the caller has already set one.
    Returns the directory in use.
    """
    existing = os.environ.get("WARP_CACHE_PATH")
    if existing:
        Path(existing).mkdir(parents=True, exist_ok=True)
        return existing

    if directory is None:
        directory = os.environ.get("GLOF_CACHE_DIR", Path.cwd() / ".warp_cache")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(path)
    os.environ.setdefault("WARP_QUIET", "1")

    # The environment variable is only read when ``warp.config`` is first imported.
    # If something already imported Warp in this process (a notebook cell, a test
    # collecting another module) the variable arrives too late, so set the resolved
    # attribute as well. Both paths are needed: the variable for fresh subprocesses,
    # the attribute for an already-initialised interpreter.
    try:
        import warp

        warp.config.kernel_cache_dir = str(path)
        warp.config.quiet = True
    except Exception:  # noqa: BLE001 - Warp absent or restructured; env var still applies
        pass
    return str(path)


# Must precede any physicsnemo.models import in this process.
configure_warp_cache()


def _spec_exists(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def physicsnemo_available() -> bool:
    return _spec_exists("physicsnemo")


def earth2studio_available() -> bool:
    return _spec_exists("earth2studio")


def omniverse_available() -> bool:
    """True when USD authoring is possible (``usd-core`` or an Omniverse kit)."""
    return _spec_exists("pxr")


def nucleus_available() -> bool:
    """True when ``omni.client`` is present for publishing to a Nucleus server."""
    return _spec_exists("omni.client")


def nvidia_versions() -> dict[str, str | None]:
    """Installed versions of the NVIDIA stack, for the run manifest."""
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for distribution in ("nvidia-physicsnemo", "earth2studio", "torch", "usd-core", "warp-lang"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def require(*components: str) -> None:
    """Raise a single actionable error if any requested component is missing.

    ``components`` may name ``physicsnemo``, ``earth2studio`` or ``omniverse``.
    """
    checks = {
        "physicsnemo": (physicsnemo_available, "nvidia-physicsnemo>=2.2.1"),
        "earth2studio": (earth2studio_available, "earth2studio>=0.18.0"),
        "omniverse": (omniverse_available, "usd-core>=24.05"),
    }
    missing = []
    for component in components:
        if component not in checks:
            raise ValueError(f"Unknown component {component!r}; expected one of {sorted(checks)}.")
        available, requirement = checks[component]
        if not available():
            missing.append(requirement)
    if missing:
        raise RuntimeError(
            "Missing NVIDIA components: " + ", ".join(missing) + ". Install with "
            "`pip install -r requirements-nvidia.txt`. PhysicsNeMo 2.2.1 requires torch>=2.10."
        )
