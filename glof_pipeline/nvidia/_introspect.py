"""Signature-filtered construction of third-party objects.

The NVIDIA APIs move between releases -- arguments are renamed, added and
retired. Passing a fixed keyword set therefore breaks on upgrade with a
``TypeError`` that names one argument and hides the rest. These helpers pass only
the keywords the *installed* callable accepts and report precisely what was
dropped, so a version change surfaces as a logged note rather than a crash, and a
genuinely missing required argument raises an error that prints the real
signature.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable


def supported_kwargs(target: Callable[..., Any]) -> set[str] | None:
    """Keyword names ``target`` accepts, or ``None`` when it accepts ``**kwargs``."""
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # builtins without introspectable signatures
        return None
    names: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.name != "self":
            names.add(parameter.name)
    return names


def filter_kwargs(
    target: Callable[..., Any], candidates: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Split ``candidates`` into those ``target`` accepts and those it does not."""
    accepted = supported_kwargs(target)
    if accepted is None:
        return dict(candidates), []
    kept = {key: value for key, value in candidates.items() if key in accepted}
    dropped = sorted(set(candidates) - set(kept))
    return kept, dropped


def missing_required(target: Callable[..., Any], provided: dict[str, Any]) -> list[str]:
    """Required parameters of ``target`` that ``provided`` does not cover."""
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return []
    missing = []
    for parameter in signature.parameters.values():
        if parameter.name in ("self", "args", "kwargs"):
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect.Parameter.empty and parameter.name not in provided:
            missing.append(parameter.name)
    return missing


def describe(target: Callable[..., Any]) -> str:
    """Readable signature of ``target``, used in error messages."""
    try:
        return f"{getattr(target, '__qualname__', target)}{inspect.signature(target)}"
    except (TypeError, ValueError):
        return str(target)


def construct(target: Callable[..., Any], /, **candidates: Any) -> tuple[Any, list[str]]:
    """Instantiate ``target`` with the subset of ``candidates`` it accepts.

    Returns ``(instance, dropped_keyword_names)`` and raises a ``RuntimeError``
    quoting the installed signature when a required argument is unmet.
    """
    kept, dropped = filter_kwargs(target, candidates)
    unmet = missing_required(target, kept)
    if unmet:
        raise RuntimeError(
            f"Cannot construct {describe(target)}: the installed version requires "
            f"{unmet}, which this adapter does not supply. Update "
            "glof_pipeline/nvidia/ for the installed release."
        )
    return target(**kept), dropped
