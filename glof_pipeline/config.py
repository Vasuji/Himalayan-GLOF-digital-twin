"""Configuration loading: YAML with includes, deep merge, dot-path access, and hashing.

A run is reproducible only if the exact configuration that produced it can be
recovered, so :class:`Config` carries a stable content hash that the pipeline writes
into every run manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is declared
    raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

_MISSING = object()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping, got {type(data).__name__}.")
    return data


def _resolve_includes(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular configuration include detected at {path}.")
    seen.add(path)

    raw = _load_yaml_file(path)
    includes = raw.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]

    merged: dict[str, Any] = {}
    for item in includes:
        merged = deep_merge(merged, _resolve_includes(path.parent / item, set(seen)))
    return deep_merge(merged, raw)


class Config:
    """Immutable-by-convention view over a nested configuration mapping."""

    def __init__(self, data: Mapping[str, Any], source: Path | None = None):
        self._data = copy.deepcopy(dict(data))
        self.source = Path(source) if source is not None else None

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
        path = Path(path)
        data = _resolve_includes(path, set())
        if overrides:
            data = deep_merge(data, overrides)
        return cls(data, source=path)

    # -- access -------------------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Fetch ``a.b.c``; raise :class:`KeyError` when absent and no default is given."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"Missing configuration key {dotted!r} (at {part!r}).")
                return default
            node = node[part]
        return copy.deepcopy(node)

    def section(self, dotted: str) -> Config:
        value = self.get(dotted)
        if not isinstance(value, Mapping):
            raise TypeError(f"Configuration key {dotted!r} is not a section.")
        return Config(value, source=self.source)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        return Config(deep_merge(self._data, overrides), source=self.source)

    def set_dotted(self, dotted: str, value: Any) -> Config:
        """Return a copy with ``a.b.c`` set to ``value``."""
        parts = dotted.split(".")
        patch: dict[str, Any] = {}
        node = patch
        for part in parts[:-1]:
            node[part] = {}
            node = node[part]
        node[parts[-1]] = value
        return self.with_overrides(patch)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- provenance ---------------------------------------------------------
    def hash(self, length: int = 12) -> str:
        payload = json.dumps(self._data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:length]

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, None) is not None

    def __repr__(self) -> str:
        return f"Config(source={self.source}, hash={self.hash()})"


def parse_cli_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Turn ``["a.b=3", "c.d=true"]`` into a nested override mapping.

    Values are parsed as YAML scalars, so ``3`` is an int, ``true`` a bool and
    ``[1, 2]`` a list.
    """
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must be of the form key.path=value.")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = overrides
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides
