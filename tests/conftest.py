"""Shared fixtures. Adds the repository root to ``sys.path`` for in-place testing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glof_pipeline.config import Config  # noqa: E402
from glof_pipeline.terrain.mesh_builder import build_moraine_graph  # noqa: E402
from glof_pipeline.terrain.synthetic_dem import build_synthetic_valley  # noqa: E402


@pytest.fixture(scope="session")
def config() -> Config:
    return Config.load(ROOT / "configs" / "toy.yaml")


@pytest.fixture(scope="session")
def smoke_config() -> Config:
    return Config.load(ROOT / "configs" / "smoke.yaml")


@pytest.fixture(scope="session")
def terrain(config):
    return build_synthetic_valley(config.get("domain.synthetic"))


@pytest.fixture(scope="session")
def graph(terrain):
    return build_moraine_graph(terrain, neighbours=8)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
