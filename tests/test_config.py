"""Configuration loading, includes, overrides and hashing."""

from __future__ import annotations

import pytest

from glof_pipeline.config import Config, deep_merge, parse_cli_overrides


def test_includes_are_merged(config: Config) -> None:
    # Sections come from four separate component files plus toy.yaml itself.
    assert config.get("atmosphere.prognostic_model") == "FCN3"
    assert config.get("moraine.friction_angle_deg") == pytest.approx(38.0)
    assert config.get("routing.physics.gravity") == pytest.approx(9.81)
    assert config.get("sensors.assimilation.method") == "enkf"
    assert config.get("runtime.tier") == "toy"


def test_production_overrides_toy() -> None:
    from tests.conftest import ROOT

    production = Config.load(ROOT / "configs" / "production.yaml")
    assert production.get("runtime.tier") == "production"
    assert production.get("surrogates.fno.num_fno_modes") == 32
    # Inherited from toy.yaml through the include chain.
    assert production.get("glaciology.ddf_ice_mm_per_c_per_day") == pytest.approx(7.0)


def test_deep_merge_is_recursive_and_non_mutating() -> None:
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"c": 3, "d": 4}})
    assert merged == {"a": {"b": 1, "c": 3, "d": 4}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_cli_overrides_parse_yaml_scalars() -> None:
    overrides = parse_cli_overrides(["runtime.seed=3", "training.fno.epochs=2", "runtime.tier=toy"])
    assert overrides["runtime"]["seed"] == 3
    assert overrides["training"]["fno"]["epochs"] == 2


def test_missing_key_raises_and_default_works(config: Config) -> None:
    with pytest.raises(KeyError):
        config.get("does.not.exist")
    assert config.get("does.not.exist", 42) == 42


def test_hash_changes_with_content(config: Config) -> None:
    assert config.hash() != config.set_dotted("runtime.seed", 999).hash()
