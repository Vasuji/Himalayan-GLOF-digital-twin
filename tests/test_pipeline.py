"""End-to-end smoke run: every stage executes and produces the expected artifacts.

Marked ``slow`` because it trains three networks, but the smoke configuration
shrinks each of them so the whole graph runs in well under a minute on CPU. This
is the test that would catch a stage-to-stage interface break, which unit tests
by construction cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# GPU tier: this run trains the MeshGraphNet, the FNO and a two-stage CorrDiff,
# which is real compute even at smoke size, and PhysicsNeMo's MeshGraphNet needs a
# working torch_scatter. Deselected by default; run it with `pytest -m gpu`.
pytestmark = [pytest.mark.slow, pytest.mark.gpu]

from glof_pipeline.evaluate.benchmark import benchmark_table  # noqa: E402
from glof_pipeline.pipeline import Pipeline  # noqa: E402


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory, smoke_config):
    directory = tmp_path_factory.mktemp("smoke")
    config = smoke_config.with_overrides(
        {
            "runtime": {
                "output_dir": str(directory / "outputs"),
                "checkpoint_dir": str(directory / "checkpoints"),
            },
            "datasets": {
                "moraine": {"path": str(directory / "moraine.npz")},
                "swe": {"path": str(directory / "swe.npz")},
                "downscaling": {"path": str(directory / "downscaling.npz")},
            },
            "visualization": {"usd_path": str(directory / "flood.usda")},
        }
    )
    pipeline = Pipeline(config)
    manifest = pipeline.run()
    return pipeline, manifest, directory


@pytest.mark.slow
def test_manifest_records_provenance(smoke_run) -> None:
    _, manifest, _ = smoke_run
    assert manifest["tier"] == "toy"
    assert len(manifest["config_hash"]) == 12
    assert "physicsnemo" in manifest["backends"]
    assert manifest["environment"]["packages"]["torch"] is not None
    assert Path(manifest["manifest_path"]).is_file()
    reloaded = json.loads(Path(manifest["manifest_path"]).read_text())
    assert reloaded["config_hash"] == manifest["config_hash"]


@pytest.mark.slow
def test_every_stage_ran(smoke_run) -> None:
    _, manifest, _ = smoke_run
    expected = {
        "terrain", "atmosphere", "downscaling", "mass_balance",
        "moraine_surrogate", "moraine_assessment", "breach",
        "assimilation", "evaluation", "products",
    }
    assert expected.issubset(set(manifest["timings_s"]))
    assert all(value >= 0.0 for value in manifest["timings_s"].values())


@pytest.mark.slow
def test_terrain_and_forcing_are_physical(smoke_run) -> None:
    pipeline, manifest, _ = smoke_run
    terrain = manifest["artifacts"]["terrain"]
    assert terrain["initial_lake_volume_m3"] > 0.0
    assert terrain["moraine_nodes"] > 0
    for member in manifest["artifacts"]["mass_balance"]["members"]:
        assert np.isfinite(member["peak_inflow_m3_per_s"])
        assert member["cumulative_pdd_c_day"] >= 0.0
    assert len(pipeline.forcings) == int(pipeline.config.get("downscaling.samples"))


@pytest.mark.slow
def test_surrogates_trained_and_checkpointed(smoke_run) -> None:
    _, manifest, directory = smoke_run
    assert (directory / "checkpoints" / "mgn_moraine.pt").is_file()
    assert (directory / "checkpoints" / "downscaler.pt").is_file()
    mgn = manifest["artifacts"]["mgn_training"]
    assert np.isfinite(mgn["val_rmse_factor_of_safety"])
    downscaler = manifest["artifacts"]["downscaler_training"]
    # The diffusion stage should restore variance the regression stage smooths out.
    assert downscaler["generated_std"] > downscaler["regression_std"]


@pytest.mark.slow
def test_breach_and_routing_are_consistent(smoke_run) -> None:
    _, manifest, _ = smoke_run
    breach = manifest["artifacts"]["breach"]
    if not breach.get("breached"):
        pytest.skip("Smoke forcing did not breach the dam; routing assertions do not apply.")
    assert breach["hydrograph_mass_balance_error"] < 0.05
    routing = manifest["artifacts"]["routing"]
    assert "solver" in routing
    assert routing["solver"]["peak_depth_m"] > 0.0
    assert routing["solver"]["mass_conservation_error"] < 1e-6
    if "fno" in routing:
        benchmark = manifest["artifacts"]["benchmark"]
        assert "fno" in benchmark["comparisons"]
        assert np.isfinite(benchmark["comparisons"]["fno"]["speedup_vs_solver"])
        assert isinstance(benchmark_table(benchmark), str)


@pytest.mark.slow
def test_assimilation_improves_the_state(smoke_run) -> None:
    _, manifest, _ = smoke_run
    assimilation = manifest["artifacts"]["assimilation"]
    assert assimilation["enabled"]
    assert assimilation["analysis_normalised_rmse"] < assimilation["prior_normalised_rmse"]


@pytest.mark.slow
def test_usd_export_is_a_valid_scene(smoke_run) -> None:
    _, manifest, directory = smoke_run
    usd_path = manifest["artifacts"]["products"].get("usd")
    if usd_path is None:
        pytest.skip("USD export disabled or no routing performed.")
    text = Path(usd_path).read_text()
    assert text.startswith("#usda 1.0") or Path(usd_path).suffix in (".usd", ".usdc")
    assert "FloodWaterSurface" in text
    assert "faceVertexIndices" in text
