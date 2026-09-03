"""Integration checks against the real NVIDIA stack.

Scope is deliberately narrow: **imports and tensor shapes only**. No training, no
model-package downloads, no GPU work. The point is to prove that the adapters in
:mod:`glof_pipeline.nvidia` construct the genuine PhysicsNeMo modules and that
their forward passes agree with the shapes the pipeline feeds them. Everything
expensive belongs on a GPU host.

The whole module skips when the NVIDIA stack is absent, so the suite still runs on
a plain install.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("physicsnemo")

from glof_pipeline.nvidia import (  # noqa: E402
    configure_warp_cache,
    nvidia_versions,
    physicsnemo_available,
)

pytestmark = pytest.mark.nvidia


def test_warp_cache_points_at_a_writable_directory() -> None:
    """physicsnemo.models imports call warp.init(), which needs a writable cache.

    configure_warp_cache honours an already-set WARP_CACHE_PATH, so the contract to
    test is that whatever directory is in force exists and accepts a write -- not
    that a particular path was chosen.
    """
    import os
    from pathlib import Path as _Path

    directory = _Path(configure_warp_cache())
    assert directory.is_dir()
    probe = directory / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    assert probe.read_text(encoding="utf-8") == "ok"
    probe.unlink()
    assert os.environ.get("WARP_CACHE_PATH") == str(directory)


def test_versions_are_reported_for_the_manifest() -> None:
    versions = nvidia_versions()
    assert versions["nvidia-physicsnemo"] is not None
    assert versions["torch"] is not None
    # PhysicsNeMo 2.2.x requires torch >= 2.10; a mismatch breaks the models import.
    major, minor = (int(part) for part in versions["torch"].split(".")[:2])
    assert (major, minor) >= (2, 10), f"torch {versions['torch']} is too old for PhysicsNeMo 2.2.x"


def test_physicsnemo_is_required_not_optional() -> None:
    """The tier resolver must require PhysicsNeMo for both tiers."""
    from glof_pipeline.backends import require_physicsnemo, resolve_tier

    assert physicsnemo_available()
    require_physicsnemo()  # must not raise
    assert resolve_tier("toy") == "toy"


def test_real_physicsnemo_fno_forward_shape() -> None:
    """The genuine physicsnemo FNO must accept the solver state and return three fields."""
    from glof_pipeline.nvidia.physicsnemo_models import FNO, build_fno

    cfg = {
        "in_channels": 4, "out_channels": 3, "dimension": 2, "num_fno_layers": 2,
        "num_fno_modes": 6, "latent_channels": 8, "decoder_layers": 1,
        "decoder_layer_size": 16, "padding": 2, "coord_features": True,
    }
    model, dropped = build_fno(cfg)
    assert isinstance(model, FNO)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 24, 24))
    assert out.shape == (1, 3, 24, 24)
    assert isinstance(dropped, list)


def test_real_physicsnemo_meshgraphnet_forward_shape() -> None:
    """MeshGraphNet takes a PyTorch Geometric Data object since PhysicsNeMo 2.x.

    Its message passing also needs ``torch_scatter``, which is compiled against the
    installed torch; without a C++ toolchain the forward pass raises an ImportError
    from inside PhysicsNeMo, so this skips rather than failing.
    """
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")
    from torch_geometric.data import Data

    from glof_pipeline.nvidia.physicsnemo_models import MeshGraphNet, build_meshgraphnet

    cfg = {
        "node_features": 9, "edge_features": 4, "output_dim": 2, "processor_size": 2,
        "hidden_dim_processor": 16, "hidden_dim_node_encoder": 16,
        "hidden_dim_edge_encoder": 16, "hidden_dim_node_decoder": 16,
    }
    model, _ = build_meshgraphnet(cfg)
    assert isinstance(model, MeshGraphNet)
    # Construction succeeds without torch_scatter; only the forward pass needs it.
    n_nodes, n_edges = 24, 72
    graph = Data(edge_index=torch.randint(0, n_nodes, (2, n_edges)), num_nodes=n_nodes)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(n_nodes, 9), torch.randn(n_edges, 4), graph)
    assert out.shape == (n_nodes, 2)


def test_moraine_graph_converts_to_pytorch_geometric(terrain, graph) -> None:
    pytest.importorskip("torch_geometric")
    data = graph.to_pyg("cpu")
    assert int(data.num_nodes) == graph.num_nodes
    assert tuple(data.edge_index.shape) == (2, graph.num_edges)


def test_corrdiff_uses_the_modern_diffusion_api() -> None:
    """The adapter must build on EDMPreconditioner, not the legacy preconditioner."""
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner

    from glof_pipeline.nvidia.corrdiff import build_corrdiff

    model = build_corrdiff(
        resolution=(32, 32), condition_channels=3, output_channels=2,
        cfg={"model_channels": 16, "channel_mult": [1, 2], "num_blocks": 1,
             "sampler_steps": 2, "attn_resolutions": []},
    )
    assert isinstance(model.denoiser, EDMPreconditioner)
    condition = torch.randn(1, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        mean = model.regress(condition)
    assert mean.shape == (1, 2, 32, 32)


def test_corrdiff_diffusion_loss_is_finite_and_differentiable() -> None:
    from glof_pipeline.nvidia.corrdiff import build_corrdiff

    model = build_corrdiff(
        resolution=(32, 32), condition_channels=3, output_channels=2,
        cfg={"model_channels": 16, "channel_mult": [1, 2], "num_blocks": 1,
             "sampler_steps": 2, "attn_resolutions": []},
    )
    condition = torch.randn(2, 3, 32, 32)
    residual = torch.randn(2, 2, 32, 32)
    loss = model.diffusion_loss(residual, condition)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.denoiser.parameters())


def test_physicsnemo_crps_matches_the_local_estimator() -> None:
    """Cross-check the local CRPS against PhysicsNeMo's kernel estimator."""
    from glof_pipeline.evaluate.metrics import crps_ensemble
    from glof_pipeline.nvidia.statistics import crps_physicsnemo

    rng = np.random.default_rng(0)
    ensemble = rng.normal(0.0, 1.0, size=(200, 1))
    observation = np.array([0.3])
    nvidia_value = crps_physicsnemo(ensemble, observation)
    local_value = crps_ensemble(ensemble.ravel(), 0.3)
    assert nvidia_value > 0.0
    assert abs(nvidia_value - local_value) < 0.05, (nvidia_value, local_value)


@pytest.mark.skipif(not physicsnemo_available(), reason="needs earth2studio")
def test_earth2_registries_expose_the_configured_models() -> None:
    """The models named in configuration must exist in the installed Earth2Studio."""
    pytest.importorskip("earth2studio")
    from glof_pipeline.nvidia.earth2 import (
        available_data_sources,
        available_downscalers,
        available_prognostics,
    )

    assert "FCN3" in available_prognostics()
    assert "GFS" in available_data_sources()
    assert "CorrDiff" in available_downscalers()


def test_foreign_domain_corrdiff_is_refused() -> None:
    """A Taiwan-trained CorrDiff must not be loadable for the Himalaya by default."""
    pytest.importorskip("earth2studio")
    from glof_pipeline.nvidia.earth2 import load_downscaler

    with pytest.raises(RuntimeError, match="Hindu Kush Himalaya"):
        load_downscaler("CorrDiffTaiwan", checkpoint=None)


def test_probabilistic_report_labels_its_backend() -> None:
    from glof_pipeline.evaluate.metrics import probabilistic_report

    rng = np.random.default_rng(1)
    ensemble = rng.normal(0.0, 1.0, size=(16, 8))
    observation = rng.normal(0.0, 1.0, size=8)
    report = probabilistic_report(ensemble, observation, thresholds=[0.0, 1.0])
    assert report["members"] == 16
    assert report["backend"] in ("physicsnemo", "physicsnemo+earth2studio")
    assert np.isfinite(report["crps"])
