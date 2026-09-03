"""Surrogate wrappers, losses and normalisation.

The networks themselves are PhysicsNeMo's, so these tests cover the layer this
repository actually owns: the operator wrappers, the loss composition, the
standardiser, and the graph conversion. Forward-pass shape checks against the real
PhysicsNeMo modules live in ``test_nvidia_integration.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("physicsnemo")

from glof_pipeline.backends import (  # noqa: E402
    build_fno,
    torch_geometric_available,
    torch_scatter_available,
)
from glof_pipeline.surrogates.losses import (  # noqa: E402
    combined_operator_loss,
    mass_conservation_loss,
    relative_l2,
    spectral_loss,
)
from glof_pipeline.surrogates.normalization import Standardizer  # noqa: E402

FNO_CFG = {
    "in_channels": 4, "out_channels": 3, "dimension": 2, "num_fno_layers": 2,
    "num_fno_modes": 6, "latent_channels": 8, "decoder_layers": 1,
    "decoder_layer_size": 16, "padding": 2, "coord_features": True,
}
MGN_CFG = {
    "node_features": 9, "edge_features": 4, "output_dim": 2, "processor_size": 2,
    "hidden_dim_processor": 16, "hidden_dim_node_encoder": 16,
    "hidden_dim_edge_encoder": 16, "hidden_dim_node_decoder": 16,
}


# --- backend policy --------------------------------------------------------
def test_build_fno_returns_the_physicsnemo_model() -> None:
    """build_fno must return a PhysicsNeMo FNO instance."""
    from physicsnemo.models.fno import FNO

    model, backend = build_fno(FNO_CFG)
    assert backend == "physicsnemo"
    assert isinstance(model, FNO)


def test_meshgraphnet_reports_its_missing_dependency_clearly() -> None:
    """Without torch_scatter the failure must name it, not fail inside PhysicsNeMo."""
    from glof_pipeline.backends import build_meshgraphnet

    if torch_scatter_available() and torch_geometric_available():
        model, backend = build_meshgraphnet(MGN_CFG)
        assert backend == "physicsnemo"
    else:
        with pytest.raises(RuntimeError, match="torch-scatter|torch-geometric"):
            build_meshgraphnet(MGN_CFG)


# --- losses ----------------------------------------------------------------
def test_relative_l2_is_zero_for_an_exact_match() -> None:
    field = torch.randn(2, 3, 16, 16)
    assert float(relative_l2(field, field)) == pytest.approx(0.0, abs=1e-6)


def test_relative_l2_grows_with_error() -> None:
    target = torch.randn(2, 3, 16, 16)
    near = target + 0.01 * torch.randn_like(target)
    far = target + 0.5 * torch.randn_like(target)
    assert float(relative_l2(near, target)) < float(relative_l2(far, target))


def test_spectral_loss_penalises_smoothing() -> None:
    """A blurred field matches in the mean but loses high-wavenumber power."""
    target = torch.randn(2, 3, 32, 32)
    kernel = torch.ones(3, 1, 3, 3) / 9.0
    blurred = torch.nn.functional.conv2d(target, kernel, padding=1, groups=3)
    assert float(spectral_loss(blurred, target)) > float(spectral_loss(target, target))


def test_mass_conservation_loss_detects_lost_water() -> None:
    target = torch.rand(2, 3, 8, 8)
    assert float(mass_conservation_loss(target, target)) == pytest.approx(0.0, abs=1e-6)
    assert float(mass_conservation_loss(0.5 * target, target)) == pytest.approx(0.5, rel=1e-3)


def test_combined_loss_reports_its_components() -> None:
    prediction = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 16, 16)
    total, parts = combined_operator_loss(prediction, target)
    assert set(parts) == {"relative_l2", "spectral", "mass", "total"}
    assert float(total) == pytest.approx(parts["total"], rel=1e-6)
    assert torch.isfinite(total)


# --- normalisation ---------------------------------------------------------
def test_standardizer_round_trips() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(64, 5))
    scaler = Standardizer.fit(data)
    encoded = scaler.transform(data)
    assert np.allclose(encoded.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(encoded.std(axis=0), 1.0, atol=1e-6)
    assert np.allclose(scaler.inverse(encoded), data, atol=1e-9)


def test_standardizer_survives_serialisation() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(32, 3))
    scaler = Standardizer.fit(data)
    restored = Standardizer.from_dict(scaler.to_dict())
    assert np.allclose(restored.transform(data), scaler.transform(data))


def test_standardizer_handles_a_constant_column() -> None:
    """A zero-variance feature must not divide by zero."""
    data = np.column_stack([np.ones(16), np.arange(16.0)])
    scaler = Standardizer.fit(data)
    encoded = scaler.transform(data)
    assert np.all(np.isfinite(encoded))


# --- graph conversion ------------------------------------------------------
def test_moraine_graph_converts_to_pytorch_geometric(graph) -> None:
    pytest.importorskip("torch_geometric")
    data = graph.to_pyg("cpu")
    assert int(data.num_nodes) == graph.num_nodes
    assert tuple(data.edge_index.shape) == (2, graph.num_edges)
