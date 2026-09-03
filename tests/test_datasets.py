"""Dataset builders produce arrays with the shapes and coverage training expects."""

from __future__ import annotations

import numpy as np
import pytest

from glof_pipeline.datasets.builders import (
    build_downscaling_dataset,
    build_moraine_dataset,
    build_swe_dataset,
)


def test_moraine_dataset_spans_the_decision_boundary(terrain, graph, config, rng) -> None:
    dataset = build_moraine_dataset(terrain, graph, config.get("moraine"), 48, rng)
    assert dataset["node_features"].shape == (48, graph.num_nodes, 9)
    assert dataset["targets"].shape == (48, graph.num_nodes, 2)
    assert dataset["edge_features"].shape == (graph.num_edges, 4)
    assert np.all(np.isfinite(dataset["targets"]))
    # Pore-pressure ratio is a physical fraction.
    assert dataset["targets"][:, :, 1].min() >= 0.0
    assert dataset["targets"][:, :, 1].max() <= 1.0
    # Both stable and failing scenarios must be represented, or the surrogate
    # never sees the boundary it is supposed to reproduce.
    min_fos = dataset["scenario_scalars"][:, 3]
    assert (min_fos < 1.0).any() and (min_fos >= 1.0).any()


def test_downscaling_pairs_are_consistent(terrain, config, rng) -> None:
    dataset = build_downscaling_dataset(terrain, config.get("atmosphere"), 6, rng)
    coarse_shape = tuple(config.get("atmosphere.toy_generator.coarse_shape"))
    assert dataset["fine"].shape == (6, 2, *terrain.shape)
    assert dataset["coarse"].shape == (6, 2, *coarse_shape)
    assert dataset["fine"][:, 1].min() >= 0.0  # precipitation cannot be negative
    # Block-averaging must preserve the field mean to numerical precision.
    assert np.allclose(dataset["fine"].mean(axis=(2, 3)), dataset["coarse"].mean(axis=(2, 3)), rtol=1e-4)
    # The fine field must contain structure the coarse grid cannot represent.
    assert dataset["fine"][:, 1].std() > dataset["coarse"][:, 1].std()


@pytest.mark.slow
def test_swe_dataset_runs_the_reference_solver(terrain, config, rng) -> None:
    dataset = build_swe_dataset(
        terrain, config.get("routing"), config.get("breach"),
        n_scenarios=2, frames_per_scenario=4, resolution=24, rng=rng,
    )
    assert dataset["depth"].shape == (2, 5, 24, 24)
    assert dataset["bed"].shape == (24, 24)
    assert np.all(dataset["depth"] >= 0.0)
    assert np.all(np.isfinite(dataset["momentum_x"]))
    # Water must actually arrive: a dataset of dry frames teaches nothing.
    assert dataset["depth"].max() > 0.05
    # Depth grows as the hydrograph is injected.
    assert dataset["depth"][:, -1].sum() > dataset["depth"][:, 0].sum()
