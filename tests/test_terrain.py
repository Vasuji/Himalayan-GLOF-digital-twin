"""Geometric invariants of the synthetic valley and the moraine mesh."""

from __future__ import annotations

import numpy as np
import pytest

from glof_pipeline.terrain.mesh_builder import distance_to_lake, slope_and_aspect
from glof_pipeline.terrain.synthetic_dem import lake_hypsometry, level_from_volume


def test_lake_sits_behind_the_crest(terrain) -> None:
    lake_rows = np.nonzero(terrain.lake_mask)[0]
    assert lake_rows.size > 0
    assert lake_rows.max() < terrain.moraine_row


def test_crest_is_above_the_lake_surface(terrain) -> None:
    assert terrain.crest_elevation > terrain.initial_lake_level
    freeboard = float(terrain.z[terrain.moraine_row].min()) - terrain.initial_lake_level
    assert freeboard == pytest.approx(10.0, abs=1e-6)  # domain.synthetic.freeboard_m


def test_valley_descends_downstream_of_the_dam(terrain) -> None:
    thalweg = terrain.z[np.arange(terrain.shape[0]), terrain.thalweg_col]
    downstream = thalweg[terrain.moraine_row + 2 :]
    assert np.all(np.diff(downstream) < 0.0)


def test_stage_storage_curve_is_monotone_and_invertible(terrain) -> None:
    levels, areas, volumes = lake_hypsometry(terrain)
    assert np.all(np.diff(volumes) >= 0.0)
    assert np.all(np.diff(areas) >= 0.0)
    target = 0.5 * (volumes[0] + volumes[-1])
    recovered = level_from_volume(levels, volumes, target)
    assert volumes[0] <= float(np.interp(recovered, levels, volumes)) <= volumes[-1]
    assert float(np.interp(recovered, levels, volumes)) == pytest.approx(target, rel=0.02)


def test_lake_volume_matches_hypsometry(terrain) -> None:
    levels, _, volumes = lake_hypsometry(terrain)
    direct = terrain.lake_volume(terrain.initial_lake_level)
    interpolated = float(np.interp(terrain.initial_lake_level, levels, volumes))
    assert direct == pytest.approx(interpolated, rel=0.05)


def test_moraine_graph_is_symmetric_and_connected_to_the_dam(terrain, graph) -> None:
    assert graph.num_nodes > 20
    assert graph.num_edges > graph.num_nodes
    edges = {(int(a), int(b)) for a, b in zip(*graph.edge_index, strict=True)}
    # Grid adjacency is built in both directions.
    assert all((b, a) in edges for a, b in list(edges)[:200])
    assert not np.any(terrain.lake_mask[graph.node_rc[:, 0], graph.node_rc[:, 1]])


def test_distance_to_lake_is_zero_at_the_upstream_face(terrain, graph) -> None:
    distance = distance_to_lake(terrain, graph)
    assert distance.min() <= terrain.dx * 1.5
    assert distance.max() > distance.min()


def test_slope_is_physical(terrain) -> None:
    beta, aspect = slope_and_aspect(terrain.z, terrain.dx)
    assert np.all(beta >= 0.0)
    assert np.all(beta < np.pi / 2)
    assert np.all(np.abs(aspect) <= np.pi + 1e-9)
