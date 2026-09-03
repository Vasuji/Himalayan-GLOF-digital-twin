"""Twin experiment: does assimilating the network actually improve the state?"""

from __future__ import annotations

import numpy as np
import pytest

from glof_pipeline.assimilation.enkf import EnsembleKalmanFilter, StateSpec
from glof_pipeline.assimilation.sensors import build_sensor_network, synthesise_observations


def test_network_places_instruments_where_they_belong(terrain, graph, config, rng) -> None:
    network = build_sensor_network(terrain, graph, config.get("sensors"), rng)
    counts = network.counts()
    assert counts["piezometers"] > 0
    assert counts["stage_gauges"] > 0
    # Piezometers sit on the moraine mesh; stage gauges sit on the lake.
    moraine_cells = {(int(r), int(c)) for r, c in graph.node_rc}
    assert all((int(r), int(c)) in moraine_cells for r, c in network.piezometers)
    assert all(terrain.lake_mask[int(r), int(c)] for r, c in network.stage_gauges)


def test_enkf_reduces_error_in_a_twin_experiment(terrain, graph, config, rng) -> None:
    sensors_cfg = config.get("sensors")
    sensors_cfg["dropout_probability"] = 0.0
    network = build_sensor_network(terrain, graph, sensors_cfg, rng)
    spec = StateSpec.from_config(sensors_cfg["assimilation"])

    truth_values = {
        "lake_level_m": float(terrain.initial_lake_level + 0.8),
        "pore_pressure_ratio": 0.42,
        "ddf_ice_mm_per_c_per_day": 8.1,
    }
    truth = np.array([truth_values[name] for name in spec.names])
    background = truth + spec.prior_std * np.array([1.8, -1.5, 1.4])

    filter_ = EnsembleKalmanFilter(spec, sensors_cfg["assimilation"], rng)
    filter_.initialise(background)
    prior_rmse = filter_.rmse_against(truth)

    for _ in range(4):
        filter_.analysis(synthesise_observations(network, truth_values, sensors_cfg, rng))
    posterior_rmse = filter_.rmse_against(truth)

    assert posterior_rmse < prior_rmse
    assert posterior_rmse < 0.5 * prior_rmse


def test_enkf_is_a_no_op_without_observations(config, rng) -> None:
    spec = StateSpec.from_config(config.get("sensors.assimilation"))
    filter_ = EnsembleKalmanFilter(spec, config.get("sensors.assimilation"), rng)
    ensemble = filter_.initialise(np.array([5000.0, 0.4, 7.0]))
    unchanged = filter_.analysis([])
    assert np.allclose(ensemble, unchanged)


def test_state_bounds_are_enforced(config, rng) -> None:
    spec = StateSpec.from_config(config.get("sensors.assimilation"))
    filter_ = EnsembleKalmanFilter(spec, config.get("sensors.assimilation"), rng)
    filter_.initialise(np.array([5000.0, 0.02, 7.0]))
    ratio_index = spec.names.index("pore_pressure_ratio")
    assert filter_.ensemble[:, ratio_index].min() >= 0.0
    assert filter_.ensemble[:, ratio_index].max() <= 1.0


def test_observation_operator_ignores_unassimilated_kinds(config, rng) -> None:
    from glof_pipeline.assimilation.sensors import Observation

    spec = StateSpec.from_config(config.get("sensors.assimilation"))
    filter_ = EnsembleKalmanFilter(spec, config.get("sensors.assimilation"), rng)
    filter_.initialise(np.array([5000.0, 0.4, 7.0]))
    H, y, r = filter_.observation_operator(
        [
            Observation("lake_stage", 5000.5, 0.05, (0, 0)),
            Observation("strain", 120.0, 15.0, (1, 1)),  # not part of the state
        ]
    )
    assert H.shape == (1, spec.size)
    assert y.size == 1 and r.size == 1
