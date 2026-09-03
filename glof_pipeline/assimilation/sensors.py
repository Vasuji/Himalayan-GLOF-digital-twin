"""Synthetic in-situ network and its observation operator.

Three instrument classes are represented because each constrains a different part
of the state vector:

* **stage gauges** on the lake constrain the lake level directly;
* **piezometers** in the moraine constrain the pore-pressure ratio, which is the
  term that actually moves the factor of safety; and
* **stream gauges** below the dam constrain the melt parameter through the inflow.

Strain gauges and acoustic emission sensors are placed and recorded but not
assimilated: they are anomaly detectors, not state observations, and treating them
as the latter would require a deformation model the twin does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glof_pipeline.terrain.mesh_builder import MoraineGraph
from glof_pipeline.terrain.synthetic_dem import ValleyTerrain


@dataclass
class Observation:
    """A single assimilated measurement."""

    kind: str            # lake_stage | piezometric_head | discharge
    value: float
    error_std: float
    location_rc: tuple[int, int]


@dataclass
class SensorNetwork:
    """Instrument positions on the DEM grid."""

    weather_stations: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    piezometers: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    strain_gauges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    stream_gauges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    stage_gauges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))

    def counts(self) -> dict[str, int]:
        return {
            "weather_stations": int(self.weather_stations.shape[0]),
            "piezometers": int(self.piezometers.shape[0]),
            "strain_gauges": int(self.strain_gauges.shape[0]),
            "stream_gauges": int(self.stream_gauges.shape[0]),
            "stage_gauges": int(self.stage_gauges.shape[0]),
        }


def _sample_cells(mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.empty((0, 2), dtype=int)
    picks = rng.choice(rows.size, size=min(count, rows.size), replace=False)
    return np.stack([rows[picks], cols[picks]], axis=1)


def build_sensor_network(
    terrain: ValleyTerrain, graph: MoraineGraph, cfg: dict[str, Any], rng: np.random.Generator
) -> SensorNetwork:
    """Place instruments on the lake, the moraine and the downstream channel."""
    network_cfg = cfg["network"]
    ny, nx = terrain.shape

    moraine_mask = np.zeros(terrain.shape, dtype=bool)
    moraine_mask[graph.node_rc[:, 0], graph.node_rc[:, 1]] = True

    downstream = np.zeros(terrain.shape, dtype=bool)
    for row in range(terrain.moraine_row + 3, ny):
        downstream[row, int(terrain.thalweg_col[row])] = True

    valley = np.zeros(terrain.shape, dtype=bool)
    valley[:, max(nx // 2 - nx // 4, 0) : nx // 2 + nx // 4] = True

    return SensorNetwork(
        weather_stations=_sample_cells(valley, int(network_cfg["weather_stations"]), rng),
        piezometers=_sample_cells(moraine_mask, int(network_cfg["piezometers"]), rng),
        strain_gauges=_sample_cells(moraine_mask, int(network_cfg["strain_gauges"]), rng),
        stream_gauges=_sample_cells(downstream, int(network_cfg["stream_gauges"]), rng),
        stage_gauges=_sample_cells(terrain.lake_mask, max(2, int(network_cfg["stream_gauges"]) // 2), rng),
    )


def synthesise_observations(
    network: SensorNetwork,
    truth: dict[str, float],
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> list[Observation]:
    """Draw noisy observations of a known truth, with telemetry dropout.

    Used both for the twin experiment in the tests and for the toy tier's
    closed-loop demonstration.
    """
    noise = cfg["noise"]
    dropout = float(cfg["dropout_probability"])
    observations: list[Observation] = []

    def maybe_add(kind: str, value: float, sigma: float, location: np.ndarray) -> None:
        if rng.random() < dropout:
            return
        observations.append(
            Observation(kind=kind, value=float(value + rng.normal(0.0, sigma)), error_std=float(sigma),
                        location_rc=(int(location[0]), int(location[1])))
        )

    for location in network.stage_gauges:
        maybe_add("lake_stage", truth["lake_level_m"], float(noise["lake_stage_m"]), location)
    for location in network.piezometers:
        # Piezometric head expressed as a fraction of the overburden, i.e. r_u.
        maybe_add(
            "piezometric_head",
            truth["pore_pressure_ratio"],
            float(noise["piezometric_head_m"]) / 25.0,
            location,
        )
    for location in network.stream_gauges:
        maybe_add("discharge", truth["ddf_ice_mm_per_c_per_day"], 0.8, location)
    return observations
