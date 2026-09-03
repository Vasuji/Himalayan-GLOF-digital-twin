""""Million Eyes": in-situ network, observation operator, and ensemble assimilation.

The forecast-only pipeline propagates whatever bias the atmospheric model and the
geotechnical parameters carry. Assimilating stage, piezometric and discharge
observations turns it into a filtered estimate, which is what makes a warning
system trustworthy between events as well as during one.
"""

from .enkf import EnsembleKalmanFilter, StateSpec
from .sensors import Observation, SensorNetwork, build_sensor_network, synthesise_observations

__all__ = [
    "EnsembleKalmanFilter",
    "Observation",
    "SensorNetwork",
    "StateSpec",
    "build_sensor_network",
    "synthesise_observations",
]
