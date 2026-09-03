"""Neural surrogates: FNO flood routing, MeshGraphNet moraine mechanics.

The operator wrappers in this package own configuration, normalisation, training
state and checkpoint layout; the networks themselves come from PhysicsNeMo via
:mod:`glof_pipeline.backends`. A checkpoint written on one host therefore loads on
another, including in a PhysicsNeMo training job on a GPU host.
"""

from .losses import mass_conservation_loss, relative_l2, spectral_loss
from .normalization import Standardizer

__all__ = [
    "Standardizer",
    "mass_conservation_loss",
    "relative_l2",
    "spectral_loss",
]
