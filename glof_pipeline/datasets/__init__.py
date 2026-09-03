"""Training-set construction from the reference physics.

Each surrogate is trained against the model it replaces: the MeshGraphNet against
the limit-equilibrium stability field, the FNO against the finite-volume
shallow-water solver, and the downscaler against coarsened high-resolution
orographic fields. The datasets are therefore reproducible from configuration and
a seed, with no external download.
"""

from .builders import (
    build_downscaling_dataset,
    build_moraine_dataset,
    build_swe_dataset,
)

__all__ = ["build_downscaling_dataset", "build_moraine_dataset", "build_swe_dataset"]
