"""Himalayan GLOF digital twin.

Two execution tiers share one code path:

* ``toy``        - self-contained reference physics plus small neural surrogates that
                   train and run on a CPU laptop. Every number the pipeline reports is
                   computed, not asserted.
* ``production`` - the same stages backed by NVIDIA Earth-2 (``earth2studio``) for the
                   atmosphere and NVIDIA PhysicsNeMo (``physicsnemo``) for the
                   MeshGraphNet and Fourier Neural Operator surrogates.

The tier is selected in configuration (``runtime.tier``); no module imports the NVIDIA
stack at import time, so the toy tier has no GPU dependency.
"""

__version__ = "0.2.0"
__all__ = ["__version__"]
