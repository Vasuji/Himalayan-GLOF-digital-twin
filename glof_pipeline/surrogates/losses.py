"""Training objectives, built from PhysicsNeMo's metric implementations.

Nothing here re-derives a metric that PhysicsNeMo ships. ``relative_l2`` and the
power spectrum come from ``physicsnemo.metrics.general``; only the two terms that
are specific to this problem -- a spectral-band penalty framed on PhysicsNeMo's
azimuthally-averaged spectrum, and water-volume conservation -- are assembled
here, and both are compositions of library primitives.
"""

from __future__ import annotations

import torch
from torch import Tensor

from physicsnemo.metrics.general.power_spectrum import power_spectrum
from physicsnemo.metrics.general.relative_error import relative_l2 as _pn_relative_l2

__all__ = [
    "combined_operator_loss",
    "mass_conservation_loss",
    "relative_l2",
    "spectral_loss",
]


def relative_l2(prediction: Tensor, target: Tensor) -> Tensor:
    """Relative L2 error -- ``physicsnemo.metrics.general.relative_error.relative_l2``.

    The conventional neural-operator accuracy measure. Reduced to a scalar so it can
    be used directly as a loss.
    """
    return _pn_relative_l2(prediction, target).mean()


def spectral_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Disagreement between azimuthally-averaged power spectra.

    A pointwise loss alone lets an operator win by smoothing: it loses the
    high-wavenumber content that carries the flood front. Penalising the spectrum
    directly keeps that content, and the spectrum is PhysicsNeMo's
    ``power_spectrum`` (2-D FFT followed by azimuthal averaging).
    """
    _, predicted_power = power_spectrum(prediction)
    _, target_power = power_spectrum(target)
    log_predicted = torch.log1p(predicted_power.clamp_min(0.0))
    log_target = torch.log1p(target_power.clamp_min(0.0))
    return torch.nn.functional.mse_loss(log_predicted, log_target)


def mass_conservation_loss(
    prediction: Tensor, target: Tensor, depth_channel: int = 0, eps: float = 1e-8
) -> Tensor:
    """Relative disagreement in total water volume.

    Problem-specific and deliberately kept: a flood surrogate that loses water is
    wrong in a way no generic accuracy metric penalises.
    """
    predicted_volume = prediction[:, depth_channel].sum(dim=(-2, -1))
    target_volume = target[:, depth_channel].sum(dim=(-2, -1))
    return ((predicted_volume - target_volume).abs() / target_volume.abs().clamp_min(eps)).mean()


def combined_operator_loss(
    prediction: Tensor,
    target: Tensor,
    spectral_weight: float = 0.15,
    mass_weight: float = 0.05,
) -> tuple[Tensor, dict[str, float]]:
    """Weighted sum of the three terms, with the components returned for logging."""
    data_term = relative_l2(prediction, target)
    spectral_term = spectral_loss(prediction, target)
    mass_term = mass_conservation_loss(prediction, target)
    total = data_term + spectral_weight * spectral_term + mass_weight * mass_term
    return total, {
        "relative_l2": float(data_term.detach()),
        "spectral": float(spectral_term.detach()),
        "mass": float(mass_term.detach()),
        "total": float(total.detach()),
    }
