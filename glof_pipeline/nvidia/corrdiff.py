"""CorrDiff downscaling built on PhysicsNeMo's diffusion stack.

CorrDiff (Mardani et al., *Nature Communications* 2025) is two-stage: a
deterministic regression predicts the conditional mean of the high-resolution
field, and a diffusion model generates the residual around it. This module builds
exactly that from PhysicsNeMo components rather than reimplementing them:

* backbone      -- ``physicsnemo.models.diffusion_unets.SongUNetPosEmbd``, the
                   positional-embedding Song UNet CorrDiff itself uses;
* preconditioner -- ``physicsnemo.diffusion.preconditioners.EDMPreconditioner``,
                   the current (non-legacy) EDM parameterisation;
* noise schedule -- ``physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler``;
* sampler        -- ``physicsnemo.diffusion.samplers.sample`` with the Heun solver.

The legacy ``EDMPrecondSuperResolution`` and ``deterministic_sampler`` are
deliberately **not** used: PhysicsNeMo 2.2.1 emits a ``LegacyFeatureWarning`` for
both and marks them for removal.

Conditioning contract
---------------------
``EDMPreconditioner.forward`` calls its wrapped model as
``model(c_in * x, c_noise, condition=condition)``, while ``SongUNetPosEmbd.forward``
accepts ``(x, noise_labels, ...)`` and knows nothing about a ``condition``
argument. :class:`_ConditionedUNet` bridges the two by concatenating the condition
onto the channel axis, which is how CorrDiff conditions its diffusion stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from glof_pipeline.nvidia import require
from glof_pipeline.nvidia._introspect import construct

require("physicsnemo")

from physicsnemo.diffusion.metrics import MSEDSMLoss  # noqa: E402
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler  # noqa: E402
from physicsnemo.diffusion.preconditioners import EDMPreconditioner  # noqa: E402
from physicsnemo.diffusion.samplers import sample as diffusion_sample  # noqa: E402
from physicsnemo.models.diffusion_unets import SongUNetPosEmbd  # noqa: E402

__all__ = ["CorrDiffDownscaler", "build_condition", "build_corrdiff"]


class _ConditionedUNet(nn.Module):
    """Adapt a Song UNet to the preconditioner's ``(x, c_noise, condition=)`` call."""

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x: Tensor, noise_labels: Tensor, condition: Tensor | None = None, **kwargs: Any) -> Tensor:
        if condition is not None:
            x = torch.cat([x, condition], dim=1)
        return self.unet(x, noise_labels, **kwargs)


class _RegressionUNet(nn.Module):
    """Deterministic stage: a Song UNet evaluated at zero noise.

    CorrDiff's regression stage is the same architecture as the diffusion stage
    run without noise, which is what the zero ``noise_labels`` vector expresses.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, condition: Tensor) -> Tensor:
        zeros = torch.zeros(condition.shape[0], device=condition.device, dtype=condition.dtype)
        return self.unet(condition, zeros)


def _build_unet(resolution: tuple[int, int], data_channels: int, out_channels: int, cfg: dict[str, Any]):
    """Construct a ``SongUNetPosEmbd`` with only the arguments this release accepts.

    ``in_channels`` must count the positional-embedding grid channels the UNet
    concatenates onto its own input, not just the data channels. Declaring only the
    data channels builds a first convolution that is ``N_grid_channels`` too narrow
    and fails at the first forward pass, which is how CorrDiff's own configuration
    files are written (``img_in_channels`` includes the grid channels).
    """
    n_grid_channels = int(cfg.get("n_grid_channels", 4))
    unet, dropped = construct(
        SongUNetPosEmbd,
        img_resolution=list(resolution),
        in_channels=int(data_channels) + n_grid_channels,
        out_channels=int(out_channels),
        model_channels=int(cfg.get("model_channels", 32)),
        channel_mult=list(cfg.get("channel_mult", [1, 2, 2])),
        num_blocks=int(cfg.get("num_blocks", 2)),
        attn_resolutions=list(cfg.get("attn_resolutions", [])),
        dropout=float(cfg.get("dropout", 0.0)),
        embedding_type=str(cfg.get("embedding_type", "positional")),
        encoder_type=str(cfg.get("encoder_type", "standard")),
        decoder_type=str(cfg.get("decoder_type", "standard")),
        # "sinusoidal" is flagged legacy upstream (no exact octave doublings); it is
        # only correct when loading a pre-trained checkpoint that used it.
        gridtype=str(cfg.get("gridtype", "sinusoidal_octave")),
        N_grid_channels=n_grid_channels,
        bottleneck_attention=bool(cfg.get("bottleneck_attention", False)),
    )
    return unet, dropped


class CorrDiffDownscaler(nn.Module):
    """Two-stage CorrDiff: PhysicsNeMo regression UNet plus EDM residual diffusion."""

    def __init__(
        self,
        resolution: tuple[int, int],
        condition_channels: int,
        output_channels: int,
        cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        cfg = dict(cfg or {})
        self.resolution = tuple(int(v) for v in resolution)
        self.condition_channels = int(condition_channels)
        self.output_channels = int(output_channels)
        self.num_steps = int(cfg.get("sampler_steps", 18))
        self.solver = str(cfg.get("solver", "heun"))

        regression_unet, dropped_regression = _build_unet(
            self.resolution, self.condition_channels, self.output_channels, cfg
        )
        self.regression = _RegressionUNet(regression_unet)

        diffusion_unet, dropped_diffusion = _build_unet(
            self.resolution,
            self.output_channels + self.condition_channels,
            self.output_channels,
            cfg,
        )
        self.denoiser, _ = construct(
            EDMPreconditioner,
            model=_ConditionedUNet(diffusion_unet),
            sigma_data=float(cfg.get("sigma_data", 0.5)),
        )
        self.scheduler, _ = construct(
            EDMNoiseScheduler,
            sigma_min=float(cfg.get("sigma_min", 0.002)),
            sigma_max=float(cfg.get("sigma_max", 80.0)),
            rho=float(cfg.get("rho", 7.0)),
            sigma_data=float(cfg.get("sigma_data", 0.5)),
            P_mean=float(cfg.get("p_mean", -1.2)),
            P_std=float(cfg.get("p_std", 1.2)),
        )
        # PhysicsNeMo owns time sampling, noise injection and loss weighting.
        self.loss_fn = MSEDSMLoss(
            model=self.denoiser,
            noise_scheduler=self.scheduler,
            prediction_type=str(cfg.get("prediction_type", "x0")),
        )
        self.dropped_arguments = sorted(set(dropped_regression) | set(dropped_diffusion))

    # -- stage 1 ------------------------------------------------------------
    def regress(self, condition: Tensor) -> Tensor:
        """Conditional mean of the high-resolution field."""
        return self.regression(condition)

    # -- stage 2 ------------------------------------------------------------
    def diffusion_loss(self, residual: Tensor, condition: Tensor) -> Tensor:
        """Denoising score-matching loss -- ``physicsnemo.diffusion.metrics.MSEDSMLoss``.

        The PhysicsNeMo loss owns the training step: it draws the diffusion time from
        the noise scheduler, injects noise through ``add_noise``, and applies the
        scheduler's ``loss_weight``, so the weighting stays consistent with the
        sampler used at inference.
        """
        return self.loss_fn(residual, condition=condition)

    @torch.no_grad()
    def sample_residual(self, condition: Tensor) -> Tensor:
        """Draw one residual realisation with PhysicsNeMo's Heun sampler."""
        shape = (condition.shape[0], self.output_channels, *condition.shape[-2:])
        sigma_max = float(getattr(self.scheduler, "sigma_max", 80.0))
        latents = torch.randn(shape, device=condition.device, dtype=condition.dtype) * sigma_max

        bound = _BoundDenoiser(self.denoiser, condition)
        return diffusion_sample(
            denoiser=bound,
            xN=latents,
            noise_scheduler=self.scheduler,
            num_steps=self.num_steps,
            solver=self.solver,
        )

    @torch.no_grad()
    def forward(self, condition: Tensor, samples: int = 1) -> Tensor:
        """Generate ``samples`` downscaled realisations, shaped ``(S, B, C, H, W)``."""
        mean = self.regress(condition)
        return torch.stack([mean + self.sample_residual(condition) for _ in range(samples)])


class _BoundDenoiser(nn.Module):
    """Freeze the conditioning so the sampler only has to supply ``(x, t)``.

    ``physicsnemo.diffusion.samplers.sample`` calls the denoiser positionally; binding
    the condition here avoids depending on whether a given release forwards extra
    keyword arguments through the solver.
    """

    def __init__(self, denoiser: nn.Module, condition: Tensor):
        super().__init__()
        self.denoiser = denoiser
        self.condition = condition

    def forward(self, x: Tensor, t: Tensor, **kwargs: Any) -> Tensor:
        return self.denoiser(x, t, condition=self.condition, **kwargs)


    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        """Write a PhysicsNeMo-format checkpoint.

        ``physicsnemo.utils.checkpoint.save_checkpoint`` writes the archive layout
        PhysicsNeMo's own recipes read, so a downscaler trained here is loadable by a
        PhysicsNeMo job on a GPU host without translation. It treats the path as a
        directory and manages the filenames inside it.
        """
        from glof_pipeline.nvidia.launch import save_checkpoint

        directory = Path(str(path).removesuffix(".pt"))
        metadata = {
            "resolution": list(self.resolution),
            "condition_channels": self.condition_channels,
            "output_channels": self.output_channels,
            "sampler_steps": self.num_steps,
            "solver": self.solver,
            **{k: v for k, v in (extra or {}).items() if not isinstance(v, (list, dict))},
        }
        return save_checkpoint(directory, models=self, metadata=metadata)

    def load(self, path: str | Path, device: str = "cpu") -> int:
        """Restore weights from a PhysicsNeMo checkpoint directory."""
        from glof_pipeline.nvidia.launch import load_checkpoint

        return load_checkpoint(Path(str(path).removesuffix(".pt")), models=self, device=device)


def build_corrdiff(
    resolution: tuple[int, int],
    condition_channels: int,
    output_channels: int,
    cfg: dict[str, Any] | None = None,
) -> CorrDiffDownscaler:
    """Factory used by the downscaling stage."""
    return CorrDiffDownscaler(resolution, condition_channels, output_channels, cfg)


def build_condition(
    coarse_fields: "np.ndarray", orography: "np.ndarray", target_shape: tuple[int, int]
) -> Tensor:
    """Assemble the CorrDiff conditioning tensor.

    ``coarse_fields`` is ``(B, C, h, w)``; it is placed on the target grid with
    ``torch.nn.functional.interpolate`` and stacked with the fine-resolution
    orography, which is the field carrying the signal the diffusion stage exploits.
    """
    import numpy as np
    import torch.nn.functional as F

    coarse = torch.as_tensor(np.asarray(coarse_fields), dtype=torch.float32)
    if coarse.ndim == 3:
        coarse = coarse.unsqueeze(0)
    upsampled = F.interpolate(coarse, size=target_shape, mode="bilinear", align_corners=False)
    topography = torch.as_tensor(np.asarray(orography), dtype=torch.float32)
    if topography.ndim == 2:
        topography = topography.unsqueeze(0).unsqueeze(0)
    topography = topography.expand(upsampled.shape[0], 1, *target_shape)
    return torch.cat([upsampled, topography], dim=1)
