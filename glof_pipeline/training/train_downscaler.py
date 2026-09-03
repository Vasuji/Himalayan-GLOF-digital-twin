"""Train the two-stage residual-diffusion downscaler.

Stage one (regression) is fitted first to convergence-ish, then frozen while the
diffusion stage learns the residual. Training the diffusion model against a moving
regression target is the usual way to get a downscaler that produces plausible
texture with the wrong mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from glof_pipeline.nvidia.corrdiff import CorrDiffDownscaler, build_condition
from glof_pipeline.surrogates.normalization import Standardizer
from glof_pipeline.utils.runtime import Timer, get_training_logger

LOGGER = get_training_logger("glof.train.downscaler")


def train_downscaler(
    dataset: dict[str, np.ndarray],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    device: str = "cpu",
    checkpoint: str | Path | None = None,
    seed: int = 0,
    val_fraction: float = 0.2,
) -> tuple[CorrDiffDownscaler, dict[str, Any], Standardizer]:
    """Fit regression then diffusion; return the model, a report and the scaler."""
    fine = np.asarray(dataset["fine"], dtype=np.float32)      # (N, 2, ny, nx)
    coarse = np.asarray(dataset["coarse"], dtype=np.float32)  # (N, 2, h, w)
    orography = np.asarray(dataset["orography"], dtype=np.float32)
    n_samples, _, ny, nx = fine.shape

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_samples)
    n_val = max(1, int(round(val_fraction * n_samples)))
    val_idx, train_idx = permutation[:n_val], permutation[n_val:]
    if train_idx.size == 0:
        train_idx = val_idx

    # Standardise both channels on the training split so the diffusion residual has
    # roughly unit scale, which is what the EDM sigma schedule assumes.
    scaler = Standardizer.fit(fine[train_idx].transpose(0, 2, 3, 1).reshape(-1, 2))
    mean = torch.as_tensor(scaler.mean, dtype=torch.float32, device=device).view(1, 2, 1, 1)
    std = torch.as_tensor(scaler.std, dtype=torch.float32, device=device).view(1, 2, 1, 1)

    # build_condition stacks the upsampled coarse channels with the fine orography.
    condition_channels = fine.shape[1] + 1
    model = CorrDiffDownscaler(
        resolution=(ny, nx),
        condition_channels=condition_channels,
        output_channels=int(fine.shape[1]),
        cfg=model_cfg,
    ).to(device)

    condition_all = build_condition(coarse, orography, (ny, nx)).to(device)
    condition_all[:, :2] = (condition_all[:, :2] - mean) / std
    fine_all = (torch.as_tensor(fine, device=device) - mean) / std

    # No random cropping: SongUNetPosEmbd carries a positional embedding sized to
    # img_resolution, so it must see the resolution it was constructed for.
    def sample_batch(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return condition_all[indices], fine_all[indices]

    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    history: list[dict[str, float]] = []

    with Timer("downscaler_training", logger=LOGGER) as timer:
        # -- stage 1: deterministic regression ------------------------------
        optimizer = torch.optim.Adam(model.regression.parameters(), lr=float(train_cfg["learning_rate"]))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(train_cfg.get("lr_gamma", 1.0)))
        for epoch in range(epochs):
            model.train()
            order = rng.permutation(train_idx)
            total, batches = 0.0, 0
            for start in range(0, order.size, batch_size):
                condition, target = sample_batch(order[start : start + batch_size])
                optimizer.zero_grad()
                loss = F.mse_loss(model.regress(condition), target)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            scheduler.step()
            history.append({"stage": 0.0, "epoch": epoch, "loss": total / max(batches, 1)})

        for parameter in model.regression.parameters():
            parameter.requires_grad_(False)

        # -- stage 2: residual diffusion ------------------------------------
        optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=float(train_cfg["learning_rate"]))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(train_cfg.get("lr_gamma", 1.0)))
        for epoch in range(epochs):
            model.train()
            order = rng.permutation(train_idx)
            total, batches = 0.0, 0
            for start in range(0, order.size, batch_size):
                condition, target = sample_batch(order[start : start + batch_size])
                with torch.no_grad():
                    residual = target - model.regress(condition)
                optimizer.zero_grad()
                loss = model.diffusion_loss(residual, condition)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            scheduler.step()
            history.append({"stage": 1.0, "epoch": epoch, "loss": total / max(batches, 1)})

    # Validation in physical units: RMSE of the regression mean and of one sample.
    model.eval()
    with torch.no_grad():
        condition = condition_all[val_idx]
        target = fine_all[val_idx]
        regression = model.regress(condition)
        generated = regression + model.sample_residual(condition)
        rmse_regression = float(torch.sqrt(((regression - target) ** 2 * std**2).mean()))
        rmse_generated = float(torch.sqrt(((generated - target) ** 2 * std**2).mean()))
        # Does the diffusion stage restore the variance the regression smooths away?
        target_std = float(target.std())
        regression_std = float(regression.std())
        generated_std = float(generated.std())

    report = {
        "epochs": epochs,
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "val_rmse_regression": rmse_regression,
        "val_rmse_generated": rmse_generated,
        "target_std": target_std,
        "regression_std": regression_std,
        "generated_std": generated_std,
        "resolution": [int(ny), int(nx)],
        "wall_time_s": timer.elapsed,
        "history": history,
    }
    LOGGER.info(
        f"Downscaler: regression RMSE {rmse_regression:.4f}, generative RMSE {rmse_generated:.4f}, "
        f"std target/reg/gen {target_std:.3f}/{regression_std:.3f}/{generated_std:.3f}"
    )
    if checkpoint is not None:
        model.save(checkpoint, extra={**report, "scaler": scaler.to_dict()})
        report["checkpoint"] = str(checkpoint)
    return model, report, scaler
