"""Train the Fourier Neural Operator flood-routing surrogate.

Two details make the difference between a surrogate that survives a rollout and
one that does not:

* the network predicts the **increment** over one output interval, not the next
  state, and
* training uses a short autoregressive rollout ("pushforward"), so the model sees
  its own predictions as inputs and learns to damp its own error growth rather
  than only the one-step error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from glof_pipeline.surrogates.fno_swe import FloodOperator
from glof_pipeline.surrogates.losses import combined_operator_loss
from glof_pipeline.utils.runtime import Timer, get_training_logger

LOGGER = get_training_logger("glof.train.fno")


def _sequences(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the solver output into ``(n_scenarios, n_frames, 3, ny, nx)``."""
    return np.stack(
        [dataset["depth"], dataset["momentum_x"], dataset["momentum_y"]], axis=2
    ).astype(np.float32)


def train_fno(
    dataset: dict[str, np.ndarray],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    device: str = "cpu",
    checkpoint: str | Path | None = None,
    seed: int = 0,
    val_fraction: float = 0.25,
) -> tuple[FloodOperator, dict[str, Any]]:
    """Fit the operator to the finite-volume solver's trajectories."""
    sequences = _sequences(dataset)
    bed = np.asarray(dataset["bed"], dtype=np.float32)
    n_scenarios, n_frames = sequences.shape[0], sequences.shape[1]
    rollout = int(min(model_cfg.get("rollout_steps", 4), n_frames - 1))
    if rollout < 1:
        raise ValueError("Need at least two frames per scenario to train a time-advance operator.")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_scenarios)
    n_val = max(1, int(round(val_fraction * n_scenarios)))
    val_idx, train_idx = permutation[:n_val], permutation[n_val:]
    if train_idx.size == 0:  # tiny smoke datasets
        train_idx = val_idx

    operator = FloodOperator(model_cfg, device=device)
    # Reference scales from the training set: depth by its 99th percentile so the
    # network sees O(1) inputs even though most of the domain is dry.
    wet = sequences[:, :, 0]
    operator.scales.depth_ref_m = float(max(np.percentile(wet[wet > 1e-3], 99) if (wet > 1e-3).any() else 1.0, 0.1))
    operator.scales.bed_offset_m = float(bed.mean())
    operator.scales.bed_ref_m = float(max(bed.std(), 1.0))

    bed_tensor = torch.as_tensor(
        (bed - operator.scales.bed_offset_m) / operator.scales.bed_ref_m,
        dtype=torch.float32, device=device,
    )[None, None]

    def to_state(batch: np.ndarray) -> torch.Tensor:
        s = operator.scales
        scaled = batch.copy()
        scaled[:, 0] /= s.depth_ref_m
        scaled[:, 1] /= s.momentum_ref
        scaled[:, 2] /= s.momentum_ref
        return torch.as_tensor(scaled, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        operator.model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(train_cfg.get("lr_gamma", 1.0))
    )
    spectral_weight = float(train_cfg.get("spectral_loss_weight", 0.15))
    mass_weight = float(train_cfg.get("mass_loss_weight", 0.05))
    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])

    def rollout_loss(indices: np.ndarray, starts: np.ndarray) -> tuple[torch.Tensor, dict[str, float]]:
        state = to_state(sequences[indices, starts])
        bed_batch = bed_tensor.expand(state.shape[0], -1, -1, -1)
        total = torch.zeros((), device=device)
        components: dict[str, float] = {}
        for k in range(1, rollout + 1):
            inputs = torch.cat([state, bed_batch], dim=1)
            state = operator.forward_increment(inputs)
            target = to_state(sequences[indices, starts + k])
            step_loss, components = combined_operator_loss(state, target, spectral_weight, mass_weight)
            total = total + step_loss
        return total / rollout, components

    history: list[dict[str, float]] = []
    with Timer("fno_training", logger=LOGGER) as timer:
        for epoch in range(epochs):
            operator.model.train()
            order = rng.permutation(train_idx)
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, order.size, batch_size):
                indices = order[start : start + batch_size]
                starts = rng.integers(0, n_frames - rollout, size=indices.size)
                optimizer.zero_grad()
                loss, _ = rollout_loss(indices, starts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(operator.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            scheduler.step()

            operator.model.eval()
            with torch.no_grad():
                val_starts = np.zeros(val_idx.size, dtype=int)
                val_loss, components = rollout_loss(val_idx, val_starts)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / max(n_batches, 1),
                    "val_loss": float(val_loss),
                    **{f"val_{k}": v for k, v in components.items()},
                }
            )
            if epoch % max(1, epochs // 6) == 0 or epoch == epochs - 1:
                LOGGER.info(
                    f"epoch {epoch:3d} | train {history[-1]['train_loss']:.4f} | "
                    f"val {history[-1]['val_loss']:.4f} "
                    f"(rel L2 {history[-1].get('val_relative_l2', float('nan')):.4f})"
                )

    report = {
        "backend": operator.backend,
        "epochs": epochs,
        "rollout_steps": rollout,
        "train_scenarios": int(train_idx.size),
        "val_scenarios": int(val_idx.size),
        "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
        "final_val_relative_l2": history[-1].get("val_relative_l2") if history else None,
        "depth_ref_m": operator.scales.depth_ref_m,
        "wall_time_s": timer.elapsed,
        "history": history,
    }
    operator.metadata.update(report)
    if checkpoint is not None:
        operator.save(checkpoint, extra=report)
        report["checkpoint"] = str(checkpoint)
    return operator, report
