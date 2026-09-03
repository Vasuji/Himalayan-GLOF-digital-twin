"""Train the MeshGraphNet moraine-stability surrogate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from glof_pipeline.surrogates.mgn_moraine import MoraineOperator
from glof_pipeline.surrogates.normalization import Standardizer
from glof_pipeline.terrain.mesh_builder import MoraineGraph
from glof_pipeline.utils.runtime import Timer, get_training_logger

LOGGER = get_training_logger("glof.train.mgn")


def train_mgn(
    dataset: dict[str, np.ndarray],
    graph: MoraineGraph,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    device: str = "cpu",
    checkpoint: str | Path | None = None,
    seed: int = 0,
) -> tuple[MoraineOperator, dict[str, Any]]:
    """Fit the surrogate to the limit-equilibrium stability fields.

    Scenarios are the batch dimension: one graph, many hydrological states. The
    loss is a plain MSE on the standardised ``[log FS, r_u]`` targets, because both
    targets are already dimensionless and comparably scaled after the transform.
    """
    node_features = np.asarray(dataset["node_features"])
    targets = np.asarray(dataset["targets"])
    edge_features = np.asarray(dataset["edge_features"])
    n_scenarios = node_features.shape[0]

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_scenarios)
    n_val = max(1, int(round(float(train_cfg.get("val_fraction", 0.2)) * n_scenarios)))
    val_idx, train_idx = permutation[:n_val], permutation[n_val:]

    operator = MoraineOperator(model_cfg, device=device)
    operator.node_scaler = Standardizer.fit(node_features[train_idx].reshape(-1, node_features.shape[-1]))
    operator.edge_scaler = Standardizer.fit(edge_features)
    operator.target_scaler = Standardizer.fit(targets[train_idx].reshape(-1, targets.shape[-1]))

    graph_object = operator.graph_object(graph)
    edge_tensor = torch.as_tensor(
        operator.edge_scaler.transform(edge_features), dtype=torch.float32, device=device
    )
    node_tensor = torch.as_tensor(
        operator.node_scaler.transform(node_features), dtype=torch.float32, device=device
    )
    target_tensor = torch.as_tensor(
        operator.target_scaler.transform(targets), dtype=torch.float32, device=device
    )

    optimizer = torch.optim.Adam(
        operator.model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(train_cfg.get("lr_gamma", 1.0))
    )
    criterion = torch.nn.MSELoss()
    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])

    history: list[dict[str, float]] = []
    with Timer("mgn_training", logger=LOGGER) as timer:
        for epoch in range(epochs):
            operator.model.train()
            order = torch.as_tensor(rng.permutation(train_idx.size), device=device)
            train_loss = 0.0
            for start in range(0, order.numel(), batch_size):
                batch = [int(train_idx[int(k)]) for k in order[start : start + batch_size]]
                optimizer.zero_grad()
                loss = torch.zeros((), device=device)
                for index in batch:
                    prediction = operator.model(node_tensor[index], edge_tensor, graph_object)
                    loss = loss + criterion(prediction, target_tensor[index])
                loss = loss / max(len(batch), 1)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach()) * len(batch)
            scheduler.step()

            operator.model.eval()
            with torch.no_grad():
                val_loss = float(
                    np.mean(
                        [
                            float(criterion(operator.model(node_tensor[i], edge_tensor, graph_object),
                                            target_tensor[i]))
                            for i in val_idx
                        ]
                    )
                )
            history.append(
                {"epoch": epoch, "train_mse": train_loss / max(train_idx.size, 1), "val_mse": val_loss}
            )
            if epoch % max(1, epochs // 6) == 0 or epoch == epochs - 1:
                LOGGER.info(f"epoch {epoch:3d} | train {history[-1]['train_mse']:.5f} | val {val_loss:.5f}")

    # Report the error in the physical variable, which is what a reviewer reads.
    with torch.no_grad():
        predictions = np.stack(
            [operator.model(node_tensor[i], edge_tensor, graph_object).cpu().numpy() for i in val_idx]
        )
    predictions = operator.target_scaler.inverse(predictions.reshape(-1, 2))
    truth = targets[val_idx].reshape(-1, 2)
    fos_rmse = float(np.sqrt(np.mean((np.exp(predictions[:, 0]) - np.exp(truth[:, 0])) ** 2)))
    ru_rmse = float(np.sqrt(np.mean((predictions[:, 1] - truth[:, 1]) ** 2)))

    report = {
        "backend": operator.backend,
        "epochs": epochs,
        "train_scenarios": int(train_idx.size),
        "val_scenarios": int(val_idx.size),
        "final_val_mse": history[-1]["val_mse"] if history else float("nan"),
        "val_rmse_factor_of_safety": fos_rmse,
        "val_rmse_pore_pressure_ratio": ru_rmse,
        "wall_time_s": timer.elapsed,
        "history": history,
    }
    operator.metadata.update(report)
    if checkpoint is not None:
        operator.save(checkpoint, extra=report)
        report["checkpoint"] = str(checkpoint)
    LOGGER.info(f"MeshGraphNet validation RMSE: FS {fos_rmse:.4f}, r_u {ru_rmse:.4f}")
    return operator, report
