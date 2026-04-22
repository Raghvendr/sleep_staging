from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .data import NightSequenceDataset, SequenceTorchDataset
from .metrics import compute_classification_metrics


@dataclass
class PaperTrainingConfig:
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.25
    epochs: int = 20
    device: str = "cpu"
    l1_lambda: float | None = None
    num_workers: int = 0


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    num_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, num_classes)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1).float()

    losses = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-100)
    losses = losses * mask_flat
    denom = mask_flat.sum().clamp(min=1.0)
    return losses.sum() / denom


def _l1_regularization(model: nn.Module) -> torch.Tensor:
    reg = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            term = module.weight.abs().sum()
            reg = term if reg is None else reg + term
    if reg is None:
        return torch.tensor(0.0)
    return reg


def _make_dataloader(dataset: NightSequenceDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        SequenceTorchDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    l1_lambda: float = 0.0,
) -> float:
    model.train()
    loss_total = 0.0
    batch_count = 0

    for batch in dataloader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.long)
        mask = batch["mask"].to(device=device, dtype=torch.float32)

        optimizer.zero_grad()
        logits = model(x)
        loss = masked_cross_entropy_loss(logits=logits, targets=y, mask=mask)
        if l1_lambda > 0:
            loss = loss + l1_lambda * _l1_regularization(model).to(device)
        loss.backward()
        optimizer.step()

        loss_total += float(loss.item())
        batch_count += 1

    return loss_total / max(batch_count, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: NightSequenceDataset,
    device: str = "cpu",
    batch_size: int = 2,
    num_workers: int = 0,
) -> dict[str, object]:
    model.eval()
    torch_device = torch.device(device)
    dataloader = _make_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    mask_all: list[np.ndarray] = []
    nightly_metrics: list[dict[str, float]] = []

    for batch in dataloader:
        x = batch["x"].to(device=torch_device, dtype=torch.float32)
        y = batch["y"].cpu().numpy()
        mask = batch["mask"].cpu().numpy()
        logits = model(x).cpu().numpy()
        logits_all.append(logits)
        y_all.append(y)
        mask_all.append(mask)

    logits_all_np = np.concatenate(logits_all, axis=0)
    y_all_np = np.concatenate(y_all, axis=0)
    mask_all_np = np.concatenate(mask_all, axis=0)
    pred_all_np = logits_all_np.argmax(axis=-1)

    valid = mask_all_np.astype(bool)
    y_true = y_all_np[valid]
    y_pred = pred_all_np[valid]
    overall = compute_classification_metrics(y_true=y_true, y_pred=y_pred, stage_names=dataset.stage_names)

    for night_idx in range(len(dataset.X)):
        valid_night = valid[night_idx]
        if valid_night.sum() == 0:
            continue
        night_metrics = compute_classification_metrics(
            y_true=y_all_np[night_idx][valid_night],
            y_pred=pred_all_np[night_idx][valid_night],
            stage_names=dataset.stage_names,
        )
        nightly_metrics.append(
            {
                "accuracy": float(night_metrics["accuracy"]),
                "kappa": float(night_metrics["kappa"]),
            }
        )

    if nightly_metrics:
        overall["nightly_accuracy_mean"] = float(np.mean([item["accuracy"] for item in nightly_metrics]))
        overall["nightly_accuracy_std"] = float(np.std([item["accuracy"] for item in nightly_metrics]))
        overall["nightly_kappa_mean"] = float(np.mean([item["kappa"] for item in nightly_metrics]))
        overall["nightly_kappa_std"] = float(np.std([item["kappa"] for item in nightly_metrics]))
    else:
        overall["nightly_accuracy_mean"] = 0.0
        overall["nightly_accuracy_std"] = 0.0
        overall["nightly_kappa_mean"] = 0.0
        overall["nightly_kappa_std"] = 0.0

    return overall


def fit_model(
    model: nn.Module,
    train_dataset: NightSequenceDataset,
    val_dataset: NightSequenceDataset | None = None,
    config: PaperTrainingConfig | None = None,
) -> dict[str, object]:
    config = config or PaperTrainingConfig()
    device = torch.device(config.device)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    train_loader = _make_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    history: list[dict[str, float]] = []
    best_val_kappa = float("-inf")
    best_state_dict = None

    for epoch in range(config.epochs):
        l1_lambda = config.l1_lambda if config.l1_lambda is not None else config.weight_decay
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            l1_lambda=l1_lambda,
        )
        record: dict[str, float] = {"epoch": float(epoch + 1), "train_loss": float(train_loss)}

        if val_dataset is not None:
            val_metrics = evaluate_model(
                model=model,
                dataset=val_dataset,
                device=config.device,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
            record["val_accuracy"] = float(val_metrics["accuracy"])
            record["val_kappa"] = float(val_metrics["kappa"])
            if float(val_metrics["kappa"]) > best_val_kappa:
                best_val_kappa = float(val_metrics["kappa"])
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(record)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_val_kappa": best_val_kappa if val_dataset is not None else None,
    }
