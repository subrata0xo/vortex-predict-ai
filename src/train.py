"""
Training loop helpers for both LSTM and hybrid models.
"""

import torch
import torch.nn as nn

from src.losses import multi_task_loss
from src.metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, grad_clip: float = 1.0,
                    hybrid: bool = False, **loss_kwargs) -> float:
    """
    Train for one epoch.
    Args:
        model: CycloneLSTM or HybridCycloneModel
        hybrid: if True, loader yields (track, era5, y) instead of (X, y)
    Returns:
        mean training loss
    """
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        if hybrid:
            track, era5, y = batch
            preds = model(track, era5)
        else:
            X, y = batch
            preds = model(X)

        optimizer.zero_grad()
        loss, _ = multi_task_loss(preds, y, **loss_kwargs)

        if not torch.isnan(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, hybrid: bool = False,
             **loss_kwargs) -> tuple:
    """
    Validate model on a DataLoader.
    Returns:
        (mean_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_preds = [[], [], [], []]
    all_y = []
    n = 0

    for batch in loader:
        if hybrid:
            track, era5, y = batch
            preds = model(track, era5)
        else:
            X, y = batch
            preds = model(X)

        loss, _ = multi_task_loss(preds, y, **loss_kwargs)
        if not torch.isnan(loss):
            total_loss += loss.item()
            n += 1

        for i, p in enumerate(preds):
            all_preds[i].append(p)
        all_y.append(y)

    all_preds = [torch.cat(p) for p in all_preds]
    all_y = torch.cat(all_y)
    metrics = compute_metrics(all_preds, all_y)

    return total_loss / max(n, 1), metrics
