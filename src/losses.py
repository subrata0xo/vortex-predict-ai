"""
Loss functions for multi-task cyclone prediction.
- Haversine loss for track forecasting
- Focal BCE for rare-event RI detection
- Multi-task weighted loss combining all heads
"""

import torch
import torch.nn as nn


def haversine_loss(pred_latlon: torch.Tensor,
                   true_latlon: torch.Tensor) -> torch.Tensor:
    """
    Mean great-circle distance in km across 24h / 48h / 72h horizons.
    pred/true shape: (batch, 6) — [lat24, lon24, lat48, lon48, lat72, lon72]
    """
    R = 6371.0
    losses = []
    for i in range(0, 6, 2):
        lat1 = torch.deg2rad(true_latlon[:, i])
        lon1 = torch.deg2rad(true_latlon[:, i + 1])
        lat2 = torch.deg2rad(pred_latlon[:, i])
        lon2 = torch.deg2rad(pred_latlon[:, i + 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
        c = 2 * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))
        losses.append((R * c).mean())
    return torch.stack(losses).mean()


def focal_bce(logits: torch.Tensor, targets: torch.Tensor,
              gamma: float = 2.0, pos_weight: float = 10.0) -> torch.Tensor:
    """
    Focal loss for binary classification.
    Down-weights easy negatives, focuses on hard positives.
    Better than plain BCE for rare events like RI.
    """
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pw
    )
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal = ((1 - p_t) ** gamma) * bce
    return focal.mean()


def multi_task_loss(preds: tuple, targets: torch.Tensor,
                    lambda_track: float = 1.0, lambda_wind: float = 0.5,
                    lambda_ri: float = 2.0, lambda_landfall: float = 1.0,
                    ri_idx: int = 10, landfall_idx: int = 11,
                    use_focal: bool = False,
                    focal_gamma: float = 2.0) -> tuple:
    """
    Combined multi-task loss for all output heads.

    Args:
        preds: (track_pred, wind_pred, ri_pred, landfall_pred)
        targets: (batch, 12) full label tensor
        use_focal: use focal BCE for RI instead of plain BCE

    Returns:
        (total_loss, detail_dict)
    """
    track_pred, wind_pred, ri_pred, lf_pred = preds

    track_true = targets[:, 0:6]
    wind_true  = targets[:, 6:9]
    ri_true    = targets[:, ri_idx].unsqueeze(1)
    lf_true    = targets[:, landfall_idx].unsqueeze(1)

    # Valid masks (skip NaN labels)
    valid_track = ~torch.isnan(track_true).any(dim=1)
    valid_wind  = ~torch.isnan(wind_true).any(dim=1)
    valid_ri    = ~torch.isnan(ri_true).any(dim=1)
    valid_lf    = ~torch.isnan(lf_true).any(dim=1)

    # Track loss (haversine)
    loss_track = (
        haversine_loss(track_pred[valid_track], track_true[valid_track])
        if valid_track.sum() > 0 else torch.tensor(0.0, device=targets.device)
    )

    # Wind loss (MSE)
    loss_wind = (
        nn.functional.mse_loss(wind_pred[valid_wind], wind_true[valid_wind])
        if valid_wind.sum() > 0 else torch.tensor(0.0, device=targets.device)
    )

    # RI loss (BCE or focal)
    if valid_ri.sum() > 0:
        if use_focal:
            loss_ri = focal_bce(ri_pred[valid_ri], ri_true[valid_ri],
                                gamma=focal_gamma, pos_weight=10.0)
        else:
            pw = torch.tensor([10.0], device=targets.device)
            loss_ri = nn.functional.binary_cross_entropy_with_logits(
                ri_pred[valid_ri], ri_true[valid_ri], pos_weight=pw
            )
    else:
        loss_ri = torch.tensor(0.0, device=targets.device)

    # Landfall loss (BCE)
    loss_lf = (
        nn.functional.binary_cross_entropy_with_logits(
            lf_pred[valid_lf], lf_true[valid_lf]
        )
        if valid_lf.sum() > 0 else torch.tensor(0.0, device=targets.device)
    )

    total = (
        lambda_track * loss_track
        + lambda_wind * loss_wind
        + lambda_ri * loss_ri
        + lambda_landfall * loss_lf
    )

    detail = {
        "track_km": loss_track.item(),
        "wind_mse": loss_wind.item(),
        "ri_bce":   loss_ri.item(),
        "lf_bce":   loss_lf.item(),
    }

    return total, detail
