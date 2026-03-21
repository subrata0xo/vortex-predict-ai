"""
Evaluation metrics for cyclone prediction models.
Track error (km), wind MAE (kt), RI F1 score.
"""

import torch


def compute_metrics(preds: tuple, targets: torch.Tensor,
                    ri_idx: int = 10) -> dict:
    """
    Compute evaluation metrics for all output heads.

    Args:
        preds: (track_pred, wind_pred, ri_pred, landfall_pred)
        targets: (batch, 12) full label tensor

    Returns:
        dict with track_24h_km, track_48h_km, track_72h_km,
        wind_mae_kt, ri_f1
    """
    track_pred, wind_pred, ri_pred, _ = preds

    track_true = targets[:, 0:6]
    wind_true  = targets[:, 6:9]
    ri_true    = targets[:, ri_idx]

    R = 6371.0

    # ── Track errors (haversine km) ──────────────────────────────────────────
    track_errors = {}
    for step, h in enumerate([24, 48, 72]):
        i = step * 2
        valid = ~(torch.isnan(track_true[:, i]) | torch.isnan(track_true[:, i + 1]))
        if valid.sum() == 0:
            track_errors[f"track_{h}h_km"] = float("nan")
            continue
        lat1 = torch.deg2rad(track_true[valid, i])
        lon1 = torch.deg2rad(track_true[valid, i + 1])
        lat2 = torch.deg2rad(track_pred[valid, i])
        lon2 = torch.deg2rad(track_pred[valid, i + 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
        d = 2 * R * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))
        track_errors[f"track_{h}h_km"] = d.mean().item()

    # ── Wind MAE (knots, 24h only) ──────────────────────────────────────────
    valid_w = ~torch.isnan(wind_true[:, 0])
    wind_mae = (
        (wind_pred[valid_w, 0] - wind_true[valid_w, 0]).abs().mean().item()
        if valid_w.sum() > 0 else float("nan")
    )

    # ── RI F1 score ──────────────────────────────────────────────────────────
    valid_r = ~torch.isnan(ri_true)
    if valid_r.sum() > 0:
        ri_prob = torch.sigmoid(ri_pred[valid_r, 0])
        ri_bin  = (ri_prob > 0.5).float()
        ri_gt   = ri_true[valid_r]
        tp = ((ri_bin == 1) & (ri_gt == 1)).sum().float()
        fp = ((ri_bin == 1) & (ri_gt == 0)).sum().float()
        fn = ((ri_bin == 0) & (ri_gt == 1)).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        ri_f1 = (2 * prec * rec / (prec + rec + 1e-8)).item()
    else:
        ri_f1 = 0.0

    return {**track_errors, "wind_mae_kt": wind_mae, "ri_f1": ri_f1}
