"""
Inference helpers — load a checkpoint and run predictions.
"""

import numpy as np
import torch
from pathlib import Path

from src.model import CycloneLSTM, HybridCycloneModel
from src.features import FEATURE_COLS, LABEL_COLS, N_FEATURES


def load_model(checkpoint_path: str, model_type: str = "lstm",
               device: str = "cpu") -> torch.nn.Module:
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: path to .pt file
        model_type: "lstm" or "hybrid"
        device: "cpu" or "cuda"

    Returns:
        model in eval mode
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if model_type == "hybrid":
        model = HybridCycloneModel()
    else:
        model = CycloneLSTM()

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model


def load_scaler(processed_dir: str = "data/processed") -> dict:
    """Load the normalization scaler from disk."""
    dp = Path(processed_dir)
    return {
        "mean": np.load(dp / "scaler_mean.npy"),
        "std":  np.load(dp / "scaler_std.npy"),
    }


@torch.no_grad()
def predict(model, track_sequence: np.ndarray,
            scaler: dict = None, era5_sequence: np.ndarray = None,
            device: str = "cpu") -> dict:
    """
    Run inference on a single track sequence.

    Args:
        model: trained CycloneLSTM or HybridCycloneModel
        track_sequence: (lookback, n_features) numpy array — raw features
        scaler: {"mean": ..., "std": ...} normalization params
        era5_sequence: (4, 8, 20, 20) numpy array for hybrid model (optional)
        device: "cpu" or "cuda"

    Returns:
        dict with track forecasts, wind, RI probability, landfall probability
    """
    # Normalize
    X = track_sequence.astype(np.float32)
    if scaler is not None:
        mu  = scaler["mean"].squeeze()
        std = scaler["std"].squeeze()
        X = (X - mu) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_tensor = torch.from_numpy(X).unsqueeze(0).float().to(device)

    # Run model
    is_hybrid = isinstance(model, HybridCycloneModel)
    if is_hybrid:
        if era5_sequence is not None:
            era5 = torch.from_numpy(
                np.nan_to_num(era5_sequence, nan=0.0)
            ).unsqueeze(0).float().to(device)
        else:
            era5 = torch.zeros(1, 4, 8, 20, 20, device=device)
        track_pred, wind_pred, ri_pred, lf_pred = model(X_tensor, era5)
    else:
        track_pred, wind_pred, ri_pred, lf_pred = model(X_tensor)

    # Parse outputs
    track = track_pred[0].cpu().numpy()
    wind  = wind_pred[0].cpu().numpy()
    ri_prob = torch.sigmoid(ri_pred[0, 0]).cpu().item()
    lf_prob = torch.sigmoid(lf_pred[0, 0]).cpu().item()

    return {
        "track": {
            "24h": {"lat": float(track[0]), "lon": float(track[1])},
            "48h": {"lat": float(track[2]), "lon": float(track[3])},
            "72h": {"lat": float(track[4]), "lon": float(track[5])},
        },
        "wind": {
            "24h_kt": float(wind[0]),
            "48h_kt": float(wind[1]),
            "72h_kt": float(wind[2]),
        },
        "ri_probability": ri_prob,
        "landfall_probability": lf_prob,
    }
