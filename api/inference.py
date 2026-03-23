"""
Model server — loads checkpoints on startup, provides predict() method.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path so src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import CycloneLSTM, HybridCycloneModel
from src.features import FEATURE_COLS, N_FEATURES


class ModelServer:
    """Manages model loading and inference."""

    def __init__(self, checkpoint_dir: str = "checkpoints",
                 processed_dir: str = "data/processed"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.processed_dir = Path(processed_dir)
        self.models = {}
        self.scaler = None
        self._load_scaler()
        self._load_models()

    def _load_scaler(self):
        """Load normalization scaler from processed data."""
        mean_path = self.processed_dir / "scaler_mean.npy"
        std_path  = self.processed_dir / "scaler_std.npy"
        if mean_path.exists() and std_path.exists():
            self.scaler = {
                "mean": np.load(mean_path),
                "std":  np.load(std_path),
            }
            print("[✓] Scaler loaded")
        else:
            print("[!] Scaler files not found — predictions will use raw features")

    def _load_models(self):
        """Load available model checkpoints."""
        # LSTM baseline
        lstm_path = self.checkpoint_dir / "best.pt"
        if lstm_path.exists():
            try:
                model = CycloneLSTM()
                ckpt = torch.load(lstm_path, map_location="cpu", weights_only=True)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self.models["lstm"] = model
                print(f"[✓] LSTM model loaded from {lstm_path}")
            except Exception as e:
                print(f"[!] Failed to load LSTM: {e}")

        # Hybrid model
        hybrid_path = self.checkpoint_dir / "hybrid_best.pt"
        if hybrid_path.exists():
            try:
                model = HybridCycloneModel()
                ckpt = torch.load(hybrid_path, map_location="cpu", weights_only=True)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self.models["hybrid"] = model
                print(f"[✓] Hybrid model loaded from {hybrid_path}")
            except Exception as e:
                print(f"[!] Failed to load hybrid: {e}")

    @property
    def available_models(self) -> list:
        return list(self.models.keys())

    def _build_features(self, track_points: list) -> np.ndarray:
        """
        Convert a list of TrackPoint dicts to a (lookback, N_FEATURES) array.
        Matches the feature engineering from 01_data_loader.py.
        """
        n = len(track_points)
        features = np.zeros((n, N_FEATURES), dtype=np.float32)

        for i, pt in enumerate(track_points):
            lat = pt.get("lat", 0.0)
            lon = pt.get("lon", 0.0)
            wind = pt.get("wind", 0.0)
            pres = pt.get("pressure", 0.0) or 0.0
            dist = pt.get("dist2land", 500.0) or 500.0

            features[i, 0] = lat
            features[i, 1] = lon
            features[i, 2] = wind
            features[i, 3] = pres
            features[i, 12] = abs(lat)       # lat_abs
            features[i, 11] = dist           # dist2land

        # Compute deltas
        for i in range(1, n):
            features[i, 4] = features[i, 0] - features[i-1, 0]  # dLAT
            features[i, 5] = features[i, 1] - features[i-1, 1]  # dLON
            features[i, 8] = features[i, 2] - features[i-1, 2]  # dWIND
            features[i, 9] = features[i, 3] - features[i-1, 3]  # dPRES

        # Second-order deltas
        for i in range(2, n):
            features[i, 6] = features[i, 4] - features[i-1, 4]  # dLAT_2
            features[i, 7] = features[i, 5] - features[i-1, 5]  # dLON_2

        # Translation speed (km/h)
        R = 6371.0
        for i in range(1, n):
            lat1 = np.radians(features[i-1, 0])
            lat2 = np.radians(features[i, 0])
            dlt  = np.radians(features[i, 4])
            dln  = np.radians(features[i, 5])
            a = np.sin(dlt/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dln/2)**2
            features[i, 10] = 2 * R * np.arcsin(min(np.sqrt(max(a, 0)), 1)) / 6.0

        # Temporal features (use timestamp if available, else dummy)
        import math
        ts = track_points[-1].get("timestamp")
        if ts:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                month = dt.month
                jday  = dt.timetuple().tm_yday
            except Exception:
                month, jday = 6, 180
        else:
            month, jday = 6, 180

        for i in range(n):
            features[i, 13] = math.sin(2 * math.pi * month / 12)
            features[i, 14] = math.cos(2 * math.pi * month / 12)
            features[i, 15] = math.sin(2 * math.pi * jday / 365)
            features[i, 16] = math.cos(2 * math.pi * jday / 365)

        return features[-8:]  # last 8 timesteps (lookback)

    @torch.no_grad()
    def predict(self, track_points: list, model_type: str = "hybrid") -> dict:
        """
        Run prediction from raw track points.

        Args:
            track_points: list of dicts with lat, lon, wind, pressure, etc.
            model_type: "lstm" or "hybrid"

        Returns:
            dict with track forecasts, wind, RI probability, landfall probability
        """
        if model_type not in self.models:
            available = self.available_models
            if not available:
                raise RuntimeError("No models loaded")
            model_type = available[0]

        model = self.models[model_type]
        features = self._build_features(track_points)

        # Normalize
        if self.scaler is not None:
            mu  = self.scaler["mean"].squeeze()
            std = self.scaler["std"].squeeze()
            features = (features - mu) / std

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = torch.from_numpy(features).unsqueeze(0).float()

        # Run inference
        is_hybrid = isinstance(model, HybridCycloneModel)
        if is_hybrid:
            # Stage 2: Provide spatial placeholders for ERA5 and GridSat
            era5 = torch.zeros(1, 4, 8, 20, 20)           # (B, T, C, H, W)
            gridsat = torch.zeros(1, 6, 3, 128, 128)     # (B, T, C, H, W)
            preds = model(X, era5, gridsat)
        else:
            preds = model(X)

        track_pred, wind_pred, ri_pred, lf_pred = preds

        # Stage 5 Decoding
        track = track_pred[0].numpy()
        
        # Wind head: [spd24, spd48, spd72, c0...c5]
        wind_raw = wind_pred[0].numpy()
        spd24, spd48, spd72 = wind_raw[0], wind_raw[1], wind_raw[2]
        cat_logits = wind_raw[3:]
        cat_idx = int(np.argmax(cat_logits))
        
        ri_prob = torch.sigmoid(ri_pred[0, 0]).item()
        
        # Landfall head: [lat, lon, time]
        lf_raw = lf_pred[0].numpy()
        lf_lat, lf_lon, lf_time = lf_raw[0], lf_raw[1], lf_raw[2]
        lf_prob = 1.0 if lf_time < 72 else 0.0 # Heuristic for dashboard compat

        return {
            "model_type": model_type,
            "track": {
                "24h": {"lat": round(float(track[0]), 4),
                        "lon": round(float(track[1]), 4)},
                "48h": {"lat": round(float(track[2]), 4),
                        "lon": round(float(track[3]), 4)},
                "72h": {"lat": round(float(track[4]), 4),
                        "lon": round(float(track[5]), 4)},
            },
            "wind": {
                "24h_kt": round(float(spd24), 1),
                "48h_kt": round(float(spd48), 1),
                "72h_kt": round(float(spd72), 1),
                "category": cat_idx
            },
            "ri_probability": round(ri_prob, 4),
            "landfall_probability": round(lf_prob, 4),
            "landfall_details": {
                "lat": round(float(lf_lat), 2),
                "lon": round(float(lf_lon), 2),
                "time_h": round(float(lf_time), 1)
            },
            "ri_alert": ri_prob > 0.5,
        }
