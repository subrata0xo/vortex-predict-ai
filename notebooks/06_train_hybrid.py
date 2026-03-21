"""
Cyclone Prediction — Hybrid Model Training
LSTM (track) + ConvLSTM (ERA5 grids) + Transformer fusion

Run from project root:
    python notebooks/06_train_hybrid.py

Outputs:
    checkpoints/hybrid_best.pt
    checkpoints/hybrid_last.pt
    experiments/hybrid_log.csv
    experiments/hybrid_test_results.json
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import json
import csv

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CFG = {
    # Paths
    "data_dir":       Path("data/processed"),
    "era5_dir":       Path("data/era5"),
    "checkpoint_dir": Path("checkpoints"),
    "experiment_dir": Path("experiments"),

    # Track encoder (LSTM)
    "track_input":    17,
    "track_hidden":   128,
    "track_layers":   2,

    # ERA5 encoder (ConvLSTM)
    "era5_channels":  8,     # SST, u10, v10, msl, q, t500, u500, v500
    "era5_hidden":    32,    # ConvLSTM hidden channels
    "era5_seq_len":   4,     # use 4 consecutive ERA5 patches (24h history)

    # Transformer fusion
    "fusion_dim":     192,   # 128 (LSTM) + 64 (ConvLSTM flat)
    "n_heads":        4,
    "n_layers":       2,
    "dropout":        0.2,

    # Training
    "epochs":         100,
    "batch_size":     16,
    "lr":             1e-4,
    "weight_decay":   1e-4,
    "patience":       20,
    "grad_clip":      1.0,

    # Loss weights
    "lambda_track":    1.0,
    "lambda_wind":     0.5,
    "lambda_ri":       2.0,
    "lambda_landfall": 1.0,
    "focal_gamma":     2.0,   # focal loss gamma for RI

    # Label indices
    "ri_idx":       10,
    "landfall_idx": 11,
}

# ─── DATASET ─────────────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    """
    Each sample:
        track_seq : (lookback=8, 17)  — IBTrACS features
        era5_seq  : (4, 8, 20, 20)    — ERA5 spatial patches
        y         : (12,)             — labels
    """
    def __init__(self, split: str):
        dp  = CFG["data_dir"]
        ep  = CFG["era5_dir"]

        # Track features
        X = np.load(dp / f"{split}_X.npy")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(X).float()

        # Labels
        self.y = torch.from_numpy(
            np.load(dp / f"{split}_y.npy")
        ).float()

        # ERA5 patches — load if available, else zeros
        era5_path = ep / "era5_patches_all.npy"
        meta_path = dp / f"{split}_meta.csv"

        self.era5 = None
        if era5_path.exists():
            try:
                import pandas as pd
                all_meta  = self._load_all_meta(dp)
                split_meta = pd.read_csv(meta_path, parse_dates=["ISO_TIME"])
                era5_all   = np.load(era5_path)
                era5_all   = np.nan_to_num(era5_all, nan=0.0)

                # Match split rows to all_meta indices
                merged = split_meta.merge(
                    all_meta.reset_index().rename(columns={"index": "era5_idx"}),
                    on=["SID", "ISO_TIME"], how="left"
                )
                idx = merged["era5_idx"].fillna(-1).astype(int).values

                # Build era5 sequences of length 4 (last 4 patches per sample)
                N = len(self.X)
                C = CFG["era5_channels"]
                era5_seq = np.zeros((N, 4, C, 20, 20), dtype=np.float32)

                for i, ei in enumerate(idx):
                    if ei < 0:
                        continue
                    # Use 4 consecutive ERA5 patches ending at current step
                    for t, offset in enumerate(range(-3, 1)):
                        j = ei + offset
                        if 0 <= j < len(era5_all):
                            p = era5_all[j]
                            if p.shape[0] >= C:
                                era5_seq[i, t] = p[:C]

                self.era5 = torch.from_numpy(era5_seq).float()
                print(f"  {split:5s}: ERA5 loaded  shape={self.era5.shape}")
            except Exception as e:
                print(f"  {split:5s}: ERA5 load failed ({e}) — using zeros")

        if self.era5 is None:
            N = len(self.X)
            self.era5 = torch.zeros(N, 4, CFG["era5_channels"], 20, 20)

        ri_rate = torch.nanmean(self.y[:, CFG["ri_idx"]]).item()
        print(f"  {split:5s}: {len(self.X):,} samples | RI rate: {ri_rate:.2%}")

    @staticmethod
    def _load_all_meta(dp):
        import pandas as pd
        dfs = []
        for s in ("train", "val", "test"):
            p = dp.parent / "processed" / f"{s}_meta.csv"
            if not p.exists():
                p = dp / f"{s}_meta.csv"
            if p.exists():
                dfs.append(pd.read_csv(p, parse_dates=["ISO_TIME"]))
        return pd.concat(dfs, ignore_index=True).drop_duplicates(
            subset=["SID", "ISO_TIME"]
        ).reset_index(drop=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.era5[idx], self.y[idx]


def get_loaders():
    loaders = {}
    for split in ("train", "val", "test"):
        ds = HybridDataset(split)
        loaders[split] = DataLoader(
            ds,
            batch_size=CFG["batch_size"],
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
            drop_last=(split == "train"),
        )
    return loaders


# ─── MODEL COMPONENTS ─────────────────────────────────────────────────────────

class TrackEncoder(nn.Module):
    """LSTM encoder for IBTrACS track + feature sequence."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=CFG["track_input"],
            hidden_size=CFG["track_hidden"],
            num_layers=CFG["track_layers"],
            batch_first=True,
            dropout=CFG["dropout"] if CFG["track_layers"] > 1 else 0,
        )
        self.dropout = nn.Dropout(CFG["dropout"])

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.dropout(out[:, -1])   # (B, 128)


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size, padding=pad
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates    = self.conv(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ERA5Encoder(nn.Module):
    """
    ConvLSTM encoder for ERA5 spatial patches.
    Input : (B, T=4, C=8, H=20, W=20)
    Output: (B, 64) flattened feature vector
    """
    def __init__(self):
        super().__init__()
        hc = CFG["era5_hidden"]   # 32

        # Spatial feature extractor (applied per timestep)
        self.cnn = nn.Sequential(
            nn.Conv2d(CFG["era5_channels"], 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, hc, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),   # -> (B, 32, 5, 5)
        )

        # ConvLSTM over time
        self.clstm = ConvLSTMCell(hc, hc, kernel_size=3)

        self.flatten = nn.Sequential(
            nn.Flatten(),        # (B, 32*5*5) = (B, 800)
            nn.Linear(hc * 5 * 5, 64),
            nn.ReLU(),
            nn.Dropout(CFG["dropout"]),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        hc = CFG["era5_hidden"]

        # Init hidden state
        h = torch.zeros(B, hc, 5, 5, device=x.device)
        c = torch.zeros(B, hc, 5, 5, device=x.device)

        for t in range(T):
            feat = self.cnn(x[:, t])   # (B, hc, 5, 5)
            h, c = self.clstm(feat, h, c)

        return self.flatten(h)   # (B, 64)


class HybridCycloneModel(nn.Module):
    """
    Full hybrid model:
        TrackEncoder (LSTM)   -> 128-dim
        ERA5Encoder (ConvLSTM)-> 64-dim
        Concat                -> 192-dim
        Transformer           -> 192-dim
        4 output heads
    """
    def __init__(self):
        super().__init__()

        self.track_enc = TrackEncoder()
        self.era5_enc  = ERA5Encoder()

        fd = CFG["fusion_dim"]   # 192

        # Transformer encoder for fusion
        enc_layer = nn.TransformerEncoderLayer(
            d_model=fd,
            nhead=CFG["n_heads"],
            dim_feedforward=fd * 2,
            dropout=CFG["dropout"],
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=CFG["n_layers"]
        )

        self.dropout = nn.Dropout(CFG["dropout"])

        # Output heads
        self.track_head = nn.Sequential(
            nn.Linear(fd, 96), nn.ReLU(), nn.Linear(96, 6)
        )
        self.wind_head = nn.Sequential(
            nn.Linear(fd, 48), nn.ReLU(), nn.Linear(48, 3)
        )
        self.ri_head = nn.Sequential(
            nn.Linear(fd, 48), nn.ReLU(), nn.Linear(48, 1)
        )
        self.landfall_head = nn.Sequential(
            nn.Linear(fd, 48), nn.ReLU(), nn.Linear(48, 1)
        )

    def forward(self, track, era5):
        track_feat = self.track_enc(track)    # (B, 128)
        era5_feat  = self.era5_enc(era5)      # (B, 64)

        fused = torch.cat([track_feat, era5_feat], dim=1)   # (B, 192)
        fused = fused.unsqueeze(1)                           # (B, 1, 192)
        fused = self.transformer(fused).squeeze(1)           # (B, 192)
        fused = self.dropout(fused)

        track    = self.track_head(fused)
        wind     = self.wind_head(fused)
        ri       = self.ri_head(fused)
        landfall = self.landfall_head(fused)

        return track, wind, ri, landfall


# ─── FOCAL LOSS FOR RI ────────────────────────────────────────────────────────

def focal_bce(logits, targets, gamma=2.0, pos_weight=10.0):
    """
    Focal loss for binary classification.
    Down-weights easy negatives, focuses on hard positives.
    Better than plain BCE for rare events like RI.
    """
    pw  = torch.tensor([pos_weight], device=logits.device)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pw
    )
    prob  = torch.sigmoid(logits)
    p_t   = prob * targets + (1 - prob) * (1 - targets)
    focal = ((1 - p_t) ** gamma) * bce
    return focal.mean()


# ─── LOSS ────────────────────────────────────────────────────────────────────

def haversine_loss(pred, true):
    R = 6371.0
    losses = []
    for i in range(0, 6, 2):
        lat1 = torch.deg2rad(true[:, i])
        lon1 = torch.deg2rad(true[:, i + 1])
        lat2 = torch.deg2rad(pred[:, i])
        lon2 = torch.deg2rad(pred[:, i + 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
        c = 2 * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))
        losses.append((R * c).mean())
    return torch.stack(losses).mean()


def multi_task_loss(preds, targets):
    track_pred, wind_pred, ri_pred, lf_pred = preds

    track_true = targets[:, 0:6]
    wind_true  = targets[:, 6:9]
    ri_true    = targets[:, CFG["ri_idx"]].unsqueeze(1)
    lf_true    = targets[:, CFG["landfall_idx"]].unsqueeze(1)

    valid_track = ~torch.isnan(track_true).any(dim=1)
    valid_wind  = ~torch.isnan(wind_true).any(dim=1)
    valid_ri    = ~torch.isnan(ri_true).any(dim=1)
    valid_lf    = ~torch.isnan(lf_true).any(dim=1)

    loss_track = (
        haversine_loss(track_pred[valid_track], track_true[valid_track])
        if valid_track.sum() > 0 else torch.tensor(0.0)
    )
    loss_wind = (
        nn.functional.mse_loss(wind_pred[valid_wind], wind_true[valid_wind])
        if valid_wind.sum() > 0 else torch.tensor(0.0)
    )
    loss_ri = (
        focal_bce(ri_pred[valid_ri], ri_true[valid_ri],
                  gamma=CFG["focal_gamma"], pos_weight=10.0)
        if valid_ri.sum() > 0 else torch.tensor(0.0)
    )
    loss_lf = (
        nn.functional.binary_cross_entropy_with_logits(
            lf_pred[valid_lf], lf_true[valid_lf]
        )
        if valid_lf.sum() > 0 else torch.tensor(0.0)
    )

    total = (
        CFG["lambda_track"]    * loss_track
        + CFG["lambda_wind"]   * loss_wind
        + CFG["lambda_ri"]     * loss_ri
        + CFG["lambda_landfall"] * loss_lf
    )

    return total, {
        "track_km": loss_track.item(),
        "wind_mse": loss_wind.item(),
        "ri_focal": loss_ri.item(),
        "lf_bce":   loss_lf.item(),
    }


# ─── METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets):
    track_pred, wind_pred, ri_pred, _ = preds

    track_true = targets[:, 0:6]
    wind_true  = targets[:, 6:9]
    ri_true    = targets[:, CFG["ri_idx"]]

    R = 6371.0
    track_errors = {}
    for step, h in enumerate([24, 48, 72]):
        i = step * 2
        valid = ~(torch.isnan(track_true[:, i]) | torch.isnan(track_true[:, i+1]))
        if valid.sum() == 0:
            track_errors[f"track_{h}h_km"] = float("nan")
            continue
        lat1 = torch.deg2rad(track_true[valid, i])
        lon1 = torch.deg2rad(track_true[valid, i+1])
        lat2 = torch.deg2rad(track_pred[valid, i])
        lon2 = torch.deg2rad(track_pred[valid, i+1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (torch.sin(dlat/2)**2
             + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2)
        d = 2 * R * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))
        track_errors[f"track_{h}h_km"] = d.mean().item()

    valid_w  = ~torch.isnan(wind_true[:, 0])
    wind_mae = (
        (wind_pred[valid_w, 0] - wind_true[valid_w, 0]).abs().mean().item()
        if valid_w.sum() > 0 else float("nan")
    )

    valid_r = ~torch.isnan(ri_true)
    ri_prob = torch.sigmoid(ri_pred[valid_r, 0])
    ri_bin  = (ri_prob > 0.5).float()
    ri_gt   = ri_true[valid_r]
    tp   = ((ri_bin == 1) & (ri_gt == 1)).sum().float()
    fp   = ((ri_bin == 1) & (ri_gt == 0)).sum().float()
    fn   = ((ri_bin == 0) & (ri_gt == 1)).sum().float()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    ri_f1 = (2 * prec * rec / (prec + rec + 1e-8)).item()

    return {**track_errors, "wind_mae_kt": wind_mae, "ri_f1": ri_f1}


# ─── TRAIN / VALIDATE ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    n = 0
    for track, era5, y in loader:
        optimizer.zero_grad()
        preds = model(track, era5)
        loss, _ = multi_task_loss(preds, y)
        if not torch.isnan(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total     = 0.0
    all_preds = [[], [], [], []]
    all_y     = []
    n = 0
    for track, era5, y in loader:
        preds = model(track, era5)
        loss, _ = multi_task_loss(preds, y)
        if not torch.isnan(loss):
            total += loss.item()
            n += 1
        for i, p in enumerate(preds):
            all_preds[i].append(p)
        all_y.append(y)
    all_preds = [torch.cat(p) for p in all_preds]
    all_y     = torch.cat(all_y)
    metrics   = compute_metrics(all_preds, all_y)
    return total / max(n, 1), metrics


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    CFG["checkpoint_dir"].mkdir(exist_ok=True)
    CFG["experiment_dir"].mkdir(exist_ok=True)

    print("\n" + "=" * 55)
    print("  CYCLONE HYBRID MODEL — LSTM + ConvLSTM + Transformer")
    print("=" * 55)
    print(f"  Device    : CPU")
    print(f"  Epochs    : {CFG['epochs']}")
    print(f"  Batch     : {CFG['batch_size']}")
    print(f"  Fusion dim: {CFG['fusion_dim']}")
    print(f"  RI loss   : Focal (gamma={CFG['focal_gamma']})")
    print("=" * 55 + "\n")

    print("[1/4] Loading datasets ...")
    loaders = get_loaders()

    print("\n[2/4] Building model ...")
    model = HybridCycloneModel()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )

    log_path = CFG["experiment_dir"] / "hybrid_log.csv"
    log_fields = [
        "epoch", "train_loss", "val_loss",
        "track_24h_km", "track_48h_km", "track_72h_km",
        "wind_mae_kt", "ri_f1", "lr", "elapsed_s",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    print("\n[3/4] Training ...\n")
    best_val_loss  = float("inf")
    patience_count = 0
    train_start    = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        ep_start   = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimizer)
        val_loss, val_metrics = validate(model, loaders["val"])
        scheduler.step(val_loss)

        elapsed       = time.time() - ep_start
        total_elapsed = time.time() - train_start

        t24 = val_metrics.get("track_24h_km", float("nan"))
        t48 = val_metrics.get("track_48h_km", float("nan"))
        t72 = val_metrics.get("track_72h_km", float("nan"))
        wnd = val_metrics.get("wind_mae_kt",  float("nan"))
        f1  = val_metrics.get("ri_f1", 0.0)

        print(
            f"Ep {epoch:3d}/{CFG['epochs']} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"trk24h {t24:6.1f}km | trk72h {t72:6.1f}km | "
            f"wind {wnd:5.1f}kt | RI_F1 {f1:.3f} | "
            f"{elapsed:.0f}s"
        )

        row = {
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss, 6),
            "track_24h_km": round(t24, 2),
            "track_48h_km": round(t48, 2),
            "track_72h_km": round(t72, 2),
            "wind_mae_kt":  round(wnd, 2),
            "ri_f1":        round(f1, 4),
            "lr":           optimizer.param_groups[0]["lr"],
            "elapsed_s":    round(total_elapsed, 1),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_metrics": val_metrics,
                "cfg":         {k: str(v) for k, v in CFG.items()},
            }, CFG["checkpoint_dir"] / "hybrid_best.pt")
            print(f"  -> Saved hybrid_best.pt  (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= CFG["patience"]:
                print(f"\n[!] Early stopping at epoch {epoch}")
                break

    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "cfg":         {k: str(v) for k, v in CFG.items()},
    }, CFG["checkpoint_dir"] / "hybrid_last.pt")

    print("\n[4/4] Evaluating on test set ...")
    best_path = CFG["checkpoint_dir"] / "hybrid_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print("  Loaded hybrid_best.pt")
    _, test_metrics = validate(model, loaders["test"])

    print("\n" + "=" * 55)
    print("  FINAL TEST RESULTS — HYBRID MODEL")
    print("=" * 55)
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # Compare vs baseline
    print("\n  vs LSTM baseline:")
    baseline = {
        "track_24h_km": 135.5, "track_48h_km": 178.7,
        "track_72h_km": 231.6, "wind_mae_kt": 7.0, "ri_f1": 0.059
    }
    for k, bv in baseline.items():
        hv = test_metrics.get(k, float("nan"))
        if k == "ri_f1":
            diff = hv - bv
            sign = "+" if diff > 0 else ""
            print(f"  {k:20s}: {sign}{diff:.4f}  ({'better' if diff > 0 else 'worse'})")
        else:
            diff = hv - bv
            sign = "+" if diff > 0 else ""
            print(f"  {k:20s}: {sign}{diff:.1f} km  ({'better' if diff < 0 else 'worse'})")

    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Total time    : {(time.time() - train_start) / 60:.1f} min")
    print("=" * 55 + "\n")

    results_path = CFG["experiment_dir"] / "hybrid_test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[OK] Results saved to {results_path}")
    print("[OK] Hybrid training complete!")


if __name__ == "__main__":
    main()
