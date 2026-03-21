"""
Cyclone Prediction — LSTM Baseline Training
Optimized for CPU overnight training (Python 3.11)

Run from project root:
    python notebooks/04_train.py

Outputs:
    checkpoints/best.pt           <- best validation model
    checkpoints/last.pt           <- last epoch model
    experiments/train_log.csv     <- loss/metric history per epoch
    experiments/test_results.json <- final test metrics
"""

from shlex import split

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
    "checkpoint_dir": Path("checkpoints"),
    "experiment_dir": Path("experiments"),

    # Model
    "input_size":   17,    # features per timestep (matches 01_data_loader.py)
    "hidden_size":  128,   # LSTM hidden units — kept small for CPU
    "num_layers":   2,     # LSTM depth
    "dropout":      0.2,

    # Training
    "epochs":       100,
    "batch_size":   16,
    "lr":           5e-4,
    "weight_decay": 1e-4,
    "patience":     20,    # early stopping
    "grad_clip":    1.0,

    # Loss weights
    "lambda_track":    1.0,
    "lambda_wind":     0.5,
    "lambda_ri":       2.0,  # upweighted — RI is rare class
    "lambda_landfall": 1.0,

    # Label indices (matches LABEL_COLS in 01_data_loader.py)
    "lat_24h_idx":  0,
    "lon_24h_idx":  1,
    "lat_48h_idx":  2,
    "lon_48h_idx":  3,
    "lat_72h_idx":  4,
    "lon_72h_idx":  5,
    "wind_24h_idx": 6,
    "wind_48h_idx": 7,
    "wind_72h_idx": 8,
    "ss_cat_idx":   9,
    "ri_idx":       10,
    "landfall_idx": 11,
}

# ─── DATASET ─────────────────────────────────────────────────────────────────

class CycloneDataset(Dataset):
    def __init__(self, split: str):
        dp = CFG["data_dir"]
        X = np.load(dp / f"{split}_X.npy")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(np.load(dp / f"{split}_y.npy")).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders():
    loaders = {}
    for split in ("train", "val", "test"):
        ds = CycloneDataset(split)
        loaders[split] = DataLoader(
            ds,
            batch_size=CFG["batch_size"],
            shuffle=(split == "train"),
            num_workers=0,     # 0 = safe on Windows
            pin_memory=False,  # CPU only
            drop_last=(split == "train"),
        )
        ri_rate = ds.y[:, CFG["ri_idx"]].mean().item()
        print(f"  {split:5s}: {len(ds):,} samples | RI rate: {ri_rate:.2%}")
    return loaders


# ─── MODEL ───────────────────────────────────────────────────────────────────

class CycloneLSTM(nn.Module):
    """
    Baseline LSTM: shared encoder + 4 task-specific output heads.

    Input  : (batch, lookback=8, features=17)
    Outputs: track (6), wind (3), ri (1), landfall (1)
    """
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=CFG["input_size"],
            hidden_size=CFG["hidden_size"],
            num_layers=CFG["num_layers"],
            batch_first=True,
            dropout=CFG["dropout"] if CFG["num_layers"] > 1 else 0,
        )

        self.dropout = nn.Dropout(CFG["dropout"])

        h = CFG["hidden_size"]

        # lat/lon at 24h, 48h, 72h -> 6 values
        self.track_head = nn.Sequential(
            nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 6)
        )

        # wind speed at 24h, 48h, 72h -> 3 values
        self.wind_head = nn.Sequential(
            nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 3)
        )

        # rapid intensification -> 1 logit (binary)
        self.ri_head = nn.Sequential(
            nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        # landfall within 72h -> 1 logit (binary)
        self.landfall_head = nn.Sequential(
            nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = self.dropout(out[:, -1])  # last timestep only

        track    = self.track_head(feat)
        wind     = self.wind_head(feat)
        ri       = self.ri_head(feat)
        landfall = self.landfall_head(feat)

        return track, wind, ri, landfall


# ─── LOSS ────────────────────────────────────────────────────────────────────

def haversine_loss(pred_latlon, true_latlon):
    """Mean great-circle distance in km across 24h / 48h / 72h horizons."""
    R = 6371.0
    losses = []
    for i in range(0, 6, 2):
        lat1 = torch.deg2rad(true_latlon[:, i])
        lon1 = torch.deg2rad(true_latlon[:, i + 1])
        lat2 = torch.deg2rad(pred_latlon[:, i])
        lon2 = torch.deg2rad(pred_latlon[:, i + 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        )
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

    pos_weight = torch.tensor([10.0])
    loss_ri = (
        nn.functional.binary_cross_entropy_with_logits(
            ri_pred[valid_ri], ri_true[valid_ri], pos_weight=pos_weight
        )
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

    detail = {
        "track_km": loss_track.item(),
        "wind_mse": loss_wind.item(),
        "ri_bce":   loss_ri.item(),
        "lf_bce":   loss_lf.item(),
    }

    return total, detail


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
        valid = ~(
            torch.isnan(track_true[:, i]) | torch.isnan(track_true[:, i + 1])
        )
        if valid.sum() == 0:
            track_errors[f"track_{h}h_km"] = float("nan")
            continue
        lat1 = torch.deg2rad(track_true[valid, i])
        lon1 = torch.deg2rad(track_true[valid, i + 1])
        lat2 = torch.deg2rad(track_pred[valid, i])
        lon2 = torch.deg2rad(track_pred[valid, i + 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        )
        d = 2 * R * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))
        track_errors[f"track_{h}h_km"] = d.mean().item()

    valid_w = ~torch.isnan(wind_true[:, 0])
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
    total_loss = 0.0
    n = 0
    for X, y in loader:
        optimizer.zero_grad()
        preds = model(X)
        loss, _ = multi_task_loss(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds  = [[], [], [], []]
    all_y      = []
    n = 0
    for X, y in loader:
        preds = model(X)
        loss, _ = multi_task_loss(preds, y)
        total_loss += loss.item()
        for i, p in enumerate(preds):
            all_preds[i].append(p)
        all_y.append(y)
        n += 1
    all_preds = [torch.cat(p) for p in all_preds]
    all_y     = torch.cat(all_y)
    metrics   = compute_metrics(all_preds, all_y)
    return total_loss / n, metrics


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    CFG["checkpoint_dir"].mkdir(exist_ok=True)
    CFG["experiment_dir"].mkdir(exist_ok=True)

    print("\n" + "=" * 55)
    print("  CYCLONE LSTM BASELINE — CPU TRAINING")
    print("=" * 55)
    print(f"  Device : CPU")
    print(f"  Epochs : {CFG['epochs']}")
    print(f"  Batch  : {CFG['batch_size']}")
    print(f"  LR     : {CFG['lr']}")
    print("=" * 55 + "\n")

    # ── Data ──────────────────────────────────────────────────
    print("[1/4] Loading datasets ...")
    loaders = get_loaders()

    # ── Model ─────────────────────────────────────────────────
    print("\n[2/4] Building model ...")
    model = CycloneLSTM()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Log file ──────────────────────────────────────────────
    log_path = CFG["experiment_dir"] / "train_log.csv"
    log_fields = [
        "epoch", "train_loss", "val_loss",
        "track_24h_km", "track_48h_km", "track_72h_km",
        "wind_mae_kt", "ri_f1", "lr", "elapsed_s",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # ── Training loop ─────────────────────────────────────────
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

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_metrics": val_metrics,
                "cfg":         {k: str(v) for k, v in CFG.items()},
            }, CFG["checkpoint_dir"] / "best.pt")
            print(f"  -> Saved best.pt  (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= CFG["patience"]:
                print(f"\n[!] Early stopping triggered at epoch {epoch}")
                break

    # Save last checkpoint
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_loss":    val_loss,
        "cfg":         {k: str(v) for k, v in CFG.items()},
    }, CFG["checkpoint_dir"] / "last.pt")

    # ── Test evaluation ───────────────────────────────────────
    print("\n[4/4] Evaluating on test set ...")
    best_path = CFG["checkpoint_dir"] / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print("  Loaded best.pt checkpoint.")
    else:
        print("  No best.pt found — evaluating with current model weights.")
    _, test_metrics = validate(model, loaders["test"])

    print("\n" + "=" * 55)
    print("  FINAL TEST RESULTS")
    print("=" * 55)
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Total time    : {(time.time() - train_start) / 60:.1f} min")
    print(f"  Log saved     : {log_path}")
    print(f"  Checkpoint    : checkpoints/best.pt")
    print("=" * 55 + "\n")

    results_path = CFG["experiment_dir"] / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[OK] Test results saved to {results_path}")
    print("[OK] Training complete!")


if __name__ == "__main__":
    main()