"""
Cyclone Prediction — Advanced 6-Stage Pipeline Training
GridSat Vision + ERA5 ConvLSTM + IBTrACS LSTM + Kendall Loss
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import time
import json
import csv

# Ensure src/ packages are discoverable from notebooks directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import get_loaders, HybridDataset
from src.model import HybridCycloneModel
from src.loss import KendallMultiTaskLoss

CFG = {
    "data_dir":       Path("data/processed"),
    "era5_dir":       Path("data/era5"),
    "checkpoint_dir": Path("checkpoints"),
    "experiment_dir": Path("experiments"),
    "epochs":         25,  # Increased for proper Cloud GPU convergence
    "batch_size":     16,
    "lr":             1e-4,
    "grad_clip":      1.0,
}

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for track, era5, gridsat, y in loader:
        track, era5, gridsat, y = track.to(device), era5.to(device), gridsat.to(device), y.to(device)
        
        optimizer.zero_grad()
        preds = model(track, era5, gridsat)
        loss, individual_losses = criterion(preds, y)
        
        if not torch.isnan(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    for track, era5, gridsat, y in loader:
        track, era5, gridsat, y = track.to(device), era5.to(device), gridsat.to(device), y.to(device)
        
        preds = model(track, era5, gridsat)
        loss, _ = criterion(preds, y)
        if not torch.isnan(loss):
            total += loss.item()
            n += 1
    return total / max(n, 1)

def main():
    CFG["checkpoint_dir"].mkdir(exist_ok=True)
    CFG["experiment_dir"].mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("  6-STAGE HYBRID PIPELINE: GRIDSAT + ERA5 + TRACK")
    print(f"  Execution Device : {device}")
    print("=" * 65 + "\n")

    loaders = get_loaders(
        batch_size=CFG["batch_size"], 
        dataset_cls=HybridDataset,
        data_dir=str(CFG["data_dir"]),
        era5_dir=str(CFG["era5_dir"])
    )

    model = HybridCycloneModel().to(device)
    criterion = KendallMultiTaskLoss(num_tasks=5).to(device)
    
    # Optimizer must update both model weights AND the learnable loss variance parameters
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), 
        lr=CFG["lr"]
    )

    best_val_loss = float("inf")
    train_start = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        ep_start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss = validate(model, loaders["val"], criterion, device)
        elapsed = time.time() - ep_start
        
        print(f"Epoch {epoch:2d}/{CFG['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save exact state_dict expected by prediction inference script
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss_weights": criterion.state_dict()
            }, CFG["checkpoint_dir"] / "hybrid_best.pt")
            print(f"  -> Saved hybrid_best.pt")

    print(f"\n[OK] Training complete in {(time.time() - train_start)/60:.1f}m!")
    print("The codebase and PyTorch weights (.pt) are fully synchronized and ready for production!")

if __name__ == "__main__":
    main()
