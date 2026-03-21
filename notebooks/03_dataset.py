"""
PyTorch Dataset & DataLoader wrappers
Plugs directly into training loop (04_train.py)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

SAVE_DIR = Path("data/processed")


class CycloneDataset(Dataset):
    """
    Each sample:
        X  : (lookback, n_features)  float32  — track + atmospheric history
        y  : (n_labels,)             float32  — multi-task targets
    Label index map (LABEL_COLS order from 01_data_loader.py):
        0–1   lat_24h, lon_24h
        2–3   lat_48h, lon_48h
        4–5   lat_72h, lon_72h
        6–8   wind_24h, wind_48h, wind_72h
        9     SS_cat
        10    RI_label   (binary)
        11    landfall_72h (binary)
    """
    def __init__(self, split: str = "train"):
        X_path = SAVE_DIR / f"{split}_X.npy"
        y_path = SAVE_DIR / f"{split}_y.npy"
        if not X_path.exists():
            raise FileNotFoundError(f"Run 01_data_loader.py first. Missing: {X_path}")
        self.X = torch.from_numpy(np.load(X_path))
        self.y = torch.from_numpy(np.load(y_path))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(batch_size: int = 64, num_workers: int = 0) -> dict:
    """Return train / val / test DataLoaders."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = CycloneDataset(split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
        print(f"  {split:5s} loader: {len(ds):,} samples  |  {len(loaders[split])} batches")
    return loaders


# ── Label helper constants (use these in train.py) ───────────────────────────

TRACK_IDX    = slice(0, 6)   # lat/lon at 24h, 48h, 72h
WIND_IDX     = slice(6, 9)   # wind at 24h, 48h, 72h
SS_IDX       = 9             # Saffir-Simpson category
RI_IDX       = 10            # Rapid Intensification binary
LANDFALL_IDX = 11            # Landfall within 72h binary

N_FEATURES   = 17            # must match FEATURE_COLS in 01_data_loader.py
N_LABELS     = 12
LOOKBACK     = 8             # 48h history


if __name__ == "__main__":
    print("Checking dataset integrity …")
    loaders = get_loaders(batch_size=32)
    Xb, yb = next(iter(loaders["train"]))
    print(f"\n  Batch X shape : {tuple(Xb.shape)}")
    print(f"  Batch y shape : {tuple(yb.shape)}")
    print(f"  RI label rate : {yb[:, RI_IDX].mean():.2%}")
    print("\n[✓] Dataset ready for training.")