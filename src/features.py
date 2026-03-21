"""
Feature / label column definitions, PyTorch Dataset classes, and DataLoader factories.
Extracted from notebooks/01_data_loader.py and notebooks/03_dataset.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ─── Feature and label column definitions ────────────────────────────────────

FEATURE_COLS = [
    "LAT", "LON", "WIND", "WMO_PRES",
    "dLAT", "dLON", "dLAT_2", "dLON_2",
    "dWIND", "dPRES", "spd_kmh",
    "dist2land", "lat_abs",
    "month_sin", "month_cos", "jday_sin", "jday_cos",
]

LABEL_COLS = [
    "lat_24h", "lon_24h",
    "lat_48h", "lon_48h",
    "lat_72h", "lon_72h",
    "wind_24h", "wind_48h", "wind_72h",
    "SS_cat", "RI_label", "landfall_72h",
]

# ─── Label index helpers ─────────────────────────────────────────────────────

TRACK_IDX    = slice(0, 6)   # lat/lon at 24h, 48h, 72h
WIND_IDX     = slice(6, 9)   # wind at 24h, 48h, 72h
SS_IDX       = 9             # Saffir-Simpson category
RI_IDX       = 10            # Rapid Intensification binary
LANDFALL_IDX = 11            # Landfall within 72h binary

N_FEATURES = len(FEATURE_COLS)  # 17
N_LABELS   = len(LABEL_COLS)    # 12
LOOKBACK   = 8                  # 48h history (6-hourly steps)

# Saffir-Simpson bins for classification
SS_BINS   = [-np.inf, 33, 63, 82, 95, 112, np.inf]
SS_LABELS = [0, 1, 2, 3, 4, 5]


# ─── PyTorch Datasets ────────────────────────────────────────────────────────

class CycloneDataset(Dataset):
    """
    Basic dataset for LSTM-only models.
    Each sample: X (lookback, n_features), y (n_labels,).
    """

    def __init__(self, split: str, data_dir: str = "data/processed"):
        dp = Path(data_dir)
        X = np.load(dp / f"{split}_X.npy")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(np.load(dp / f"{split}_y.npy")).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HybridDataset(Dataset):
    """
    Dataset for hybrid LSTM + ConvLSTM models.
    Each sample: track_seq (lookback, 17), era5_seq (4, 8, 20, 20), y (12,).
    """

    ERA5_CHANNELS = 8
    ERA5_SEQ_LEN  = 4

    def __init__(self, split: str, data_dir: str = "data/processed",
                 era5_dir: str = "data/era5"):
        dp = Path(data_dir)
        ep = Path(era5_dir)

        # Track features
        X = np.load(dp / f"{split}_X.npy")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(np.load(dp / f"{split}_y.npy")).float()

        # ERA5 patches
        self.era5 = self._load_era5(dp, ep, split)

    def _load_era5(self, dp, ep, split):
        era5_path = ep / "era5_patches_all.npy"
        meta_path = dp / f"{split}_meta.csv"

        if not era5_path.exists():
            return torch.zeros(len(self.X), self.ERA5_SEQ_LEN,
                               self.ERA5_CHANNELS, 20, 20)
        try:
            import pandas as pd
            all_meta   = self._load_all_meta(dp)
            split_meta = pd.read_csv(meta_path, parse_dates=["ISO_TIME"])
            era5_all   = np.nan_to_num(np.load(era5_path), nan=0.0)

            merged = split_meta.merge(
                all_meta.reset_index().rename(columns={"index": "era5_idx"}),
                on=["SID", "ISO_TIME"], how="left"
            )
            idx = merged["era5_idx"].fillna(-1).astype(int).values

            C = self.ERA5_CHANNELS
            era5_seq = np.zeros((len(self.X), self.ERA5_SEQ_LEN, C, 20, 20),
                                dtype=np.float32)
            for i, ei in enumerate(idx):
                if ei < 0:
                    continue
                for t, offset in enumerate(range(-3, 1)):
                    j = ei + offset
                    if 0 <= j < len(era5_all):
                        p = era5_all[j]
                        if p.shape[0] >= C:
                            era5_seq[i, t] = p[:C]

            return torch.from_numpy(era5_seq).float()
        except Exception:
            return torch.zeros(len(self.X), self.ERA5_SEQ_LEN,
                               self.ERA5_CHANNELS, 20, 20)

    @staticmethod
    def _load_all_meta(dp):
        import pandas as pd
        dfs = []
        for s in ("train", "val", "test"):
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


# ─── DataLoader factories ────────────────────────────────────────────────────

def get_loaders(batch_size: int = 64, num_workers: int = 0,
                data_dir: str = "data/processed",
                dataset_cls=CycloneDataset, **ds_kwargs) -> dict:
    """Return train / val / test DataLoaders."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = dataset_cls(split, data_dir=data_dir, **ds_kwargs)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
        print(f"  {split:5s}: {len(ds):,} samples | {len(loaders[split])} batches")
    return loaders
