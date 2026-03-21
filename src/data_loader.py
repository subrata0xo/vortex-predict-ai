"""
IBTrACS data loading, cleaning, feature engineering, and processing pipeline.
Extracted from notebooks/01_data_loader.py.
"""

import numpy as np
import pandas as pd
import requests
from pathlib import Path

from src.features import FEATURE_COLS, LABEL_COLS, SS_BINS, SS_LABELS

# ─── CONFIG ──────────────────────────────────────────────────────────────────

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)


def download_ibtracs(data_dir: str = "data/raw",
                     force: bool = False) -> Path:
    """Download IBTrACS NI basin CSV if not already cached."""
    dest_dir = Path(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "ibtracs_NI.csv"

    if dest.exists() and not force:
        print(f"[✓] Found cached file: {dest}")
        return dest

    print("[↓] Downloading IBTrACS NI basin ...")
    r = requests.get(IBTRACS_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[✓] Saved to {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load IBTrACS CSV, merge wind columns, clean, and filter."""
    df = pd.read_csv(path, skiprows=[1], low_memory=False)

    # Merge wind from agency columns (priority order)
    wind_cols = ["USA_WIND", "NEWDELHI_WIND", "WMO_WIND", "TOKYO_WIND"]
    for c in wind_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["WIND"] = np.nan
    for c in wind_cols:
        if c in df.columns:
            df["WIND"] = df["WIND"].fillna(df[c])

    # Numeric columns
    num_cols = ["LAT", "LON", "WMO_PRES", "DIST2LAND", "STORM_SPEED", "STORM_DIR"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON", "ISO_TIME"])

    # Keep storms with ≥8 obs for lookback window
    counts = df.groupby("SID")["LAT"].count()
    valid = counts[counts >= 8].index
    df = df[df["SID"].isin(valid)].copy()
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    print(f"[✓] Cleaned → {df['SID'].nunique()} storms | {len(df):,} observations")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add motion vectors, speed, deltas, temporal features, and labels."""
    out = []

    for sid, grp in df.groupby("SID"):
        grp = grp.reset_index(drop=True)

        # Motion vectors
        grp["dLAT"]   = grp["LAT"].diff()
        grp["dLON"]   = grp["LON"].diff()
        grp["dLAT_2"] = grp["dLAT"].diff()
        grp["dLON_2"] = grp["dLON"].diff()

        # Translation speed
        grp["spd_kmh"] = _haversine_speed(grp)

        # Interpolate wind/pressure
        grp["WIND"] = grp["WIND"].interpolate(method="linear", limit=4)
        grp["WIND"] = grp["WIND"].ffill().bfill()
        if "WMO_PRES" in grp.columns:
            grp["WMO_PRES"] = grp["WMO_PRES"].interpolate(method="linear", limit=4)
            grp["WMO_PRES"] = grp["WMO_PRES"].ffill().bfill()

        grp["dWIND"] = grp["WIND"].diff()
        grp["dPRES"] = grp["WMO_PRES"].diff() if "WMO_PRES" in grp.columns else 0.0

        # RI label
        grp["wind_24h_future"] = grp["WIND"].shift(-4)
        grp["RI_label"] = ((grp["wind_24h_future"] - grp["WIND"]) >= 35).astype(float)

        # Saffir-Simpson category
        grp["SS_cat"] = pd.cut(
            grp["WIND"], bins=SS_BINS, labels=SS_LABELS, right=True
        ).astype(float)

        # Temporal features
        grp["month"]      = grp["ISO_TIME"].dt.month
        grp["julian_day"] = grp["ISO_TIME"].dt.day_of_year
        grp["month_sin"]  = np.sin(2 * np.pi * grp["month"] / 12)
        grp["month_cos"]  = np.cos(2 * np.pi * grp["month"] / 12)
        grp["jday_sin"]   = np.sin(2 * np.pi * grp["julian_day"] / 365)
        grp["jday_cos"]   = np.cos(2 * np.pi * grp["julian_day"] / 365)

        # Geographic features
        grp["lat_abs"]   = grp["LAT"].abs()
        grp["dist2land"] = grp["DIST2LAND"].fillna(
            grp["DIST2LAND"].median() if grp["DIST2LAND"].notna().any() else 500.0
        )

        # Future track/wind labels
        for h, steps in [(24, 4), (48, 8), (72, 12)]:
            grp[f"lat_{h}h"]  = grp["LAT"].shift(-steps)
            grp[f"lon_{h}h"]  = grp["LON"].shift(-steps)
            grp[f"wind_{h}h"] = grp["WIND"].shift(-steps)

        # Landfall within 72h
        grp["landfall_72h"] = (
            grp["DIST2LAND"].shift(-12).fillna(999) == 0
        ).astype(float)

        out.append(grp)

    result = pd.concat(out, ignore_index=True)
    result = result.dropna(subset=["dLAT", "dLON"])
    result = result.dropna(subset=["lat_24h", "lon_24h"])

    print(f"[✓] Features engineered → {len(result):,} usable rows")
    return result


def make_sequences(df: pd.DataFrame, lookback: int = 8) -> dict:
    """Create sliding-window sequences for training."""
    X_list, y_list, meta_list = [], [], []

    for sid, grp in df.groupby("SID"):
        grp = grp.reset_index(drop=True)
        feat = grp[FEATURE_COLS].ffill().bfill().values.astype(np.float32)
        lab  = grp[LABEL_COLS].values.astype(np.float32)

        for i in range(lookback, len(grp)):
            X_list.append(feat[i - lookback: i])
            y_list.append(lab[i])
            meta_list.append({
                "SID":      sid,
                "NAME":     grp.loc[i, "NAME"],
                "ISO_TIME": grp.loc[i, "ISO_TIME"],
                "SEASON":   grp.loc[i, "SEASON"],
            })

    X    = np.stack(X_list).astype(np.float32)
    y    = np.stack(y_list).astype(np.float32)
    meta = pd.DataFrame(meta_list)

    print(f"[✓] Sequences → X: {X.shape} | y: {y.shape}")
    return {"X": X, "y": y, "meta": meta}


def temporal_split(data: dict, val_year: int = 2018,
                   test_year: int = 2020) -> dict:
    """Split data by year — train / val / test."""
    years = data["meta"]["ISO_TIME"].dt.year
    split = {}
    for name, mask in [
        ("train", years < val_year),
        ("val",   (years >= val_year) & (years < test_year)),
        ("test",  years >= test_year),
    ]:
        split[name] = {
            "X":    data["X"][mask],
            "y":    data["y"][mask],
            "meta": data["meta"][mask].reset_index(drop=True),
        }
        print(f"  {name:5s}: {mask.sum():5,} samples")
    return split


def normalize(split: dict) -> tuple:
    """Normalize features using train-set statistics."""
    X_train = split["train"]["X"]
    mu  = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(X_train,  axis=(0, 1), keepdims=True) + 1e-8

    scaler = {"mean": mu, "std": std,
              "feature_cols": FEATURE_COLS, "label_cols": LABEL_COLS}

    for key in split:
        split[key]["X"] = ((split[key]["X"] - mu) / std).astype(np.float32)

    print("[✓] Normalisation applied (fit on train only)")
    return split, scaler


def save_artifacts(split: dict, scaler: dict,
                   save_dir: str = "data/processed") -> None:
    """Save processed data to disk."""
    sd = Path(save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    np.save(sd / "scaler_mean.npy", scaler["mean"])
    np.save(sd / "scaler_std.npy",  scaler["std"])
    for key in split:
        np.save(sd / f"{key}_X.npy", split[key]["X"])
        np.save(sd / f"{key}_y.npy", split[key]["y"])
        split[key]["meta"].to_csv(sd / f"{key}_meta.csv", index=False)
    print(f"[✓] Saved all artifacts to {sd}/")


def _haversine_speed(grp: pd.DataFrame) -> pd.Series:
    """Translation speed in km/h from consecutive lat/lon."""
    R = 6371.0
    lat1 = np.radians(grp["LAT"].shift(1))
    lat2 = np.radians(grp["LAT"])
    dlt  = np.radians(grp["LAT"] - grp["LAT"].shift(1))
    dln  = np.radians(grp["LON"] - grp["LON"].shift(1))
    a    = np.sin(dlt / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dln / 2)**2
    dist = 2 * R * np.arcsin(np.sqrt(a.clip(0, 1)))
    return dist / 6.0
