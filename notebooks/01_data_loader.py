"""
IBTrACS North Indian Basin — Data Loader (v2)
Fixed: wind column merging, cleaner label strategy

Run from project root:
    python notebooks/01_data_loader.py
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/raw")
SAVE_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)

SS_BINS   = [-np.inf, 33, 63, 82, 95, 112, np.inf]
SS_LABELS = [0, 1, 2, 3, 4, 5]

# ─── STEP 1: DOWNLOAD ────────────────────────────────────────────────────────

def download_ibtracs(force: bool = False) -> Path:
    dest = DATA_DIR / "ibtracs_NI.csv"
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


# ─── STEP 2: LOAD & CLEAN ────────────────────────────────────────────────────

def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1], low_memory=False)

    # ── Merge wind from all available agency columns ──────────────────────────
    # Priority: USA > NEWDELHI > WMO > TOKYO
    wind_cols = ["USA_WIND", "NEWDELHI_WIND", "WMO_WIND", "TOKYO_WIND"]
    for c in wind_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Take first non-NaN across agency columns
    df["WIND"] = np.nan
    for c in wind_cols:
        if c in df.columns:
            df["WIND"] = df["WIND"].fillna(df[c])

    print(f"  Wind coverage: {df['WIND'].notna().sum():,} / {len(df):,} rows "
          f"({df['WIND'].notna().mean():.1%})")

    # ── Other numeric columns ─────────────────────────────────────────────────
    num_cols = ["LAT", "LON", "WMO_PRES", "DIST2LAND", "STORM_SPEED", "STORM_DIR"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

    # Drop rows with no position or time
    df = df.dropna(subset=["LAT", "LON", "ISO_TIME"])

    # Keep storms with at least 8 obs (~48h) so lookback window fits
    counts = df.groupby("SID")["LAT"].count()
    valid  = counts[counts >= 8].index
    df     = df[df["SID"].isin(valid)].copy()

    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    print(f"[✓] Cleaned → {df['SID'].nunique()} storms | {len(df):,} observations")
    print(f"    Date range: {df['ISO_TIME'].min().date()} → {df['ISO_TIME'].max().date()}")
    return df


# ─── STEP 3: FEATURE ENGINEERING ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    for sid, grp in df.groupby("SID"):
        grp = grp.reset_index(drop=True)

        # ── Motion vectors ────────────────────────────────────────────────────
        grp["dLAT"]   = grp["LAT"].diff()
        grp["dLON"]   = grp["LON"].diff()
        grp["dLAT_2"] = grp["dLAT"].diff()
        grp["dLON_2"] = grp["dLON"].diff()

        # ── Translation speed (km/h) ──────────────────────────────────────────
        grp["spd_kmh"] = _haversine_speed(grp)

        # ── Interpolate wind within storm before computing deltas ─────────────
        # Linear interpolation fills gaps between known obs; limit=4 steps max
        grp["WIND"] = grp["WIND"].interpolate(method="linear", limit=4)
        grp["WIND"] = grp["WIND"].ffill().bfill()   # fill any remaining edges

        # ── Pressure — interpolate similarly ─────────────────────────────────
        if "WMO_PRES" in grp.columns:
            grp["WMO_PRES"] = grp["WMO_PRES"].interpolate(method="linear", limit=4)
            grp["WMO_PRES"] = grp["WMO_PRES"].ffill().bfill()

        grp["dWIND"] = grp["WIND"].diff()
        grp["dPRES"] = grp["WMO_PRES"].diff() if "WMO_PRES" in grp.columns else 0.0

        # ── Rapid Intensification label (≥35 kt in next 24h = 4 steps) ───────
        grp["wind_24h_future"] = grp["WIND"].shift(-4)
        grp["RI_label"] = (
            (grp["wind_24h_future"] - grp["WIND"]) >= 35
        ).astype(float)

        # ── Saffir-Simpson category ───────────────────────────────────────────
        grp["SS_cat"] = pd.cut(
            grp["WIND"], bins=SS_BINS, labels=SS_LABELS, right=True
        ).astype(float)

        # ── Temporal features (cyclical) ──────────────────────────────────────
        grp["month"]      = grp["ISO_TIME"].dt.month
        grp["julian_day"] = grp["ISO_TIME"].dt.day_of_year
        grp["month_sin"]  = np.sin(2 * np.pi * grp["month"] / 12)
        grp["month_cos"]  = np.cos(2 * np.pi * grp["month"] / 12)
        grp["jday_sin"]   = np.sin(2 * np.pi * grp["julian_day"] / 365)
        grp["jday_cos"]   = np.cos(2 * np.pi * grp["julian_day"] / 365)

        # ── Geographic features ───────────────────────────────────────────────
        grp["lat_abs"]   = grp["LAT"].abs()
        grp["dist2land"] = grp["DIST2LAND"].fillna(
            grp["DIST2LAND"].median() if grp["DIST2LAND"].notna().any() else 500.0
        )

        # ── Future track labels (regression targets) ──────────────────────────
        for h, steps in [(24, 4), (48, 8), (72, 12)]:
            grp[f"lat_{h}h"]  = grp["LAT"].shift(-steps)
            grp[f"lon_{h}h"]  = grp["LON"].shift(-steps)
            grp[f"wind_{h}h"] = grp["WIND"].shift(-steps)

        # ── Landfall within 72h ───────────────────────────────────────────────
        grp["landfall_72h"] = (
            grp["DIST2LAND"].shift(-12).fillna(999) == 0
        ).astype(float)

        out.append(grp)

    result = pd.concat(out, ignore_index=True)

    # Drop rows missing motion vectors (first row of each storm)
    result = result.dropna(subset=["dLAT", "dLON"])

    # Only require 24h track labels — allow 48h/72h/wind to be NaN
    # (the training loop already masks NaN labels per task)
    result = result.dropna(subset=["lat_24h", "lon_24h"])

    ri_rate = result["RI_label"].mean()
    print(f"[✓] Features engineered → {len(result):,} usable rows")
    print(f"    RI rate: {ri_rate:.2%}")
    print(f"    Wind NaN after interpolation: "
          f"{result['WIND'].isna().sum()} / {len(result)}")
    return result


def _haversine_speed(grp: pd.DataFrame) -> pd.Series:
    R    = 6371.0
    lat1 = np.radians(grp["LAT"].shift(1))
    lat2 = np.radians(grp["LAT"])
    dlt  = np.radians(grp["LAT"] - grp["LAT"].shift(1))
    dln  = np.radians(grp["LON"] - grp["LON"].shift(1))
    a    = np.sin(dlt / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dln / 2)**2
    dist = 2 * R * np.arcsin(np.sqrt(a.clip(0, 1)))
    return dist / 6.0


# ─── STEP 4: FEATURE & LABEL COLUMNS ─────────────────────────────────────────

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


# ─── STEP 5: SLIDING WINDOW SEQUENCES ────────────────────────────────────────

def make_sequences(df: pd.DataFrame, lookback: int = 8) -> dict:
    X_list, y_list, meta_list = [], [], []

    for sid, grp in df.groupby("SID"):
        grp  = grp.reset_index(drop=True)

        # Fill feature NaNs within storm
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

    # Report NaN coverage in labels
    print(f"[✓] Sequences → X: {X.shape} | y: {y.shape}")
    print("    Label NaN rates:")
    for i, col in enumerate(LABEL_COLS):
        rate = np.isnan(y[:, i]).mean()
        bar  = "█" * int((1 - rate) * 20)
        print(f"      {col:15s}: {rate:.1%} NaN  |{bar}|")

    return {"X": X, "y": y, "meta": meta}


# ─── STEP 6: TRAIN / VAL / TEST SPLIT ────────────────────────────────────────

def temporal_split(data: dict, val_year: int = 2018, test_year: int = 2020) -> dict:
    years = data["meta"]["ISO_TIME"].dt.year
    split = {}
    for name, mask in [
        ("train", years <  val_year),
        ("val",   (years >= val_year) & (years < test_year)),
        ("test",  years >= test_year),
    ]:
        split[name] = {
            "X":    data["X"][mask],
            "y":    data["y"][mask],
            "meta": data["meta"][mask].reset_index(drop=True),
        }
        ri = data["y"][mask][:, LABEL_COLS.index("RI_label")]
        ri_rate = np.nanmean(ri)
        print(f"  {name:5s}: {mask.sum():5,} samples | RI rate: {ri_rate:.2%}")
    return split


# ─── STEP 7: NORMALISE ───────────────────────────────────────────────────────

def normalize(split: dict) -> tuple:
    X_train = split["train"]["X"]
    mu  = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(X_train,  axis=(0, 1), keepdims=True) + 1e-8

    scaler = {"mean": mu, "std": std,
              "feature_cols": FEATURE_COLS, "label_cols": LABEL_COLS}

    for key in split:
        split[key]["X"] = ((split[key]["X"] - mu) / std).astype(np.float32)

    print("[✓] Normalisation applied (fit on train only)")
    return split, scaler


# ─── STEP 8: SAVE ─────────────────────────────────────────────────────────────

def save_artifacts(split: dict, scaler: dict) -> None:
    np.save(SAVE_DIR / "scaler_mean.npy", scaler["mean"])
    np.save(SAVE_DIR / "scaler_std.npy",  scaler["std"])
    for key in split:
        np.save(SAVE_DIR / f"{key}_X.npy", split[key]["X"])
        np.save(SAVE_DIR / f"{key}_y.npy", split[key]["y"])
        split[key]["meta"].to_csv(SAVE_DIR / f"{key}_meta.csv", index=False)
    print(f"[✓] Saved all artifacts to {SAVE_DIR}/")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  IBTRACS NI BASIN — DATA PIPELINE v2")
    print("=" * 55 + "\n")

    path    = download_ibtracs()
    df_raw  = load_and_clean(path)
    df_eng  = engineer_features(df_raw)
    data    = make_sequences(df_eng, lookback=8)

    print("\nTemporal split:")
    split   = temporal_split(data)

    print("\nNormalising:")
    split, scaler = normalize(split)

    save_artifacts(split, scaler)
    print("\n[✓] Pipeline complete — run 04_train.py next.\n")