"""
ERA5 Data Downloader — NI Basin Cyclone Prediction
Downloads atmospheric variables for each storm in the dataset.

Run from project root:
    python notebooks/05_era5_download.py

Output:
    data/era5/          <- one .nc file per year
    data/era5_patches/  <- per-storm numpy patches ready for training
"""

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import time

# ─── CONFIG ──────────────────────────────────────────────────────────────────

ERA5_DIR    = Path("data/era5")
PATCH_DIR   = Path("data/era5_patches")
META_DIR    = Path("data/processed")

ERA5_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)

# Variables to download — chosen for cyclone intensity prediction
VARIABLES = [
    "sea_surface_temperature",           # SST — key for intensification
    "10m_u_component_of_wind",           # surface wind u
    "10m_v_component_of_wind",           # surface wind v
    "mean_sea_level_pressure",           # MSLP
    "specific_humidity",                 # moisture
    "temperature",                       # upper-level temp (200 hPa)
    "u_component_of_wind",               # upper-level wind shear
    "v_component_of_wind",               # upper-level wind shear
]

# Pressure levels for upper-atmosphere variables
PRESSURE_LEVELS = ["200", "500", "850"]

# Patch size around storm center in degrees
PATCH_DEG = 10.0    # ± 10° = 20° × 20° box around storm

# Downsample to 1° resolution to keep file sizes manageable on CPU
GRID_RES  = "1.0/1.0"

# ─── STEP 1: LOAD STORM METADATA ─────────────────────────────────────────────

def load_storm_meta() -> pd.DataFrame:
    """Load train+val+test metadata to know which storms/times to download."""
    dfs = []
    for split in ("train", "val", "test"):
        path = META_DIR / f"{split}_meta.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["ISO_TIME"])
            df["split"] = split
            dfs.append(df)
    meta = pd.concat(dfs, ignore_index=True)
    meta = meta.sort_values("ISO_TIME").drop_duplicates(subset=["SID", "ISO_TIME"])
    print(f"[✓] Loaded {len(meta):,} storm obs | "
          f"{meta['SID'].nunique()} storms | "
          f"years {meta['ISO_TIME'].dt.year.min()}–{meta['ISO_TIME'].dt.year.max()}")
    return meta


# ─── STEP 2: DOWNLOAD ERA5 BY YEAR ───────────────────────────────────────────

def download_era5_year(year: int, months: list, client: cdsapi.Client) -> Path:
    """Download ERA5 for a given year over the NI basin domain."""
    out_path = ERA5_DIR / f"era5_{year}.nc"
    if out_path.exists():
        print(f"  [skip] era5_{year}.nc already exists")
        return out_path

    print(f"  [↓] Downloading ERA5 {year} ...")

    # Single-level variables (surface)
    sl_path = ERA5_DIR / f"era5_{year}_sl.nc"
    if not sl_path.exists():
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "sea_surface_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",
                    "specific_humidity",
                ],
                "year":  str(year),
                "month": [f"{m:02d}" for m in months],
                "day":   [f"{d:02d}" for d in range(1, 29)],
                "time":  ["00:00", "06:00", "12:00", "18:00"],
                "area":  [35, 40, -5, 110],   # N, W, S, E — NI basin
                "grid":  GRID_RES,
                "format": "netcdf",
            },
            str(sl_path),
        )
        print(f"    [✓] Surface vars saved ({sl_path.stat().st_size/1e6:.1f} MB)")

    # Pressure-level variables (upper atmosphere)
    pl_path = ERA5_DIR / f"era5_{year}_pl.nc"
    if not pl_path.exists():
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type":   "reanalysis",
                "variable":       ["temperature", "u_component_of_wind",
                                   "v_component_of_wind"],
                "pressure_level": PRESSURE_LEVELS,
                "year":  str(year),
                "month": [f"{m:02d}" for m in months],
                "day":   [f"{d:02d}" for d in range(1, 29)],
                "time":  ["00:00", "06:00", "12:00", "18:00"],
                "area":  [35, 40, -5, 110],
                "grid":  GRID_RES,
                "format": "netcdf",
            },
            str(pl_path),
        )
        print(f"    [✓] Pressure vars saved ({pl_path.stat().st_size/1e6:.1f} MB)")

    return sl_path   # caller uses both


# ─── STEP 3: EXTRACT STORM PATCHES ───────────────────────────────────────────

def extract_patches(meta: pd.DataFrame, year: int) -> int:
    """
    For each storm observation in this year, extract a spatial patch
    from ERA5 centered on the storm and save as .npy.
    Returns number of patches saved.
    """
    sl_path = ERA5_DIR / f"era5_{year}_sl.nc"
    pl_path = ERA5_DIR / f"era5_{year}_pl.nc"

    if not sl_path.exists() or not pl_path.exists():
        print(f"  [skip] ERA5 files missing for {year}")
        return 0

    year_meta = meta[meta["ISO_TIME"].dt.year == year].copy()
    if len(year_meta) == 0:
        return 0

    # Load ERA5 for this year
    try:
        ds_sl = xr.open_dataset(sl_path)
        ds_pl = xr.open_dataset(pl_path)
    except Exception as e:
        print(f"  [!] Could not open ERA5 for {year}: {e}")
        return 0

    # Load IBTrACS to get lat/lon per obs
    ibtracs = pd.read_csv(
        "data/raw/ibtracs_NI.csv", skiprows=[1], low_memory=False
    )
    ibtracs["ISO_TIME"] = pd.to_datetime(ibtracs["ISO_TIME"], errors="coerce")
    ibtracs["LAT"] = pd.to_numeric(ibtracs["LAT"], errors="coerce")
    ibtracs["LON"] = pd.to_numeric(ibtracs["LON"], errors="coerce")
    ibtracs = ibtracs.dropna(subset=["LAT", "LON", "ISO_TIME"])

    # Merge lat/lon into metadata
    year_meta = year_meta.merge(
        ibtracs[["SID", "ISO_TIME", "LAT", "LON"]],
        on=["SID", "ISO_TIME"], how="left"
    ).dropna(subset=["LAT", "LON"])

    saved = 0
    for _, row in year_meta.iterrows():
        patch_path = PATCH_DIR / f"{row['SID']}_{row['ISO_TIME'].strftime('%Y%m%d%H')}.npy"
        if patch_path.exists():
            saved += 1
            continue

        try:
            patch = _extract_one_patch(ds_sl, ds_pl, row["LAT"], row["LON"],
                                       row["ISO_TIME"])
            if patch is not None:
                np.save(patch_path, patch.astype(np.float32))
                saved += 1
        except Exception:
            pass   # skip individual failures silently

    ds_sl.close()
    ds_pl.close()
    return saved


def _extract_one_patch(ds_sl, ds_pl, lat, lon, time):
    """
    Extract a (C, H, W) patch centered on (lat, lon) at given time.
    C = number of ERA5 channels
    H = W = patch size in grid cells (20deg / 1deg = 20 cells)
    Handles ERA5 new API which uses 'valid_time' and 'pressure_level'.
    """
    # Snap time to nearest 6h ERA5 timestamp
    t = pd.Timestamp(time).round("6h")

    lat_min = lat - PATCH_DEG
    lat_max = lat + PATCH_DEG
    lon_min = lon - PATCH_DEG
    lon_max = lon + PATCH_DEG

    # ERA5 new API uses 'valid_time' instead of 'time'
    sl_time_dim = "valid_time" if "valid_time" in ds_sl.dims else "time"
    pl_time_dim = "valid_time" if "valid_time" in ds_pl.dims else "time"

    # ERA5 new API uses 'pressure_level' instead of 'level'
    pl_lev_dim  = "pressure_level" if "pressure_level" in ds_pl.dims else "level"

    try:
        sl = ds_sl.sel(
            {sl_time_dim: t}, method="nearest"
        ).sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )
        pl = ds_pl.sel(
            {pl_time_dim: t}, method="nearest"
        ).sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        ).sel({pl_lev_dim: 500}, method="nearest")
    except Exception:
        return None

    channels = []

    # Surface channels — only use what exists (q may be missing)
    for var in ["sst", "u10", "v10", "msl", "q"]:
        if var in sl:
            arr = sl[var].values
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                arr = _resize_patch(np.nan_to_num(arr, nan=0.0), 20, 20)
                channels.append(arr)

    # Pressure level channels (500 hPa)
    for var in ["t", "u", "v"]:
        if var in pl:
            arr = pl[var].values
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                arr = _resize_patch(np.nan_to_num(arr, nan=0.0), 20, 20)
                channels.append(arr)

    if len(channels) == 0:
        return None

    patch = np.stack(channels, axis=0)   # (C, 20, 20)
    return patch


def _resize_patch(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Simple nearest-neighbour resize to (h, w)."""
    from scipy.ndimage import zoom
    if arr.shape == (h, w):
        return arr
    zy = h / arr.shape[0]
    zx = w / arr.shape[1]
    return zoom(arr, (zy, zx), order=1)


# ─── STEP 4: BUILD ALIGNED DATASET ───────────────────────────────────────────

def build_era5_array(meta: pd.DataFrame) -> tuple:
    """
    For every row in meta, load the corresponding ERA5 patch by filename.
    Returns (era5_array, valid_mask).
    era5_array shape: (N, C, 20, 20)
    valid_mask shape: (N,)
    """
    N = len(meta)
    C = 8
    era5  = np.full((N, C, 20, 20), np.nan, dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    meta = meta.reset_index(drop=True)

    for i, row in meta.iterrows():
        sid       = row["SID"]
        iso_time  = pd.Timestamp(row["ISO_TIME"])
        fname     = f"{sid}_{iso_time.strftime('%Y%m%d%H')}.npy"
        patch_path = PATCH_DIR / fname

        if patch_path.exists():
            try:
                patch = np.load(patch_path)
                c = min(patch.shape[0], C)
                era5[i, :c] = patch[:c]
                valid[i]    = True
            except Exception:
                pass

    coverage = valid.mean()
    print(f"[✓] ERA5 patch coverage: {valid.sum():,}/{N:,} ({coverage:.1%})")
    return era5, valid


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ERA5 DOWNLOADER — NI BASIN")
    print("=" * 55 + "\n")

    meta   = load_storm_meta()
    client = cdsapi.Client()

    # Get unique years and their active months
    meta["year"]  = meta["ISO_TIME"].dt.year
    meta["month"] = meta["ISO_TIME"].dt.month

    years = sorted(y for y in meta["year"].unique() if int(y) >= 1980)
    print(f"\n[1/3] Downloading ERA5 for {len(years)} years: {years[0]}–{years[-1]}")
    print("      (This takes 20–40 minutes — grab a coffee)\n")

    for year in years:
        months = sorted(meta[meta["year"] == year]["month"].unique().tolist())
        try:
            download_era5_year(int(year), months, client)
        except Exception as e:
            print(f"  [!] Failed year {year}: {e}")
        time.sleep(2)   # be polite to CDS API

    print(f"\n[2/3] Extracting storm patches ...")
    total_patches = 0
    for year in years:
        n = extract_patches(meta, int(year))
        print(f"  {year}: {n} patches saved")
        total_patches += n
    print(f"\n  Total patches: {total_patches:,}")

    print(f"\n[3/3] Verifying patch coverage ...")

    # Load all meta combined
    all_meta = meta.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    era5_array, valid_mask = build_era5_array(all_meta)

    # Save combined ERA5 array aligned with processed splits
    np.save(ERA5_DIR / "era5_patches_all.npy",  era5_array)
    np.save(ERA5_DIR / "era5_valid_mask.npy",   valid_mask)

    print(f"\n[✓] ERA5 data ready.")
    print(f"    era5_patches_all.npy : {era5_array.shape}")
    print(f"    era5_valid_mask.npy  : {valid_mask.sum():,} valid patches")
    print(f"\n    Next: python notebooks/06_train_hybrid.py\n")