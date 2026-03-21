"""
IBTrACS NI Basin — Exploratory Data Analysis
Run this AFTER 01_data_loader.py to understand your data before modelling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

DATA_DIR = Path("data/raw")
SAVE_DIR = Path("data/eda_plots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SS_BINS   = [-np.inf, 33, 63, 82, 95, 112, np.inf]
SS_LABELS = [0, 1, 2, 3, 4, 5]
SS_NAMES  = {0:"TD/TS", 1:"Cat 1", 2:"Cat 2", 3:"Cat 3", 4:"Cat 4", 5:"Cat 5"}
SS_COLORS = {0:"#94a3b8", 1:"#fde68a", 2:"#fb923c", 3:"#ef4444", 4:"#b91c1c", 5:"#7f1d1d"}


def load_clean() -> pd.DataFrame:
    path = DATA_DIR / "ibtracs_NI.csv"
    if not path.exists():
        raise FileNotFoundError("Run 01_data_loader.py first to download data.")
    df = pd.read_csv(path, skiprows=[1], low_memory=False)
    num = ["LAT","LON","WMO_WIND","WMO_PRES","DIST2LAND"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df["SEASON"]   = pd.to_numeric(df["SEASON"], errors="coerce")
    df = df.dropna(subset=["LAT","LON","ISO_TIME"])
    df["month"] = df["ISO_TIME"].dt.month
    df["SS_cat"] = pd.cut(df["WMO_WIND"], bins=SS_BINS,
                          labels=SS_LABELS, right=True).astype(float)
    return df


def plot_storm_tracks(df: pd.DataFrame):
    """Map of all NI basin storm tracks coloured by intensity."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0a1628")
    ax.set_facecolor("#0a1628")

    for sid, grp in df.groupby("SID"):
        for i in range(len(grp) - 1):
            r = grp.iloc[i]
            cat = int(r["SS_cat"]) if not np.isnan(r["SS_cat"]) else 0
            ax.plot([grp.iloc[i]["LON"], grp.iloc[i+1]["LON"]],
                    [grp.iloc[i]["LAT"], grp.iloc[i+1]["LAT"]],
                    color=SS_COLORS[cat], lw=0.6, alpha=0.5)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=SS_COLORS[k], lw=2, label=SS_NAMES[k])
               for k in SS_NAMES]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              facecolor="#1e3a5f", labelcolor="white", framealpha=0.8)

    ax.set_xlim(40, 110); ax.set_ylim(-5, 35)
    ax.set_xlabel("Longitude", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Latitude",  color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    ax.set_title("North Indian Basin — Storm Tracks 1980–present\n(colour = Saffir-Simpson category)",
                 color="white", fontsize=13, pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "01_storm_tracks.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] Saved 01_storm_tracks.png")


def plot_seasonal_distribution(df: pd.DataFrame):
    """Monthly storm frequency — classic double-peak pattern."""
    storm_months = (df.groupby("SID")
                      .agg(month=("month","first"))
                      .reset_index())
    counts = storm_months["month"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a1628")
    ax.set_facecolor("#0a1628")
    bars = ax.bar(counts.index, counts.values,
                  color="#06b6d4", edgecolor="#0e7490", linewidth=0.8)

    # Annotate peak months
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, str(val),
                ha="center", va="bottom", color="white", fontsize=9)

    ax.set_xticks(range(1,13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"],
                        color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylabel("Number of storms", color="#94a3b8")
    ax.set_title("Seasonal distribution of NI Basin storms\n(Pre-monsoon: May–Jun | Post-monsoon: Oct–Nov peak)",
                 color="white", fontsize=12)
    ax.set_facecolor("#0a1628")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "02_seasonal_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] Saved 02_seasonal_distribution.png")


def plot_intensity_distribution(df: pd.DataFrame):
    """Histogram of WMO wind speeds + RI threshold line."""
    winds = df["WMO_WIND"].dropna()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a1628")
    ax.set_facecolor("#0a1628")
    n, bins, patches = ax.hist(winds, bins=40, color="#0e7490",
                                edgecolor="#065a82", linewidth=0.5)
    # Colour bars by SS category
    for patch, left in zip(patches, bins[:-1]):
        cat = pd.cut([left], bins=SS_BINS, labels=SS_LABELS)[0]
        patch.set_facecolor(SS_COLORS.get(int(cat) if not np.isnan(cat) else 0, "#94a3b8"))

    ax.axvline(63, color="#fbbf24", lw=1.5, linestyle="--", label="Hurricane (63 kt)")
    ax.axvline(33, color="#6ee7b7", lw=1.5, linestyle="--", label="Tropical storm (33 kt)")

    ax.set_xlabel("Max sustained wind (knots)", color="#94a3b8")
    ax.set_ylabel("Observations", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.set_title("Wind speed distribution — NI Basin", color="white", fontsize=12)
    ax.legend(fontsize=9, facecolor="#1e3a5f", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "03_intensity_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] Saved 03_intensity_distribution.png")


def plot_ri_analysis(df: pd.DataFrame):
    """Rapid intensification events — how often does RI happen?"""
    df = df.sort_values(["SID","ISO_TIME"])
    df["wind_24h_future"] = df.groupby("SID")["WMO_WIND"].shift(-4)
    df["delta_wind_24h"]  = df["wind_24h_future"] - df["WMO_WIND"]
    df["RI"] = df["delta_wind_24h"] >= 35

    ri_by_month = df.groupby("month")["RI"].mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0a1628")

    for ax in axes:
        ax.set_facecolor("#0a1628")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")
        ax.tick_params(colors="#94a3b8")

    # Left: RI rate by month
    axes[0].bar(ri_by_month.index, ri_by_month.values,
                color="#ef4444", edgecolor="#b91c1c", linewidth=0.8)
    axes[0].set_xticks(range(1,13))
    axes[0].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                             color="#94a3b8")
    axes[0].set_ylabel("RI events (%)", color="#94a3b8")
    axes[0].set_title("RI rate by month", color="white")

    # Right: delta wind distribution
    axes[1].hist(df["delta_wind_24h"].dropna(), bins=50,
                 color="#0e7490", edgecolor="#065a82", linewidth=0.5)
    axes[1].axvline(35, color="#ef4444", lw=2, linestyle="--", label="RI threshold (35 kt)")
    axes[1].set_xlabel("Δ wind speed over 24h (knots)", color="#94a3b8")
    axes[1].set_ylabel("Count", color="#94a3b8")
    axes[1].set_title("24h wind change distribution", color="white")
    axes[1].legend(fontsize=9, facecolor="#1e3a5f", labelcolor="white")

    plt.suptitle("Rapid Intensification Analysis — NI Basin", color="white",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "04_ri_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] Saved 04_ri_analysis.png")


def print_summary(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  NI BASIN — DATA SUMMARY")
    print("="*55)
    print(f"  Total unique storms   : {df['SID'].nunique()}")
    print(f"  Total observations    : {len(df):,}")
    ymin = int(df["SEASON"].min()); ymax = int(df["SEASON"].max())
    print(f"  Year range            : {ymin} – {ymax}")
    print(f"  Max wind speed (kt)   : {df['WMO_WIND'].max():.0f}")
    print(f"  Min pressure (hPa)    : {df['WMO_PRES'].min():.0f}")
    cat_counts = df.groupby("SID")["SS_cat"].max().value_counts().sort_index()
    print("\n  Storms by peak category:")
    for cat, cnt in cat_counts.items():
        print(f"    {SS_NAMES.get(int(cat),'?'):8s}  {cnt:3d}  {'█' * int(cnt//2)}")
    print("="*55 + "\n")


if __name__ == "__main__":
    df = load_clean()
    print_summary(df)
    plot_storm_tracks(df)
    plot_seasonal_distribution(df)
    plot_intensity_distribution(df)
    plot_ri_analysis(df)
    print(f"\n[✓] All EDA plots saved to {SAVE_DIR}/")