"""
Cyclone Prediction Dashboard — Streamlit App

Run:
    cd d:/Cyclone AI
    python -m streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import json
from streamlit_folium import st_folium

from dashboard.map_viz import render_storm_map
from dashboard.charts import (
    wind_forecast_chart, track_error_chart, ri_gauge, model_comparison_chart,
)
from api.inference import ModelServer

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cyclone Prediction — NI Basin",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0c111d; /* Sleek slate dark */
    }
    
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    h1, h2, h3, h4 { color: #f3f4f6 !important; font-weight: 600; }
    p { color: #cbd5e1; }
    
    /* Segmented Control / Radio styling for "tabs" feeling */
    div.row-widget.stRadio > div {
        display: flex;
        flex-direction: row;
        background: rgba(31, 41, 55, 0.5);
        border-radius: 12px;
        padding: 4px;
        gap: 8px;
    }
    div.row-widget.stRadio > div > label {
        background: transparent;
        padding: 10px 24px;
        border-radius: 8px;
        transition: 0.2s;
        cursor: pointer;
    }
    div.row-widget.stRadio > div > label:hover {
        background: rgba(59, 130, 246, 0.1);
    }
    div.row-widget.stRadio > div > label[data-checked="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }
    div.row-widget.stRadio p {
        margin: 0;
        font-weight: 500;
        font-size: 15px;
        color: inherit;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(31, 41, 55, 0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 20px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .metric-value {
        font-size: 34px;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 13px;
        color: #9ca3af;
        margin-top: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .ri-alert {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.8) 0%, rgba(185, 28, 28, 0.8) 100%);
        backdrop-filter: blur(4px);
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fee2e2;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.95; transform: scale(0.99); }
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        background: rgba(17, 24, 39, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─── Load data ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_server():
    """Load the model server (cached across reruns)."""
    return ModelServer(
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints"),
        processed_dir=str(PROJECT_ROOT / "data" / "processed"),
    )

@st.cache_data
def load_test_meta():
    """Load test set metadata for storm selection."""
    path = PROJECT_ROOT / "data" / "processed" / "test_meta.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["ISO_TIME"])
    return pd.DataFrame()

@st.cache_data
def load_raw_data():
    """Load raw IBTrACS data for track visualization."""
    path = PROJECT_ROOT / "data" / "raw" / "ibtracs_NI.csv"
    if not path.exists():
        # Download on the fly if running on cloud
        from src.data_loader import download_ibtracs
        try:
            download_ibtracs(data_dir=str(PROJECT_ROOT / "data" / "raw"))
        except Exception as e:
            st.error(f"Failed to download required dataset: {e}")
            return pd.DataFrame()
            
    df = pd.read_csv(path, skiprows=[1], low_memory=False)
    for c in ["LAT", "LON", "WMO_WIND", "WMO_PRES", "DIST2LAND"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    return df

@st.cache_data
def load_test_results():
    """Load test results for both models."""
    results = {}
    for name, path in [
        ("lstm", PROJECT_ROOT / "experiments" / "test_results.json"),
        ("hybrid", PROJECT_ROOT / "experiments" / "hybrid_test_results.json"),
    ]:
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


# ─── Initialize ──────────────────────────────────────────────────────────────

server = load_model_server()
test_meta = load_test_meta()
raw_data = load_raw_data()
test_results = load_test_results()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌀 Cyclone AI")
    st.markdown("**Bay of Bengal Forecast**")
    st.divider()

    st.markdown("### ⚙️ Global Settings")
    model_type = st.selectbox(
        "AI Forecast Model",
        options=server.available_models if server.available_models else ["lstm"],
        index=(server.available_models.index("hybrid") if "hybrid" in server.available_models else 0),
        help="Select the underlying AI model for track and intensity prediction.",
    )

    st.divider()
    st.markdown("### ℹ️ Active Model Error")
    if test_results.get(model_type):
        tr = test_results[model_type]
        st.markdown(f"- **Track (24h)**: {tr.get('track_24h_km', 0):.0f} km")
        st.markdown(f"- **Wind (MAE)**: {tr.get('wind_mae_kt', 0):.1f} kt")
    else:
        st.markdown("*No test data available for this model.*")

    st.divider()
    st.markdown("### 💡 About")
    st.caption(
        "This dashboard integrates ERA5 atmospheric data and IBTrACS history "
        "with deep learning models (LSTM/Hybrid) to forecast cyclone paths and "
        "intensities in the North Indian Ocean basin, prioritizing live monitoring "
        "over the Bay of Bengal."
    )


# ─── Main content ────────────────────────────────────────────────────────────

# Header & Mode Selection row
st.markdown("# 🌀 Cyclone Tracking & Prediction")
st.markdown("*Real-time monitoring and advanced AI trajectory forecasts.*")

st.markdown("<br>", unsafe_allow_html=True)

# The Mode Selector acting like tabs visually
dashboard_mode = st.radio(
    "Mode",
    ["📡 Live Monitor", "✍️ Custom Storm Input", "📚 Historical Archive"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

storm_name = None
track_lat = []
track_lon = []
track_wind = []
prediction = None
is_ready_to_render = False

selected_sid = None

if dashboard_mode == "📡 Live Monitor":
    st.markdown("### 🌊 Local Bay of Bengal Radar")
    try:
        from dashboard.live_tracker import fetch_live_storms
        live_storms = fetch_live_storms()
        st.session_state["live_storms_cache"] = live_storms
        live_options = {f"🔴 {sname}": f"LIVE_{sname}" for sname in reversed(list(live_storms.keys()))}
    except Exception:
        live_options = {}
        
    if live_options:
        colA, colB = st.columns([1, 3])
        with colA:
            selected_label = st.selectbox("Active Cyclone:", options=list(live_options.keys()))
            selected_sid = live_options[selected_label]
            st.session_state["active_live_storm"] = str(selected_sid).replace("LIVE_", "")
            selected_sid = "CUSTOM"
    else:
        st.success("✓ No active cyclones detected. Automatically routing to clear Bay of Bengal view.")
        selected_sid = "LIVE_NONE"

elif dashboard_mode == "✍️ Custom Storm Input":
    st.markdown("### ✍️ Enter Manual Coordinates")
    st.markdown("Provide the **last 48 hours** of track data to predict the future AI path.")
    selected_sid = "CUSTOM"
    
elif dashboard_mode == "📚 Historical Archive":
    st.markdown("### 📚 Review Past Cyclones")
    colA, colB = st.columns([1, 2])
    with colA:
        if not test_meta.empty:
            storms = test_meta.drop_duplicates(subset=["SID"])
            storm_options = {
                f"{row['NAME']} ({row['SID'][:4]})": row["SID"]
                for _, row in storms.iterrows() if pd.notna(row.get("NAME"))
            }
            if not storm_options:
                storm_options = {row["SID"]: row["SID"] for _, row in storms.iterrows()}
                
            selected_label = st.selectbox("Select Storm", options=list(storm_options.keys()))
            selected_sid = storm_options[selected_label]
        else:
            st.warning("No historical metadata found")
            selected_sid = None


st.divider()

# Logic implementation based on state
if selected_sid == "LIVE_NONE":
    # Empty default Bay of Bengal view
    is_ready_to_render = True
    storm_name = "Bay of Bengal (Clear)"
    
elif selected_sid == "CUSTOM":
    # Pre-fill with dummy track or live track
    dlats = [12.0, 12.5, 13.0, 13.6, 14.2, 14.9, 15.5, 16.0]
    dlons = [80.0, 79.5, 79.0, 78.4, 77.8, 77.1, 76.5, 76.0]
    dwinds = [25.0, 30.0, 35.0, 40.0, 50.0, 55.0, 60.0, 65.0]
    dpress = [1000.0] * 8
    
    # Auto-fill if from live radar
    active_ls = st.session_state.get("active_live_storm")
    if dashboard_mode == "📡 Live Monitor" and active_ls and "live_storms_cache" in st.session_state:
        pts = st.session_state["live_storms_cache"].get(active_ls, [])
        if len(pts) == 8:
            dlats = [p.get("lat", 0) for p in pts]
            dlons = [p.get("lon", 0) for p in pts]
            dwinds = [p.get("wind", 0) for p in pts]
            dpress = [p.get("pressure", 1000) for p in pts]
            st.success(f"Loaded live track for **{active_ls}** from GDACS system.")
    
    form_expanded = (dashboard_mode == "✍️ Custom Storm Input")
    
    with st.expander("📝 Track Data Payload", expanded=form_expanded):
        with st.form("custom_storm_form"):
            cols = st.columns(4)
            cols[0].markdown("**Lat (°N)**", help="Latitude")
            cols[1].markdown("**Lon (°E)**", help="Longitude")
            cols[2].markdown("**Wind (kt)**", help="Maximum sustained wind speed")
            cols[3].markdown("**Press (hPa)**", help="Central pressure")
                
            inputs = []
            for i in range(8):
                hr = -(8 - i - 1) * 6
                lbl = "Now" if hr == 0 else f"{hr}h"
                c = st.columns(4)
                lat = c[0].number_input(f"Lat {lbl}", value=dlats[i], step=0.1, label_visibility="collapsed", key=f"lat_{i}")
                lon = c[1].number_input(f"Lon {lbl}", value=dlons[i], step=0.1, label_visibility="collapsed", key=f"lon_{i}")
                wind = c[2].number_input(f"Wind {lbl}", value=float(dwinds[i]), step=1.0, label_visibility="collapsed", key=f"wnd_{i}")
                pres = c[3].number_input(f"Pres {lbl}", value=float(dpress[i]), step=1.0, label_visibility="collapsed", key=f"prs_{i}")
                inputs.append({"lat": lat, "lon": lon, "wind": wind, "pressure": pres, "dist2land": 500.0, "timestamp": None})
                
            submitted = st.form_submit_button("Run AI Prediction 🚀" if dashboard_mode == "✍️ Custom Storm Input" else "Recalculate Path")

    if submitted or (dashboard_mode == "📡 Live Monitor"):
        # We auto-run if in Live Monitor mode since we just mounted it, 
        # but if custom, we wait for form submit.
        storm_name = active_ls if (dashboard_mode == "📡 Live Monitor" and active_ls) else "Custom Storm"
        track_lat = [pt["lat"] for pt in inputs]
        track_lon = [pt["lon"] for pt in inputs]
        track_wind = [pt["wind"] for pt in inputs]
        try:
            with st.spinner("Analyzing storm vectors with Deep Learning..."):
                prediction = server.predict(inputs, model_type=model_type)
            is_ready_to_render = True
        except Exception as e:
            st.error(f"Prediction inference failed: {e}")
            
elif selected_sid and not raw_data.empty:
    storm_data = raw_data[raw_data["SID"] == selected_sid].sort_values("ISO_TIME")
    if not storm_data.empty:
        storm_name = str(storm_data.iloc[0].get("NAME", selected_sid))
        track_lat = storm_data["LAT"].dropna().tolist()
        track_lon = storm_data["LON"].dropna().tolist()
        track_wind = storm_data["WMO_WIND"].fillna(0).tolist()
        
        if len(storm_data) >= 8:
            last_obs = storm_data.tail(8)
            track_points = []
            for _, row in last_obs.iterrows():
                track_points.append({
                    "lat": row["LAT"], "lon": row["LON"],
                    "wind": row.get("WMO_WIND", 0) or 0,
                    "pressure": row.get("WMO_PRES", None),
                    "dist2land": row.get("DIST2LAND", None),
                    "timestamp": str(row["ISO_TIME"]) if pd.notna(row["ISO_TIME"]) else None,
                })
            try:
                with st.spinner("Recalculating historical prediction..."):
                    prediction = server.predict(track_points, model_type=model_type)
                is_ready_to_render = True
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Historical storm has fewer than 8 observations for prediction.")
    else:
        st.warning(f"No data found for storm {selected_sid}")


# ─── Display Output ──────────────────────────────────────────────────────────

if is_ready_to_render:
    
    if prediction:
        # RI alert banner at top
        if prediction.get("ri_alert", False):
            st.markdown(
                '<div class="ri-alert">⚠️ RAPID INTENSIFICATION ALERT — '
                'Storm may cross 35+ kt threshold in the next 24 hours</div>',
                unsafe_allow_html=True,
            )
            
        # Metrics row
        cols = st.columns(4)
        with cols[0]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{prediction["wind"]["24h_kt"]:.0f} kt</div>'
                f'<div class="metric-label">Max Wind (24h)</div>'
                f'</div>', unsafe_allow_html=True
            )
        with cols[1]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#a855f7">'
                f'{prediction["track"]["24h"]["lat"]:.1f}°N</div>'
                f'<div class="metric-label">Latitude Track (24h)</div>'
                f'</div>', unsafe_allow_html=True
            )
        with cols[2]:
            ri_pct = prediction["ri_probability"] * 100
            ri_color = "#ef4444" if ri_pct > 50 else "#fbbf24" if ri_pct > 20 else "#10b981"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{ri_color}">{ri_pct:.0f}%</div>'
                f'<div class="metric-label">Rapid Int. Chance</div>'
                f'</div>', unsafe_allow_html=True
            )
        with cols[3]:
            lf_pct = prediction["landfall_probability"] * 100
            lf_color = "#ef4444" if lf_pct > 50 else "#8b5cf6"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{lf_color}">{lf_pct:.0f}%</div>'
                f'<div class="metric-label">Landfall Probability</div>'
                f'</div>', unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Wrap the map & charts in nice containers
    map_container = st.container(border=True)
    with map_container:
        st.markdown(f"### 🗺️ Live Trajectory: {storm_name}")
        forecast_track = prediction["track"] if prediction else None
        storm_map = render_storm_map(
            track_lat, track_lon, track_wind,
            forecast=forecast_track,
            storm_name=str(storm_name),
        )
        st_folium(storm_map, height=550, use_container_width=True)

    if prediction:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            with st.container(border=True):
                # Wind forecast chart
                current_wind = track_wind[-1] if track_wind else 0
                fig_wind = wind_forecast_chart(
                    current_wind=current_wind,
                    forecast_wind=prediction["wind"],
                    history_wind=track_wind[-10:] if len(track_wind) >= 10 else track_wind,
                )
                st.plotly_chart(fig_wind, use_container_width=True)
                
        with chart_col2:
            with st.container(border=True):
                # RI gauge
                fig_ri = ri_gauge(prediction["ri_probability"])
                st.plotly_chart(fig_ri, use_container_width=True)


    if dashboard_mode == "📚 Historical Archive":
        st.divider()
        st.markdown("### 📊 Architecture Performance Analysis")

        if len(test_results) >= 2:
            fig_comp = model_comparison_chart(
                test_results.get("lstm", {}),
                test_results.get("hybrid", {}),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        elif test_results:
            key = list(test_results.keys())[0]
            fig_err = track_error_chart(test_results[key])
            st.plotly_chart(fig_err, use_container_width=True)

    if prediction and dashboard_mode == "✍️ Custom Storm Input":
        with st.expander("📋 View Raw JSON Model Output"):
            st.json(prediction)


st.divider()

# Footer
st.markdown(
    '<div style="text-align:center; color:#64748b; font-size:13px; font-weight:500;">'
    'Cyclone Prediction AI System — Protected by MIT License | '
    'Dual Stack: LSTM & ConvLSTM-Transformer | '
    'Real-time ERA5 Feed'
    '</div>',
    unsafe_allow_html=True,
)
