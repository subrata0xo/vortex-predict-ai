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
    .stApp {
        background-color: #0a1628;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f1f38;
        border-right: 1px solid #1e3a5f;
    }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #0f1f38 0%, #1e3a5f 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d4a6f;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #06b6d4;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .ri-alert {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
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
    st.markdown("## 🌀 Cyclone Prediction")
    st.markdown("**NI Basin — AI Forecast System**")
    st.divider()

    # Model selector
    model_type = st.selectbox(
        "Model",
        options=server.available_models if server.available_models else ["lstm"],
        index=(server.available_models.index("hybrid")
               if "hybrid" in server.available_models else 0),
        help="Select the prediction model",
    )

    st.divider()

    # Mode selector
    dashboard_mode = st.radio(
        "Dashboard Mode",
        ["Live Monitor 📡", "Custom Input ✍️", "Historical Analysis 📚"],
        help="Switch between live operational tracking and historical model evaluation."
    )
    
    st.divider()

    selected_sid = None
    
    if dashboard_mode == "Live Monitor 📡":
        st.markdown("### 🌊 Active Cyclones")
        try:
            from dashboard.live_tracker import fetch_live_storms
            live_storms = fetch_live_storms()
            st.session_state["live_storms_cache"] = live_storms
            live_options = {f"🔴 {sname}": f"LIVE_{sname}" for sname in reversed(list(live_storms.keys()))}
        except Exception:
            live_options = {}
            
        if live_options:
            selected_label = st.selectbox("Tracking Now:", options=list(live_options.keys()))
            selected_sid = live_options[selected_label]
        else:
            st.success("✓ No active cyclones detected currently.")
            selected_sid = "LIVE_NONE"
            
    elif dashboard_mode == "Custom Input ✍️":
        st.markdown("### ✍️ Custom Storm")
        st.info("Enter manual coordinates in the main panel.")
        selected_sid = "CUSTOM"
        
    elif dashboard_mode == "Historical Analysis 📚":
        st.markdown("### 🌊 Historical Storms")
        if not test_meta.empty:
            storms = test_meta.drop_duplicates(subset=["SID"])
            storm_options = {
                f"{row['NAME']} ({row['SID'][:4]})": row["SID"]
                for _, row in storms.iterrows()
                if pd.notna(row.get("NAME"))
            }
            if not storm_options:
                storm_options = {row["SID"]: row["SID"] for _, row in storms.iterrows()}
                
            selected_label = st.selectbox("Select past storm", options=list(storm_options.keys()))
            selected_sid = storm_options[selected_label]
        else:
            st.warning("No test metadata found")
            selected_sid = None

    st.divider()
    st.markdown("### ℹ️ Model Info")
    st.markdown(f"- **Models loaded**: {len(server.available_models)}")
    st.markdown(f"- **Active**: {model_type}")
    if test_results.get(model_type):
        tr = test_results[model_type]
        st.markdown(f"- **Track 24h**: {tr.get('track_24h_km', 0):.0f} km")
        st.markdown(f"- **Wind MAE**: {tr.get('wind_mae_kt', 0):.1f} kt")


# ─── Main content ────────────────────────────────────────────────────────────

st.markdown("# 🌀 Cyclone Prediction Dashboard")
st.markdown("*AI-powered cyclone track & intensity forecasting for the North Indian Basin*")

storm_name = None
track_lat = []
track_lon = []
track_wind = []
prediction = None
is_ready_to_render = False

if str(selected_sid).startswith("LIVE_") and selected_sid != "LIVE_NONE":
    st.session_state["active_live_storm"] = str(selected_sid).replace("LIVE_", "")
    selected_sid = "CUSTOM"
elif selected_sid == "LIVE_NONE":
    st.session_state["active_live_storm"] = None
else:
    st.session_state["active_live_storm"] = None

if selected_sid == "LIVE_NONE":
    # Empty default Bay of Bengal view
    is_ready_to_render = True
    storm_name = "Bay of Bengal (Clear)"
    
elif selected_sid == "CUSTOM":
    st.markdown("### ✍️ Enter Live Storm Data")
    st.markdown("Provide the **last 48 hours** of track data (8 observations at 6-hour intervals) to predict the future path.")
    
    # Pre-fill with a dummy track just to show how it works
    dlats = [12.0, 12.5, 13.0, 13.6, 14.2, 14.9, 15.5, 16.0]
    dlons = [80.0, 79.5, 79.0, 78.4, 77.8, 77.1, 76.5, 76.0]
    dwinds = [25.0, 30.0, 35.0, 40.0, 50.0, 55.0, 60.0, 65.0]
    dpress = [1000.0] * 8
    
    # Auto-fill if a LIVE storm was selected
    active_ls = st.session_state.get("active_live_storm")
    if active_ls and "live_storms_cache" in st.session_state:
        pts = st.session_state["live_storms_cache"].get(active_ls, [])
        if len(pts) == 8:
            dlats = [p.get("lat", 0) for p in pts]
            dlons = [p.get("lon", 0) for p in pts]
            dwinds = [p.get("wind", 0) for p in pts]
            dpress = [p.get("pressure", 1000) for p in pts]
            st.success(f"Successfully fetched live historical track for **{active_ls}** from GDACS.")
    
    with st.expander("📝 Manual Data Entry Form", expanded=True):
        with st.form("custom_storm_form"):
            cols = st.columns(4)
            cols[0].markdown("**Latitude (°N)**")
            cols[1].markdown("**Longitude (°E)**")
            cols[2].markdown("**Wind (knots)**")
            cols[3].markdown("**Pressure (hPa)**")
                
            inputs = []
            for i in range(8):
                hr = -(8 - i - 1) * 6
                lbl = "Now" if hr == 0 else f"{hr}h"
                c = st.columns(4)
                lat = c[0].number_input(f"Lat {lbl}", value=dlats[i], step=0.1, label_visibility="collapsed", key=f"lat_{i}")
                lon = c[1].number_input(f"Lon {lbl}", value=dlons[i], step=0.1, label_visibility="collapsed", key=f"lon_{i}")
                wind = c[2].number_input(f"Wind {lbl}", value=dwinds[i], step=1.0, label_visibility="collapsed", key=f"wnd_{i}")
                pres = c[3].number_input(f"Pres {lbl}", value=float(dpress[i]), step=1.0, label_visibility="collapsed", key=f"prs_{i}")
                inputs.append({"lat": lat, "lon": lon, "wind": wind, "pressure": pres, "dist2land": 500.0, "timestamp": None})
                
            submitted = st.form_submit_button("Run AI Prediction 🚀")
            
    if submitted:
        storm_name = "Custom Live Storm Forecast"
        track_lat = [pt["lat"] for pt in inputs]
        track_lon = [pt["lon"] for pt in inputs]
        track_wind = [pt["wind"] for pt in inputs]
        try:
            with st.spinner("Analyzing storm vectors..."):
                prediction = server.predict(inputs, model_type=model_type)
            is_ready_to_render = True
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            
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
                prediction = server.predict(track_points, model_type=model_type)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Historical storm has fewer than 8 observations for prediction.")
            
        is_ready_to_render = True
    else:
        st.warning(f"No data found for storm {selected_sid}")
else:
    st.info("Select a storm from the sidebar to view predictions.")

if is_ready_to_render:
    # ── Metrics row ──────────────────────────────────────────────────────
    if prediction:
        cols = st.columns(4)
        with cols[0]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{prediction["wind"]["24h_kt"]:.0f} kt</div>'
                f'<div class="metric-label">Wind (24h forecast)</div>'
                f'</div>', unsafe_allow_html=True
            )
        with cols[1]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#8b5cf6">'
                f'{prediction["track"]["24h"]["lat"]:.1f}°N</div>'
                f'<div class="metric-label">Latitude (24h forecast)</div>'
                f'</div>', unsafe_allow_html=True
            )
        with cols[2]:
            ri_pct = prediction["ri_probability"] * 100
            ri_color = "#ef4444" if ri_pct > 50 else "#fbbf24" if ri_pct > 20 else "#6ee7b7"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{ri_color}">{ri_pct:.0f}%</div>'
                f'<div class="metric-label">RI Probability</div>'
                f'</div>', unsafe_allow_html=True
            )
        with cols[3]:
            lf_pct = prediction["landfall_probability"] * 100
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#f43f5e">{lf_pct:.0f}%</div>'
                f'<div class="metric-label">Landfall (72h)</div>'
                f'</div>', unsafe_allow_html=True
            )

        # RI alert banner
        if prediction.get("ri_alert", False):
            st.markdown(
                '<div class="ri-alert">⚠️ RAPID INTENSIFICATION ALERT — '
                'Storm may intensify ≥35 kt in the next 24 hours</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Map + Charts layout ──────────────────────────────────────────────
    map_col, chart_col = st.columns([3, 2])

    with map_col:
        st.markdown(f"### 🗺️ Storm Track — {storm_name}")
        forecast_track = prediction["track"] if prediction else None
        storm_map = render_storm_map(
            track_lat, track_lon, track_wind,
            forecast=forecast_track,
            storm_name=str(storm_name),
        )
        st_folium(storm_map, height=500, use_container_width=True)

    with chart_col:
        if prediction:
            # Wind forecast chart
            current_wind = track_wind[-1] if track_wind else 0
            fig_wind = wind_forecast_chart(
                current_wind=current_wind,
                forecast_wind=prediction["wind"],
                history_wind=track_wind[-10:] if len(track_wind) >= 10 else track_wind,
            )
            st.plotly_chart(fig_wind, use_container_width=True)

            # RI gauge
            fig_ri = ri_gauge(prediction["ri_probability"])
            st.plotly_chart(fig_ri, use_container_width=True)

    # ── Model comparison ─────────────────────────────────────────────────
    if selected_sid != "CUSTOM":
        st.divider()
        st.markdown("### 📊 Model Performance")

        if len(test_results) >= 2:
            fig_comp = model_comparison_chart(
                test_results.get("lstm", {}),
                test_results.get("hybrid", {}),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        elif test_results:
            # Single model
            key = list(test_results.keys())[0]
            fig_err = track_error_chart(test_results[key])
            st.plotly_chart(fig_err, use_container_width=True)

    # ── Raw prediction data ──────────────────────────────────────────────
    if prediction:
        with st.expander("📋 Raw Prediction Data"):
            st.json(prediction)

    # ── Animated Wind Flow (Earth Nullschool) ────────────────────────────
    st.divider()
    st.markdown("### 🌪️ Live Global Wind Flow")
    st.markdown("Interactive 3D atmospheric wind animation synced to the current map center.")
    
    wind_lat = track_lat[-1] if track_lat else 15.0
    wind_lon = track_lon[-1] if track_lon else 90.0
    
    nullschool_url = f"https://earth.nullschool.net/#current/wind/surface/level/orthographic={wind_lon:.2f},{wind_lat:.2f},1500"
    
    st.components.v1.iframe(nullschool_url, height=500, scrolling=False)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center; color:#475569; font-size:12px;">'
    'Cyclone Prediction System — NI Basin | '
    'Data: IBTrACS + ERA5 | '
    'Models: LSTM + ConvLSTM + Transformer'
    '</div>',
    unsafe_allow_html=True,
)
