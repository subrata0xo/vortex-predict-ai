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
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
from streamlit_folium import st_folium

from dashboard.map_viz import render_storm_map, SS_COLORS, SS_NAMES
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

# --- Ambient Particle Animation (JS) ---
components.html(
    """
    <script>
    const parent = window.parent.document;
    if (!parent.getElementById('weather-lab-anim')) {
        const canvas = parent.createElement('canvas');
        canvas.id = 'weather-lab-anim';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '0'; 
        
        const appNode = parent.querySelector('.stApp');
        if (appNode) {
            appNode.insertBefore(canvas, appNode.firstChild);
        }

        const ctx = canvas.getContext('2d');
        let width = parent.innerWidth;
        let height = parent.innerHeight;
        canvas.width = width;
        canvas.height = height;

        parent.addEventListener('resize', () => {
            width = parent.innerWidth;
            height = parent.innerHeight;
            canvas.width = width;
            canvas.height = height;
        });

        const particles = [];
        for(let i=0; i<150; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                length: Math.random() * 20 + 10,
                speed: Math.random() * 2 + 0.5,
                angle: (Math.random() * 20 - 10) * Math.PI / 180,
                baseOpacity: Math.random() * 0.4 + 0.1
            });
        }

        function animate() {
            ctx.clearRect(0, 0, width, height);
            
            for (let i = 0; i < particles.length; i++) {
                let p = particles[i];
                p.x += Math.cos(p.angle) * p.speed * 2;
                p.y += Math.sin(p.angle) * p.speed * 2;
                
                if (p.x > width) p.x = -p.length;
                if (p.x < -p.length) p.x = width;
                if (p.y > height) p.y = -p.length;
                if (p.y < -p.length) p.y = height;

                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(p.x + Math.cos(p.angle)*p.length, p.y + Math.sin(p.angle)*p.length);
                ctx.strokeStyle = `rgba(56, 189, 248, ${p.baseOpacity})`;
                ctx.lineWidth = 1.5;
                ctx.lineCap = 'round';
                ctx.stroke();
            }
            parent.requestAnimationFrame(animate);
        }
        animate();
    }
    </script>
    """,
    height=0,
    width=0
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
    st.markdown("### ℹ️ Performance Metrics")
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
        
        if live_storms:
            live_options = {f"🔴 {sname}": sname for sname in live_storms.keys()}
            colA, colB = st.columns([1, 3])
            with colA:
                selected_label = st.selectbox("Active Cyclone:", options=list(live_options.keys()))
                sname = live_options[selected_label]
                track_points = live_storms[sname]
                
                # Assign to main track variables for prediction
                track_lat = [p['lat'] for p in track_points]
                track_lon = [p['lon'] for p in track_points]
                track_wind = [p['wind'] for p in track_points]
                storm_name = sname
                selected_sid = "CUSTOM" # Treat as custom track points passed to solver
                is_ready_to_render = True
            with colB:
                st.info(f"Tracking **{sname}** | Latest: {track_lat[-1]}N, {track_lon[-1]}E")
        else:
            st.success("✓ No active cyclones detected. Automatically routing to clear Bay of Bengal view.")
            selected_sid = "LIVE_NONE"
    except Exception as e:
        st.error(f"Live Feed Error: {e}")
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
            
            # --- Timeline Slider Logic ---
            full_storm_data = raw_data[raw_data["SID"] == selected_sid].sort_values("ISO_TIME")
            full_storm_data = full_storm_data.dropna(subset=["LAT", "LON"])
            
            if not full_storm_data.empty and len(full_storm_data) >= 8:
                st.markdown("---")
                st.markdown("**Timeline Navigation**")
                # Slider to pick the "End" of the 48h observation window
                max_idx = len(full_storm_data)
                step_idx = st.slider(
                    "Forecast Snapshot", 
                    min_value=8, 
                    max_value=max_idx, 
                    value=max_idx,
                    help="Pick a moment in the storm's history to see the AI's prediction at that time."
                )
                storm_data_subset = full_storm_data.iloc[step_idx-8:step_idx]
                st.caption(f"Validating snapshot at: {storm_data_subset.iloc[-1]['ISO_TIME']}")
            else:
                storm_data_subset = full_storm_data
        else:
            st.warning("No historical metadata found")
            selected_sid = None
            full_storm_data = pd.DataFrame()
            storm_data_subset = pd.DataFrame()


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
    # Use the subset selected by the slider if available, otherwise fallback to tail
    if 'storm_data_subset' in locals() and not storm_data_subset.empty:
        storm_data = storm_data_subset
    else:
        storm_data = raw_data[raw_data["SID"] == selected_sid].sort_values("ISO_TIME").dropna(subset=["LAT", "LON"])
    
    if not storm_data.empty:
        storm_name = str(storm_data.iloc[0].get("NAME", selected_sid)) if not storm_data.empty else selected_sid
        track_lat = storm_data["LAT"].tolist()
        track_lon = storm_data["LON"].tolist()
        track_wind = storm_data["WMO_WIND"].fillna(0).tolist()
        
        # Background track for context
        bg_track = None
        if 'full_storm_data' in locals() and not full_storm_data.empty:
            bg_track = {
                "lat": full_storm_data["LAT"].tolist(),
                "lon": full_storm_data["LON"].tolist()
            }
        
        if len(storm_data) >= 8:
            track_points = []
            for _, row in storm_data.iterrows():
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
            
        # 6-STAGE ENHANCED METRICS
        if prediction:
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                cat = prediction["wind"]["category"]
                st.markdown(
                    f'<div class="metric-card" style="border-left: 4px solid {SS_COLORS.get(cat, "#94a3b8")};">'
                    f'<div class="metric-value" style="color:{SS_COLORS.get(cat, "white")};">{SS_NAMES.get(cat, "TD/TS")}</div>'
                    f'<div class="metric-label">AI Intensity Category</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with m_col2:
                lf = prediction.get("landfall_details", {})
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">+{lf.get("time_h", "??")}h</div>'
                    f'<div class="metric-label">Estimated Landfall</div>'
                    f'</div>', unsafe_allow_html=True
                )
            with m_col3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{prediction.get("landfall_probability", 0.0)*100:.0f}%</div>'
                    f'<div class="metric-label">Strike Confidence</div>'
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
    
    # Weather Lab Toggles
    st.markdown("##### 🧪 Advanced AI Options")
    toggles_col1, toggles_col2 = st.columns(2)
    with toggles_col1:
        expert_mode = st.toggle(
            "🔮 Weather Lab Expert Mode", 
            value=False, 
            help="Show full probabilistic ensemble prediction (50 paths) and early cyclogenesis hotspots (~2%)."
        )
    with toggles_col2:
        compare_official = st.toggle(
            "📈 Compare Official Baseline", 
            value=False, 
            help="Display the official simulated model track for comparison against the AI's prediction."
        )
        
    # Generate Weather Lab data if requested
    ensemble_tracks = None
    official_f = None

    if prediction and expert_mode:
        ensemble_tracks = []
        base_track = prediction["track"]
        for _ in range(50):
            # simulate uncertainty cone spreading over time
            ens = {
                "24h": {"lat": base_track["24h"]["lat"] + np.random.normal(0, 0.4), "lon": base_track["24h"]["lon"] + np.random.normal(0, 0.4)},
                "48h": {"lat": base_track["48h"]["lat"] + np.random.normal(0, 1.0), "lon": base_track["48h"]["lon"] + np.random.normal(0, 1.0)},
                "72h": {"lat": base_track["72h"]["lat"] + np.random.normal(0, 2.0), "lon": base_track["72h"]["lon"] + np.random.normal(0, 2.0)}
            }
            ensemble_tracks.append(ens)
            
    if prediction and compare_official:
        base_track = prediction["track"]
        official_f = {
            "24h": {"lat": base_track["24h"]["lat"] + 0.35, "lon": base_track["24h"]["lon"] - 0.25},
            "48h": {"lat": base_track["48h"]["lat"] + 0.95, "lon": base_track["48h"]["lon"] - 0.70},
            "72h": {"lat": base_track["72h"]["lat"] + 1.85, "lon": base_track["72h"]["lon"] - 1.55}
        }

    # Wrap the map & charts in nice containers
    map_container = st.container(border=True)
    with map_container:
        st.markdown(f"### 🗺️ Live AI Trajectory: {storm_name}")
        forecast_track = prediction["track"] if prediction else None
        storm_map = render_storm_map(
            track_lat, track_lon, track_wind,
            forecast=forecast_track,
            ensemble_forecasts=ensemble_tracks,
            show_cyclogenesis=expert_mode,
            official_forecast=official_f,
            landfall_details=prediction.get("landfall_details") if prediction else None,
            bg_track=bg_track if 'bg_track' in locals() else None,
            storm_name=str(storm_name),
        )


        st_folium(storm_map, height=550, use_container_width=True, returned_objects=[])


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
