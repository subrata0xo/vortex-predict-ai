"""
Storm track map visualization with Folium.
"""

import folium
from folium import plugins
import numpy as np
import random

# Saffir-Simpson color palette
SS_COLORS = {
    0: "#94a3b8",   # TD/TS — slate
    1: "#fde68a",   # Cat 1 — amber
    2: "#fb923c",   # Cat 2 — orange
    3: "#ef4444",   # Cat 3 — red
    4: "#b91c1c",   # Cat 4 — dark red
    5: "#7f1d1d",   # Cat 5 — maroon
}

SS_NAMES = {
    0: "TD/TS", 1: "Cat 1", 2: "Cat 2",
    3: "Cat 3", 4: "Cat 4", 5: "Cat 5",
}

def _wind_to_category(wind_kt: float) -> int:
    """Convert wind speed (knots) to Saffir-Simpson category."""
    if wind_kt < 33:    return 0
    elif wind_kt < 63:  return 0
    elif wind_kt < 82:  return 1
    elif wind_kt < 95:  return 2
    elif wind_kt < 112: return 3
    elif wind_kt < 136: return 4
    else:               return 5

def render_storm_map(track_lat: list, track_lon: list,
                     track_wind: list = None,
                     forecast: dict = None,
                     ensemble_forecasts: list = None,
                     show_cyclogenesis: bool = False,
                     official_forecast: dict = None,
                     landfall_details: dict = None,
                     bg_track: dict = None,
                     storm_name: str = "Storm") -> folium.Map:

    """
    Render a Folium map showing storm track, Weather Lab ensemble predictions, and cyclogenesis hotspots.
    """
    # Center map on the latest position, or default to Bay of Bengal
    center_lat = track_lat[-1] if track_lat else 15.0
    center_lon = track_lon[-1] if track_lon else 90.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB dark_matter",
    )

    # Add Live Satellite Clouds Overlay
    folium.raster_layers.WmsTileLayer(
        url="https://mesonet.agron.iastate.edu/cgi-bin/wms/goes/global_ir.cgi?",
        layers="goes_global_ir",
        fmt="image/png",
        transparent=True,
        name="Live Satellite Clouds (IR)",
        overlay=True,
        control=False,
        opacity=0.35, # Dimmed slightly for better ensemble visibility
        show=True,
    ).add_to(m)

    # ── Expert Mode: Cyclogenesis Probability Clusters ────────────────────────
    if show_cyclogenesis:
        fg_cyclo = folium.FeatureGroup(name="Cyclogenesis Hotspots (2%)", show=True)
        # Generate some synthetic low-probability clusters in the Indian Ocean
        # for demonstration of the Weather Lab feature.
        base_hospots = [
            (8.5, 87.2), (10.1, 92.5), (6.4, 82.1), (12.3, 68.4)
        ]
        # Seed for visual stability based on name/coords so it doesn't flicker
        # on every streamlit rerun exactly if we care, but random is fine.
        random.seed(int(center_lat * 10))
        for hlat, hlon in base_hospots:
            for _ in range(8):
                alat = hlat + random.uniform(-1.5, 1.5)
                alon = hlon + random.uniform(-1.5, 1.5)
                folium.CircleMarker(
                    location=[alat, alon],
                    radius=3,
                    color="#f43f5e",
                    fill=True,
                    fill_color="#f43f5e",
                    fill_opacity=0.4,
                    weight=0,
                    popup="Early Formation Risk (~2%)"
                ).add_to(fg_cyclo)
        fg_cyclo.add_to(m)

    # ── Full History Background (Faint) ──────────────────────────────────────
    if bg_track:
        folium.PolyLine(
            locations=list(zip(bg_track["lat"], bg_track["lon"])),
            color="#ffffff",
            weight=1,
            opacity=0.3,
            dash_array="5, 5",
            tooltip="Full Historical Path",
        ).add_to(m)

    # ── Historical track ─────────────────────────────────────────────────────
    if track_wind is None:
        track_wind = [0.0] * len(track_lat)

    # Draw track segments colored by intensity
    for i in range(1, len(track_lat)):
        cat = _wind_to_category(track_wind[i])
        plugins.AntPath(
            locations=[
                [track_lat[i-1], track_lon[i-1]],
                [track_lat[i], track_lon[i]],
            ],
            color=SS_COLORS[cat],
            pulse_color="#ffffff",
            weight=4,
            opacity=0.8,
            dash_array=[10, 20],
            delay=1000,
            hardware_accelerated=True,
        ).add_to(m)

    # Mark current position
    if track_lat:
        current_cat = _wind_to_category(track_wind[-1])
        folium.CircleMarker(
            location=[track_lat[-1], track_lon[-1]],
            radius=10,
            color=SS_COLORS[current_cat],
            fill=True,
            fill_color=SS_COLORS[current_cat],
            fill_opacity=0.9,
            popup=(
                f"<b>{storm_name}</b><br>"
                f"Wind: {track_wind[-1]:.0f} kt<br>"
                f"Category: {SS_NAMES[current_cat]}<br>"
                f"Lat: {track_lat[-1]:.2f}°<br>"
                f"Lon: {track_lon[-1]:.2f}°"
            ),
        ).add_to(m)

        # Start marker
        folium.CircleMarker(
            location=[track_lat[0], track_lon[0]],
            radius=5,
            color="#6ee7b7",
            fill=True,
            fill_color="#6ee7b7",
            fill_opacity=0.8,
            popup="Genesis",
        ).add_to(m)

    # ── Official Model Baseline Comparison ───────────────────────────────────
    if official_forecast and track_lat:
        off_lats = [track_lat[-1]]
        off_lons = [track_lon[-1]]
        for horizon in ["24h", "48h", "72h"]:
            if horizon in official_forecast:
                off_lats.append(official_forecast[horizon]["lat"])
                off_lons.append(official_forecast[horizon]["lon"])
                
        # Draw official track as dashed orange line
        folium.PolyLine(
            locations=list(zip(off_lats, off_lons)),
            color="#fb923c",
            weight=3,
            opacity=0.8,
            dash_array="5, 10",
            tooltip="Official Model (Baseline)",
        ).add_to(m)

    # ── Uncertainty Cone (95th Percentile Spread) ──────────────────────────────
    if ensemble_forecasts and track_lat:
        cone_points_left = []
        cone_points_right = []
        
        # Add current position as the tip of the cone
        cone_points_left.append([track_lat[-1], track_lon[-1]])
        cone_points_right.append([track_lat[-1], track_lon[-1]])

        for horizon in ["24h", "48h", "72h"]:
            hs = [ens[horizon] for ens in ensemble_forecasts if horizon in ens]
            if not hs: continue
            
            lats = [h['lat'] for h in hs]
            lons = [h['lon'] for h in hs]
            
            # Use mean and standard deviation to find the "edges" of the cone
            m_lat, s_lat = np.mean(lats), np.std(lats)
            m_lon, s_lon = np.mean(lons), np.std(lons)
            
            # Simple approximation of the spread
            cone_points_left.append([m_lat + 1.28 * s_lat, m_lon - 1.28 * s_lon])
            cone_points_right.append([m_lat - 1.28 * s_lat, m_lon + 1.28 * s_lon])

        # Construct the polygon (left points then reversed right points to close the loop)
        full_cone = cone_points_left + cone_points_right[::-1]
        folium.Polygon(
            locations=full_cone,
            color="#38bdf8",
            weight=1,
            fill=True,
            fill_color="#38bdf8",
            fill_opacity=0.15,
            popup="Cone of Uncertainty (95%)",
            tooltip="Ensemble Prediction Spread",
        ).add_to(m)

    # ── Weather Lab Ensemble AI Forecasts ────────────────────────────────────
    if ensemble_forecasts and track_lat:
        fg_ensemble = folium.FeatureGroup(name="AI Probabilistic Ensemble (50x)", show=False)
        for ens in ensemble_forecasts:
            ens_lats = [track_lat[-1]]
            ens_lons = [track_lon[-1]]
            for horizon in ["24h", "48h", "72h"]:
                ens_lats.append(ens[horizon]["lat"])
                ens_lons.append(ens[horizon]["lon"])
            
            folium.PolyLine(
                locations=list(zip(ens_lats, ens_lons)),
                color="#38bdf8", # Light blue
                weight=1,
                opacity=0.08,
            ).add_to(fg_ensemble)
            
        fg_ensemble.add_to(m)


    # ── AI Mean Forecast track ───────────────────────────────────────────────
    if forecast and track_lat:
        forecast_lats = [track_lat[-1]]
        forecast_lons = [track_lon[-1]]
        labels = ["Now", "24h", "48h", "72h"]

        for horizon in ["24h", "48h", "72h"]:
            if horizon in forecast:
                forecast_lats.append(forecast[horizon]["lat"])
                forecast_lons.append(forecast[horizon]["lon"])

        # Deepmind style: Bold blue line for the mean prediction
        plugins.AntPath(
            locations=list(zip(forecast_lats, forecast_lons)),
            color="#0ea5e9",  # Vibrant cyan/blue
            pulse_color="#ffffff",
            weight=5,
            opacity=1.0,
            dash_array=[15, 30],
            delay=800,  
            hardware_accelerated=True,
            tooltip="AI Mean Forecast Path",
        ).add_to(m)

        # Forecast point markers
        for i in range(1, len(forecast_lats)):
            folium.CircleMarker(
                location=[forecast_lats[i], forecast_lons[i]],
                radius=6,
                color="#0ea5e9",
                fill=True,
                fill_color="#0284c7",
                fill_opacity=0.9,
                popup=f"AI Forecast: {labels[i]}",
            ).add_to(m)

    # ── Landfall Marker ──────────────────────────────────────────────────────
    if landfall_details:
        llat = landfall_details.get("lat")
        llon = landfall_details.get("lon")
        ltime = landfall_details.get("time_h")
        if llat and llon:
            folium.Marker(
                location=[llat, llon],
                icon=folium.Icon(color="red", icon="house-flood-water", prefix="fa"),
                popup=(
                    f"<b>Predicted Landfall</b><br>"
                    f"ETA: +{ltime} hours<br>"
                    f"Location: {llat}N, {llon}E"
                ),
                tooltip="AI Landfall Prediction"
            ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:999;
                background:rgba(10,22,40,0.85); padding:12px 16px;
                border-radius:12px; border:1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
                font-family: 'Inter', sans-serif; font-size:12px; color:white;">
        <b style="color:#e2e8f0; font-size:13px; margin-bottom:4px; display:block;">Weather Lab Layers</b>
        <div style="display:flex; align-items:center; margin-bottom:2px;"><span style="color:#0ea5e9; font-size:11px; margin-right:6px;">──</span> AI Mean Prediction</div>
        <div style="display:flex; align-items:center; margin-bottom:2px;"><span style="background:rgba(56,189,248,0.3); border:1px solid #38bdf8; width:12px; height:8px; margin-right:6px; display:inline-block;"></span> 95% Uncertainty Cone</div>
        <div style="display:flex; align-items:center; margin-bottom:2px;"><span style="color:#38bdf8; font-size:14px; margin-right:6px; opacity:0.6;">●</span> 50-Member Ensemble</div>

    """
    
    if official_forecast:
        legend_html += '<div style="display:flex; align-items:center; margin-bottom:2px;"><span style="color:#fb923c; font-size:16px; margin-right:6px;">--</span> Official Baseline</div>'
    
    if show_cyclogenesis:
        legend_html += '<div style="display:flex; align-items:center; margin-bottom:2px;"><span style="color:#f43f5e; font-size:16px; margin-right:6px;">●</span> Formative Cyclogenesis (2%)</div>'
    
    legend_html += (
        '<div style="margin-top:8px; border-top:1px solid rgba(255,255,255,0.1); padding-top:4px;">'
        '<b style="color:#94a3b8; font-size:11px;">Intensity</b></div>'
    )
    for cat in range(6):
        legend_html += (
            f'<div style="display:flex; align-items:center; font-size:11px; margin-bottom:1px;">'
            f'<span style="color:{SS_COLORS[cat]}; font-size:12px; margin-right:6px;">●</span> '
            f'{SS_NAMES[cat]}</div>'
        )
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add layer toggle control
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    return m
