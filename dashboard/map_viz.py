"""
Storm track map visualization with Folium.
"""

import folium
from folium import plugins
import numpy as np


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
                     storm_name: str = "Storm") -> folium.Map:
    """
    Render a Folium map showing storm track and optional forecast.

    Args:
        track_lat: list of historical latitudes
        track_lon: list of historical longitudes
        track_wind: list of wind speeds (knots) for coloring
        forecast: dict with "24h", "48h", "72h" keys, each {"lat", "lon"}
        storm_name: name for the popup

    Returns:
        folium.Map object
    """
    # Center map on the latest position
    center_lat = track_lat[-1] if track_lat else 15.0
    center_lon = track_lon[-1] if track_lon else 80.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB dark_matter",
    )

    # ── Historical track ─────────────────────────────────────────────────────
    if track_wind is None:
        track_wind = [0.0] * len(track_lat)

    # Draw track segments colored by intensity
    for i in range(1, len(track_lat)):
        cat = _wind_to_category(track_wind[i])
        folium.PolyLine(
            locations=[
                [track_lat[i-1], track_lon[i-1]],
                [track_lat[i], track_lon[i]],
            ],
            color=SS_COLORS[cat],
            weight=3,
            opacity=0.8,
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

    # ── Forecast track (dashed) ──────────────────────────────────────────────
    if forecast and track_lat:
        forecast_lats = [track_lat[-1]]
        forecast_lons = [track_lon[-1]]
        labels = ["Now", "24h", "48h", "72h"]

        for horizon in ["24h", "48h", "72h"]:
            if horizon in forecast:
                forecast_lats.append(forecast[horizon]["lat"])
                forecast_lons.append(forecast[horizon]["lon"])

        # Dashed forecast line
        folium.PolyLine(
            locations=list(zip(forecast_lats, forecast_lons)),
            color="#fbbf24",
            weight=2,
            opacity=0.7,
            dash_array="8 4",
        ).add_to(m)

        # Forecast point markers
        for i in range(1, len(forecast_lats)):
            folium.CircleMarker(
                location=[forecast_lats[i], forecast_lons[i]],
                radius=6,
                color="#fbbf24",
                fill=True,
                fill_color="#fbbf24",
                fill_opacity=0.7,
                popup=f"Forecast: {labels[i]}",
            ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:999;
                background:rgba(10,22,40,0.9); padding:12px 16px;
                border-radius:8px; border:1px solid #1e3a5f;
                font-family:monospace; font-size:12px; color:white;">
        <b>Intensity</b><br>
    """
    for cat in range(6):
        legend_html += (
            f'<span style="color:{SS_COLORS[cat]}">●</span> '
            f'{SS_NAMES[cat]}<br>'
        )
    legend_html += (
        '<span style="color:#fbbf24">- - -</span> Forecast'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
