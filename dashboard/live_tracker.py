import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour to avoid spamming the API
def fetch_live_storms():
    """
    Fetches currently active Tropical Cyclones from the Global Disaster 
    Alert and Coordination System (GDACS).
    
    Returns:
        dict: { "Storm Name": [{"lat": ..., "lon": ..., "wind": ..., "pressure": ...}, ...], ... }
    """
    try:
        # 1. Fetch GDACS RSS feed
        rss_url = "https://gdacs.org/xml/rss.xml"
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        # Namespaces in GDACS XML
        ns = {"gdacs": "http://www.gdacs.org"}
        
        active_storms = {}
        
        for item in root.findall(".//item"):
            event_type = item.find("gdacs:eventtype", ns)
            if event_type is not None and event_type.text == "TC":
                event_id = item.find("gdacs:eventid", ns).text
                storm_name = item.find("gdacs:eventname", ns)
                storm_name = storm_name.text if storm_name is not None else f"Cyclone {event_id}"
                
                # Fetch detailed GeoJSON for the storm track
                geojson_url = f"https://www.gdacs.org/datareport/resources/TC/{event_id}/geojson_{event_id}.geojson"
                try:
                    geo_resp = requests.get(geojson_url, timeout=10)
                    if geo_resp.status_code == 200:
                        geo_data = geo_resp.json()
                        track_points = []
                        
                        # Parse the GeoJSON features
                        for feature in geo_data.get("features", []):
                            props = feature.get("properties", {})
                            geom = feature.get("geometry", {})
                            
                            # We only want actual observed/historical track points, not the long-term forecast points
                            # GDACS features have "tracktype": "past" or "forecast"
                            if props.get("tracktype", "").lower() in ["past", "observed", ""]:
                                if geom.get("type") == "Point":
                                    coords = geom.get("coordinates", [0, 0])
                                    lon, lat = coords[0], coords[1]
                                    
                                    # KMH to Knots conversion roughly = kmh / 1.852
                                    wind_kmh = props.get("stormwind", 0)
                                    wind_kt = wind_kmh / 1.852 if wind_kmh else 0
                                    
                                    pressure = props.get("centralpressure")
                                    
                                    # Try to parse date string
                                    date_str = props.get("todate", "")
                                    
                                    track_points.append({
                                        "lat": float(lat),
                                        "lon": float(lon),
                                        "wind": float(wind_kt),
                                        "pressure": float(pressure) if pressure else 1000.0,
                                        "timestamp": date_str
                                    })
                                    
                        # Sort chronologically
                        if track_points:
                            track_points.reverse() # GeoJSON usually newest first
                            
                            # Since we need exactly 8 points (48 hours), grab the last 8
                            if len(track_points) >= 8:
                                active_storms[storm_name] = track_points[-8:]
                            else:
                                # Interpolate or pad if less than 8
                                # For live demo, we pad the missing history with the earliest known point
                                padded = track_points.copy()
                                while len(padded) < 8:
                                    padded.insert(0, padded[0])
                                active_storms[storm_name] = padded
                                
                except Exception as e:
                    print(f"Failed to fetch GeoJSON for {storm_name}: {e}")
                    continue
                    
        return active_storms

    except Exception as e:
        print(f"GDACS Fetch Error: {e}")
        return {}

def interpolate_live_track(track_points):
    """Ensure exactly 8 points are returned by padding if needed."""
    if not track_points:
        return []
    if len(track_points) >= 8:
        return track_points[-8:]
        
    padded = track_points.copy()
    while len(padded) < 8:
        padded.insert(0, padded[0])
    return padded
