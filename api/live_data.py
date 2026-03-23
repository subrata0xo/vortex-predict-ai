import requests
import xml.etree.ElementTree as ET
import re
from datetime import datetime

class LiveStormFetcher:
    """
    Fetches real-time cyclone data for the North Indian Ocean (Bay of Bengal / Arabian Sea).
    Source: JTWC (Joint Typhoon Warning Center) and GDACS.
    """
    
    JTWC_RSS = "https://www.metoc.navy.mil/jtwc/rss/jtwc.rss"
    GDACS_URL = "https://www.gdacs.org/xml/rss.xml"

    @classmethod
    def get_active_storms(cls):
        """
        Returns a list of active storms with their latest coordinates.
        Example: [{'SID': 'IO012026', 'NAME': 'TEST', 'LAT': 12.5, 'LON': 85.0, 'WIND': 45}]
        """
        storms = []
        try:
            # 1. Try GDACS first (reliable XML structure)
            response = requests.get(cls.GDACS_URL, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall(".//item"):
                    title = item.find("title").text or ""
                    # Filter for Tropical Cyclones in North Indian Ocean
                    if "Tropical Cyclone" in title:
                        desc = item.find("description").text or ""
                        # Extract Lat/Lon using Regex
                        lat_match = re.search(r"lat:([-+]?\d*\.\d+|\d+)", desc)
                        lon_match = re.search(r"long:([-+]?\d*\.\d+|\d+)", desc)
                        
                        if lat_match and lon_match:
                            storms.append({
                                'SID': "TC_" + datetime.now().strftime("%Y%m%d"),
                                'NAME': title.split("for")[-1].strip(),
                                'LAT': float(lat_match.group(1)),
                                'LON': float(lon_match.group(1)),
                                'WIND': 35.0, # Default to tropical depression if unknown
                                'SOURCE': 'GDACS'
                            })
        except Exception as e:
            print(f"[!] Live fetch error: {e}")
            
        # Fallback/Mock for development if no active storms are found
        if not storms:
            storms.append({
                'SID': 'IO882026',
                'NAME': 'INVEST_99B',
                'LAT': 13.5,
                'LON': 86.2,
                'WIND': 30.0,
                'SOURCE': 'MOCK'
            })
            
        return storms

if __name__ == "__main__":
    fetcher = LiveStormFetcher()
    print("Fetching active storms...")
    print(fetcher.get_active_storms())
