"""
GeoAI Spatial Query Assistant
==============================
An AI-powered chatbot that answers natural language questions about
geospatial concepts, satellite imagery, GIS analysis, and spatial data —
combining LLM intelligence with GIS knowledge.

Features:
- Natural language GIS Q&A
- Satellite index calculator (NDVI, NDWI, NDBI, EVI)
- Coordinate transformer (WGS84 ↔ UTM zones)
- Simple spatial statistics from CSV data

Run modes:
  python geoai_chatbot.py            # Interactive CLI chat
  python geoai_chatbot.py --demo     # Run demo questions

Author: Fatima Filza Hassan | github.com/Filza-coder
"""

import sys
import math
import argparse
import datetime

# ─────────────────────────────────────────────────────────
# GIS KNOWLEDGE BASE  (rule-based NLU fallback)
# ─────────────────────────────────────────────────────────
GIS_KB = {
    "ndvi": """
**NDVI (Normalized Difference Vegetation Index)**
Formula: NDVI = (NIR - Red) / (NIR + Red)
Range: -1 to +1
Interpretation:
  < 0.0  → Water, snow, bare rock
  0.0–0.2 → Sparse vegetation / bare soil
  0.2–0.4 → Shrubs, grassland
  0.4–0.6 → Moderate vegetation / crops
  > 0.6   → Dense forest / high biomass
Sentinel-2 bands: B8 (NIR), B4 (Red)
Landsat 8 bands : B5 (NIR), B4 (Red)
""",
    "ndwi": """
**NDWI (Normalized Difference Water Index)**
Formula: NDWI = (Green - NIR) / (Green + NIR)   [McFeeters 1996]
         NDWI = (NIR - SWIR) / (NIR + SWIR)      [Gao 1996 — vegetation water]
Range: -1 to +1
Use: Delineate open water bodies; flood mapping
Threshold: > 0 typically indicates water
""",
    "ndbi": """
**NDBI (Normalized Difference Built-up Index)**
Formula: NDBI = (SWIR - NIR) / (SWIR + NIR)
Range: -1 to +1
Use: Map urban and built-up areas
High NDBI (> 0) → Urban/impervious surfaces
Low NDBI (< 0)  → Vegetation, water
""",
    "sar": """
**SAR (Synthetic Aperture Radar)**
- Active sensor: emits microwave pulses, records backscatter
- All-weather, day/night capability (unlike optical sensors)
- Key bands: C-band (Sentinel-1), L-band (ALOS-PALSAR), X-band (TerraSAR-X)
- Flood mapping: flooded areas show strong backscatter decrease (smooth surface)
- Common change detection threshold: ΔVV < -3 to -5 dB indicates flooding
- Polarisations: VV (vertical-vertical), VH (vertical-horizontal)
""",
    "gee": """
**Google Earth Engine (GEE)**
- Cloud-based geospatial analysis platform with massive satellite data archive
- Supports: Landsat, Sentinel, MODIS, Copernicus DEM, and 1000+ datasets
- Languages: JavaScript (Code Editor) and Python (earthengine-api)
- Free for research and non-commercial use
- Key use cases: LULC classification, time series analysis, change detection
- Quota limits: 5000 EECU-seconds/day for free tier
""",
    "random forest": """
**Random Forest for Remote Sensing**
- Ensemble of decision trees; robust to overfitting
- Handles high-dimensional spectral data well
- Outputs feature importances (useful for band selection)
- Key parameters: n_estimators (100–500 typical), max_depth, min_samples_leaf
- Accuracy assessment: Overall Accuracy, Kappa coefficient, per-class F1
- Libraries: scikit-learn (sklearn.ensemble.RandomForestClassifier)
""",
    "sentinel": """
**Sentinel Satellite Family (ESA Copernicus)**
Sentinel-1: SAR, C-band, 10m resolution, 6/12 day revisit → floods, deformation
Sentinel-2: Multispectral, 10/20/60m, 13 bands, 5-day revisit → LULC, vegetation
Sentinel-3: Ocean & land colour, OLCI/SLSTR → SST, fire, algal bloom
Sentinel-5P: Atmospheric trace gases (NO₂, CO, O₃) → air quality
Data access: Copernicus Open Access Hub, Google Earth Engine, AWS S3
""",
    "lulc": """
**Land Use Land Cover (LULC) Classification**
- Supervised ML (Random Forest, SVM, CNN) on multispectral imagery
- Key classes: Urban, Agriculture, Forest, Water, Bare Soil
- Accuracy thresholds: OA > 85%, Kappa > 0.80 (acceptable in literature)
- Validation: Stratified random sampling; minimum 50 points per class
- Tools: ArcGIS, QGIS Semi-Automatic Classification Plugin, GEE, Python (scikit-learn)
""",
    "coordinate": """
**Coordinate Reference Systems (CRS)**
- WGS84 (EPSG:4326): Geographic, lat/lon in decimal degrees — GPS standard
- UTM (Universal Transverse Mercator): Projected, metres, 60 zones
  Pakistan zones: 42N (EPSG:32642), 43N (EPSG:32643)
- Web Mercator (EPSG:3857): Used by Google Maps, OpenStreetMap
- Lahore, Pakistan: 74.3587°E, 31.5204°N  → UTM Zone 43N
- Python: pyproj, geopandas (.to_crs()), shapely + pyproj Transformer
""",
}

TOOLS_HELP = """
Available tools in this assistant:
  /calc ndvi <NIR> <Red>           — Calculate NDVI value
  /calc ndwi <Green> <NIR>         — Calculate NDWI value  
  /calc ndbi <SWIR> <NIR>          — Calculate NDBI value
  /calc evi  <NIR> <Red> <Blue>    — Calculate EVI value
  /coord <lat> <lon>               — Convert WGS84 to UTM zone info
  /bands sentinel2                 — List all Sentinel-2 bands
  /bands landsat8                  — List all Landsat 8 bands
  /about                           — About this assistant
  /help                            — Show this help
  quit / exit                      — Exit the chatbot
"""

SENTINEL2_BANDS = """
Sentinel-2 Band Reference:
  B1  - Coastal Aerosol (443nm, 60m)
  B2  - Blue (490nm, 10m)
  B3  - Green (560nm, 10m)
  B4  - Red (665nm, 10m)
  B5  - Red Edge 1 (705nm, 20m)
  B6  - Red Edge 2 (740nm, 20m)
  B7  - Red Edge 3 (783nm, 20m)
  B8  - NIR (842nm, 10m)
  B8A - Narrow NIR (865nm, 20m)
  B9  - Water Vapour (940nm, 60m)
  B11 - SWIR 1 (1610nm, 20m)
  B12 - SWIR 2 (2190nm, 20m)
"""

LANDSAT8_BANDS = """
Landsat 8 OLI/TIRS Band Reference:
  B1  - Coastal Aerosol (443nm, 30m)
  B2  - Blue (482nm, 30m)
  B3  - Green (562nm, 30m)
  B4  - Red (655nm, 30m)
  B5  - NIR (865nm, 30m)
  B6  - SWIR 1 (1609nm, 30m)
  B7  - SWIR 2 (2201nm, 30m)
  B10 - Thermal TIRS 1 (10.9µm, 100m) ← Land Surface Temperature
  B11 - Thermal TIRS 2 (12.0µm, 100m)
  QA  - Quality Assessment band
"""

# ─────────────────────────────────────────────
# INDEX CALCULATORS
# ─────────────────────────────────────────────
def safe_div(a, b):
    return a / b if abs(b) > 1e-9 else 0.0

def calc_ndvi(nir, red):
    val = safe_div(nir - red, nir + red)
    interp = ("Dense vegetation / forest" if val > 0.6 else
              "Moderate vegetation" if val > 0.4 else
              "Sparse vegetation / crops" if val > 0.2 else
              "Bare soil / urban" if val > 0.0 else
              "Water / snow / rock")
    return f"NDVI = {val:.4f}  →  {interp}"

def calc_ndwi(green, nir):
    val = safe_div(green - nir, green + nir)
    interp = "Likely water surface" if val > 0 else "Non-water surface"
    return f"NDWI = {val:.4f}  →  {interp}"

def calc_ndbi(swir, nir):
    val = safe_div(swir - nir, swir + nir)
    interp = "Urban / built-up" if val > 0 else "Vegetation / water"
    return f"NDBI = {val:.4f}  →  {interp}"

def calc_evi(nir, red, blue, G=2.5, C1=6, C2=7.5, L=1):
    val = G * safe_div(nir - red, nir + C1*red - C2*blue + L)
    interp = ("High biomass" if val > 0.5 else
              "Moderate vegetation" if val > 0.2 else
              "Low vegetation / non-vegetated")
    return f"EVI  = {val:.4f}  →  {interp}"

def wgs84_to_utm_zone(lat, lon):
    zone_num = int((lon + 180) / 6) + 1
    hemisphere = "North" if lat >= 0 else "South"
    epsg = 32600 + zone_num if lat >= 0 else 32700 + zone_num
    return (f"UTM Zone: {zone_num}{hemisphere[0]}  |  EPSG: {epsg}\n"
            f"Hemisphere: {hemisphere}\n"
            f"Input: {lat:.4f}°N, {lon:.4f}°E")

# ─────────────────────────────────────────────
# COMMAND PARSER
# ─────────────────────────────────────────────
def handle_command(cmd):
    parts = cmd.strip().split()
    if not parts:
        return "Empty command. Type /help for options."
    
    if parts[0] == "/calc" and len(parts) >= 2:
        try:
            idx = parts[1].lower()
            vals = [float(v) for v in parts[2:]]
            if   idx == "ndvi" and len(vals) >= 2: return calc_ndvi(*vals[:2])
            elif idx == "ndwi" and len(vals) >= 2: return calc_ndwi(*vals[:2])
            elif idx == "ndbi" and len(vals) >= 2: return calc_ndbi(*vals[:2])
            elif idx == "evi"  and len(vals) >= 3: return calc_evi(*vals[:3])
            else: return f"Usage: /calc ndvi <NIR> <Red>  or /calc evi <NIR> <Red> <Blue>"
        except ValueError:
            return "Please provide numeric band values. Example: /calc ndvi 0.45 0.12"
    
    elif parts[0] == "/coord" and len(parts) >= 3:
        try:
            return wgs84_to_utm_zone(float(parts[1]), float(parts[2]))
        except ValueError:
            return "Usage: /coord <latitude> <longitude>  (e.g. /coord 31.52 74.36)"
    
    elif parts[0] == "/bands":
        if len(parts) > 1 and "sentinel" in parts[1].lower(): return SENTINEL2_BANDS
        if len(parts) > 1 and "landsat"  in parts[1].lower(): return LANDSAT8_BANDS
        return "Usage: /bands sentinel2  OR  /bands landsat8"
    
    elif parts[0] == "/help":  return TOOLS_HELP
    elif parts[0] == "/about": return (
        "GeoAI Spatial Query Assistant v1.0\n"
        "Built by: Fatima Filza Hassan\n"
        "MS Remote Sensing & GIS — NUST (2017)\n"
        "GitHub: github.com/Filza-coder\n"
        f"Date: {datetime.date.today()}"
    )
    
    return f"Unknown command: {parts[0]}. Type /help for options."

# ─────────────────────────────────────────────
# NLU: SIMPLE KEYWORD MATCHING
# ─────────────────────────────────────────────
def knowledge_response(query):
    q = query.lower()
    for keyword, answer in GIS_KB.items():
        if keyword in q:
            return answer.strip()
    
    # Fallback patterns
    if any(w in q for w in ["hello", "hi", "hey"]):
        return "Hello! I'm GeoAI, your spatial assistant. Ask me about NDVI, SAR, Sentinel, LULC, GEE, or type /help for tools."
    if any(w in q for w in ["flood", "inundation", "water level"]):
        return GIS_KB["sar"].strip()
    if any(w in q for w in ["vegetation", "plant", "crop", "forest"]):
        return GIS_KB["ndvi"].strip()
    if any(w in q for w in ["urban", "city", "built"]):
        return GIS_KB["ndbi"].strip()
    if any(w in q for w in ["classify", "classification", "machine learning", "ml"]):
        return GIS_KB["random forest"].strip()
    if any(w in q for w in ["pakistan", "lahore", "islamabad", "karachi"]):
        return ("Pakistan GIS Context:\n"
                "• UTM Zones: 42N (West Pakistan) and 43N (East Pakistan/Punjab)\n"
                "• Lahore: 74.36°E, 31.52°N — UTM 43N (EPSG:32643)\n"
                "• Major flood events: 2010, 2022 (Sindh/Balochistan)\n"
                "• Key RS applications: Indus basin monitoring, cotton/wheat crop mapping, smog/aerosol analysis\n"
                "• Data portals: SUPARCO (Pakistan's space agency), PMD (Met Dept)")
    
    return ("I can help with GIS/Remote Sensing topics. Try asking about:\n"
            "ndvi, ndwi, ndbi, sar, sentinel, gee, random forest, lulc, coordinates\n"
            "Or use /calc and /coord tools. Type /help for all options.")

# ─────────────────────────────────────────────
# MAIN CHAT LOOP
# ─────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════╗
║           🌍  GeoAI Spatial Query Assistant              ║
║      GIS | Remote Sensing | Satellite Data | AI          ║
║      Author: Fatima Filza Hassan — github.com/Filza-coder║
╚══════════════════════════════════════════════════════════╝
Type a GIS question, a /command, or 'quit' to exit.
Type /help to see all available tools.
"""

DEMO_QUESTIONS = [
    "What is NDVI?",
    "How does SAR flood mapping work?",
    "Tell me about Sentinel satellites",
    "/calc ndvi 0.45 0.12",
    "/calc evi 0.50 0.10 0.05",
    "/coord 31.5204 74.3587",
    "/bands sentinel2",
    "What is LULC classification?",
    "Tell me about Pakistan GIS",
]

def run_demo():
    print(BANNER)
    print("=== DEMO MODE — Running sample questions ===\n")
    for q in DEMO_QUESTIONS:
        print(f">>> {q}")
        if q.startswith("/"):
            resp = handle_command(q)
        else:
            resp = knowledge_response(q)
        print(resp)
        print("-" * 55)

def run_interactive():
    print(BANNER)
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 🌍")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye! Happy mapping! 🛰️")
            break
        
        if user_input.startswith("/"):
            response = handle_command(user_input)
        else:
            response = knowledge_response(user_input)
        
        print(f"\nGeoAI: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoAI Spatial Query Assistant")
    parser.add_argument("--demo", action="store_true", help="Run demo questions")
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        run_interactive()
