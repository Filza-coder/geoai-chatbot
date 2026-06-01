# 🤖 GeoAI Spatial Query Assistant — Chatbot

> **Natural language chatbot for GIS, remote sensing, and spatial analysis queries**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python)](https://python.org)
[![No Dependencies](https://img.shields.io/badge/Dependencies-None%20required-brightgreen?style=flat)]()

---

## 📌 Overview

A command-line AI assistant that answers **natural language questions about geospatial topics** — covering satellite indices, sensor specifications, classification methods, coordinate systems, and Pakistan-specific GIS context.

Runs entirely in Python standard library (no external APIs needed), with optional LLM integration.

---

## 💬 Capabilities

### Natural Language Q&A
Ask about: NDVI, NDWI, NDBI, SAR, Sentinel-1/2, Landsat 8, Google Earth Engine, Random Forest, LULC, coordinate systems, Pakistan GIS

### Built-in Tools
```
/calc ndvi <NIR> <Red>        → Calculates NDVI + interpretation
/calc ndwi <Green> <NIR>      → Calculates NDWI + interpretation  
/calc ndbi <SWIR> <NIR>       → Calculates NDBI + interpretation
/calc evi  <NIR> <Red> <Blue> → Calculates EVI + interpretation
/coord <lat> <lon>            → WGS84 → UTM zone lookup
/bands sentinel2              → Full S2 band reference
/bands landsat8               → Full Landsat 8 band reference
```

---

## 🚀 Run

```bash
# Interactive mode
python geoai_chatbot.py

# Demo mode (runs 10 sample questions)
python geoai_chatbot.py --demo
```

### Example Session
```
You: What is NDVI?
GeoAI: NDVI = (NIR - Red) / (NIR + Red) ...

You: /calc ndvi 0.45 0.12
GeoAI: NDVI = 0.5789  →  Moderate vegetation / crops

You: /coord 31.5204 74.3587
GeoAI: UTM Zone: 43N  |  EPSG: 32643
```

---

## 🔮 Extensions

- [ ] Connect to OpenAI / Claude API for full LLM responses
- [ ] Add real-time satellite data queries via GEE Python API
- [ ] Build web interface with Flask/Streamlit
- [ ] Add spatial query execution with GeoPandas

---

## 📦 Requirements

```
No external dependencies — uses Python standard library only
```

---

## 👩‍🔬 Author

**Fatima Filza Hassan** — MS GIS & Remote Sensing, NUST | [Portfolio](https://Filza-coder.github.io)
