# 🌀 Cyclone Prediction — NI Basin

AI-powered cyclone track, intensity, and rapid intensification forecasting for the North Indian Basin.

## Architecture

```
Cyclone AI/
├── src/                    # Reusable Python modules
│   ├── data_loader.py      #   IBTrACS download, clean, feature engineering
│   ├── features.py         #   Feature/label definitions, PyTorch Datasets
│   ├── model.py            #   CycloneLSTM + HybridCycloneModel
│   ├── losses.py           #   Haversine loss, focal BCE, multi-task loss
│   ├── metrics.py          #   Track error, wind MAE, RI F1
│   ├── train.py            #   Training loop helpers
│   └── predict.py          #   Inference helpers
├── models/                 # Model registry
│   ├── lstm.py             #   LSTM baseline factory
│   ├── hybrid.py           #   Hybrid model factory
│   └── full.py             #   Unified build_model()
├── api/                    # FastAPI backend
│   ├── main.py             #   App + routes (/health, /predict, /models)
│   ├── schemas.py          #   Pydantic request/response models
│   └── inference.py        #   Model loading + prediction server
├── dashboard/              # Streamlit dashboard
│   ├── app.py              #   Main app with dark theme
│   ├── map_viz.py          #   Folium storm track map
│   └── charts.py           #   Plotly charts (wind, RI gauge, comparison)
├── notebooks/              # Original development notebooks
│   ├── 01_data_loader.py   #   Data pipeline
│   ├── 02_eda.py           #   Exploratory data analysis
│   ├── 03_dataset.py       #   Dataset verification
│   ├── 04_train.py         #   LSTM baseline training
│   ├── 05_era5_download.py #   ERA5 data download
│   └── 06_train_hybrid.py  #   Hybrid model training
├── data/                   # Data directory
│   ├── raw/                #   Raw IBTrACS CSV
│   ├── processed/          #   Train/val/test splits (.npy, .csv)
│   ├── era5/               #   ERA5 NetCDF files
│   ├── era5_patches/       #   Extracted ERA5 patches
│   └── eda_plots/          #   EDA visualizations
├── checkpoints/            # Trained model weights
├── experiments/            # Training logs & test results
├── config.yaml             # Centralized configuration
└── requirements.txt        # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

### Additional dashboard dependencies

```bash
pip install streamlit streamlit-folium folium plotly
```

## Run Order

### 1. Data pipeline (already completed)

```bash
cd notebooks
python 01_data_loader.py   # Download, clean, feature engineering → data/processed/
python 02_eda.py           # EDA plots → data/eda_plots/
python 03_dataset.py       # Verify PyTorch Dataset
```

### 2. Model training (already completed)

```bash
python 04_train.py         # LSTM baseline → checkpoints/best.pt
python 05_era5_download.py # ERA5 data → data/era5/
python 06_train_hybrid.py  # Hybrid model → checkpoints/hybrid_best.pt
```

### 3. Start the API

```bash
cd "d:/Cyclone AI"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 4. Launch the dashboard

```bash
cd "d:/Cyclone AI"
python -m streamlit run dashboard/app.py
```

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "track_history": [
      {"lat": 12.0, "lon": 80.0, "wind": 25},
      {"lat": 12.5, "lon": 79.5, "wind": 30},
      {"lat": 13.0, "lon": 79.0, "wind": 35},
      {"lat": 13.6, "lon": 78.4, "wind": 40},
      {"lat": 14.2, "lon": 77.8, "wind": 50},
      {"lat": 14.9, "lon": 77.1, "wind": 55},
      {"lat": 15.5, "lon": 76.5, "wind": 60},
      {"lat": 16.0, "lon": 76.0, "wind": 65}
    ],
    "model_type": "hybrid"
  }'
```

## Data Description

| File | Shape | Description |
|------|-------|-------------|
| `train_X.npy` | (N, 8, 17) | Training sequences |
| `train_y.npy` | (N, 12) | Training labels |
| `val_X.npy`   | (N, 8, 17) | Validation sequences |
| `val_y.npy`   | (N, 12) | Validation labels |
| `test_X.npy`  | (N, 8, 17) | Test sequences |
| `test_y.npy`  | (N, 12) | Test labels |
| `scaler_mean.npy` | (1,1,17) | Feature mean |
| `scaler_std.npy`  | (1,1,17) | Feature std |

## Label Index Map

| Index | Label | Type |
|-------|-------|------|
| 0–1 | lat_24h, lon_24h | regression |
| 2–3 | lat_48h, lon_48h | regression |
| 4–5 | lat_72h, lon_72h | regression |
| 6–8 | wind_24h/48h/72h | regression |
| 9   | SS_cat (0–5)     | classification |
| 10  | RI_label         | binary |
| 11  | landfall_72h     | binary |

## Feature Columns (dim=17)

LAT, LON, WMO_WIND, WMO_PRES, dLAT, dLON, dLAT_2, dLON_2,
dWIND, dPRES, spd_kmh, dist2land, lat_abs,
month_sin, month_cos, jday_sin, jday_cos

## Models

| Model | Architecture | Track 24h | Wind MAE |
|-------|-------------|-----------|----------|
| LSTM Baseline | 2-layer LSTM + task heads | ~200 km | ~15 kt |
| Hybrid | LSTM + ConvLSTM (ERA5) + Transformer | ~180 km | ~14 kt |

## Temporal Split

- **Train**: before 2018
- **Val**: 2018–2019
- **Test**: 2020 onwards