"""
Cyclone Prediction API — FastAPI backend.

Run:
    cd d:/Cyclone AI
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    PredictRequest, PredictResponse, HealthResponse, TrackForecast,
)
from api.inference import ModelServer

# ─── Global model server ─────────────────────────────────────────────────────
server: ModelServer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global server
    print("\n" + "=" * 50)
    print("  CYCLONE PREDICTION API — Starting")
    print("=" * 50 + "\n")
    server = ModelServer(
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints"),
        processed_dir=str(PROJECT_ROOT / "data" / "processed"),
    )
    print(f"\n  Available models: {server.available_models}")
    print("=" * 50 + "\n")
    yield
    print("\n[API] Shutting down")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cyclone Prediction API",
    description="NI Basin cyclone track, intensity, and RI prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — verify the API is running and models are loaded."""
    return HealthResponse(
        status="ok",
        models_loaded=server.available_models if server else [],
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run cyclone prediction from track history.

    Provide at least 8 sequential 6-hourly observations.
    Returns track forecast (24/48/72h), wind speed, RI probability,
    and landfall probability.
    """
    if server is None or not server.available_models:
        raise HTTPException(status_code=503, detail="No models loaded")

    if request.model_type not in server.available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' not available. "
                   f"Choose from: {server.available_models}"
        )

    # Convert Pydantic models to dicts
    track_dicts = [
        {
            "lat": pt.lat,
            "lon": pt.lon,
            "wind": pt.wind,
            "pressure": pt.pressure,
            "dist2land": pt.dist2land,
            "timestamp": pt.timestamp,
        }
        for pt in request.track_history
    ]

    try:
        result = server.predict(track_dicts, model_type=request.model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictResponse(
        model_type=result["model_type"],
        track={
            k: TrackForecast(**v) for k, v in result["track"].items()
        },
        wind=result["wind"],
        ri_probability=result["ri_probability"],
        landfall_probability=result["landfall_probability"],
        ri_alert=result["ri_alert"],
    )


@app.get("/models")
async def list_models():
    """List available models and their info."""
    return {
        "models": server.available_models if server else [],
        "default": "hybrid" if "hybrid" in (server.available_models if server else []) else "lstm",
    }
