"""
Pydantic request/response schemas for the cyclone prediction API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TrackPoint(BaseModel):
    """A single observation point in a storm's track history."""
    lat: float = Field(..., description="Latitude (degrees)")
    lon: float = Field(..., description="Longitude (degrees)")
    wind: float = Field(0.0, description="Max sustained wind (knots)")
    pressure: Optional[float] = Field(None, description="Sea-level pressure (hPa)")
    dist2land: Optional[float] = Field(None, description="Distance to land (km)")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")


class PredictRequest(BaseModel):
    """
    Prediction request — provide recent track history (minimum 8 points, 6-hourly).
    """
    track_history: list[TrackPoint] = Field(
        ..., min_length=8,
        description="At least 8 sequential 6-hourly observations"
    )
    model_type: str = Field(
        "hybrid", description="Model to use: 'lstm' or 'hybrid'"
    )


class TrackForecast(BaseModel):
    """Predicted lat/lon at a specific forecast horizon."""
    lat: float
    lon: float


class PredictResponse(BaseModel):
    """Full prediction response with all forecast outputs."""
    model_type: str
    track: dict[str, TrackForecast] = Field(
        ..., description="Track forecasts at 24h, 48h, 72h"
    )
    wind: dict[str, float] = Field(
        ..., description="Wind speed forecasts (kt) at 24h, 48h, 72h"
    )
    ri_probability: float = Field(
        ..., description="Rapid intensification probability (0-1)"
    )
    landfall_probability: float = Field(
        ..., description="Landfall within 72h probability (0-1)"
    )
    ri_alert: bool = Field(
        ..., description="True if RI probability > 0.5"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: list[str]
    version: str
