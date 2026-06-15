from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_age_group: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: str
    class_probabilities: dict[str, float]
    model_version: str
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    device: str | None = None
