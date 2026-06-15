from __future__ import annotations

from pydantic import BaseModel, Field


class TwilioRecordingPrediction(BaseModel):
    recording_url: str
    predicted_age_group: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: str


class TwilioRecordingLog(TwilioRecordingPrediction):
    call_sid: str | None = None
    recording_sid: str | None = None
    recording_duration: float | None = None
    confidence_level: str
    class_probabilities: dict[str, float]
    model_version: str
    processing_time_ms: int
