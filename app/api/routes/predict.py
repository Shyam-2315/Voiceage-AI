from __future__ import annotations

import time

from fastapi import APIRouter, File, UploadFile

from app.schemas.prediction import PredictionResponse
from app.services.audio_service import preprocess_upload
from app.services.model_service import model_service


router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict-age", response_model=PredictionResponse)
async def predict_age(file: UploadFile = File(...)) -> PredictionResponse:
    start = time.perf_counter()
    audio = await preprocess_upload(file)
    response = model_service.predict(audio)
    total_ms = int(round((time.perf_counter() - start) * 1000))
    return response.copy(update={"processing_time_ms": total_ms})
