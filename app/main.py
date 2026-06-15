from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes.predict import router as predict_router
from app.api.routes.twilio import router as twilio_router
from app.core.config import settings
from app.schemas.prediction import HealthResponse
from app.services.model_service import model_service


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    version=settings.model_version,
    description="VoiceAge AI Wav2Vec2 inference API.",
)
app.include_router(predict_router)
app.include_router(twilio_router)


@app.on_event("startup")
def startup_event() -> None:
    try:
        model_service.load_model()
    except Exception:
        logging.getLogger(__name__).exception("Startup model load failed; prediction endpoint will return 503.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        model_version=settings.model_version,
        device=str(model_service.device),
    )
