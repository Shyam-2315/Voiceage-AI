from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from threading import Lock
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException, status
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app.core.config import settings
from app.schemas.prediction import PredictionResponse


logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.feature_extractor: Any | None = None
        self.model: Any | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = self.device.type == "cuda"
        self.load_error: str | None = None
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.feature_extractor is not None

    def load_model(self) -> None:
        if self.is_loaded:
            return

        with self._lock:
            if self.is_loaded:
                return

            try:
                if not settings.model_path.exists():
                    raise FileNotFoundError(f"Model path not found: {settings.model_path}")

                logger.info("Loading Wav2Vec2 model from %s", settings.model_path)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(settings.model_path)
                self.model = AutoModelForAudioClassification.from_pretrained(settings.model_path)
                self._validate_label_mapping()
                self.model.to(self.device)
                self.model.eval()

                if self.device.type == "cuda":
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.cuda.empty_cache()

                self.load_error = None
                logger.info("Model loaded on %s; fp16=%s", self.device, self.fp16)
            except Exception as exc:
                self.feature_extractor = None
                self.model = None
                self.load_error = str(exc)
                logger.exception("Model loading failed")
                raise

    def _validate_label_mapping(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        raw_id2label = getattr(self.model.config, "id2label", {}) or {}
        id2label = {int(idx): str(label) for idx, label in raw_id2label.items()}
        expected = {idx: label for idx, label in enumerate(settings.class_labels)}
        if id2label and id2label != expected:
            raise ValueError(f"Model labels do not match API labels. Expected {expected}, found {id2label}")

    def ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        try:
            self.load_model()
        except Exception as exc:
            detail = f"Model loading error: {self.load_error or exc}"
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from exc

    def predict(self, audio: np.ndarray) -> PredictionResponse:
        self.ensure_loaded()
        if self.model is None or self.feature_extractor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available.",
            )

        start = time.perf_counter()
        inputs = self.feature_extractor(
            audio,
            sampling_rate=settings.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self.fp16
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_context:
                logits = self.model(**model_inputs).logits

        probabilities = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()[0]
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        class_probabilities = {
            label: float(probabilities[idx])
            for idx, label in enumerate(settings.class_labels)
        }
        processing_time_ms = int(round((time.perf_counter() - start) * 1000))

        return PredictionResponse(
            predicted_age_group=settings.class_labels[predicted_idx],
            confidence=confidence,
            confidence_level=confidence_level(confidence),
            class_probabilities=class_probabilities,
            model_version=settings.model_version,
            processing_time_ms=processing_time_ms,
        )


def confidence_level(confidence: float) -> str:
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"


model_service = ModelService()
