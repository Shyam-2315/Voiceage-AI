from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from xml.sax.saxutils import escape

import requests
from fastapi import HTTPException, Request, status

from app.core.config import settings
from app.schemas.twilio import TwilioRecordingLog, TwilioRecordingPrediction
from app.services.audio_service import AudioProcessingError, decode_audio_bytes
from app.services.model_service import model_service

try:
    from twilio.request_validator import RequestValidator
except ImportError:  # pragma: no cover - optional until twilio is installed
    RequestValidator = None


logger = logging.getLogger(__name__)


def public_url(path: str) -> str:
    if settings.public_base_url:
        return f"{settings.public_base_url.rstrip('/')}{path}"
    return path


def recording_callback_url() -> str:
    return public_url("/api/twilio/recording-complete")


def recording_action_url() -> str:
    return public_url("/api/twilio/recording-action")


def build_voice_twiml() -> str:
    callback_url = escape(recording_callback_url())
    action_url = escape(recording_action_url())
    max_length = settings.twilio_recording_max_seconds
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Say voice=\"alice\">"
        "VoiceAge AI will estimate an age group from your voice. "
        "After the beep, please speak naturally for ten to twenty seconds."
        "</Say>"
        f"<Record maxLength=\"{max_length}\" timeout=\"5\" playBeep=\"true\" "
        f"action=\"{action_url}\" method=\"POST\" "
        f"recordingStatusCallback=\"{callback_url}\" "
        'recordingStatusCallbackMethod="POST" '
        'recordingStatusCallbackEvent="completed" />'
        "</Response>"
    )


def build_recording_action_twiml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Say voice=\"alice\">Thank you. Your recording will now be analyzed.</Say>"
        "<Hangup />"
        "</Response>"
    )


async def validate_twilio_signature(request: Request, form_data: dict[str, Any]) -> None:
    if not settings.twilio_auth_token:
        logger.warning("TWILIO_AUTH_TOKEN is not set; skipping Twilio signature validation.")
        return
    if RequestValidator is None:
        logger.warning("twilio package is not installed; skipping Twilio signature validation.")
        return

    signature = request.headers.get("X-Twilio-Signature")
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing Twilio request signature.",
        )

    validator = RequestValidator(settings.twilio_auth_token)
    url = str(request.url)
    if settings.public_base_url:
        url = f"{settings.public_base_url.rstrip('/')}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

    if not validator.validate(url, form_data, signature):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Twilio request signature.",
        )


def media_url_for_recording(recording_url: str) -> str:
    parsed = urlparse(recording_url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid RecordingUrl received from Twilio.",
        )
    if Path(parsed.path).suffix.lower() in {".wav", ".mp3"}:
        return recording_url
    return urlunparse(parsed._replace(path=f"{parsed.path}.wav"))


def download_recording(recording_url: str) -> bytes:
    media_url = media_url_for_recording(recording_url)
    auth = None
    if settings.twilio_account_sid and settings.twilio_auth_token:
        auth = (settings.twilio_account_sid, settings.twilio_auth_token)

    try:
        response = requests.get(
            media_url,
            auth=auth,
            timeout=settings.twilio_download_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download Twilio recording: {exc}",
        ) from exc

    if not response.content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Twilio recording download returned an empty file.",
        )
    return response.content


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_log_token(value: str | None) -> str:
    if not value:
        return "recording"
    safe = "".join(char for char in value if char.isalnum() or char in {"_", "-"})
    return safe or "recording"


def save_prediction_log(payload: TwilioRecordingLog) -> Path:
    settings.twilio_predictions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_recording_sid = safe_log_token(payload.recording_sid)
    log_path = settings.twilio_predictions_dir / f"{timestamp}_{safe_recording_sid}.json"
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    else:
        data = payload.dict()
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return log_path


def analyze_recording(form_data: dict[str, Any]) -> TwilioRecordingPrediction:
    recording_url = str(form_data.get("RecordingUrl") or "").strip()
    if not recording_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing RecordingUrl in Twilio callback.",
        )

    content = download_recording(recording_url)
    try:
        audio = decode_audio_bytes(content, "wav")
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not process Twilio recording: {exc}",
        ) from exc

    prediction = model_service.predict(audio)
    timestamp = datetime.now(UTC).isoformat()
    log_payload = TwilioRecordingLog(
        recording_url=recording_url,
        predicted_age_group=prediction.predicted_age_group,
        confidence=prediction.confidence,
        timestamp=timestamp,
        call_sid=form_data.get("CallSid"),
        recording_sid=form_data.get("RecordingSid"),
        recording_duration=parse_float(form_data.get("RecordingDuration")),
        confidence_level=prediction.confidence_level,
        class_probabilities=prediction.class_probabilities,
        model_version=prediction.model_version,
        processing_time_ms=prediction.processing_time_ms,
    )
    log_path = save_prediction_log(log_payload)
    logger.info("Saved Twilio prediction log: %s", log_path)

    return TwilioRecordingPrediction(
        recording_url=recording_url,
        predicted_age_group=prediction.predicted_age_group,
        confidence=prediction.confidence,
        timestamp=timestamp,
    )
