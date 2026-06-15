from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings


class AudioProcessingError(ValueError):
    """Raised when uploaded audio cannot be decoded or prepared."""


def validate_upload_file(file: UploadFile) -> str:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Supported formats: {allowed}",
        )
    return extension


async def read_upload_bytes(file: UploadFile) -> bytes:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds {settings.max_upload_mb} MB limit.",
        )
    return content


def decode_audio_bytes(content: bytes, suffix: str) -> np.ndarray:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        audio, _ = librosa.load(
            temp_path,
            sr=settings.target_sample_rate,
            mono=True,
            duration=settings.max_duration_seconds,
        )
    except Exception as exc:
        raise AudioProcessingError("Could not decode uploaded audio file.") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    if audio.size == 0:
        raise AudioProcessingError("Uploaded audio contains no samples.")
    if not np.isfinite(audio).all():
        raise AudioProcessingError("Uploaded audio contains invalid sample values.")

    max_samples = int(settings.max_duration_seconds * settings.target_sample_rate)
    audio = audio[:max_samples].astype(np.float32, copy=False)
    if np.max(np.abs(audio)) == 0:
        raise AudioProcessingError("Uploaded audio is silent.")
    return audio


async def preprocess_upload(file: UploadFile) -> np.ndarray:
    suffix = validate_upload_file(file)
    content = await read_upload_bytes(file)
    try:
        return decode_audio_bytes(content, suffix)
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
