from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str = "VoiceAge AI"
    model_path: Path = PROJECT_ROOT / "models" / "wav2vec_50k" / "best"
    model_version: str = "wav2vec_50k"
    target_sample_rate: int = 16_000
    max_duration_seconds: float = 8.0
    max_upload_mb: int = 25
    allowed_extensions: tuple[str, ...] = ("wav", "mp3", "m4a")
    class_labels: tuple[str, ...] = ("Adult", "Middle_Age", "Senior", "Teen")
    twilio_account_sid: str | None = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token: str | None = os.getenv("TWILIO_AUTH_TOKEN")
    public_base_url: str | None = os.getenv("PUBLIC_BASE_URL")
    twilio_predictions_dir: Path = PROJECT_ROOT / "data" / "twilio_predictions"
    twilio_recording_max_seconds: int = 20
    twilio_download_timeout_seconds: int = 60


settings = Settings()
