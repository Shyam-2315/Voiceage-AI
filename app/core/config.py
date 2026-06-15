from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse


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
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_realtime_model: str = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
    azure_openai_api_key: str | None = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_realtime_endpoint: str | None = os.getenv("AZURE_OPENAI_REALTIME_ENDPOINT")
    azure_openai_realtime_deployment: str | None = os.getenv("AZURE_OPENAI_REALTIME_DEPLOYMENT")
    azure_openai_api_version: str | None = os.getenv("AZURE_OPENAI_API_VERSION")
    realtime_voice: str = os.getenv("REALTIME_VOICE", "alloy")
    realtime_conversations_dir: Path = PROJECT_ROOT / "data" / "realtime_conversations"
    realtime_audio_capture_seconds: int = int(os.getenv("REALTIME_AUDIO_CAPTURE_SECONDS", "20"))

    @property
    def use_azure_openai_realtime(self) -> bool:
        return bool(self.azure_openai_api_key or self.azure_openai_realtime_endpoint)

    @property
    def azure_openai_endpoint_query(self) -> dict[str, str]:
        if not self.azure_openai_realtime_endpoint:
            return {}

        parsed = urlparse(self.azure_openai_realtime_endpoint)
        return {key: values[-1] for key, values in parse_qs(parsed.query).items() if values}

    @property
    def azure_openai_effective_realtime_deployment(self) -> str | None:
        return self.azure_openai_realtime_deployment or self.azure_openai_endpoint_query.get("deployment")

    @property
    def azure_openai_effective_api_version(self) -> str | None:
        return self.azure_openai_api_version or self.azure_openai_endpoint_query.get("api-version")

    @property
    def realtime_model_name(self) -> str:
        return self.azure_openai_effective_realtime_deployment or self.openai_realtime_model


settings = Settings()
