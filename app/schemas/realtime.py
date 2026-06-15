from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TwilioMediaStreamEvent = Literal["connected", "start", "media", "stop"]


class TwilioStreamStart(BaseModel):
    stream_sid: str | None = Field(default=None, alias="streamSid")
    call_sid: str | None = Field(default=None, alias="callSid")
    account_sid: str | None = Field(default=None, alias="accountSid")
    media_format: dict[str, Any] = Field(default_factory=dict, alias="mediaFormat")
    custom_parameters: dict[str, Any] = Field(default_factory=dict, alias="customParameters")


class TwilioMediaPayload(BaseModel):
    track: str | None = None
    chunk: str | None = None
    timestamp: str | None = None
    payload: str


class RealtimeConversationMetadata(BaseModel):
    call_sid: str | None = None
    stream_sid: str | None = None
    started_at: str
    ended_at: str | None = None
    openai_model: str
    voice: str
    audio_format: str = "mulaw"
    sample_rate_hz: int = 8000
    captured_audio_seconds: float = 0.0
    captured_audio_ulaw_path: str | None = None
    captured_audio_wav_path: str | None = None


class ConversationLogEvent(BaseModel):
    timestamp: str
    source: Literal["twilio", "openai", "bridge"]
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
