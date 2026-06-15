from __future__ import annotations

import asyncio
import base64
import json
import logging
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.schemas.realtime import ConversationLogEvent, RealtimeConversationMetadata

try:
    import audioop
except ImportError:  # pragma: no cover - Python 3.13+ compatibility
    audioop = None


logger = logging.getLogger(__name__)


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def safe_log_token(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    safe = "".join(char for char in value if char.isalnum() or char in {"_", "-"})
    return safe or fallback


def pydantic_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class RealtimeConversationLogger:
    def __init__(
        self,
        root_dir: Path | None = None,
        capture_seconds: int | None = None,
    ) -> None:
        self.root_dir = root_dir or settings.realtime_conversations_dir
        self.capture_seconds = capture_seconds or settings.realtime_audio_capture_seconds
        self.capture_limit_bytes = max(1, self.capture_seconds) * 8000
        self.started_at = utc_timestamp()
        self.call_sid: str | None = None
        self.stream_sid: str | None = None
        self.session_dir: Path | None = None
        self.events_path: Path | None = None
        self.metadata_path: Path | None = None
        self.audio_ulaw_path: Path | None = None
        self.audio_wav_path: Path | None = None
        self._captured_audio = bytearray()
        self._lock = asyncio.Lock()

    @property
    def captured_audio_seconds(self) -> float:
        return len(self._captured_audio) / 8000.0

    async def start(self, call_sid: str | None, stream_sid: str | None) -> None:
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        if self.session_dir is not None:
            await self.log_event(
                "bridge",
                "session.context_updated",
                {"call_sid": call_sid, "stream_sid": stream_sid},
            )
            return

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        safe_call = safe_log_token(call_sid, "call")
        safe_stream = safe_log_token(stream_sid, "stream")
        self.session_dir = self.root_dir / f"{timestamp}_{safe_call}_{safe_stream}"
        self.events_path = self.session_dir / "events.jsonl"
        self.metadata_path = self.session_dir / "metadata.json"
        self.audio_ulaw_path = self.session_dir / "caller_capture.ulaw"
        self.audio_wav_path = self.session_dir / "caller_capture.wav"
        await asyncio.to_thread(self.session_dir.mkdir, parents=True, exist_ok=True)
        await self.log_event(
            "bridge",
            "session.started",
            {"call_sid": call_sid, "stream_sid": stream_sid},
        )

    async def log_event(self, source: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if self.events_path is None:
            return

        event = ConversationLogEvent(
            timestamp=utc_timestamp(),
            source=source,
            event_type=event_type,
            payload=payload or {},
        )
        line = json.dumps(pydantic_dump(event), ensure_ascii=False) + "\n"

        async with self._lock:
            await asyncio.to_thread(self._append_text, self.events_path, line)

    async def capture_twilio_payload(self, payload: str) -> None:
        if len(self._captured_audio) >= self.capture_limit_bytes:
            return

        try:
            audio = base64.b64decode(payload)
        except ValueError:
            await self.log_event("bridge", "capture.invalid_payload", {})
            return

        remaining = self.capture_limit_bytes - len(self._captured_audio)
        self._captured_audio.extend(audio[:remaining])

    async def finalize(self) -> None:
        if self.session_dir is None:
            await self.start(self.call_sid, self.stream_sid)

        await self.log_event(
            "bridge",
            "session.ended",
            {"captured_audio_seconds": round(self.captured_audio_seconds, 3)},
        )

        await asyncio.to_thread(self._write_audio_files)
        await asyncio.to_thread(self._write_metadata)
        logger.info("Saved realtime conversation log: %s", self.session_dir)

    @staticmethod
    def _append_text(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _write_audio_files(self) -> None:
        if not self._captured_audio or self.audio_ulaw_path is None:
            return

        raw_audio = bytes(self._captured_audio)
        self.audio_ulaw_path.write_bytes(raw_audio)

        if audioop is None or self.audio_wav_path is None:
            return

        pcm16 = audioop.ulaw2lin(raw_audio, 2)
        with wave.open(str(self.audio_wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(8000)
            wav_file.writeframes(pcm16)

    def _write_metadata(self) -> None:
        if self.metadata_path is None:
            return

        metadata = RealtimeConversationMetadata(
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            started_at=self.started_at,
            ended_at=utc_timestamp(),
            openai_model=settings.realtime_model_name,
            voice=settings.realtime_voice,
            captured_audio_seconds=round(self.captured_audio_seconds, 3),
            captured_audio_ulaw_path=str(self.audio_ulaw_path) if self.audio_ulaw_path else None,
            captured_audio_wav_path=str(self.audio_wav_path) if self.audio_wav_path else None,
        )
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(pydantic_dump(metadata), handle, indent=2)
