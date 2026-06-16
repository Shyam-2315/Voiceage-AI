from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
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
ULAW_SAMPLE_RATE_HZ = 8000
VOICEAGE_READY_SECONDS = 5.0
VOICEAGE_MAX_CAPTURE_SECONDS = 10.0


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def current_time_mark() -> tuple[str, float]:
    return utc_timestamp(), time.perf_counter()


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
        configured_capture_seconds = capture_seconds or settings.realtime_audio_capture_seconds
        self.capture_seconds = min(
            VOICEAGE_MAX_CAPTURE_SECONDS,
            max(VOICEAGE_READY_SECONDS, float(configured_capture_seconds)),
        )
        self.capture_limit_bytes = int(self.capture_seconds * ULAW_SAMPLE_RATE_HZ)
        self.started_at = utc_timestamp()
        self.call_sid: str | None = None
        self.stream_sid: str | None = None
        self.session_dir: Path | None = None
        self.events_path: Path | None = None
        self.metadata_path: Path | None = None
        self.latency_metrics_path: Path | None = None
        self.caller_audio_ulaw_path: Path | None = None
        self.caller_audio_wav_path: Path | None = None
        self.assistant_audio_ulaw_path: Path | None = None
        self._caller_only_audio = bytearray()
        self._assistant_audio = bytearray()
        self._caller_audio_ready_logged = False
        self._latency_timestamps: dict[str, str] = {}
        self._latency_marks: dict[str, float] = {}
        self._twilio_media_frames = 0
        self._lock = asyncio.Lock()

    @property
    def captured_audio_seconds(self) -> float:
        return self.caller_audio_duration_seconds

    @property
    def caller_audio_duration_seconds(self) -> float:
        return len(self._caller_only_audio) / float(ULAW_SAMPLE_RATE_HZ)

    @property
    def assistant_audio_duration_seconds(self) -> float:
        return len(self._assistant_audio) / float(ULAW_SAMPLE_RATE_HZ)

    @property
    def caller_audio_ready_for_voiceage(self) -> bool:
        return self.caller_audio_duration_seconds >= VOICEAGE_READY_SECONDS

    async def start(self, call_sid: str | None, stream_sid: str | None) -> None:
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        if self.session_dir is not None:
            await self._move_to_call_dir(call_sid, stream_sid)
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
        self._set_session_paths(self.session_dir)
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

    async def mark_latency_event(self, event_name: str) -> bool:
        if self.events_path is None:
            await self.start(self.call_sid, self.stream_sid)

        timestamp, monotonic_mark = current_time_mark()

        async with self._lock:
            if event_name in self._latency_timestamps:
                return False
            self._latency_timestamps[event_name] = timestamp
            self._latency_marks[event_name] = monotonic_mark
            return True

    async def has_latency_event(self, event_name: str) -> bool:
        async with self._lock:
            return event_name in self._latency_timestamps

    async def increment_twilio_media_frame(self) -> int:
        async with self._lock:
            self._twilio_media_frames += 1
            return self._twilio_media_frames

    async def capture_caller_payload(self, payload: str) -> bool:
        if len(self._caller_only_audio) >= self.capture_limit_bytes:
            return self.caller_audio_ready_for_voiceage

        try:
            audio = base64.b64decode(payload)
        except ValueError:
            await self.log_event("bridge", "caller_capture.invalid_payload", {})
            return False

        ready_before = self.caller_audio_ready_for_voiceage
        remaining = self.capture_limit_bytes - len(self._caller_only_audio)
        self._caller_only_audio.extend(audio[:remaining])
        ready_after = self.caller_audio_ready_for_voiceage

        if ready_after and not ready_before and not self._caller_audio_ready_logged:
            self._caller_audio_ready_logged = True
            await self.log_event(
                "bridge",
                "voiceage.caller_audio_ready",
                {
                    "audio_source": "caller_only",
                    "caller_audio_duration_seconds": round(self.caller_audio_duration_seconds, 3),
                    "minimum_required_seconds": VOICEAGE_READY_SECONDS,
                    "assistant_audio_excluded": True,
                },
            )
        return ready_after

    async def capture_twilio_payload(self, payload: str) -> None:
        await self.capture_caller_payload(payload)

    async def capture_assistant_payload(self, payload: str) -> None:
        try:
            audio = base64.b64decode(payload)
        except ValueError:
            await self.log_event("bridge", "assistant_capture.invalid_payload", {})
            return

        self._assistant_audio.extend(audio)

    async def write_caller_audio_snapshot(self) -> Path | None:
        if self.session_dir is None:
            await self.start(self.call_sid, self.stream_sid)
        await asyncio.to_thread(self._write_caller_audio_files)
        return self.caller_audio_wav_path

    async def finalize(self) -> None:
        if self.session_dir is None:
            await self.start(self.call_sid, self.stream_sid)

        await self.log_event(
            "bridge",
            "session.ended",
            {
                "caller_audio_duration_seconds": round(self.caller_audio_duration_seconds, 3),
                "assistant_audio_duration_seconds": round(self.assistant_audio_duration_seconds, 3),
                "assistant_audio_excluded_from_voiceage": True,
            },
        )

        await asyncio.to_thread(self._write_audio_files)
        await asyncio.to_thread(self._write_metadata)
        await asyncio.to_thread(self._write_latency_metrics)
        logger.info("Saved realtime conversation log: %s", self.session_dir)

    def _set_session_paths(self, session_dir: Path) -> None:
        self.events_path = session_dir / "events.jsonl"
        self.metadata_path = session_dir / "metadata.json"
        self.latency_metrics_path = session_dir / "latency_metrics.json"
        self.caller_audio_ulaw_path = session_dir / "caller_only_audio.ulaw"
        self.caller_audio_wav_path = session_dir / "caller_only_audio.wav"
        self.assistant_audio_ulaw_path = session_dir / "assistant_audio.ulaw"

    async def _move_to_call_dir(self, call_sid: str | None, stream_sid: str | None) -> None:
        if call_sid is None or self.session_dir is None:
            return

        safe_call = safe_log_token(call_sid, "call")
        target_dir = self.root_dir / safe_call
        if target_dir == self.session_dir:
            return
        if target_dir.exists():
            safe_stream = safe_log_token(stream_sid, "stream")
            target_dir = self.root_dir / f"{safe_call}_{safe_stream}"
            if target_dir == self.session_dir or target_dir.exists():
                return

        async with self._lock:
            if self.session_dir is None or target_dir == self.session_dir or target_dir.exists():
                return
            await asyncio.to_thread(self.session_dir.rename, target_dir)
            self.session_dir = target_dir
            self._set_session_paths(target_dir)

    @staticmethod
    def _append_text(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _write_audio_files(self) -> None:
        self._write_caller_audio_files()
        self._write_assistant_audio_file()

    def _write_caller_audio_files(self) -> None:
        if not self._caller_only_audio or self.caller_audio_ulaw_path is None:
            return

        raw_audio = bytes(self._caller_only_audio)
        self.caller_audio_ulaw_path.write_bytes(raw_audio)

        if audioop is None or self.caller_audio_wav_path is None:
            return

        pcm16 = audioop.ulaw2lin(raw_audio, 2)
        with wave.open(str(self.caller_audio_wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(ULAW_SAMPLE_RATE_HZ)
            wav_file.writeframes(pcm16)

    def _write_assistant_audio_file(self) -> None:
        if not self._assistant_audio or self.assistant_audio_ulaw_path is None:
            return
        self.assistant_audio_ulaw_path.write_bytes(bytes(self._assistant_audio))

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
            captured_audio_seconds=round(self.caller_audio_duration_seconds, 3),
            captured_audio_ulaw_path=str(self.caller_audio_ulaw_path) if self.caller_audio_ulaw_path else None,
            captured_audio_wav_path=str(self.caller_audio_wav_path) if self.caller_audio_wav_path else None,
        )
        payload = pydantic_dump(metadata)
        payload.update(
            {
                "audio_source": "caller_only",
                "caller_only_audio_seconds": round(self.caller_audio_duration_seconds, 3),
                "caller_only_audio_ulaw_path": str(self.caller_audio_ulaw_path)
                if self.caller_audio_ulaw_path
                else None,
                "caller_only_audio_wav_path": str(self.caller_audio_wav_path)
                if self.caller_audio_wav_path
                else None,
                "caller_audio_ready_for_voiceage": self.caller_audio_ready_for_voiceage,
                "voiceage_minimum_ready_seconds": VOICEAGE_READY_SECONDS,
                "voiceage_capture_limit_seconds": self.capture_seconds,
                "assistant_audio_excluded_from_voiceage": True,
                "assistant_audio_ulaw_path": str(self.assistant_audio_ulaw_path)
                if self.assistant_audio_ulaw_path and self._assistant_audio
                else None,
                "assistant_audio_seconds": round(self.assistant_audio_duration_seconds, 3),
            }
        )
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_latency_metrics(self) -> None:
        if self.latency_metrics_path is None:
            return

        def elapsed_ms(start: str, end: str) -> float | None:
            if start not in self._latency_marks or end not in self._latency_marks:
                return None
            return round((self._latency_marks[end] - self._latency_marks[start]) * 1000, 3)

        metrics = {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "started_at": self.started_at,
            "ended_at": utc_timestamp(),
            "timestamps": {
                "speech_started": self._latency_timestamps.get("speech_started"),
                "speech_stopped": self._latency_timestamps.get("speech_stopped"),
                "response_created": self._latency_timestamps.get("response_created"),
                "first_response_audio_delta": self._latency_timestamps.get("first_response_audio_delta"),
                "first_twilio_audio_send": self._latency_timestamps.get("first_twilio_audio_send"),
            },
            "durations_ms": {
                "speech_started_to_speech_stopped": elapsed_ms("speech_started", "speech_stopped"),
                "speech_stopped_to_response_created": elapsed_ms("speech_stopped", "response_created"),
                "response_created_to_first_response_audio_delta": elapsed_ms(
                    "response_created",
                    "first_response_audio_delta",
                ),
                "first_response_audio_delta_to_first_twilio_audio_send": elapsed_ms(
                    "first_response_audio_delta",
                    "first_twilio_audio_send",
                ),
                "speech_stopped_to_first_twilio_audio_send": elapsed_ms(
                    "speech_stopped",
                    "first_twilio_audio_send",
                ),
                "response_created_to_first_audio_delta": elapsed_ms(
                    "response_created",
                    "first_response_audio_delta",
                ),
                "first_audio_delta_to_twilio_send": elapsed_ms(
                    "first_response_audio_delta",
                    "first_twilio_audio_send",
                ),
                "total_response_latency": elapsed_ms(
                    "speech_stopped",
                    "first_twilio_audio_send",
                ),
            },
            "twilio_media_frames": self._twilio_media_frames,
            "audio_source": "caller_only",
            "caller_audio_duration_seconds": round(self.caller_audio_duration_seconds, 3),
            "assistant_audio_duration_seconds": round(self.assistant_audio_duration_seconds, 3),
            "assistant_audio_excluded_from_voiceage": True,
        }
        with self.latency_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
