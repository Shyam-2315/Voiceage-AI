from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import deque
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from xml.sax.saxutils import escape

import websockets
from fastapi import HTTPException, WebSocket, WebSocketDisconnect, status
from websockets.exceptions import ConnectionClosed

from app.core.config import settings
from app.services.conversation_style_service import (
    ConversationStyle,
    build_conversation_style_instructions,
    select_conversation_style,
)
from app.services.conversation_logger import RealtimeConversationLogger
from app.services.report_service import generate_live_voiceage_prediction, generate_reports_for_call, load_json


logger = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
TWILIO_AUDIO_FORMAT = "g711_ulaw"

ASSISTANT_INSTRUCTIONS = (
    "You are VoiceAge AI, a concise and friendly phone assistant. "
    "Respond in 1 to 2 short sentences. "
    "Avoid long greetings and get to the point naturally. "
    "If audio is unclear, ask one short clarifying question."
)


def realtime_stream_url() -> str:
    if not settings.public_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PUBLIC_BASE_URL is required for Twilio Media Streams.",
        )

    parsed = urlparse(settings.public_base_url)
    if not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PUBLIC_BASE_URL must be an absolute HTTPS URL.",
        )

    return f"wss://{parsed.netloc}/api/realtime/twilio-stream"


def build_realtime_voice_twiml() -> str:
    stream_url = escape(realtime_stream_url())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Say voice=\"alice\">You are now connected to the VoiceAge AI assistant.</Say>"
        "<Connect>"
        f"<Stream url=\"{stream_url}\" />"
        "</Connect>"
        "</Response>"
    )


def assistant_instructions(conversation_style: ConversationStyle | None = None) -> str:
    if conversation_style is None:
        return ASSISTANT_INSTRUCTIONS
    return f"{ASSISTANT_INSTRUCTIONS}\n\n{build_conversation_style_instructions(conversation_style)}"


def realtime_turn_detection(conversation_style: ConversationStyle | None = None) -> dict[str, Any]:
    silence_duration_ms = settings.realtime_vad_silence_ms
    if conversation_style is not None:
        silence_duration_ms = max(silence_duration_ms, conversation_style.interruption_delay_ms)

    return {
        "type": "server_vad",
        "threshold": settings.realtime_vad_threshold,
        "prefix_padding_ms": settings.realtime_vad_prefix_ms,
        "silence_duration_ms": silence_duration_ms,
    }


def openai_session_update_event(conversation_style: ConversationStyle | None = None) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "voice": settings.realtime_voice,
            "instructions": assistant_instructions(conversation_style),
            "input_audio_format": TWILIO_AUDIO_FORMAT,
            "output_audio_format": TWILIO_AUDIO_FORMAT,
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": realtime_turn_detection(conversation_style),
        },
    }


def openai_response_create_event() -> dict[str, str]:
    return {"type": "response.create"}


def azure_openai_realtime_url() -> str:
    if not settings.azure_openai_realtime_endpoint:
        raise RuntimeError("AZURE_OPENAI_REALTIME_ENDPOINT is required for Azure OpenAI Realtime calls.")

    parsed = urlparse(settings.azure_openai_realtime_endpoint)
    if not parsed.netloc:
        raise RuntimeError("AZURE_OPENAI_REALTIME_ENDPOINT must be an absolute Azure endpoint URL.")

    path = parsed.path.rstrip("/") or "/openai/realtime"
    if path == "/openai":
        path = "/openai/realtime"

    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    api_version = settings.azure_openai_effective_api_version
    deployment = settings.azure_openai_effective_realtime_deployment
    if not api_version:
        raise RuntimeError("AZURE_OPENAI_API_VERSION is required for Azure OpenAI Realtime calls.")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_REALTIME_DEPLOYMENT is required for Azure OpenAI Realtime calls.")

    query_params["api-version"] = api_version
    query_params["deployment"] = deployment
    return urlunparse(("wss", parsed.netloc, path, "", urlencode(query_params), ""))


async def connect_openai_realtime() -> Any:
    if settings.use_azure_openai_realtime:
        if not settings.azure_openai_api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is required for Azure OpenAI Realtime calls.")

        url = azure_openai_realtime_url()
        headers = {"api-key": settings.azure_openai_api_key}
    else:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI Realtime calls.")

        url = f"{OPENAI_REALTIME_URL}?model={settings.openai_realtime_model}"
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

    connect_kwargs: dict[str, Any] = {"ping_interval": 20, "ping_timeout": 20}
    signature = inspect.signature(websockets.connect)
    header_param = "additional_headers" if "additional_headers" in signature.parameters else "extra_headers"
    connect_kwargs[header_param] = headers
    return await websockets.connect(url, **connect_kwargs)


async def send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload))


async def send_openai_event(openai_ws: Any, payload: dict[str, Any]) -> None:
    event_type = str(payload.get("type") or "")
    logger.info("Realtime outbound event=%s", event_type)
    await openai_ws.send(json.dumps(payload))


def parse_json_message(raw_message: str | bytes) -> dict[str, Any] | None:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    try:
        return json.loads(raw_message)
    except json.JSONDecodeError:
        logger.warning("Received non-JSON WebSocket message.")
        return None


def compact_openai_payload(event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("type") or "")
    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
        return {"delta_bytes_base64": len(str(event.get("delta") or ""))}
    if event_type == "response.audio_transcript.delta":
        return {"delta": event.get("delta")}
    if event_type == "conversation.item.input_audio_transcription.completed":
        return {"transcript": event.get("transcript")}
    if event_type == "error":
        return {"error": sanitize_log_payload(event.get("error") or event)}
    return {}


def sanitize_log_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("key", "token", "authorization", "secret")):
                sanitized[key] = "[redacted]"
            else:
                sanitized[key] = sanitize_log_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_log_payload(item) for item in value[:20]]
    if isinstance(value, str):
        if "Bearer " in value or "api-key" in value:
            return "[redacted]"
        return value if len(value) <= 500 else f"{value[:500]}...[truncated]"
    return value


def realtime_diagnostic_fields(event: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    response = event.get("response")
    if isinstance(response, dict):
        for key in ("id", "status", "output", "error"):
            if key in response:
                fields[f"response.{key}"] = response.get(key)

    item = event.get("item")
    if isinstance(item, dict):
        for key in ("id", "status", "role", "content"):
            if key in item:
                fields[f"item.{key}"] = item.get(key)

    return fields


def is_inbound_twilio_media(media: dict[str, Any]) -> bool:
    track = str(media.get("track") or "inbound").lower()
    return track == "inbound"


async def bridge_twilio_stream(twilio_ws: WebSocket) -> None:
    await twilio_ws.accept()
    conversation_logger = RealtimeConversationLogger()
    await conversation_logger.start(None, None)
    openai_ws = None
    stream_sid: str | None = None
    call_sid: str | None = None
    pending_openai_audio_deltas: deque[str] = deque()
    stop_event = asyncio.Event()
    voiceage_live_task: asyncio.Task[None] | None = None
    response_active = False
    assistant_speaking = False
    ignore_barge_in_until = 0.0
    skipped_twilio_media_count = 0
    applied_adaptive_age_group: str | None = None

    try:
        openai_ws = await connect_openai_realtime()
        logger.info(
            "Azure/OpenAI Realtime WebSocket connected: provider=%s model=%s vad_threshold=%s vad_prefix_ms=%s vad_silence_ms=%s",
            "azure" if settings.use_azure_openai_realtime else "openai",
            settings.realtime_model_name,
            settings.realtime_vad_threshold,
            settings.realtime_vad_prefix_ms,
            settings.realtime_vad_silence_ms,
        )
        await send_openai_event(openai_ws, openai_session_update_event())
        await conversation_logger.log_event(
            "bridge",
            "openai.connected",
            {
                "provider": "azure" if settings.use_azure_openai_realtime else "openai",
                "model": settings.realtime_model_name,
            },
        )

        async def send_twilio_audio_delta(delta: str) -> None:
            if not stream_sid:
                pending_openai_audio_deltas.append(delta)
                logger.info("Buffered OpenAI audio delta until Twilio streamSid is available.")
                return

            await send_json(
                twilio_ws,
                {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": delta,
                    },
                },
            )
            first_delta_seen = await conversation_logger.has_latency_event("first_response_audio_delta")
            first_send_seen = await conversation_logger.has_latency_event("first_twilio_audio_send")
            if first_delta_seen and not first_send_seen:
                first_send_marked = await conversation_logger.mark_latency_event("first_twilio_audio_send")
            else:
                first_send_marked = False
            if first_send_marked:
                await conversation_logger.log_event("twilio", "first_audio_send", {})
                logger.info("First assistant audio sent to Twilio.")

        async def flush_pending_openai_audio() -> None:
            while stream_sid and pending_openai_audio_deltas:
                await send_twilio_audio_delta(pending_openai_audio_deltas.popleft())

        async def apply_adaptive_conversation_update(live_report: dict[str, Any]) -> None:
            nonlocal applied_adaptive_age_group

            if not settings.enable_adaptive_conversation:
                await conversation_logger.log_event(
                    "bridge",
                    "adaptive_conversation.skipped",
                    {"reason": "disabled_by_config"},
                )
                logger.info("Adaptive conversation skipped: disabled_by_config")
                return

            if not live_report.get("prediction_success"):
                reason = live_report.get("failure_reason") or "prediction_not_successful"
                await conversation_logger.log_event(
                    "bridge",
                    "adaptive_conversation.skipped",
                    {"reason": reason},
                )
                logger.info("Adaptive conversation skipped: reason=%s", reason)
                return

            predicted_age_group = str(live_report.get("predicted_age_group") or "").strip() or None
            selection = select_conversation_style(
                predicted_age_group,
                default_age_group=settings.default_conversation_style,
            )
            if applied_adaptive_age_group == selection.selected_age_group:
                await conversation_logger.log_event(
                    "bridge",
                    "adaptive_conversation.skipped",
                    {
                        "reason": "style_already_applied",
                        "selected_age_group": selection.selected_age_group,
                    },
                )
                logger.info(
                    "Adaptive conversation skipped: style already applied selected_age_group=%s",
                    selection.selected_age_group,
                )
                return

            try:
                await send_openai_event(openai_ws, openai_session_update_event(selection.style))
            except Exception as exc:
                logger.exception(
                    "Adaptive conversation session.update failed: selected_age_group=%s",
                    selection.selected_age_group,
                )
                await conversation_logger.log_event(
                    "bridge",
                    "adaptive_conversation.update_failed",
                    {
                        "selected_age_group": selection.selected_age_group,
                        "error": str(exc),
                    },
                )
                return

            applied_adaptive_age_group = selection.selected_age_group
            style_payload = selection.log_payload()
            logger.info(
                "Adaptive conversation selected: predicted_age_group=%s selected_age_group=%s used_fallback=%s conversation_style=%s",
                selection.requested_age_group,
                selection.selected_age_group,
                selection.used_fallback,
                json.dumps(style_payload["conversation_style"], default=str),
            )
            await conversation_logger.log_event(
                "bridge",
                "adaptive_conversation.style_selected",
                style_payload,
            )
            await conversation_logger.log_event(
                "openai",
                "session.update.sent",
                {
                    "reason": "adaptive_conversation",
                    **style_payload,
                },
            )

        async def run_live_voiceage_prediction() -> None:
            try:
                audio_path = await conversation_logger.write_caller_audio_snapshot()
                if audio_path is None or conversation_logger.session_dir is None:
                    return
                live_path = await asyncio.to_thread(
                    generate_live_voiceage_prediction,
                    conversation_logger.session_dir,
                    call_sid or conversation_logger.session_dir.name,
                    audio_path,
                )
                await conversation_logger.log_event(
                    "bridge",
                    "voiceage.live_prediction.completed",
                    {"path": str(live_path), "audio_source": "caller_only"},
                )
                logger.info("VoiceAge live prediction generated: %s", live_path)
                live_report = await asyncio.to_thread(load_json, live_path)
                await apply_adaptive_conversation_update(live_report)
            except Exception as exc:
                logger.exception("VoiceAge live prediction failed")
                await conversation_logger.log_event(
                    "bridge",
                    "voiceage.live_prediction.error",
                    {"error": str(exc)},
                )

        async def receive_twilio() -> None:
            nonlocal stream_sid, call_sid, voiceage_live_task, skipped_twilio_media_count

            try:
                while not stop_event.is_set():
                    raw_message = await twilio_ws.receive_text()
                    event = parse_json_message(raw_message)
                    if event is None:
                        continue

                    event_type = event.get("event")
                    if event_type == "connected":
                        await conversation_logger.log_event("twilio", "connected", {})
                        continue

                    if event_type == "start":
                        start = event.get("start") or {}
                        stream_sid = event.get("streamSid") or start.get("streamSid")
                        call_sid = start.get("callSid")
                        await conversation_logger.start(call_sid, stream_sid)
                        await conversation_logger.log_event(
                            "twilio",
                            "start",
                            {
                                "stream_sid": stream_sid,
                                "call_sid": call_sid,
                                "media_format": start.get("mediaFormat"),
                            },
                        )
                        await flush_pending_openai_audio()
                        continue

                    if event_type == "media":
                        media = event.get("media") or {}
                        payload = media.get("payload")
                        if not payload:
                            continue
                        if not is_inbound_twilio_media(media):
                            await conversation_logger.log_event(
                                "twilio",
                                "media.ignored_non_inbound",
                                {"track": media.get("track")},
                            )
                            continue
                        media_frame_count = await conversation_logger.increment_twilio_media_frame()
                        if media_frame_count % 100 == 0:
                            logger.info(
                                "Twilio media frame received: count=%s chunk=%s timestamp=%s",
                                media_frame_count,
                                media.get("chunk"),
                                media.get("timestamp"),
                            )
                        caller_audio_ready = await conversation_logger.capture_caller_payload(payload)
                        if caller_audio_ready and voiceage_live_task is None:
                            voiceage_live_task = asyncio.create_task(run_live_voiceage_prediction())
                            await conversation_logger.log_event(
                                "bridge",
                                "voiceage.live_prediction.started",
                                {"audio_source": "caller_only"},
                            )
                        if assistant_speaking:
                            skipped_twilio_media_count += 1
                            log_payload = {
                                "count": skipped_twilio_media_count,
                                "chunk": media.get("chunk"),
                                "timestamp": media.get("timestamp"),
                            }
                            await conversation_logger.log_event(
                                "twilio",
                                "media.skipped_assistant_speaking",
                                log_payload,
                            )
                            if skipped_twilio_media_count == 1 or skipped_twilio_media_count % 100 == 0:
                                logger.info(
                                    "Skipped Twilio inbound media while assistant_speaking=True: count=%s chunk=%s timestamp=%s",
                                    skipped_twilio_media_count,
                                    media.get("chunk"),
                                    media.get("timestamp"),
                                )
                            continue
                        await send_openai_event(
                            openai_ws,
                            {
                                "type": "input_audio_buffer.append",
                                "audio": payload,
                            },
                        )
                        continue

                    if event_type == "stop":
                        await conversation_logger.log_event("twilio", "stop", {})
                        stop_event.set()
                        break

                    await conversation_logger.log_event("twilio", str(event_type or "unknown"), {})
            except WebSocketDisconnect:
                await conversation_logger.log_event("twilio", "disconnect", {})
                stop_event.set()
            except Exception as exc:
                logger.exception("Twilio receive loop failed")
                await conversation_logger.log_event("bridge", "twilio_receive.error", {"error": str(exc)})
                stop_event.set()

        async def receive_openai() -> None:
            nonlocal response_active, assistant_speaking, ignore_barge_in_until, skipped_twilio_media_count

            first_openai_event_seen = False
            greeting_requested = False
            caller_speech_stopped_seen = False
            response_terminal_events = {
                "response.audio.done",
                "response.output_audio.done",
                "response.done",
                "response.cancelled",
                "response.failed",
                "error",
            }
            diagnostic_attention_events = {
                "response.failed",
                "error",
                "rate_limits.updated",
                "conversation.item.truncated",
                "response.cancelled",
            }

            async def send_response_create(reason: str) -> None:
                payload = openai_response_create_event()
                after_caller_speech = caller_speech_stopped_seen
                logger.info(
                    "===== RESPONSE CREATE SENT ===== reason=%s after_caller_speech=%s",
                    reason,
                    after_caller_speech,
                )
                if after_caller_speech:
                    logger.info("response.create sent after caller speech: reason=%s", reason)
                await send_openai_event(openai_ws, payload)

            try:
                async for raw_message in openai_ws:
                    if stop_event.is_set():
                        break

                    event = parse_json_message(raw_message)
                    if event is None:
                        continue

                    event_type = str(event.get("type") or "")
                    compact_payload = compact_openai_payload(event)
                    logger.info(
                        "Realtime event=%s compact_payload=%s",
                        event_type,
                        json.dumps(sanitize_log_payload(compact_payload), default=str),
                    )
                    diagnostic_fields = realtime_diagnostic_fields(event)
                    if diagnostic_fields:
                        logger.info(
                            "Realtime diagnostic fields=%s",
                            json.dumps(sanitize_log_payload(diagnostic_fields), default=str),
                        )
                    if event_type in diagnostic_attention_events:
                        logger.warning(
                            "Realtime attention event=%s payload=%s",
                            event_type,
                            json.dumps(sanitize_log_payload(compact_payload), default=str),
                        )
                    if not first_openai_event_seen:
                        logger.info("First Azure/OpenAI Realtime event received: %s", event_type)
                        await conversation_logger.log_event(
                            "openai",
                            "first_received_event",
                            {"type": event_type, **compact_payload},
                        )
                        first_openai_event_seen = True

                    if event_type in {
                        "session.created",
                        "session.updated",
                        "input_audio_buffer.speech_started",
                        "input_audio_buffer.speech_stopped",
                        "response.created",
                        "response.audio.delta",
                        "response.output_audio.delta",
                        "response.audio.done",
                        "response.output_audio.done",
                        "response.done",
                        "response.cancelled",
                        "response.failed",
                        "response.audio_transcript.delta",
                        "rate_limits.updated",
                        "conversation.item.truncated",
                        "conversation.item.input_audio_transcription.completed",
                        "error",
                    }:
                        if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                            if not assistant_speaking:
                                logger.info("===== RESPONSE AUDIO START =====")
                                assistant_speaking = True
                                logger.info(
                                    "Realtime turn state changed: response_active=%s assistant_speaking=True reason=%s",
                                    response_active,
                                    event_type,
                                )
                                await conversation_logger.log_event(
                                    "bridge",
                                    "turn_state.changed",
                                    {
                                        "reason": event_type,
                                        "response_active": response_active,
                                        "assistant_speaking": assistant_speaking,
                                    },
                                )
                            response_created_seen = await conversation_logger.has_latency_event("response_created")
                            first_delta_seen = await conversation_logger.has_latency_event(
                                "first_response_audio_delta"
                            )
                            first_delta_marked = (
                                await conversation_logger.mark_latency_event("first_response_audio_delta")
                                if response_created_seen and not first_delta_seen
                                else False
                            )
                            if first_delta_marked:
                                logger.info(
                                    "First Azure/OpenAI audio delta received: event=%s delta_bytes_base64=%s",
                                    event_type,
                                    compact_payload.get("delta_bytes_base64", 0),
                                )

                        if event_type == "input_audio_buffer.speech_started":
                            logger.info("===== CALLER SPEECH START =====")
                            now = time.monotonic()
                            protected_for_ms = max(0, int((ignore_barge_in_until - now) * 1000))
                            if response_active or assistant_speaking or now < ignore_barge_in_until:
                                compact_payload.update(
                                    {
                                        "ignored": True,
                                        "response_active": response_active,
                                        "assistant_speaking": assistant_speaking,
                                        "protected_for_ms": protected_for_ms,
                                    }
                                )
                                logger.info(
                                    "Ignored speech_started during assistant response: response_active=%s assistant_speaking=%s protected_for_ms=%s",
                                    response_active,
                                    assistant_speaking,
                                    protected_for_ms,
                                )
                            else:
                                compact_payload["accepted"] = True
                                await conversation_logger.mark_latency_event("speech_started")
                                logger.info("Accepted speech_started from caller.")
                        elif event_type == "input_audio_buffer.speech_stopped":
                            logger.info("===== CALLER SPEECH STOP =====")
                            caller_speech_stopped_seen = True
                            await conversation_logger.mark_latency_event("speech_stopped")
                        elif event_type == "response.created":
                            response_active = True
                            ignore_barge_in_until = time.monotonic() + 1.0
                            skipped_twilio_media_count = 0
                            compact_payload.update(
                                {
                                    "response_active": response_active,
                                    "assistant_speaking": assistant_speaking,
                                    "ignore_barge_in_ms": 1000,
                                }
                            )
                            logger.info(
                                "Realtime turn state changed: response_active=True assistant_speaking=%s reason=response.created ignore_barge_in_ms=1000",
                                assistant_speaking,
                            )
                            await conversation_logger.log_event(
                                "bridge",
                                "turn_state.changed",
                                {
                                    "reason": "response.created",
                                    "response_active": response_active,
                                    "assistant_speaking": assistant_speaking,
                                    "ignore_barge_in_ms": 1000,
                                },
                            )
                            speech_stopped_seen = await conversation_logger.has_latency_event("speech_stopped")
                            response_created_seen = await conversation_logger.has_latency_event("response_created")
                            if speech_stopped_seen and not response_created_seen:
                                await conversation_logger.mark_latency_event("response_created")
                        elif event_type in response_terminal_events:
                            if event_type in {"response.audio.done", "response.output_audio.done", "response.done"}:
                                logger.info("===== RESPONSE AUDIO END ===== reason=%s", event_type)
                            previous_response_active = response_active
                            previous_assistant_speaking = assistant_speaking
                            response_active = False
                            assistant_speaking = False
                            compact_payload.update(
                                {
                                    "response_active": response_active,
                                    "assistant_speaking": assistant_speaking,
                                }
                            )
                            logger.info(
                                "Realtime turn state changed: response_active=False assistant_speaking=False reason=%s previous_response_active=%s previous_assistant_speaking=%s",
                                event_type,
                                previous_response_active,
                                previous_assistant_speaking,
                            )
                            await conversation_logger.log_event(
                                "bridge",
                                "turn_state.changed",
                                {
                                    "reason": event_type,
                                    "response_active": response_active,
                                    "assistant_speaking": assistant_speaking,
                                    "previous_response_active": previous_response_active,
                                    "previous_assistant_speaking": previous_assistant_speaking,
                                },
                            )

                        await conversation_logger.log_event(
                            "openai",
                            event_type,
                            compact_payload,
                        )

                    if event_type == "session.updated" and not greeting_requested:
                        await send_response_create("session.updated")
                        greeting_requested = True
                        await conversation_logger.log_event("openai", "response.create.sent", {})
                        logger.info("Sent OpenAI response.create after session.updated.")
                        continue

                    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = event.get("delta")
                        if delta:
                            await conversation_logger.capture_assistant_payload(delta)
                            await send_twilio_audio_delta(delta)
                        continue
            except ConnectionClosed:
                await conversation_logger.log_event("openai", "disconnect", {})
                stop_event.set()
            except Exception as exc:
                logger.exception("OpenAI receive loop failed")
                await conversation_logger.log_event("bridge", "openai_receive.error", {"error": str(exc)})
                stop_event.set()

        twilio_task = asyncio.create_task(receive_twilio())
        openai_task = asyncio.create_task(receive_openai())
        await stop_event.wait()

        for task in (twilio_task, openai_task):
            task.cancel()
        await asyncio.gather(twilio_task, openai_task, return_exceptions=True)
        if voiceage_live_task is not None and voiceage_live_task.done():
            await asyncio.gather(voiceage_live_task, return_exceptions=True)

    except Exception as exc:
        logger.exception("Realtime bridge failed")
        await conversation_logger.log_event("bridge", "fatal.error", {"error": str(exc)})
        if twilio_ws.client_state.name == "CONNECTED":
            await twilio_ws.close(code=1011)
    finally:
        if openai_ws is not None:
            try:
                await openai_ws.close()
            except Exception:
                logger.debug("OpenAI WebSocket cleanup failed.", exc_info=True)
        await conversation_logger.finalize()
        if conversation_logger.session_dir is not None:
            try:
                reports_dir = await asyncio.to_thread(generate_reports_for_call, conversation_logger.session_dir)
                logger.info("Post-call reports generated: %s", reports_dir)
            except Exception as exc:
                logger.exception("Post-call report generation failed")
                await conversation_logger.log_event("bridge", "report_generation.error", {"error": str(exc)})
