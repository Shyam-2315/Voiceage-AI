from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections import deque
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from xml.sax.saxutils import escape

import websockets
from fastapi import HTTPException, WebSocket, WebSocketDisconnect, status
from websockets.exceptions import ConnectionClosed

from app.core.config import settings
from app.services.conversation_logger import RealtimeConversationLogger


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


def openai_session_update_event() -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "voice": settings.realtime_voice,
            "instructions": ASSISTANT_INSTRUCTIONS,
            "input_audio_format": TWILIO_AUDIO_FORMAT,
            "output_audio_format": TWILIO_AUDIO_FORMAT,
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": settings.realtime_vad_threshold,
                "prefix_padding_ms": settings.realtime_vad_prefix_ms,
                "silence_duration_ms": settings.realtime_vad_silence_ms,
            },
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
        return {"error": event.get("error") or event}
    return {}


async def bridge_twilio_stream(twilio_ws: WebSocket) -> None:
    await twilio_ws.accept()
    conversation_logger = RealtimeConversationLogger()
    await conversation_logger.start(None, None)
    openai_ws = None
    stream_sid: str | None = None
    call_sid: str | None = None
    pending_openai_audio_deltas: deque[str] = deque()
    stop_event = asyncio.Event()

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
            first_send_marked = await conversation_logger.mark_latency_event("first_twilio_audio_send")
            if first_send_marked:
                await conversation_logger.log_event("twilio", "first_audio_send", {})
                logger.info("First assistant audio sent to Twilio.")

        async def flush_pending_openai_audio() -> None:
            while stream_sid and pending_openai_audio_deltas:
                await send_twilio_audio_delta(pending_openai_audio_deltas.popleft())

        async def receive_twilio() -> None:
            nonlocal stream_sid, call_sid

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
                        media_frame_count = await conversation_logger.increment_twilio_media_frame()
                        if media_frame_count % 100 == 0:
                            logger.info(
                                "Twilio media frame received: count=%s chunk=%s timestamp=%s",
                                media_frame_count,
                                media.get("chunk"),
                                media.get("timestamp"),
                            )
                        await conversation_logger.capture_twilio_payload(payload)
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
            first_openai_event_seen = False
            greeting_requested = False

            try:
                async for raw_message in openai_ws:
                    if stop_event.is_set():
                        break

                    event = parse_json_message(raw_message)
                    if event is None:
                        continue

                    event_type = str(event.get("type") or "")
                    compact_payload = compact_openai_payload(event)
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
                        "response.audio_transcript.delta",
                        "conversation.item.input_audio_transcription.completed",
                        "error",
                    }:
                        if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                            first_delta_marked = await conversation_logger.mark_latency_event(
                                "first_response_audio_delta"
                            )
                            if first_delta_marked:
                                logger.info(
                                    "First Azure/OpenAI audio delta received: event=%s delta_bytes_base64=%s",
                                    event_type,
                                    compact_payload.get("delta_bytes_base64", 0),
                                )
                        elif event_type == "error":
                            logger.error("Azure/OpenAI Realtime error event: %s", event)
                        else:
                            logger.info("Azure/OpenAI Realtime event: %s", event_type)

                        if event_type == "input_audio_buffer.speech_started":
                            await conversation_logger.mark_latency_event("speech_started")
                        elif event_type == "input_audio_buffer.speech_stopped":
                            await conversation_logger.mark_latency_event("speech_stopped")
                        elif event_type == "response.created":
                            await conversation_logger.mark_latency_event("response_created")

                        await conversation_logger.log_event(
                            "openai",
                            event_type,
                            compact_payload,
                        )

                    if event_type == "session.updated" and not greeting_requested:
                        await send_openai_event(openai_ws, openai_response_create_event())
                        greeting_requested = True
                        await conversation_logger.log_event("openai", "response.create.sent", {})
                        logger.info("Sent OpenAI response.create after session.updated.")
                        continue

                    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = event.get("delta")
                        if delta:
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
