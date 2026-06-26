from __future__ import annotations

import json
import logging
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from app.core.config import settings


logger = logging.getLogger(__name__)

VOICEAGE_EXPLANATION = "This predicts an age group from voice, not exact age."
VOICEAGE_CALLER_ONLY_STATEMENT = "Prediction based on caller-only audio. AI assistant voice was excluded."
VOICEAGE_COMBINED_STATEMENT = (
    "VoiceAge prediction was generated using caller-only audio. Assistant/AI voice was excluded."
)
VOICEAGE_AUDIO_SOURCE = "caller_only"
VOICEAGE_MINIMUM_REPORT_SECONDS = 3.0
MODEL_EVALUATION_REFERENCE = {
    "accuracy": "81.37%",
    "weighted_f1": "81.29%",
}
LATENCY_EXPLANATION = (
    "Latency measures how quickly the realtime assistant responds after the caller stops speaking. "
    "Lower values mean the phone conversation feels more natural."
)


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def pydantic_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def seconds_between(start: str | None, end: str | None) -> float | None:
    started = parse_datetime(start)
    ended = parse_datetime(end)
    if started is None or ended is None:
        return None
    return round((ended - started).total_seconds(), 3)


def format_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def bool_text(value: bool) -> str:
    return "Yes" if value else "No"


def wav_duration_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return round(wav_file.getnframes() / float(frame_rate), 3)
    except (wave.Error, OSError):
        return None


def first_existing_path(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def metadata_path(metadata: dict[str, Any], key: str) -> Path | None:
    value = metadata.get(key)
    return Path(value) if value else None


def report_audio_selection(
    call_dir: Path,
    metadata: dict[str, Any],
    audio_path: Path | None = None,
    audio_source: str | None = None,
) -> dict[str, Any]:
    caller_full_audio_path = first_existing_path(
        metadata_path(metadata, "caller_full_audio_wav_path"),
        call_dir / "caller_full_audio.wav",
    )
    legacy_caller_only_audio_path = first_existing_path(
        metadata_path(metadata, "caller_only_audio_wav_path"),
        call_dir / "caller_only_audio.wav",
    )
    prediction_clip_path = first_existing_path(
        audio_path if audio_source == "caller_prediction_clip" else None,
        metadata_path(metadata, "caller_prediction_clip_wav_path"),
        call_dir / "caller_prediction_clip.wav",
    )

    if audio_source == "caller_prediction_clip":
        selected_audio_path = first_existing_path(audio_path, prediction_clip_path)
        selected_audio_source = "caller_prediction_clip"
    elif caller_full_audio_path is not None:
        selected_audio_path = caller_full_audio_path
        selected_audio_source = "caller_full_audio"
    else:
        selected_audio_path = legacy_caller_only_audio_path
        selected_audio_source = "caller_only_audio_legacy"

    selected_duration = wav_duration_seconds(selected_audio_path) if selected_audio_path else None
    return {
        "selected_audio_path": selected_audio_path,
        "selected_audio_source": selected_audio_source,
        "selected_audio_duration_sec": selected_duration,
        "caller_full_audio_path": caller_full_audio_path,
        "caller_full_audio_exists": caller_full_audio_path is not None,
        "caller_full_audio_duration_sec": wav_duration_seconds(caller_full_audio_path)
        if caller_full_audio_path
        else None,
        "legacy_caller_only_audio_path": legacy_caller_only_audio_path,
        "prediction_clip_path": prediction_clip_path,
        "prediction_clip_exists": prediction_clip_path is not None,
    }


def redact_sensitive_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact_sensitive_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_sensitive_text(item) for item in value]
    if not isinstance(value, str):
        return value

    redacted = value
    if "Bearer " in redacted:
        return "Bearer [redacted]"
    sensitive_markers = ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "TWILIO_AUTH_TOKEN", "api-key", "Authorization")
    for marker in sensitive_markers:
        if marker in redacted:
            return "[redacted sensitive log text]"
    return redacted


def read_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []

    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                events.append(
                    {
                        "timestamp": utc_timestamp(),
                        "source": "bridge",
                        "event_type": "report.warning",
                        "payload": {"warning": f"Could not parse events.jsonl line {line_number}."},
                    }
                )
                continue
            events.append(redact_sensitive_text(event))
    return events


def generate_voiceage_report(
    call_dir: Path,
    call_id: str,
    metadata: dict[str, Any],
    audio_path: Path | None = None,
    audio_source: str | None = None,
) -> dict[str, Any]:
    audio_selection = report_audio_selection(call_dir, metadata, audio_path, audio_source)
    prediction_audio_path = audio_selection["selected_audio_path"]
    full_audio_path = audio_selection["caller_full_audio_path"]
    selected_audio_duration_sec = audio_selection["selected_audio_duration_sec"]
    full_call_audio_duration_sec = audio_selection["caller_full_audio_duration_sec"]
    caller_audio_duration_seconds = selected_audio_duration_sec
    caller_only_audio_available = (
        prediction_audio_path is not None
        and prediction_audio_path.exists()
        and selected_audio_duration_sec is not None
    )
    logger.info(
        "VoiceAge report audio selected: selected_audio_for_report=%s selected_audio_duration_sec=%s caller_full_audio_exists=%s prediction_clip_exists=%s",
        str(prediction_audio_path) if prediction_audio_path else None,
        selected_audio_duration_sec,
        audio_selection["caller_full_audio_exists"],
        audio_selection["prediction_clip_exists"],
    )
    report: dict[str, Any] = {
        "call_id": call_id,
        "timestamp": utc_timestamp(),
        "audio_file_used": str(prediction_audio_path) if prediction_audio_path and prediction_audio_path.exists() else None,
        "full_call_audio_file": str(full_audio_path) if full_audio_path and full_audio_path.exists() else None,
        "prediction_audio_file": str(prediction_audio_path) if prediction_audio_path and prediction_audio_path.exists() else None,
        "audio_source": audio_source or audio_selection["selected_audio_source"],
        "assistant_audio_excluded": True,
        "caller_audio_duration_seconds": caller_audio_duration_seconds,
        "full_call_audio_duration_sec": full_call_audio_duration_sec,
        "prediction_audio_duration_sec": selected_audio_duration_sec,
        "selected_audio_for_report": str(prediction_audio_path)
        if prediction_audio_path and prediction_audio_path.exists()
        else None,
        "selected_audio_duration_sec": selected_audio_duration_sec,
        "twilio_media_chunks_received": metadata.get("twilio_audio_chunks_received")
        or metadata.get("twilio_media_chunks_received")
        or metadata.get("twilio_media_frames"),
        "minimum_required_seconds": VOICEAGE_MINIMUM_REPORT_SECONDS,
        "caller_only_audio_available": caller_only_audio_available,
        "prediction_success": False,
        "voiceage_prediction_success": False,
        "predicted_age_group": None,
        "confidence": None,
        "confidence_level": None,
        "class_probabilities": {},
        "model_version": settings.model_version,
        "model_evaluation_reference": MODEL_EVALUATION_REFERENCE,
        "simple_explanation": VOICEAGE_EXPLANATION,
        "caller_only_statement": VOICEAGE_CALLER_ONLY_STATEMENT,
        "failure_reason": None,
    }

    if (
        not caller_only_audio_available
        or caller_audio_duration_seconds is None
        or selected_audio_duration_sec is None
        or selected_audio_duration_sec < VOICEAGE_MINIMUM_REPORT_SECONDS
        or prediction_audio_path is None
    ):
        report["failure_reason"] = "caller_only_audio_missing_or_too_short"
        return report

    try:
        from app.services.model_service import model_service

        if settings.background_voice_isolation_enabled:
            try:
                from app.services.audio_service import decode_audio_file_full_duration
                from app.services.background_voice_isolation_service import BackgroundVoiceIsolationService

                isolation_service = BackgroundVoiceIsolationService()
                original_audio = decode_audio_file_full_duration(prediction_audio_path)
                input_file_duration_sec = len(original_audio) / float(settings.target_sample_rate)
                filtered_audio = isolation_service.filter_audio_for_prediction(
                    original_audio,
                    input_file_duration_sec=input_file_duration_sec,
                )
                isolation_summary = isolation_service.report_summary()
                report["background_voice_isolation"] = isolation_summary
                if filtered_audio is not original_audio:
                    isolated_audio_path = call_dir / "caller_full_background_isolated.wav"
                    sf.write(isolated_audio_path, filtered_audio, settings.target_sample_rate)
                    prediction_audio_path = isolated_audio_path
                    report["audio_file_used"] = str(isolated_audio_path)
                    report["prediction_audio_file"] = str(isolated_audio_path)
                    report["audio_source"] = "caller_only_background_isolated"
                    report["prediction_audio_duration_sec"] = round(
                        len(filtered_audio) / float(settings.target_sample_rate),
                        3,
                    )
                metrics = isolation_summary.get("debug_metrics") or {}
                logger.info(
                    "Background voice isolation summary: enabled=%s reference_ready=%s kept_segments=%s rejected_segments=%s avg_similarity=%s threshold=%s fallback_used=%s fallback_reason=%s reference_source=%s reference_segment_count=%s reference_start_sec=%s reference_end_sec=%s reference_audio_duration_sec=%s input_file_duration_sec=%s full_audio_duration_before_reference_slice=%s vad_audio_duration_sec=%s total_audio_duration_sec=%s audio_rms=%s audio_peak=%s vad_segment_count=%s vad_speech_duration_sec=%s vad_speech_percentage=%s",
                    isolation_summary.get("enabled"),
                    isolation_summary.get("reference_ready"),
                    isolation_summary.get("kept_segments"),
                    isolation_summary.get("rejected_segments"),
                    isolation_summary.get("avg_similarity"),
                    isolation_summary.get("threshold"),
                    isolation_summary.get("fallback_used"),
                    isolation_summary.get("fallback_reason"),
                    metrics.get("reference_source"),
                    metrics.get("reference_segment_count"),
                    metrics.get("reference_start_sec"),
                    metrics.get("reference_end_sec"),
                    metrics.get("reference_audio_duration_sec"),
                    metrics.get("input_file_duration_sec"),
                    metrics.get("full_audio_duration_before_reference_slice"),
                    metrics.get("vad_audio_duration_sec"),
                    metrics.get("total_audio_duration_sec"),
                    metrics.get("audio_rms"),
                    metrics.get("audio_peak"),
                    metrics.get("vad_segment_count"),
                    metrics.get("vad_speech_duration_sec"),
                    metrics.get("vad_speech_percentage"),
                )
                prediction = model_service.predict(filtered_audio)
            except Exception as isolation_exc:
                logger.warning(
                    "Background voice isolation preprocessing failed; falling back to original caller audio: %s",
                    isolation_exc,
                )
                report["background_voice_isolation"] = {
                    "enabled": True,
                    "reference_ready": False,
                    "debug_metrics_enabled": settings.background_voice_debug_metrics,
                    "kept_segments": 0,
                    "rejected_segments": 0,
                    "avg_similarity": None,
                    "threshold": settings.background_voice_isolation_threshold,
                    "fallback_used": True,
                    "failure_reason": isolation_exc.__class__.__name__,
                    "fallback_reason": isolation_exc.__class__.__name__,
                }
                prediction = model_service.predict_audio_file(prediction_audio_path)
        else:
            prediction = model_service.predict_audio_file(prediction_audio_path)
    except Exception as exc:
        logger.exception("VoiceAge prediction failed while generating report for %s", call_dir)
        report["failure_reason"] = str(exc)
        return report

    prediction_payload = pydantic_dump(prediction)
    report.update(
        {
            "prediction_success": True,
            "voiceage_prediction_success": True,
            "predicted_age_group": prediction_payload["predicted_age_group"],
            "confidence": prediction_payload["confidence"],
            "confidence_level": prediction_payload["confidence_level"],
            "class_probabilities": prediction_payload["class_probabilities"],
            "model_version": prediction_payload["model_version"],
            "processing_time_ms": prediction_payload["processing_time_ms"],
        }
    )
    return report


def generate_live_voiceage_prediction(call_dir: Path, call_id: str, audio_path: Path) -> Path:
    metadata = {
        "caller_prediction_clip_wav_path": str(audio_path),
        "caller_prediction_clip_seconds": wav_duration_seconds(audio_path),
    }
    report = generate_voiceage_report(
        call_dir,
        call_id,
        metadata,
        audio_path=audio_path,
        audio_source="caller_prediction_clip",
    )
    live_path = call_dir / "voiceage_live_prediction.json"
    write_json(live_path, report)
    return live_path


def extract_conversation(events: list[dict[str, Any]]) -> dict[str, Any]:
    user_turns: list[str] = []
    assistant_turns: list[str] = []
    current_assistant: list[str] = []

    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") or {}
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(payload.get("transcript") or "").strip()
            if transcript:
                user_turns.append(transcript)
        elif event_type == "response.created":
            if current_assistant:
                assistant_turns.append("".join(current_assistant).strip())
                current_assistant = []
        elif event_type == "response.audio_transcript.delta":
            current_assistant.append(str(payload.get("delta") or ""))

    if current_assistant:
        assistant_turns.append("".join(current_assistant).strip())

    assistant_turns = [turn for turn in assistant_turns if turn]
    transcript_lines: list[str] = []
    max_turns = max(len(user_turns), len(assistant_turns))
    for index in range(max_turns):
        if index < len(user_turns):
            transcript_lines.append(f"User: {user_turns[index]}")
        if index < len(assistant_turns):
            transcript_lines.append(f"Assistant: {assistant_turns[index]}")

    return {
        "transcript": "\n".join(transcript_lines) if transcript_lines else None,
        "assistant_transcript": "\n".join(assistant_turns) if assistant_turns else None,
        "user_transcript": "\n".join(user_turns) if user_turns else None,
        "number_of_user_turns": len(user_turns),
        "number_of_assistant_turns": len(assistant_turns),
    }


def extract_errors_and_warnings(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") or {}
        lowered = event_type.lower()
        if "error" in lowered or "warning" in lowered or event_type == "error":
            findings.append(
                {
                    "timestamp": event.get("timestamp"),
                    "source": event.get("source"),
                    "event_type": event_type,
                    "detail": payload.get("error") or payload.get("warning") or payload,
                }
            )
    return findings


def build_latency_metrics(raw_metrics: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    durations = raw_metrics.get("durations_ms") or {}

    def metric(*names: str) -> float | None:
        for name in names:
            value = durations.get(name)
            if value is not None:
                return value
        return None

    latency_metrics = {
        "speech_started_to_speech_stopped_ms": metric("speech_started_to_speech_stopped"),
        "speech_stopped_to_response_created_ms": metric("speech_stopped_to_response_created"),
        "response_created_to_first_audio_delta_ms": metric(
            "response_created_to_first_response_audio_delta",
            "response_created_to_first_audio_delta",
        ),
        "first_audio_delta_to_twilio_send_ms": metric(
            "first_response_audio_delta_to_first_twilio_audio_send",
            "first_audio_delta_to_twilio_send",
        ),
        "total_response_latency_ms": metric(
            "speech_stopped_to_first_twilio_audio_send",
            "total_response_latency",
        ),
    }

    missing = [key for key, value in latency_metrics.items() if value is None]
    if missing:
        warnings.append(f"Latency metrics unavailable: {', '.join(missing)}.")

    negative = [key for key, value in latency_metrics.items() if isinstance(value, (int, float)) and value < 0]
    if negative:
        warnings.append(
            "Some latency metrics were negative, usually because older logs marked the assistant greeting "
            f"before caller speech: {', '.join(negative)}."
        )

    return latency_metrics


def generate_conversation_report(
    call_dir: Path,
    call_id: str,
    metadata: dict[str, Any],
    raw_metrics: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    warnings: list[str] = []
    conversation = extract_conversation(events)
    errors_and_warnings = extract_errors_and_warnings(events)

    if not conversation["transcript"]:
        warnings.append("Transcript unavailable.")
    if not events:
        warnings.append("Conversation events unavailable.")
    if not raw_metrics:
        warnings.append("Latency metrics file unavailable.")

    latency_metrics = build_latency_metrics(raw_metrics, warnings)
    errors_and_warnings.extend(
        {
            "timestamp": utc_timestamp(),
            "source": "report",
            "event_type": "report.warning",
            "detail": warning,
        }
        for warning in warnings
    )

    return {
        "call_id": call_id,
        "timestamp": utc_timestamp(),
        "call_duration_seconds": seconds_between(metadata.get("started_at"), metadata.get("ended_at")),
        **conversation,
        "latency_metrics": latency_metrics,
        "latency_explanation": LATENCY_EXPLANATION,
        "errors_warnings": errors_and_warnings,
    }


def event_seen(events: list[dict[str, Any]], source: str | None = None, event_types: set[str] | None = None) -> bool:
    for event in events:
        if source is not None and event.get("source") != source:
            continue
        if event_types is not None and event.get("event_type") not in event_types:
            continue
        return True
    return False


def build_call_quality_notes(
    voiceage_report: dict[str, Any],
    conversation_report: dict[str, Any],
    azure_realtime_connected: bool,
    twilio_stream_connected: bool,
) -> list[str]:
    notes: list[str] = []
    if voiceage_report.get("prediction_success"):
        notes.append("VoiceAge prediction completed from caller-only audio.")
    else:
        notes.append(f"VoiceAge prediction did not complete: {voiceage_report.get('failure_reason')}")
    notes.append("Assistant/AI audio was excluded from VoiceAge prediction.")

    if conversation_report.get("transcript"):
        notes.append("Conversation transcript was captured.")
    else:
        notes.append("Conversation transcript was not available.")

    total_latency = (conversation_report.get("latency_metrics") or {}).get("total_response_latency_ms")
    if total_latency is None:
        notes.append("End-to-end response latency was unavailable.")
    elif total_latency < 0:
        notes.append("End-to-end response latency was invalid in this log and should be rechecked.")
    elif total_latency <= 1500:
        notes.append("Realtime response latency looked demo-friendly.")
    elif total_latency <= 3000:
        notes.append("Realtime response latency was usable but may feel slightly delayed.")
    else:
        notes.append("Realtime response latency may feel slow to callers.")

    if not azure_realtime_connected:
        notes.append("Azure/OpenAI Realtime connection was not confirmed.")
    if not twilio_stream_connected:
        notes.append("Twilio Media Stream connection was not confirmed.")
    return notes


def generate_combined_report(
    call_id: str,
    voiceage_report: dict[str, Any],
    conversation_report: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    azure_realtime_connected = event_seen(
        events,
        event_types={"openai.connected", "session.created", "session.updated", "first_received_event"},
    )
    twilio_stream_connected = event_seen(events, source="twilio", event_types={"connected", "start"})
    realtime_conversation_success = bool(
        azure_realtime_connected
        and twilio_stream_connected
        and (
            conversation_report.get("number_of_user_turns", 0) > 0
            or conversation_report.get("number_of_assistant_turns", 0) > 0
        )
    )
    voiceage_prediction_success = bool(voiceage_report.get("prediction_success"))
    caller_only_audio_available = bool(voiceage_report.get("caller_only_audio_available"))
    call_quality_notes = build_call_quality_notes(
        voiceage_report,
        conversation_report,
        azure_realtime_connected,
        twilio_stream_connected,
    )
    has_errors = any(
        item.get("event_type") != "report.warning"
        for item in conversation_report.get("errors_warnings", [])
    )
    has_warnings = bool(conversation_report.get("errors_warnings"))
    latency_values = (conversation_report.get("latency_metrics") or {}).values()
    invalid_latency = any(isinstance(value, (int, float)) and value < 0 for value in latency_values)

    if voiceage_prediction_success and realtime_conversation_success and not has_errors and not has_warnings:
        recommendation = "Ready for demo"
    elif invalid_latency:
        recommendation = "Partial success"
    elif voiceage_prediction_success or realtime_conversation_success or azure_realtime_connected or twilio_stream_connected:
        recommendation = "Partial success"
    else:
        recommendation = "Needs debugging"

    voiceage_summary = (
        f"VoiceAge predicted {voiceage_report.get('predicted_age_group')} "
        f"with {format_value(voiceage_report.get('confidence'))} confidence."
        if voiceage_prediction_success
        else f"VoiceAge prediction failed: {voiceage_report.get('failure_reason')}"
    )
    conversation_summary = (
        f"The realtime assistant captured {conversation_report.get('number_of_user_turns', 0)} user turn(s) "
        f"and {conversation_report.get('number_of_assistant_turns', 0)} assistant turn(s)."
    )
    latency_value = (conversation_report.get("latency_metrics") or {}).get("total_response_latency_ms")
    latency_summary = (
        f"Measured response latency was {format_value(latency_value, ' ms')}."
        if latency_value is not None
        else "Response latency was unavailable."
    )

    return {
        "call_id": call_id,
        "timestamp": utc_timestamp(),
        "executive_summary": (
            f"{voiceage_summary} {VOICEAGE_COMBINED_STATEMENT} "
            f"{conversation_summary} {latency_summary}"
        ),
        "voiceage_result": voiceage_report,
        "conversation_result": {
            "call_duration_seconds": conversation_report.get("call_duration_seconds"),
            "number_of_user_turns": conversation_report.get("number_of_user_turns"),
            "number_of_assistant_turns": conversation_report.get("number_of_assistant_turns"),
            "transcript_available": bool(conversation_report.get("transcript")),
        },
        "latency_result": conversation_report.get("latency_metrics"),
        "call_quality_notes": call_quality_notes,
        "voiceage_audio_policy": VOICEAGE_COMBINED_STATEMENT,
        "systems_worked": {
            "voiceage_prediction_success": voiceage_prediction_success,
            "realtime_conversation_success": realtime_conversation_success,
            "azure_realtime_connected": azure_realtime_connected,
            "twilio_stream_connected": twilio_stream_connected,
            "caller_only_audio_available": caller_only_audio_available,
            "assistant_audio_excluded_from_voiceage": True,
        },
        "recommendation": recommendation,
    }


def voiceage_markdown(report: dict[str, Any]) -> str:
    probabilities = report.get("class_probabilities") or {}
    probability_rows = "\n".join(
        f"| {label} | {probability:.4f} |"
        for label, probability in probabilities.items()
    )
    if not probability_rows:
        probability_rows = "| Unavailable | Unavailable |"

    return f"""
# VoiceAge Report

## Summary
| Field | Value |
| --- | --- |
| Call ID | {report.get("call_id")} |
| Generated | {report.get("timestamp")} |
| Audio File Used | {report.get("audio_file_used") or "Unavailable"} |
| Full Call Audio File | {report.get("full_call_audio_file") or "Unavailable"} |
| Audio Source | {report.get("audio_source") or "Unavailable"} |
| Assistant Audio Excluded | {bool_text(bool(report.get("assistant_audio_excluded")))} |
| Full Call Audio Duration | {format_value(report.get("full_call_audio_duration_sec"), " seconds")} |
| Prediction Audio Duration | {format_value(report.get("prediction_audio_duration_sec"), " seconds")} |
| Twilio Media Chunks Received | {format_value(report.get("twilio_media_chunks_received"))} |
| Minimum Required Audio | {format_value(report.get("minimum_required_seconds"), " seconds")} |
| Prediction Success | {bool_text(bool(report.get("prediction_success")))} |
| Predicted Age Group | {report.get("predicted_age_group") or "Unavailable"} |
| Confidence | {format_value(report.get("confidence"))} |
| Confidence Level | {report.get("confidence_level") or "Unavailable"} |
| Model Version | {report.get("model_version") or "Unavailable"} |

## Class Probabilities
| Age Group | Probability |
| --- | --- |
{probability_rows}

## Model Reference
| Metric | Value |
| --- | --- |
| Accuracy | {MODEL_EVALUATION_REFERENCE["accuracy"]} |
| Weighted F1 | {MODEL_EVALUATION_REFERENCE["weighted_f1"]} |

## Explanation
{report.get("simple_explanation") or VOICEAGE_EXPLANATION}

{report.get("caller_only_statement") or VOICEAGE_CALLER_ONLY_STATEMENT}

## Failure Reason
{report.get("failure_reason") or "None"}
"""


def conversation_markdown(report: dict[str, Any]) -> str:
    latency = report.get("latency_metrics") or {}
    errors = report.get("errors_warnings") or []
    error_rows = "\n".join(
        f"| {item.get('timestamp') or 'Unavailable'} | {item.get('source') or 'Unavailable'} | "
        f"{item.get('event_type') or 'Unavailable'} | {item.get('detail') or 'Unavailable'} |"
        for item in errors
    )
    if not error_rows:
        error_rows = "| None | None | None | None |"

    return f"""
# Realtime Conversation Report

## Summary
| Field | Value |
| --- | --- |
| Call ID | {report.get("call_id")} |
| Generated | {report.get("timestamp")} |
| Call Duration | {format_value(report.get("call_duration_seconds"), " seconds")} |
| User Turns | {report.get("number_of_user_turns", 0)} |
| Assistant Turns | {report.get("number_of_assistant_turns", 0)} |

## Transcript
{report.get("transcript") or "Transcript unavailable."}

## User Transcript
{report.get("user_transcript") or "User transcript unavailable."}

## Assistant Transcript
{report.get("assistant_transcript") or "Assistant transcript unavailable."}

## Latency Metrics
{report.get("latency_explanation") or LATENCY_EXPLANATION}

| Metric | Value |
| --- | --- |
| Speech Started to Speech Stopped | {format_value(latency.get("speech_started_to_speech_stopped_ms"), " ms")} |
| Speech Stopped to Response Created | {format_value(latency.get("speech_stopped_to_response_created_ms"), " ms")} |
| Response Created to First Audio Delta | {format_value(latency.get("response_created_to_first_audio_delta_ms"), " ms")} |
| First Audio Delta to Twilio Send | {format_value(latency.get("first_audio_delta_to_twilio_send_ms"), " ms")} |
| Total Response Latency | {format_value(latency.get("total_response_latency_ms"), " ms")} |

## Errors and Warnings
| Timestamp | Source | Event | Detail |
| --- | --- | --- | --- |
{error_rows}
"""


def combined_markdown(report: dict[str, Any]) -> str:
    systems = report.get("systems_worked") or {}
    notes = "\n".join(f"- {note}" for note in report.get("call_quality_notes") or [])
    latency = report.get("latency_result") or {}
    voiceage = report.get("voiceage_result") or {}
    conversation = report.get("conversation_result") or {}

    return f"""
# Combined Call Report

## Executive Summary
{report.get("executive_summary")}

{report.get("voiceage_audio_policy") or VOICEAGE_COMBINED_STATEMENT}

## Recommendation
**{report.get("recommendation")}**

## VoiceAge Result
| Field | Value |
| --- | --- |
| Prediction Success | {bool_text(bool(systems.get("voiceage_prediction_success")))} |
| Predicted Age Group | {voiceage.get("predicted_age_group") or "Unavailable"} |
| Confidence | {format_value(voiceage.get("confidence"))} |
| Confidence Level | {voiceage.get("confidence_level") or "Unavailable"} |
| Audio Source | {voiceage.get("audio_source") or "Unavailable"} |
| Caller Audio Available | {bool_text(bool(systems.get("caller_only_audio_available")))} |
| Full Call Audio Duration | {format_value(voiceage.get("full_call_audio_duration_sec"), " seconds")} |
| Prediction Audio Duration | {format_value(voiceage.get("prediction_audio_duration_sec"), " seconds")} |
| Twilio Media Chunks Received | {format_value(voiceage.get("twilio_media_chunks_received"))} |
| Assistant Audio Excluded | {bool_text(bool(systems.get("assistant_audio_excluded_from_voiceage")))} |

## Conversation Result
| Field | Value |
| --- | --- |
| Realtime Conversation Success | {bool_text(bool(systems.get("realtime_conversation_success")))} |
| Call Duration | {format_value(conversation.get("call_duration_seconds"), " seconds")} |
| Transcript Available | {bool_text(bool(conversation.get("transcript_available")))} |
| User Turns | {conversation.get("number_of_user_turns", 0)} |
| Assistant Turns | {conversation.get("number_of_assistant_turns", 0)} |

## Latency Result
| Metric | Value |
| --- | --- |
| Total Response Latency | {format_value(latency.get("total_response_latency_ms"), " ms")} |
| Speech Stopped to Response Created | {format_value(latency.get("speech_stopped_to_response_created_ms"), " ms")} |
| Response Created to First Audio Delta | {format_value(latency.get("response_created_to_first_audio_delta_ms"), " ms")} |
| First Audio Delta to Twilio Send | {format_value(latency.get("first_audio_delta_to_twilio_send_ms"), " ms")} |

## System Checks
| System | Worked |
| --- | --- |
| VoiceAge Prediction | {bool_text(bool(systems.get("voiceage_prediction_success")))} |
| Realtime Conversation | {bool_text(bool(systems.get("realtime_conversation_success")))} |
| Azure/OpenAI Realtime Connected | {bool_text(bool(systems.get("azure_realtime_connected")))} |
| Twilio Stream Connected | {bool_text(bool(systems.get("twilio_stream_connected")))} |
| Caller-Only Audio Available | {bool_text(bool(systems.get("caller_only_audio_available")))} |
| Assistant Audio Excluded From VoiceAge | {bool_text(bool(systems.get("assistant_audio_excluded_from_voiceage")))} |

## Call Quality Notes
{notes or "- No call quality notes available."}
"""


def generate_reports_for_call(call_dir: str | Path) -> Path:
    call_dir = Path(call_dir)
    if not call_dir.exists() or not call_dir.is_dir():
        raise FileNotFoundError(f"Call directory not found: {call_dir}")

    metadata = load_json(call_dir / "metadata.json")
    raw_metrics = load_json(call_dir / "latency_metrics.json")
    events = read_events(call_dir / "events.jsonl")
    call_id = metadata.get("call_sid") or raw_metrics.get("call_sid") or call_dir.name
    reports_dir = call_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    voiceage_report = generate_voiceage_report(call_dir, call_id, metadata)
    conversation_report = generate_conversation_report(call_dir, call_id, metadata, raw_metrics, events)
    combined_report = generate_combined_report(call_id, voiceage_report, conversation_report, events)

    write_json(reports_dir / "voiceage_report.json", voiceage_report)
    write_text(reports_dir / "voiceage_report.md", voiceage_markdown(voiceage_report))
    write_json(reports_dir / "conversation_report.json", conversation_report)
    write_text(reports_dir / "conversation_report.md", conversation_markdown(conversation_report))
    write_json(reports_dir / "combined_call_report.json", combined_report)
    write_text(reports_dir / "combined_call_report.md", combined_markdown(combined_report))

    logger.info("Post-call reports generated: %s", reports_dir)
    return reports_dir
