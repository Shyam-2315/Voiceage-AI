from __future__ import annotations

import base64
import json
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.conversation_logger import RealtimeConversationLogger, ULAW_SAMPLE_RATE_HZ
from app.services import realtime_bridge, report_service
from app.services.conversation_style_service import get_conversation_style


def ulaw_payload(seconds: int = 1) -> str:
    return base64.b64encode(b"\xff" * ULAW_SAMPLE_RATE_HZ * seconds).decode("ascii")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return round(wav_file.getnframes() / float(wav_file.getframerate()), 3)


def write_silent_wav(path: Path, seconds: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(ULAW_SAMPLE_RATE_HZ)
        wav_file.writeframes(b"\x00\x00" * ULAW_SAMPLE_RATE_HZ * seconds)


class FakePrediction:
    def dict(self) -> dict[str, object]:
        return {
            "predicted_age_group": "Adult",
            "confidence": 0.9,
            "confidence_level": "high",
            "class_probabilities": {
                "Adult": 0.9,
                "Middle_Age": 0.05,
                "Senior": 0.03,
                "Teen": 0.02,
            },
            "model_version": "test",
            "processing_time_ms": 1,
        }


def fake_model_module() -> types.ModuleType:
    module = types.ModuleType("app.services.model_service")
    model_service = types.SimpleNamespace()
    model_service.predict_audio_file = Mock(return_value=FakePrediction())
    module.model_service = model_service
    return module


class RealtimeAudioCaptureTest(unittest.IsolatedAsyncioTestCase):
    async def test_full_caller_audio_is_not_capped_by_prediction_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RealtimeConversationLogger(root_dir=Path(tmpdir), capture_seconds=5)
            await logger.start("CA123", "MZ123")
            await logger.mark_stream_started()

            for _ in range(12):
                await logger.increment_twilio_media_event()
                ready = await logger.capture_caller_payload(ulaw_payload())

            await logger.mark_stream_closed("twilio_stop")
            await logger.finalize()

            self.assertIsNotNone(logger.session_dir)
            call_dir = logger.session_dir
            metadata = json.loads((call_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertTrue(ready)
            self.assertAlmostEqual(wav_duration(call_dir / "caller_full_audio.wav"), 12.0)
            self.assertAlmostEqual(wav_duration(call_dir / "caller_prediction_clip.wav"), 5.0)
            self.assertAlmostEqual(wav_duration(call_dir / "caller_reference_audio.wav"), 5.0)
            self.assertFalse((call_dir / "caller_only_audio.wav").exists())
            self.assertEqual(metadata["twilio_media_events"], 12)
            self.assertEqual(metadata["twilio_audio_chunks_received"], 12)
            self.assertEqual(metadata["twilio_audio_bytes_received"], 12 * ULAW_SAMPLE_RATE_HZ)
            self.assertEqual(metadata["caller_full_audio_seconds"], 12.0)
            self.assertEqual(metadata["caller_prediction_clip_seconds"], 5.0)
            self.assertEqual(metadata["stream_close_reason"], "twilio_stop")


class RealtimeReportAudioSelectionTest(unittest.TestCase):
    def test_final_voiceage_report_prefers_full_call_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            call_dir = Path(tmpdir)
            full_audio = call_dir / "caller_full_audio.wav"
            prediction_clip = call_dir / "caller_prediction_clip.wav"
            write_silent_wav(full_audio, seconds=12)
            write_silent_wav(prediction_clip, seconds=5)

            metadata = {
                "caller_full_audio_wav_path": str(full_audio),
                "caller_prediction_clip_wav_path": str(prediction_clip),
                "twilio_audio_chunks_received": 600,
            }

            module = fake_model_module()
            with patch.dict(sys.modules, {"app.services.model_service": module}):
                report = report_service.generate_voiceage_report(call_dir, "CA123", metadata)

            self.assertTrue(report["prediction_success"])
            module.model_service.predict_audio_file.assert_called_once_with(full_audio)
            self.assertEqual(report["audio_file_used"], str(full_audio))
            self.assertEqual(report["full_call_audio_file"], str(full_audio))
            self.assertEqual(report["full_call_audio_duration_sec"], 12.0)
            self.assertEqual(report["prediction_audio_duration_sec"], 12.0)
            self.assertEqual(report["selected_audio_for_report"], str(full_audio))
            self.assertEqual(report["selected_audio_duration_sec"], 12.0)
            self.assertEqual(report["twilio_media_chunks_received"], 600)

    def test_final_voiceage_report_falls_back_to_legacy_caller_only_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            call_dir = Path(tmpdir)
            legacy_audio = call_dir / "caller_only_audio.wav"
            prediction_clip = call_dir / "caller_prediction_clip.wav"
            write_silent_wav(legacy_audio, seconds=11)
            write_silent_wav(prediction_clip, seconds=5)

            metadata = {
                "caller_only_audio_wav_path": str(legacy_audio),
                "caller_prediction_clip_wav_path": str(prediction_clip),
            }

            module = fake_model_module()
            with patch.dict(sys.modules, {"app.services.model_service": module}):
                report = report_service.generate_voiceage_report(call_dir, "CA123", metadata)

            self.assertTrue(report["prediction_success"])
            module.model_service.predict_audio_file.assert_called_once_with(legacy_audio)
            self.assertEqual(report["audio_file_used"], str(legacy_audio))
            self.assertIsNone(report["full_call_audio_file"])
            self.assertEqual(report["selected_audio_for_report"], str(legacy_audio))
            self.assertEqual(report["selected_audio_duration_sec"], 11.0)

    def test_final_voiceage_report_does_not_use_prediction_clip_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            call_dir = Path(tmpdir)
            prediction_clip = call_dir / "caller_prediction_clip.wav"
            write_silent_wav(prediction_clip, seconds=5)

            metadata = {"caller_prediction_clip_wav_path": str(prediction_clip)}

            module = fake_model_module()
            with patch.dict(sys.modules, {"app.services.model_service": module}):
                report = report_service.generate_voiceage_report(call_dir, "CA123", metadata)

            self.assertFalse(report["prediction_success"])
            self.assertEqual(report["failure_reason"], "caller_only_audio_missing_or_too_short")
            module.model_service.predict_audio_file.assert_not_called()
            self.assertIsNone(report["audio_file_used"])
            self.assertIsNone(report["selected_audio_for_report"])

    def test_live_voiceage_report_uses_prediction_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            call_dir = Path(tmpdir)
            full_audio = call_dir / "caller_full_audio.wav"
            prediction_clip = call_dir / "caller_prediction_clip.wav"
            write_silent_wav(full_audio, seconds=12)
            write_silent_wav(prediction_clip, seconds=5)

            module = fake_model_module()
            with patch.dict(sys.modules, {"app.services.model_service": module}):
                live_path = report_service.generate_live_voiceage_prediction(call_dir, "CA123", prediction_clip)

            report = json.loads(live_path.read_text(encoding="utf-8"))
            self.assertTrue(report["prediction_success"])
            module.model_service.predict_audio_file.assert_called_once_with(prediction_clip)
            self.assertEqual(report["audio_file_used"], str(prediction_clip))
            self.assertEqual(report["full_call_audio_file"], str(full_audio))
            self.assertEqual(report["full_call_audio_duration_sec"], 12.0)
            self.assertEqual(report["prediction_audio_duration_sec"], 5.0)
            self.assertEqual(report["selected_audio_for_report"], str(prediction_clip))
            self.assertEqual(report["selected_audio_duration_sec"], 5.0)


class RealtimeTurnDetectionTest(unittest.TestCase):
    def test_realtime_silence_duration_defaults_to_3000_ms(self) -> None:
        settings = SimpleNamespace(
            realtime_silence_duration_ms=3000,
            realtime_vad_threshold=0.55,
            realtime_vad_prefix_ms=200,
            realtime_allow_adaptive_silence_extension=False,
            realtime_max_adaptive_silence_ms=3500,
        )

        with patch.object(realtime_bridge, "settings", settings):
            turn_detection = realtime_bridge.realtime_turn_detection()

        self.assertEqual(turn_detection["silence_duration_ms"], 3000)

    def test_adaptive_style_does_not_increase_silence_unless_allowed(self) -> None:
        settings = SimpleNamespace(
            realtime_silence_duration_ms=3000,
            realtime_vad_threshold=0.55,
            realtime_vad_prefix_ms=200,
            realtime_allow_adaptive_silence_extension=False,
            realtime_max_adaptive_silence_ms=3500,
        )

        with patch.object(realtime_bridge, "settings", settings):
            timing = realtime_bridge.realtime_turn_detection_timing(get_conversation_style("Senior"))

        self.assertEqual(timing["adaptive_interruption_delay_ms"], 1100)
        self.assertEqual(timing["final_turn_detection_silence_ms"], 3000)

    def test_adaptive_extension_is_capped_when_explicitly_allowed(self) -> None:
        senior_style = get_conversation_style("Senior")
        slow_style = type(senior_style)(
            **{
                **senior_style.as_dict(),
                "interruption_delay_ms": 10000,
            }
        )
        settings = SimpleNamespace(
            realtime_silence_duration_ms=3000,
            realtime_vad_threshold=0.55,
            realtime_vad_prefix_ms=200,
            realtime_allow_adaptive_silence_extension=True,
            realtime_max_adaptive_silence_ms=3500,
        )

        with patch.object(realtime_bridge, "settings", settings):
            timing = realtime_bridge.realtime_turn_detection_timing(slow_style)

        self.assertEqual(timing["adaptive_interruption_delay_ms"], 10000)
        self.assertEqual(timing["final_turn_detection_silence_ms"], 3500)


if __name__ == "__main__":
    unittest.main()
