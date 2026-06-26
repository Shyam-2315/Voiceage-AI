from __future__ import annotations

import logging
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.services import report_service
from app.services import background_voice_isolation_service as isolation


def fake_settings(
    enabled: bool = False,
    debug_metrics: bool = True,
    reference_seconds: int = 5,
    reference_from_vad: bool = True,
    reference_max_search_sec: int = 20,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_version="test",
        target_sample_rate=16_000,
        background_voice_isolation_enabled=enabled,
        background_voice_isolation_threshold=0.75,
        background_voice_reference_seconds=reference_seconds,
        background_voice_reference_from_vad=reference_from_vad,
        background_voice_reference_max_search_sec=reference_max_search_sec,
        background_voice_min_segment_sec=1.0,
        background_voice_debug_metrics=debug_metrics,
    )


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
    model_service.predict = Mock(return_value=FakePrediction())
    model_service.predict_audio_file = Mock(return_value=FakePrediction())
    module.model_service = model_service
    return module


def write_tone_wav(path: Path, seconds: int, sample_rate: int = 16_000) -> None:
    samples = (np.ones(seconds * sample_rate, dtype=np.int16) * 1000).tobytes()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples)


class BackgroundVoiceIsolationServiceTest(unittest.TestCase):
    def test_disabled_flag_preserves_audio_object(self) -> None:
        audio = np.ones(16_000, dtype=np.float32)
        with patch.object(isolation, "settings", fake_settings(enabled=False)):
            service = isolation.BackgroundVoiceIsolationService()

            result = service.filter_audio_for_prediction(audio)

        self.assertIs(result, audio)
        self.assertFalse(service.report_summary()["enabled"])

    def test_failure_falls_back_safely(self) -> None:
        audio = np.ones(16_000, dtype=np.float32)
        with patch.object(isolation, "settings", fake_settings(enabled=True)):
            service = isolation.BackgroundVoiceIsolationService()

            with patch.object(service, "initialize_reference", side_effect=RuntimeError("model unavailable")):
                with patch.object(
                    service,
                    "detect_speech_segments",
                    return_value=[isolation.SpeechSegment(0, 16_000)],
                ):
                    with self.assertLogs(isolation.logger.name, level=logging.WARNING):
                        result = service.filter_audio_for_prediction(audio)

        self.assertIs(result, audio)
        summary = service.report_summary()
        self.assertTrue(summary["fallback_used"])
        self.assertEqual(summary["failure_reason"], "RuntimeError")

    def test_threshold_decision(self) -> None:
        self.assertTrue(isolation.threshold_decision(0.75, 0.75))
        self.assertTrue(isolation.threshold_decision(0.9, 0.75))
        self.assertFalse(isolation.threshold_decision(0.74, 0.75))

    def test_service_initializes_without_twilio_or_azure_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(isolation, "settings", fake_settings(enabled=False)):
                service = isolation.BackgroundVoiceIsolationService()

        self.assertFalse(service.reference_ready)
        self.assertEqual(service.threshold, 0.75)

    def test_no_raw_audio_in_decision_logs(self) -> None:
        audio = np.array([0.123456, 0.654321, 0.0], dtype=np.float32)
        with patch.object(isolation, "settings", fake_settings(enabled=True)):
            service = isolation.BackgroundVoiceIsolationService()
            service.reference_embedding = np.array([1.0, 0.0], dtype=np.float32)

            with patch.object(
                service,
                "generate_embedding",
                return_value=np.array([1.0, 0.0], dtype=np.float32),
            ):
                with self.assertLogs(isolation.logger.name, level=logging.INFO) as logs:
                    keep = service.should_keep_segment(audio)

        self.assertTrue(keep)
        log_text = "\n".join(logs.output)
        self.assertIn("similarity=", log_text)
        self.assertNotIn("0.123456", log_text)
        self.assertNotIn("0.654321", log_text)

    def test_metrics_generated_for_normal_audio_array(self) -> None:
        audio = np.ones(16_000, dtype=np.float32) * 0.25
        with patch.object(isolation, "settings", fake_settings(enabled=True)):
            metrics = isolation.audio_debug_metrics(
                audio,
                sample_rate=16_000,
                min_segment_sec=1.0,
                threshold=0.75,
            )

        self.assertEqual(metrics["total_audio_duration_sec"], 1.0)
        self.assertEqual(metrics["sample_rate"], 16_000)
        self.assertEqual(metrics["num_samples"], 16_000)
        self.assertAlmostEqual(metrics["audio_rms"], 0.25)
        self.assertAlmostEqual(metrics["audio_peak"], 0.25)
        self.assertFalse(metrics["audio_is_silent"])

    def test_silent_audio_detected(self) -> None:
        audio = np.zeros(16_000, dtype=np.float32)
        with patch.object(isolation, "settings", fake_settings(enabled=True)):
            metrics = isolation.audio_debug_metrics(
                audio,
                sample_rate=16_000,
                min_segment_sec=1.0,
                threshold=0.75,
            )

        self.assertEqual(metrics["audio_rms"], 0.0)
        self.assertEqual(metrics["audio_peak"], 0.0)
        self.assertTrue(metrics["audio_is_silent"])

    def test_zero_vad_segments_produces_fallback_reason(self) -> None:
        audio = np.ones(16_000, dtype=np.float32) * 0.2
        with patch.object(isolation, "settings", fake_settings(enabled=True)):
            service = isolation.BackgroundVoiceIsolationService()

            def reference_ready(
                _audio: np.ndarray,
                vad_segments: list[isolation.SpeechSegment] | None = None,
            ) -> bool:
                service.reference_embedding = np.array([1.0, 0.0], dtype=np.float32)
                service.summary.reference_ready = True
                return True

            with patch.object(service, "initialize_reference", side_effect=reference_ready):
                with patch.object(service, "detect_speech_segments", return_value=[]):
                    with self.assertLogs(isolation.logger.name, level=logging.WARNING) as logs:
                        result = service.filter_audio_for_prediction(audio)

        self.assertIs(result, audio)
        summary = service.report_summary()
        self.assertEqual(summary["fallback_reason"], "no_speech_segments")
        self.assertEqual(summary["debug_metrics"]["vad_segment_count"], 0)
        self.assertIn("recommended_next_action", "\n".join(logs.output))

    def test_debug_metrics_can_be_disabled(self) -> None:
        audio = np.ones(16_000, dtype=np.float32) * 0.2
        with patch.object(isolation, "settings", fake_settings(enabled=True, debug_metrics=False)):
            service = isolation.BackgroundVoiceIsolationService()
            service.update_debug_metrics(audio)

        summary = service.report_summary()
        self.assertFalse(summary["debug_metrics_enabled"])
        self.assertNotIn("debug_metrics", summary)

    def test_full_sixty_second_audio_remains_sixty_seconds_for_vad(self) -> None:
        audio = np.ones(60 * 16_000, dtype=np.float32) * 0.2
        captured_vad_durations: list[float] = []

        def no_segments(vad_audio: np.ndarray) -> list[isolation.SpeechSegment]:
            captured_vad_durations.append(round(vad_audio.size / 16_000, 3))
            return []

        with patch.object(isolation, "settings", fake_settings(enabled=True, reference_seconds=3)):
            service = isolation.BackgroundVoiceIsolationService()
            service.reference_embedding = np.array([1.0, 0.0], dtype=np.float32)
            service.summary.reference_ready = True

            with patch.object(service, "detect_speech_segments", side_effect=no_segments):
                result = service.filter_audio_for_prediction(audio, input_file_duration_sec=60.0)

        self.assertIs(result, audio)
        self.assertEqual(captured_vad_durations, [60.0])
        metrics = service.report_summary()["debug_metrics"]
        self.assertEqual(metrics["input_file_duration_sec"], 60.0)
        self.assertEqual(metrics["vad_audio_duration_sec"], 60.0)
        self.assertEqual(metrics["total_audio_duration_sec"], 60.0)

    def test_reference_fallback_audio_is_sliced_to_reference_seconds(self) -> None:
        audio = np.ones(60 * 16_000, dtype=np.float32) * 0.2
        captured_reference_vad_durations: list[float] = []

        def no_reference_segments(reference_audio: np.ndarray) -> list[isolation.SpeechSegment]:
            captured_reference_vad_durations.append(round(reference_audio.size / 16_000, 3))
            return []

        with patch.object(isolation, "settings", fake_settings(enabled=True, reference_seconds=3)):
            service = isolation.BackgroundVoiceIsolationService()

            with patch.object(service, "detect_speech_segments", side_effect=no_reference_segments):
                with patch.object(
                    service,
                    "generate_embedding",
                    return_value=np.array([1.0, 0.0], dtype=np.float32),
                ):
                    reference_ready = service.initialize_reference(audio)

        self.assertTrue(reference_ready)
        self.assertEqual(captured_reference_vad_durations, [60.0])
        metrics = service.report_summary()["debug_metrics"]
        self.assertEqual(metrics["full_audio_duration_before_reference_slice"], 60.0)
        self.assertEqual(metrics["reference_audio_duration_sec"], 3.0)
        self.assertEqual(metrics["reference_source"], "first_seconds_fallback")

    def test_reference_is_built_from_first_vad_speech_segment(self) -> None:
        audio = np.zeros(6 * 16_000, dtype=np.float32)
        audio[2 * 16_000 : 4 * 16_000] = 0.3
        captured_reference: list[np.ndarray] = []

        def capture_embedding(reference_audio: np.ndarray) -> np.ndarray:
            captured_reference.append(reference_audio.copy())
            return np.array([1.0, 0.0], dtype=np.float32)

        with patch.object(isolation, "settings", fake_settings(enabled=True, reference_seconds=1)):
            service = isolation.BackgroundVoiceIsolationService()
            with patch.object(
                service,
                "detect_speech_segments",
                return_value=[isolation.SpeechSegment(2 * 16_000, 4 * 16_000)],
            ):
                with patch.object(service, "generate_embedding", side_effect=capture_embedding):
                    self.assertTrue(service.initialize_reference(audio))

        self.assertEqual(len(captured_reference), 1)
        self.assertEqual(captured_reference[0].size, 16_000)
        self.assertTrue(np.allclose(captured_reference[0], 0.3))
        metrics = service.report_summary()["debug_metrics"]
        self.assertEqual(metrics["reference_source"], "vad_speech_segments")
        self.assertEqual(metrics["reference_segment_count"], 1)
        self.assertEqual(metrics["reference_start_sec"], 2.0)
        self.assertEqual(metrics["reference_end_sec"], 3.0)

    def test_silence_before_reference_speech_is_ignored(self) -> None:
        audio = np.zeros(5 * 16_000, dtype=np.float32)
        audio[1 * 16_000 : 4 * 16_000] = 0.25
        captured_reference: list[np.ndarray] = []

        with patch.object(isolation, "settings", fake_settings(enabled=True, reference_seconds=2)):
            service = isolation.BackgroundVoiceIsolationService()
            with patch.object(
                service,
                "detect_speech_segments",
                return_value=[isolation.SpeechSegment(1 * 16_000, 4 * 16_000)],
            ):
                with patch.object(
                    service,
                    "generate_embedding",
                    side_effect=lambda reference_audio: captured_reference.append(reference_audio.copy())
                    or np.array([1.0, 0.0], dtype=np.float32),
                ):
                    self.assertTrue(service.initialize_reference(audio))

        self.assertEqual(captured_reference[0].size, 2 * 16_000)
        self.assertTrue(np.all(captured_reference[0] != 0.0))
        metrics = service.report_summary()["debug_metrics"]
        self.assertEqual(metrics["reference_source"], "vad_speech_segments")
        self.assertEqual(metrics["reference_audio_duration_sec"], 2.0)

    def test_reference_falls_back_to_first_seconds_when_vad_finds_no_speech(self) -> None:
        audio = np.zeros(5 * 16_000, dtype=np.float32)
        audio[1 * 16_000 : 5 * 16_000] = 0.4
        captured_reference: list[np.ndarray] = []

        with patch.object(isolation, "settings", fake_settings(enabled=True, reference_seconds=2)):
            service = isolation.BackgroundVoiceIsolationService()
            with patch.object(service, "detect_speech_segments", return_value=[]):
                with patch.object(
                    service,
                    "generate_embedding",
                    side_effect=lambda reference_audio: captured_reference.append(reference_audio.copy())
                    or np.array([1.0, 0.0], dtype=np.float32),
                ):
                    self.assertTrue(service.initialize_reference(audio))

        self.assertEqual(captured_reference[0].size, 2 * 16_000)
        self.assertTrue(np.allclose(captured_reference[0][:16_000], 0.0))
        self.assertTrue(np.allclose(captured_reference[0][16_000:], 0.4))
        metrics = service.report_summary()["debug_metrics"]
        self.assertEqual(metrics["reference_source"], "first_seconds_fallback")
        self.assertEqual(metrics["reference_segment_count"], 0)
        self.assertEqual(metrics["reference_start_sec"], 0.0)
        self.assertEqual(metrics["reference_end_sec"], 2.0)

    def test_reference_duration_respects_configured_reference_seconds(self) -> None:
        audio = np.zeros(8 * 16_000, dtype=np.float32)
        audio[1 * 16_000 : 3 * 16_000] = 0.2
        audio[4 * 16_000 : 7 * 16_000] = 0.5
        captured_reference: list[np.ndarray] = []

        with patch.object(isolation, "settings", fake_settings(enabled=True, reference_seconds=3)):
            service = isolation.BackgroundVoiceIsolationService()
            with patch.object(
                service,
                "detect_speech_segments",
                return_value=[
                    isolation.SpeechSegment(1 * 16_000, 3 * 16_000),
                    isolation.SpeechSegment(4 * 16_000, 7 * 16_000),
                ],
            ):
                with patch.object(
                    service,
                    "generate_embedding",
                    side_effect=lambda reference_audio: captured_reference.append(reference_audio.copy())
                    or np.array([1.0, 0.0], dtype=np.float32),
                ):
                    self.assertTrue(service.initialize_reference(audio))

        self.assertEqual(captured_reference[0].size, 3 * 16_000)
        self.assertTrue(np.allclose(captured_reference[0][: 2 * 16_000], 0.2))
        self.assertTrue(np.allclose(captured_reference[0][2 * 16_000 :], 0.5))
        metrics = service.report_summary()["debug_metrics"]
        self.assertEqual(metrics["reference_source"], "vad_speech_segments")
        self.assertEqual(metrics["reference_segment_count"], 2)
        self.assertEqual(metrics["reference_start_sec"], 1.0)
        self.assertEqual(metrics["reference_end_sec"], 5.0)
        self.assertEqual(metrics["reference_audio_duration_sec"], 3.0)

    def test_reference_search_ignores_speech_after_max_search_seconds(self) -> None:
        audio = np.zeros(30 * 16_000, dtype=np.float32)
        audio[25 * 16_000 : 27 * 16_000] = 0.6
        captured_reference: list[np.ndarray] = []

        with patch.object(
            isolation,
            "settings",
            fake_settings(enabled=True, reference_seconds=2, reference_max_search_sec=20),
        ):
            service = isolation.BackgroundVoiceIsolationService()
            with patch.object(
                service,
                "detect_speech_segments",
                return_value=[isolation.SpeechSegment(25 * 16_000, 27 * 16_000)],
            ):
                with patch.object(
                    service,
                    "generate_embedding",
                    side_effect=lambda reference_audio: captured_reference.append(reference_audio.copy())
                    or np.array([1.0, 0.0], dtype=np.float32),
                ):
                    self.assertTrue(service.initialize_reference(audio))

        self.assertTrue(np.allclose(captured_reference[0], 0.0))
        metrics = service.report_summary()["debug_metrics"]
        self.assertEqual(metrics["reference_source"], "first_seconds_fallback")

    def test_vad_input_is_not_capped_by_reference_seconds(self) -> None:
        audio = np.ones(60 * 16_000, dtype=np.float32) * 0.2
        vad_lengths: list[float] = []

        def no_segments(vad_audio: np.ndarray) -> list[isolation.SpeechSegment]:
            vad_lengths.append(round(vad_audio.size / 16_000, 3))
            return []

        with patch.object(isolation, "settings", fake_settings(enabled=True, reference_seconds=3)):
            service = isolation.BackgroundVoiceIsolationService()
            service.reference_embedding = np.array([1.0, 0.0], dtype=np.float32)
            service.summary.reference_ready = True

            with patch.object(service, "detect_speech_segments", side_effect=no_segments):
                service.filter_audio_for_prediction(audio, input_file_duration_sec=60.0)

        self.assertEqual(vad_lengths, [60.0])
        self.assertNotEqual(vad_lengths[0], service.reference_seconds)

    def test_report_background_isolation_receives_full_duration_audio(self) -> None:
        captured_durations: list[float] = []

        class FakeIsolationService:
            def filter_audio_for_prediction(
                self,
                audio: np.ndarray,
                input_file_duration_sec: float | None = None,
            ) -> np.ndarray:
                captured_durations.append(round(audio.size / 16_000, 3))
                captured_durations.append(round(float(input_file_duration_sec or 0.0), 3))
                return audio

            def report_summary(self) -> dict[str, object]:
                return {
                    "enabled": True,
                    "reference_ready": True,
                    "debug_metrics": {
                        "input_file_duration_sec": captured_durations[-1],
                        "full_audio_duration_before_reference_slice": captured_durations[0],
                        "reference_audio_duration_sec": 3.0,
                        "vad_audio_duration_sec": captured_durations[0],
                    },
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            call_dir = Path(tmpdir)
            full_audio = call_dir / "caller_full_audio.wav"
            write_tone_wav(full_audio, seconds=60)
            metadata = {"caller_full_audio_wav_path": str(full_audio)}

            module = fake_model_module()
            report_settings = SimpleNamespace(
                **{
                    **fake_settings(enabled=True, reference_seconds=3).__dict__,
                    "max_duration_seconds": 8.0,
                }
            )
            with patch.dict(sys.modules, {"app.services.model_service": module}):
                with patch.object(report_service, "settings", report_settings):
                    with patch.object(isolation, "BackgroundVoiceIsolationService", FakeIsolationService):
                        report = report_service.generate_voiceage_report(call_dir, "CA123", metadata)

        self.assertTrue(report["prediction_success"])
        self.assertEqual(captured_durations, [60.0, 60.0])
        module.model_service.predict.assert_called_once()
        module.model_service.predict_audio_file.assert_not_called()

    def test_missing_or_corrupt_audio_file_does_not_crash_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            call_dir = Path(tmpdir)
            missing_report = report_service.generate_voiceage_report(call_dir, "missing", {})

            corrupt_path = call_dir / "caller_only_audio.wav"
            corrupt_path.write_bytes(b"not a wav")
            corrupt_report = report_service.generate_voiceage_report(call_dir, "corrupt", {})

        self.assertFalse(missing_report["prediction_success"])
        self.assertEqual(missing_report["failure_reason"], "caller_only_audio_missing_or_too_short")
        self.assertFalse(corrupt_report["prediction_success"])
        self.assertEqual(corrupt_report["failure_reason"], "caller_only_audio_missing_or_too_short")


if __name__ == "__main__":
    unittest.main()
