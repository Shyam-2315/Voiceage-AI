"""Basic tests for VAD helper behavior."""

from __future__ import annotations

import unittest

import numpy as np

from src.audio_io import load_audio
from src.vad_service import SileroVADService, format_segment, merge_close_segments


class VADServiceTests(unittest.TestCase):
    def test_import_service(self) -> None:
        service = SileroVADService()
        self.assertEqual(service.sample_rate, 16_000)

    def test_missing_audio_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_audio("sample_audio/does-not-exist.wav")

    def test_segment_formatting(self) -> None:
        segment = format_segment(8_000, 16_000)

        self.assertEqual(segment.start_sample, 8_000)
        self.assertEqual(segment.end_sample, 16_000)
        self.assertEqual(segment.start_seconds, 0.5)
        self.assertEqual(segment.end_seconds, 1.0)
        self.assertEqual(segment.duration_seconds, 0.5)

    def test_merge_close_segments(self) -> None:
        segments = [
            format_segment(0, 8_000),
            format_segment(9_000, 16_000),
            format_segment(30_000, 32_000),
        ]

        merged = merge_close_segments(segments, max_gap_seconds=0.1)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].start_sample, 0)
        self.assertEqual(merged[0].end_sample, 16_000)

    def test_empty_audio_extracts_empty_speech(self) -> None:
        service = SileroVADService()

        speech = service.extract_speech_only(np.array([], dtype=np.float32), [])

        self.assertEqual(speech.size, 0)


if __name__ == "__main__":
    unittest.main()
