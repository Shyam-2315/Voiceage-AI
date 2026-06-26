"""Tests for caller isolation pipeline helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_phase4_check import missing_required_samples
from src.isolation_pipeline import (
    SegmentIsolationResult,
    build_isolation_report,
    concatenate_segments,
    default_output_paths,
    should_keep_segment,
)
from src.vad_service import format_segment


class IsolationPipelineTests(unittest.TestCase):
    def test_report_schema_generation(self) -> None:
        segment_results = [
            SegmentIsolationResult(1, 0.0, 1.0, 1.0, 0.82, "keep"),
            SegmentIsolationResult(2, 1.5, 2.0, 0.5, 0.41, "reject"),
        ]

        report = build_isolation_report(
            input_file="sample_audio/mixed_call.wav",
            reference_file="sample_audio/caller_reference.wav",
            threshold=0.75,
            total_duration_sec=3.0,
            total_speech_duration_sec=1.5,
            kept_duration_sec=1.0,
            rejected_duration_sec=0.5,
            segment_results=segment_results,
        )

        self.assertEqual(report["input_file"], "sample_audio/mixed_call.wav")
        self.assertEqual(report["reference_file"], "sample_audio/caller_reference.wav")
        self.assertEqual(report["threshold"], 0.75)
        self.assertEqual(report["total_segments"], 2)
        self.assertEqual(len(report["kept_segments"]), 1)
        self.assertEqual(len(report["rejected_segments"]), 1)
        self.assertEqual(report["kept_segments"][0]["decision"], "keep")
        self.assertIn("similarity", report["rejected_segments"][0])

    def test_keep_reject_decision_logic(self) -> None:
        self.assertTrue(should_keep_segment(0.75, threshold=0.75))
        self.assertTrue(should_keep_segment(0.9, threshold=0.75))
        self.assertFalse(should_keep_segment(0.74, threshold=0.75))

    def test_empty_segment_handling(self) -> None:
        audio = np.arange(16_000, dtype=np.float32)

        result = concatenate_segments(audio, [])

        self.assertEqual(result.size, 0)
        self.assertEqual(result.dtype, np.float32)

    def test_concatenate_segments(self) -> None:
        audio = np.arange(10, dtype=np.float32)
        segments = [
            format_segment(0, 3, sample_rate=10),
            format_segment(7, 10, sample_rate=10),
        ]

        result = concatenate_segments(audio, segments)

        np.testing.assert_array_equal(result, np.array([0, 1, 2, 7, 8, 9], dtype=np.float32))

    def test_output_path_generation(self) -> None:
        paths = default_output_paths("/tmp/poc")

        self.assertEqual(paths["caller_audio"], Path("/tmp/poc/outputs/caller_only.wav"))
        self.assertEqual(paths["rejected_audio"], Path("/tmp/poc/outputs/rejected_segments.wav"))
        self.assertEqual(paths["report"], Path("/tmp/poc/reports/isolation_report.json"))

    def test_no_sample_case_does_not_fail_harshly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "sample_audio").mkdir()

            missing = missing_required_samples(tmpdir)

        self.assertEqual(missing, ["mixed_call.wav", "caller_reference.wav"])


if __name__ == "__main__":
    unittest.main()
