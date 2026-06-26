"""Command-line caller-only isolation runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.isolation_pipeline import (
    DEFAULT_CALLER_OUTPUT,
    DEFAULT_REJECTED_OUTPUT,
    DEFAULT_REPORT_OUTPUT,
    CallerIsolationPipeline,
)


def run(input_audio: str | Path, reference_audio: str | Path, threshold: float = 0.75) -> dict:
    pipeline = CallerIsolationPipeline()
    report = pipeline.isolate(
        input_audio=input_audio,
        reference_audio=reference_audio,
        threshold=threshold,
        caller_output=DEFAULT_CALLER_OUTPUT,
        rejected_output=DEFAULT_REJECTED_OUTPUT,
        report_output=DEFAULT_REPORT_OUTPUT,
    )

    print(f"Total duration: {report['total_duration_sec']:.4f}s")
    print(f"Total speech duration: {report['total_speech_duration_sec']:.4f}s")
    print(f"Kept duration: {report['kept_duration_sec']:.4f}s")
    print(f"Rejected duration: {report['rejected_duration_sec']:.4f}s")
    print(f"Total segments: {report['total_segments']}")
    print(f"Kept segments: {len(report['kept_segments'])}")
    print(f"Rejected segments: {len(report['rejected_segments'])}")
    print(f"Threshold used: {report['threshold']:.2f}")
    print(f"Caller-only audio: {DEFAULT_CALLER_OUTPUT}")
    print(f"Rejected audio: {DEFAULT_REJECTED_OUTPUT}")
    print(f"Isolation report: {DEFAULT_REPORT_OUTPUT}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate caller speech from a mixed call WAV.")
    parser.add_argument("--input", required=True, help="Full mixed conversation WAV file.")
    parser.add_argument("--reference", required=True, help="Caller reference WAV file.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Caller similarity threshold.")
    args = parser.parse_args()

    input_path = Path(args.input)
    reference_path = Path(args.reference)

    if not input_path.is_file():
        raise SystemExit(f"Input audio file not found: {input_path}")
    if not reference_path.is_file():
        raise SystemExit(f"Reference audio file not found: {reference_path}")

    run(input_path, reference_path, args.threshold)


if __name__ == "__main__":
    main()
