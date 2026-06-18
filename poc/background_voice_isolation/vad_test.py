"""Command-line VAD test runner for one WAV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.vad_service import SileroVADService


OUTPUT_AUDIO = Path("outputs/speech_only.wav")
REPORT_PATH = Path("reports/vad_report.json")


def run(input_audio: str | Path) -> dict:
    service = SileroVADService()
    report = service.process_file(input_audio, OUTPUT_AUDIO)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Input duration: {report['input_duration_seconds']:.4f}s")
    print(f"Speech duration: {report['speech_duration_seconds']:.4f}s")
    print(f"Number of segments: {report['num_segments']}")
    print(f"Speech percentage: {report['speech_percentage']:.2f}%")
    print(f"Speech-only audio: {OUTPUT_AUDIO}")
    print(f"VAD report: {REPORT_PATH}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Silero VAD on a WAV file.")
    parser.add_argument("input_audio", help="Path to an input WAV file.")
    args = parser.parse_args()

    input_path = Path(args.input_audio)
    if not input_path.is_file():
        raise SystemExit(f"Input audio file not found: {input_path}")

    run(input_path)


if __name__ == "__main__":
    main()
