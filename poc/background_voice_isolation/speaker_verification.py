"""Command-line speaker verification with SpeechBrain ECAPA-TDNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.speaker_embedding_service import SpeakerEmbeddingService


REPORT_PATH = Path("reports/speaker_verification_report.json")


def run(reference_audio: str | Path, test_audio: str | Path, threshold: float = 0.75) -> dict:
    service = SpeakerEmbeddingService()
    report = service.verify_files(reference_audio, test_audio, threshold)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Similarity score: {report['similarity']:.6f}")
    print(f"same_speaker: {str(report['same_speaker']).lower()}")
    print(f"Threshold used: {report['threshold']:.2f}")
    print(f"Speaker verification report: {REPORT_PATH}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two WAV files for speaker similarity.")
    parser.add_argument("--reference", required=True, help="Reference caller WAV file.")
    parser.add_argument("--test", required=True, help="Test segment WAV file.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Same-speaker threshold.")
    args = parser.parse_args()

    reference_path = Path(args.reference)
    test_path = Path(args.test)

    if not reference_path.is_file():
        raise SystemExit(f"Reference audio file not found: {reference_path}")
    if not test_path.is_file():
        raise SystemExit(f"Test audio file not found: {test_path}")

    run(reference_path, test_path, args.threshold)


if __name__ == "__main__":
    main()
