#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.audio_service import decode_audio_file  # noqa: E402
from app.services.background_voice_isolation_service import (  # noqa: E402
    BackgroundVoiceIsolationService,
)
from app.services.report_service import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run background voice isolation on local WAV files.")
    parser.add_argument("--input", required=True, type=Path, help="Mixed caller audio WAV file.")
    parser.add_argument("--reference", required=True, type=Path, help="Caller reference WAV file.")
    parser.add_argument("--threshold", type=float, default=settings.background_voice_isolation_threshold)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "background_isolated.wav")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "background_isolation_dry_run.json",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input audio file not found: {args.input}")
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference audio file not found: {args.reference}")

    mixed_audio = decode_audio_file(args.input)
    reference_audio = decode_audio_file(args.reference)
    service = BackgroundVoiceIsolationService(enabled=True, threshold=args.threshold)

    reference_ready = service.initialize_reference(reference_audio)
    filtered_audio = service.filter_audio_for_prediction(mixed_audio)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, filtered_audio, settings.target_sample_rate)

    report = {
        "input": str(args.input),
        "reference": str(args.reference),
        "output": str(args.output),
        "reference_ready": reference_ready,
        "input_duration_sec": round(len(mixed_audio) / float(settings.target_sample_rate), 3),
        "output_duration_sec": round(len(filtered_audio) / float(settings.target_sample_rate), 3),
        "isolation": service.report_summary(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)

    print(f"reference_ready: {reference_ready}")
    print(f"kept_segments: {report['isolation']['kept_segments']}")
    print(f"rejected_segments: {report['isolation']['rejected_segments']}")
    print(f"avg_similarity: {report['isolation']['avg_similarity']}")
    print(f"output: {args.output}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
