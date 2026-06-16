from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.report_service import generate_reports_for_call, wav_duration_seconds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VoiceAge realtime post-call reports.")
    parser.add_argument(
        "--call-dir",
        required=False,
        type=Path,
        help="Path to data/realtime_conversations/<call_id>",
    )
    return parser.parse_args()


def latest_call_dir() -> Path | None:
    root = settings.realtime_conversations_dir
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_voiceage_status(reports_dir: Path) -> str:
    report_path = reports_dir / "voiceage_report.json"
    if not report_path.exists():
        return "missing"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("voiceage_prediction_success") or report.get("prediction_success"):
        return "success"
    return f"failed:{report.get('failure_reason') or 'unknown'}"


def main() -> int:
    args = parse_args()
    call_dir = args.call_dir or latest_call_dir()
    if call_dir is None:
        print("No realtime call folders found.", file=sys.stderr)
        return 1

    reports_dir = generate_reports_for_call(call_dir)
    caller_audio_path = call_dir / "caller_only_audio.wav"
    duration = wav_duration_seconds(caller_audio_path)

    print(f"call_dir: {call_dir}")
    print(f"caller_only_audio: {caller_audio_path}")
    print(
        "caller_only_audio_duration_seconds: "
        f"{duration:.3f}" if duration is not None else "caller_only_audio_duration_seconds: unavailable"
    )
    print(f"voiceage_prediction_status: {load_voiceage_status(reports_dir)}")
    print(f"reports_path: {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
