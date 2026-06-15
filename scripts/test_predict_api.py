#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the VoiceAge AI prediction API.")
    parser.add_argument("audio_file", type=Path, help="Path to a wav, mp3, or m4a file.")
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")

    base_url = args.url.rstrip("/")
    health = requests.get(f"{base_url}/health", timeout=10)
    print("Health:")
    print(json.dumps(health.json(), indent=2))
    health.raise_for_status()

    with args.audio_file.open("rb") as handle:
        files = {"file": (args.audio_file.name, handle, "application/octet-stream")}
        response = requests.post(f"{base_url}/api/predict-age", files=files, timeout=120)

    print("Prediction:")
    try:
        print(json.dumps(response.json(), indent=2))
    finally:
        response.raise_for_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
