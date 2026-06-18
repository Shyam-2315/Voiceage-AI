"""Phase 3 validation for SpeechBrain speaker verification."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = (
    "numpy",
    "soundfile",
    "librosa",
    "sklearn",
    "torch",
    "torchaudio",
    "speechbrain",
    "src.audio_io",
    "src.speaker_embedding_service",
)


def verify_imports() -> None:
    failures: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append(f"{module_name}: {exc}")

    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise ImportError(f"Import checks failed:\n{details}")

    print("Imports OK")


def verify_model_load() -> None:
    from src.speaker_embedding_service import SpeakerEmbeddingService

    service = SpeakerEmbeddingService()
    service.load_model()
    print("SpeechBrain ECAPA-TDNN model load OK")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    sample_dir = base_dir / "sample_audio"
    reference_audio = sample_dir / "caller_reference.wav"
    test_audio = sample_dir / "test_segment.wav"

    verify_imports()
    verify_model_load()

    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample audio folder: {sample_dir}")

    missing = [
        path.name
        for path in (reference_audio, test_audio)
        if not path.is_file()
    ]
    if missing:
        print("Sample speaker verification WAV files are missing: " + ", ".join(missing))
        print("Add caller_reference.wav and test_segment.wav into sample_audio/")
        print(
            "Then run: python speaker_verification.py "
            "--reference sample_audio/caller_reference.wav --test sample_audio/test_segment.wav"
        )
        return

    subprocess.run(
        [
            sys.executable,
            "speaker_verification.py",
            "--reference",
            str(reference_audio),
            "--test",
            str(test_audio),
        ],
        cwd=base_dir,
        check=True,
    )
    print("Phase 3 check passed.")


if __name__ == "__main__":
    main()
