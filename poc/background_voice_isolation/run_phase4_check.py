"""Phase 4 validation for caller-only isolation."""

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
    "silero_vad",
    "speechbrain",
    "src.audio_io",
    "src.vad_service",
    "src.speaker_embedding_service",
    "src.isolation_pipeline",
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
    print("Phase 2 and Phase 3 services are importable")


def required_sample_paths(base_dir: str | Path = ".") -> dict[str, Path]:
    sample_dir = Path(base_dir) / "sample_audio"
    return {
        "mixed_call.wav": sample_dir / "mixed_call.wav",
        "caller_reference.wav": sample_dir / "caller_reference.wav",
    }


def missing_required_samples(base_dir: str | Path = ".") -> list[str]:
    return [
        name
        for name, path in required_sample_paths(base_dir).items()
        if not path.is_file()
    ]


def print_sample_instructions(missing: list[str]) -> None:
    print("Sample caller isolation WAV files are missing: " + ", ".join(missing))
    print("Add mixed_call.wav and caller_reference.wav into sample_audio/")
    print(
        "Then run: python isolate_caller.py "
        "--input sample_audio/mixed_call.wav --reference sample_audio/caller_reference.wav --threshold 0.75"
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    verify_imports()

    missing = missing_required_samples(base_dir)
    if missing:
        print_sample_instructions(missing)
        return

    paths = required_sample_paths(base_dir)
    subprocess.run(
        [
            sys.executable,
            "isolate_caller.py",
            "--input",
            str(paths["mixed_call.wav"]),
            "--reference",
            str(paths["caller_reference.wav"]),
            "--threshold",
            "0.75",
        ],
        cwd=base_dir,
        check=True,
    )
    print("Phase 4 check passed.")


if __name__ == "__main__":
    main()
