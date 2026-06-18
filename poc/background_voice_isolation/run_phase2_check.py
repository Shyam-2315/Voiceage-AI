"""Phase 2 validation for Silero VAD speech detection."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = (
    "numpy",
    "soundfile",
    "librosa",
    "torch",
    "torchaudio",
    "silero_vad",
    "src.audio_io",
    "src.vad_service",
)


def verify_imports() -> None:
    failures: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append(f"{module_name}: {exc}")

    if failures:
        detail = "\n".join(f"  - {failure}" for failure in failures)
        raise ImportError(f"Import checks failed:\n{detail}")

    print("Imports OK")


def find_sample_wavs(sample_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    sample_dir = base_dir / "sample_audio"

    verify_imports()

    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample audio folder: {sample_dir}")

    sample_wavs = find_sample_wavs(sample_dir)
    if not sample_wavs:
        print("No sample WAV found in sample_audio/.")
        print("Add a WAV file into sample_audio/ and run: python vad_test.py sample_audio/<file>.wav")
        return

    input_wav = sample_wavs[0]
    print(f"Running VAD test with: {input_wav}")
    subprocess.run(
        [sys.executable, "vad_test.py", str(input_wav)],
        cwd=base_dir,
        check=True,
    )
    print("Phase 2 check passed.")


if __name__ == "__main__":
    main()
