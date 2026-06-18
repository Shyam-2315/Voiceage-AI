"""Audio loading and saving helpers for the VAD POC."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


DEFAULT_SAMPLE_RATE = 16_000


def load_audio(path: str | Path, target_sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at the requested sample rate."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if sample_rate != target_sample_rate:
        audio = librosa.resample(
            y=audio,
            orig_sr=sample_rate,
            target_sr=target_sample_rate,
        ).astype(np.float32)
        sample_rate = target_sample_rate

    return audio, sample_rate


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """Save mono audio to a WAV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)
    sf.write(output_path, audio, sample_rate)
