"""Speaker embedding and verification helpers using SpeechBrain ECAPA-TDNN."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.audio_io import DEFAULT_SAMPLE_RATE, load_audio


POC_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_MODEL_DIR = POC_ROOT / "outputs" / "speechbrain_spkrec_ecapa_voxceleb"
DEFAULT_CACHE_DIR = POC_ROOT / "outputs" / "model_cache"


def configure_local_model_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    """Keep Hugging Face and SpeechBrain runtime caches inside this POC folder."""
    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_path / "hf_home")
    os.environ["HF_HUB_CACHE"] = str(cache_path / "hf_hub")
    os.environ["XDG_CACHE_HOME"] = str(cache_path / "xdg")

    return cache_path


def normalize_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Convert in-memory audio to mono float32 at the target sample rate."""
    if audio.size == 0:
        raise ValueError("Audio is empty")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    normalized = np.asarray(audio, dtype=np.float32)

    if sample_rate != target_sample_rate:
        normalized = librosa.resample(
            y=normalized,
            orig_sr=sample_rate,
            target_sr=target_sample_rate,
        ).astype(np.float32)
        sample_rate = target_sample_rate

    return normalized, sample_rate


def cosine_similarity_score(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Compute cosine similarity for two speaker embeddings."""
    vector_a = np.asarray(embedding_a, dtype=np.float32).reshape(1, -1)
    vector_b = np.asarray(embedding_b, dtype=np.float32).reshape(1, -1)

    if not np.any(vector_a) or not np.any(vector_b):
        raise ValueError("Cosine similarity is undefined for zero vectors")

    return float(cosine_similarity(vector_a, vector_b)[0][0])


def same_speaker_decision(similarity: float, threshold: float = 0.75) -> bool:
    """Return True when a similarity score meets the speaker-match threshold."""
    return similarity >= threshold


class SpeakerEmbeddingService:
    """SpeechBrain ECAPA-TDNN wrapper for speaker verification."""

    def __init__(
        self,
        model_source: str = MODEL_SOURCE,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        device: str | None = None,
    ) -> None:
        self.model_source = model_source
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._classifier: Any | None = None

    def load_model(self) -> Any:
        """Load SpeechBrain ECAPA-TDNN lazily."""
        if self._classifier is None:
            configure_local_model_cache(self.cache_dir)
            self.model_dir.mkdir(parents=True, exist_ok=True)

            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # pragma: no cover - compatibility fallback
                from speechbrain.pretrained import EncoderClassifier

            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_source,
                savedir=str(self.model_dir),
                run_opts={"device": self.device},
            )

        return self._classifier

    def generate_embedding_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> np.ndarray:
        """Generate an ECAPA speaker embedding from in-memory audio."""
        normalized_audio, sample_rate = normalize_audio_array(
            audio,
            sample_rate,
            target_sample_rate=self.sample_rate,
        )
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, found {sample_rate} Hz")

        classifier = self.load_model()
        waveform = torch.from_numpy(normalized_audio).unsqueeze(0).to(self.device)
        wav_lens = torch.ones(waveform.shape[0], device=self.device)

        with torch.no_grad():
            embedding = classifier.encode_batch(waveform, wav_lens=wav_lens, normalize=True)

        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    def generate_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Load and normalize an audio file, then generate its speaker embedding."""
        audio, sample_rate = load_audio(audio_path, target_sample_rate=self.sample_rate)
        return self.generate_embedding_from_audio(audio, sample_rate)

    def compare_embeddings(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return cosine_similarity_score(embedding_a, embedding_b)

    def verify_files(
        self,
        reference_audio: str | Path,
        test_audio: str | Path,
        threshold: float = 0.75,
    ) -> dict[str, Any]:
        """Compare two audio files and return a speaker-verification report."""
        reference_embedding = self.generate_embedding(reference_audio)
        test_embedding = self.generate_embedding(test_audio)
        similarity = self.compare_embeddings(reference_embedding, test_embedding)

        return {
            "reference_audio": str(reference_audio),
            "test_audio": str(test_audio),
            "similarity": round(similarity, 6),
            "same_speaker": same_speaker_decision(similarity, threshold),
            "threshold": threshold,
            "model": self.model_source,
        }

    def is_same_speaker(
        self,
        reference_audio: str | Path,
        test_audio: str | Path,
        threshold: float = 0.75,
    ) -> bool:
        """Return whether two audio files appear to contain the same speaker."""
        return bool(self.verify_files(reference_audio, test_audio, threshold)["same_speaker"])


def is_same_speaker(
    reference_audio: str | Path,
    test_audio: str | Path,
    threshold: float = 0.75,
) -> bool:
    """Convenience helper for one-off speaker checks."""
    return SpeakerEmbeddingService().is_same_speaker(reference_audio, test_audio, threshold)
