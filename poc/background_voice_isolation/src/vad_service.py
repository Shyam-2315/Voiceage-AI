"""Silero VAD service for detecting and extracting speech regions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.audio_io import DEFAULT_SAMPLE_RATE, load_audio, save_wav


@dataclass(frozen=True)
class SpeechSegment:
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float


def format_segment(start_sample: int, end_sample: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> SpeechSegment:
    """Create a normalized speech segment from sample offsets."""
    if start_sample < 0 or end_sample < 0:
        raise ValueError("Segment sample offsets must be non-negative")
    if end_sample < start_sample:
        raise ValueError("Segment end must be greater than or equal to start")

    start_seconds = start_sample / sample_rate
    end_seconds = end_sample / sample_rate
    return SpeechSegment(
        start_sample=int(start_sample),
        end_sample=int(end_sample),
        start_seconds=round(start_seconds, 4),
        end_seconds=round(end_seconds, 4),
        duration_seconds=round(end_seconds - start_seconds, 4),
    )


def merge_close_segments(
    segments: list[SpeechSegment],
    max_gap_seconds: float = 0.25,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[SpeechSegment]:
    """Merge speech segments separated by short gaps."""
    if not segments:
        return []

    max_gap_samples = int(max_gap_seconds * sample_rate)
    sorted_segments = sorted(segments, key=lambda segment: segment.start_sample)
    merged: list[SpeechSegment] = [sorted_segments[0]]

    for segment in sorted_segments[1:]:
        previous = merged[-1]
        gap = segment.start_sample - previous.end_sample
        if gap <= max_gap_samples:
            merged[-1] = format_segment(previous.start_sample, segment.end_sample, sample_rate)
        else:
            merged.append(segment)

    return merged


class SileroVADService:
    """Small wrapper around Silero VAD for this isolated POC."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_merge_gap_seconds: float = 0.25,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_merge_gap_seconds = max_merge_gap_seconds
        self._model: Any | None = None
        self._get_speech_timestamps: Any | None = None

    def load_model(self) -> Any:
        """Load the Silero VAD model lazily."""
        if self._model is None or self._get_speech_timestamps is None:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps

        return self._model

    def detect_speech(self, audio: np.ndarray) -> list[SpeechSegment]:
        """Detect speech segments in mono 16 kHz audio."""
        if audio.size == 0:
            return []

        model = self.load_model()
        audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))

        timestamps = self._get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )

        segments = [
            format_segment(item["start"], item["end"], self.sample_rate)
            for item in timestamps
        ]
        return merge_close_segments(
            segments,
            max_gap_seconds=self.max_merge_gap_seconds,
            sample_rate=self.sample_rate,
        )

    def extract_speech_only(self, audio: np.ndarray, segments: list[SpeechSegment]) -> np.ndarray:
        """Concatenate detected speech segments into one speech-only waveform."""
        if not segments:
            return np.array([], dtype=np.float32)

        chunks = [
            audio[segment.start_sample:segment.end_sample]
            for segment in segments
            if segment.end_sample > segment.start_sample
        ]
        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(chunks).astype(np.float32)

    def process_file(self, input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
        """Run VAD on a WAV file and optionally write speech-only output."""
        audio, sample_rate = load_audio(input_path, target_sample_rate=self.sample_rate)
        segments = self.detect_speech(audio)
        speech_audio = self.extract_speech_only(audio, segments)

        if output_path is not None:
            save_wav(output_path, speech_audio, sample_rate)

        input_duration = len(audio) / sample_rate
        speech_duration = len(speech_audio) / sample_rate
        speech_percentage = (speech_duration / input_duration * 100.0) if input_duration else 0.0

        return {
            "input_path": str(input_path),
            "output_path": str(output_path) if output_path is not None else None,
            "sample_rate": sample_rate,
            "input_duration_seconds": round(input_duration, 4),
            "speech_duration_seconds": round(speech_duration, 4),
            "speech_percentage": round(speech_percentage, 2),
            "num_segments": len(segments),
            "segments": [asdict(segment) for segment in segments],
        }
