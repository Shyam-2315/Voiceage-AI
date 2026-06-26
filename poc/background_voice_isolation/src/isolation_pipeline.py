"""Caller-only isolation pipeline combining VAD and speaker verification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.audio_io import DEFAULT_SAMPLE_RATE, load_audio, save_wav
from src.speaker_embedding_service import SpeakerEmbeddingService
from src.vad_service import SileroVADService, SpeechSegment


DEFAULT_CALLER_OUTPUT = Path("outputs/caller_only.wav")
DEFAULT_REJECTED_OUTPUT = Path("outputs/rejected_segments.wav")
DEFAULT_REPORT_OUTPUT = Path("reports/isolation_report.json")


@dataclass(frozen=True)
class SegmentIsolationResult:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    similarity: float
    decision: str


def should_keep_segment(similarity: float, threshold: float = 0.75) -> bool:
    """Return whether a segment similarity meets the caller threshold."""
    return similarity >= threshold


def default_output_paths(base_dir: str | Path = ".") -> dict[str, Path]:
    """Return default Phase 4 output paths relative to a base directory."""
    base_path = Path(base_dir)
    return {
        "caller_audio": base_path / DEFAULT_CALLER_OUTPUT,
        "rejected_audio": base_path / DEFAULT_REJECTED_OUTPUT,
        "report": base_path / DEFAULT_REPORT_OUTPUT,
    }


def concatenate_segments(audio: np.ndarray, segments: list[SpeechSegment]) -> np.ndarray:
    """Concatenate audio chunks for the provided segment list."""
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


def build_isolation_report(
    input_file: str | Path,
    reference_file: str | Path,
    threshold: float,
    total_duration_sec: float,
    total_speech_duration_sec: float,
    kept_duration_sec: float,
    rejected_duration_sec: float,
    segment_results: list[SegmentIsolationResult],
) -> dict[str, Any]:
    """Build the JSON-serializable isolation report."""
    kept_segments = [
        asdict(result)
        for result in segment_results
        if result.decision == "keep"
    ]
    rejected_segments = [
        asdict(result)
        for result in segment_results
        if result.decision == "reject"
    ]

    return {
        "input_file": str(input_file),
        "reference_file": str(reference_file),
        "threshold": threshold,
        "total_duration_sec": round(total_duration_sec, 4),
        "total_speech_duration_sec": round(total_speech_duration_sec, 4),
        "kept_duration_sec": round(kept_duration_sec, 4),
        "rejected_duration_sec": round(rejected_duration_sec, 4),
        "total_segments": len(segment_results),
        "kept_segments": kept_segments,
        "rejected_segments": rejected_segments,
    }


class CallerIsolationPipeline:
    """End-to-end caller isolation using Silero VAD and SpeechBrain embeddings."""

    def __init__(
        self,
        vad_service: SileroVADService | None = None,
        speaker_service: SpeakerEmbeddingService | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.sample_rate = sample_rate
        self.vad_service = vad_service or SileroVADService(sample_rate=sample_rate)
        self.speaker_service = speaker_service or SpeakerEmbeddingService(sample_rate=sample_rate)

    def isolate(
        self,
        input_audio: str | Path,
        reference_audio: str | Path,
        threshold: float = 0.75,
        caller_output: str | Path = DEFAULT_CALLER_OUTPUT,
        rejected_output: str | Path = DEFAULT_REJECTED_OUTPUT,
        report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    ) -> dict[str, Any]:
        """Run caller-only isolation and write audio/report outputs."""
        full_audio, sample_rate = load_audio(input_audio, target_sample_rate=self.sample_rate)
        speech_segments = self.vad_service.detect_speech(full_audio)
        reference_embedding = self.speaker_service.generate_embedding(reference_audio)

        kept_segments: list[SpeechSegment] = []
        rejected_segments: list[SpeechSegment] = []
        segment_results: list[SegmentIsolationResult] = []

        for index, segment in enumerate(speech_segments, start=1):
            segment_audio = full_audio[segment.start_sample:segment.end_sample]
            test_embedding = self.speaker_service.generate_embedding_from_audio(
                segment_audio,
                sample_rate,
            )
            similarity = self.speaker_service.compare_embeddings(
                reference_embedding,
                test_embedding,
            )
            decision = "keep" if should_keep_segment(similarity, threshold) else "reject"

            if decision == "keep":
                kept_segments.append(segment)
            else:
                rejected_segments.append(segment)

            segment_results.append(
                SegmentIsolationResult(
                    index=index,
                    start_sec=segment.start_seconds,
                    end_sec=segment.end_seconds,
                    duration_sec=segment.duration_seconds,
                    similarity=round(float(similarity), 6),
                    decision=decision,
                )
            )

        caller_audio = concatenate_segments(full_audio, kept_segments)
        rejected_audio = concatenate_segments(full_audio, rejected_segments)

        save_wav(caller_output, caller_audio, sample_rate)
        save_wav(rejected_output, rejected_audio, sample_rate)

        total_duration_sec = len(full_audio) / sample_rate
        total_speech_duration_sec = sum(segment.duration_seconds for segment in speech_segments)
        kept_duration_sec = len(caller_audio) / sample_rate
        rejected_duration_sec = len(rejected_audio) / sample_rate

        report = build_isolation_report(
            input_file=input_audio,
            reference_file=reference_audio,
            threshold=threshold,
            total_duration_sec=total_duration_sec,
            total_speech_duration_sec=total_speech_duration_sec,
            kept_duration_sec=kept_duration_sec,
            rejected_duration_sec=rejected_duration_sec,
            segment_results=segment_results,
        )

        report_path = Path(report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return report
