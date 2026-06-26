from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from app.core.config import PROJECT_ROOT, settings


logger = logging.getLogger(__name__)

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = PROJECT_ROOT / "data" / "background_voice_isolation" / "speechbrain_spkrec_ecapa_voxceleb"
CACHE_DIR = PROJECT_ROOT / "data" / "background_voice_isolation" / "model_cache"


@dataclass(frozen=True)
class SpeechSegment:
    start_sample: int
    end_sample: int

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / float(settings.target_sample_rate)


@dataclass(frozen=True)
class ReferenceSelection:
    audio: np.ndarray
    source: str
    segment_count: int
    start_sample: int | None
    end_sample: int | None
    segments: list[SpeechSegment]


@dataclass
class IsolationSummary:
    enabled: bool
    reference_ready: bool
    debug_metrics_enabled: bool
    kept_segments: int = 0
    rejected_segments: int = 0
    similarities: list[float] | None = None
    threshold: float = settings.background_voice_isolation_threshold
    fallback_used: bool = False
    failure_reason: str | None = None
    debug_metrics: dict[str, Any] | None = None

    @property
    def avg_similarity(self) -> float | None:
        if not self.similarities:
            return None
        return round(sum(self.similarities) / len(self.similarities), 6)

    def to_report(self) -> dict[str, Any]:
        payload = {
            "enabled": self.enabled,
            "reference_ready": self.reference_ready,
            "debug_metrics_enabled": self.debug_metrics_enabled,
            "kept_segments": self.kept_segments,
            "rejected_segments": self.rejected_segments,
            "avg_similarity": self.avg_similarity,
            "threshold": self.threshold,
            "fallback_used": self.fallback_used,
            "failure_reason": self.failure_reason,
            "fallback_reason": self.failure_reason,
        }
        if self.debug_metrics_enabled:
            payload["debug_metrics"] = self.debug_metrics or {}
        return payload


def configure_model_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hf_hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))


def normalize_audio(audio: np.ndarray, sample_rate: int = settings.target_sample_rate) -> np.ndarray:
    if audio.size == 0:
        return np.array([], dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    normalized = np.asarray(audio, dtype=np.float32)
    if sample_rate != settings.target_sample_rate:
        normalized = librosa.resample(
            y=normalized,
            orig_sr=sample_rate,
            target_sr=settings.target_sample_rate,
        ).astype(np.float32)
    return normalized


def audio_debug_metrics(
    audio: np.ndarray,
    sample_rate: int,
    min_segment_sec: float,
    threshold: float,
    input_file_duration_sec: float | None = None,
    full_audio_duration_before_reference_slice: float | None = None,
    reference_duration_sec: float | None = None,
    reference_source: str | None = None,
    reference_segment_count: int | None = None,
    reference_start_sec: float | None = None,
    reference_end_sec: float | None = None,
    vad_audio_duration_sec: float | None = None,
    vad_segments: list[SpeechSegment] | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_audio(audio, sample_rate)
    num_samples = int(normalized.size)
    total_duration_sec = round(num_samples / float(settings.target_sample_rate), 4) if num_samples else 0.0
    rms = float(np.sqrt(np.mean(np.square(normalized)))) if num_samples else 0.0
    peak = float(np.max(np.abs(normalized))) if num_samples else 0.0
    segments = vad_segments or []
    vad_speech_duration_sec = round(sum(segment.duration_seconds for segment in segments), 4)
    vad_speech_percentage = (
        round((vad_speech_duration_sec / total_duration_sec) * 100.0, 2)
        if total_duration_sec > 0
        else 0.0
    )

    return {
        "input_file_duration_sec": round(input_file_duration_sec, 4)
        if input_file_duration_sec is not None
        else total_duration_sec,
        "full_audio_duration_before_reference_slice": round(full_audio_duration_before_reference_slice, 4)
        if full_audio_duration_before_reference_slice is not None
        else total_duration_sec,
        "reference_audio_duration_sec": round(reference_duration_sec, 4) if reference_duration_sec is not None else None,
        "reference_source": reference_source,
        "reference_segment_count": reference_segment_count,
        "reference_start_sec": round(reference_start_sec, 4) if reference_start_sec is not None else None,
        "reference_end_sec": round(reference_end_sec, 4) if reference_end_sec is not None else None,
        "vad_audio_duration_sec": round(vad_audio_duration_sec, 4)
        if vad_audio_duration_sec is not None
        else total_duration_sec,
        "total_audio_duration_sec": total_duration_sec,
        "sample_rate": settings.target_sample_rate,
        "num_samples": num_samples,
        "audio_rms": round(rms, 8),
        "audio_peak": round(peak, 8),
        "audio_is_silent": bool(peak <= 1e-6),
        "reference_duration_sec": round(reference_duration_sec, 4) if reference_duration_sec is not None else None,
        "vad_segment_count": len(segments),
        "vad_speech_duration_sec": vad_speech_duration_sec,
        "vad_speech_percentage": vad_speech_percentage,
        "min_segment_sec": min_segment_sec,
        "threshold": threshold,
        "fallback_reason": fallback_reason,
    }


def cosine_similarity_score(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = np.asarray(left, dtype=np.float32).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float32).reshape(-1)
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return float(np.dot(left_vector, right_vector) / (left_norm * right_norm))


def threshold_decision(similarity: float, threshold: float) -> bool:
    return similarity >= threshold


class BackgroundVoiceIsolationService:
    def __init__(
        self,
        enabled: bool | None = None,
        threshold: float | None = None,
        reference_seconds: float | None = None,
        min_segment_seconds: float | None = None,
    ) -> None:
        self.enabled = enabled if enabled is not None else settings.background_voice_isolation_enabled
        self.threshold = threshold if threshold is not None else settings.background_voice_isolation_threshold
        self.reference_seconds = (
            reference_seconds
            if reference_seconds is not None
            else settings.background_voice_reference_seconds
        )
        self.reference_from_vad = getattr(settings, "background_voice_reference_from_vad", True)
        self.reference_max_search_sec = getattr(settings, "background_voice_reference_max_search_sec", 20)
        self.min_segment_seconds = (
            min_segment_seconds
            if min_segment_seconds is not None
            else settings.background_voice_min_segment_sec
        )
        self.debug_metrics_enabled = settings.background_voice_debug_metrics
        self.sample_rate = settings.target_sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reference_embedding: np.ndarray | None = None
        self.reference_source: str | None = None
        self.reference_segment_count: int | None = None
        self.reference_start_sec: float | None = None
        self.reference_end_sec: float | None = None
        self.reference_audio_duration_sec: float | None = None
        self.reference_segments: list[SpeechSegment] = []
        self.summary = IsolationSummary(
            enabled=self.enabled,
            reference_ready=False,
            debug_metrics_enabled=self.debug_metrics_enabled,
            threshold=self.threshold,
            similarities=[],
        )
        self._vad_model: Any | None = None
        self._get_speech_timestamps: Any | None = None
        self._speaker_model: Any | None = None

    @property
    def reference_ready(self) -> bool:
        return self.reference_embedding is not None

    def load_vad_model(self) -> Any:
        if self._vad_model is None or self._get_speech_timestamps is None:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._vad_model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
        return self._vad_model

    def load_speaker_model(self) -> Any:
        if self._speaker_model is None:
            configure_model_cache()
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # pragma: no cover - SpeechBrain compatibility fallback
                from speechbrain.pretrained import EncoderClassifier

            self._speaker_model = EncoderClassifier.from_hparams(
                source=MODEL_SOURCE,
                savedir=str(MODEL_DIR),
                run_opts={"device": self.device},
            )
        return self._speaker_model

    def detect_speech_segments(self, audio: np.ndarray) -> list[SpeechSegment]:
        normalized = normalize_audio(audio, self.sample_rate)
        if normalized.size == 0:
            return []

        vad_model = self.load_vad_model()
        audio_tensor = torch.from_numpy(normalized)
        timestamps = self._get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=self.sample_rate,
            threshold=0.5,
            min_speech_duration_ms=int(self.min_segment_seconds * 1000),
            min_silence_duration_ms=100,
            speech_pad_ms=30,
            return_seconds=False,
        )
        return [
            SpeechSegment(int(item["start"]), int(item["end"]))
            for item in timestamps
            if int(item["end"]) > int(item["start"])
        ]

    def generate_embedding(self, audio: np.ndarray) -> np.ndarray:
        normalized = normalize_audio(audio, self.sample_rate)
        if normalized.size == 0:
            raise ValueError("Cannot generate speaker embedding from empty audio.")

        speaker_model = self.load_speaker_model()
        waveform = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        wav_lens = torch.ones(waveform.shape[0], device=self.device)
        with torch.no_grad():
            embedding = speaker_model.encode_batch(waveform, wav_lens=wav_lens, normalize=True)
        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    def select_reference_audio(
        self,
        normalized: np.ndarray,
        vad_segments: list[SpeechSegment],
    ) -> ReferenceSelection:
        target_samples = int(self.reference_seconds * self.sample_rate)
        if target_samples <= 0:
            return ReferenceSelection(
                audio=np.array([], dtype=np.float32),
                source="first_seconds_fallback",
                segment_count=0,
                start_sample=None,
                end_sample=None,
                segments=[],
            )

        if self.reference_from_vad:
            search_limit_samples = min(
                normalized.size,
                int(self.reference_max_search_sec * self.sample_rate),
            )
            remaining_samples = target_samples
            chunks: list[np.ndarray] = []
            selected_segments: list[SpeechSegment] = []
            for segment in sorted(vad_segments, key=lambda item: item.start_sample):
                if segment.start_sample >= search_limit_samples:
                    break
                start_sample = max(0, segment.start_sample)
                end_sample = min(segment.end_sample, search_limit_samples, normalized.size)
                if end_sample <= start_sample:
                    continue

                chunk = normalized[start_sample:end_sample]
                if chunk.size > remaining_samples:
                    chunk = chunk[:remaining_samples]
                    end_sample = start_sample + remaining_samples
                if chunk.size == 0:
                    continue

                chunks.append(chunk)
                selected_segments.append(SpeechSegment(start_sample, end_sample))
                remaining_samples -= chunk.size
                if remaining_samples <= 0:
                    break

            if chunks:
                return ReferenceSelection(
                    audio=np.concatenate(chunks).astype(np.float32),
                    source="vad_speech_segments",
                    segment_count=len(selected_segments),
                    start_sample=selected_segments[0].start_sample,
                    end_sample=selected_segments[-1].end_sample,
                    segments=selected_segments,
                )

        candidate_audio = normalized[:target_samples].astype(np.float32)
        return ReferenceSelection(
            audio=candidate_audio,
            source="first_seconds_fallback",
            segment_count=0,
            start_sample=0 if candidate_audio.size else None,
            end_sample=candidate_audio.size if candidate_audio.size else None,
            segments=[],
        )

    def _store_reference_selection(self, selection: ReferenceSelection) -> None:
        self.reference_source = selection.source
        self.reference_segment_count = selection.segment_count
        self.reference_start_sec = (
            selection.start_sample / float(self.sample_rate) if selection.start_sample is not None else None
        )
        self.reference_end_sec = (
            selection.end_sample / float(self.sample_rate) if selection.end_sample is not None else None
        )
        self.reference_audio_duration_sec = selection.audio.size / float(self.sample_rate)
        self.reference_segments = selection.segments

    def _segment_overlaps_reference(self, segment: SpeechSegment) -> bool:
        return any(
            segment.start_sample < reference.end_sample and segment.end_sample > reference.start_sample
            for reference in self.reference_segments
        )

    def initialize_reference(
        self,
        audio: np.ndarray,
        vad_segments: list[SpeechSegment] | None = None,
    ) -> bool:
        normalized = normalize_audio(audio, self.sample_rate)
        full_audio_duration_sec = normalized.size / float(self.sample_rate) if normalized.size else 0.0
        if normalized.size == 0:
            self.summary.reference_ready = False
            self.update_debug_metrics(
                normalized,
                fallback_reason="empty_reference_audio",
                full_audio_duration_before_reference_slice=full_audio_duration_sec,
                reference_duration_sec=0.0,
            )
            return False

        segments = vad_segments if vad_segments is not None else self.detect_speech_segments(normalized)
        selection = self.select_reference_audio(normalized, segments)
        self._store_reference_selection(selection)
        candidate_audio = selection.audio

        if candidate_audio.size < int(self.min_segment_seconds * self.sample_rate):
            self.summary.reference_ready = False
            self.update_debug_metrics(
                normalized,
                vad_segments=segments,
                fallback_reason="reference_too_short",
                full_audio_duration_before_reference_slice=full_audio_duration_sec,
                reference_duration_sec=self.reference_audio_duration_sec,
            )
            return False

        self.reference_embedding = self.generate_embedding(candidate_audio)
        self.summary.reference_ready = True
        self.update_debug_metrics(
            normalized,
            vad_segments=segments,
            full_audio_duration_before_reference_slice=full_audio_duration_sec,
            reference_duration_sec=self.reference_audio_duration_sec,
        )
        logger.info(
            "Background voice isolation reference initialized: reference_ready=%s reference_source=%s reference_segment_count=%s reference_start_sec=%s reference_end_sec=%s reference_audio_duration_sec=%s full_audio_duration_before_reference_slice=%s configured_reference_seconds=%s reference_max_search_sec=%s",
            self.summary.reference_ready,
            self.reference_source,
            self.reference_segment_count,
            round(self.reference_start_sec, 3) if self.reference_start_sec is not None else None,
            round(self.reference_end_sec, 3) if self.reference_end_sec is not None else None,
            round(self.reference_audio_duration_sec, 3) if self.reference_audio_duration_sec is not None else None,
            round(full_audio_duration_sec, 3),
            self.reference_seconds,
            self.reference_max_search_sec,
        )
        return True

    def update_debug_metrics(
        self,
        audio: np.ndarray,
        vad_segments: list[SpeechSegment] | None = None,
        fallback_reason: str | None = None,
        input_file_duration_sec: float | None = None,
        full_audio_duration_before_reference_slice: float | None = None,
        reference_duration_sec: float | None = None,
        vad_audio_duration_sec: float | None = None,
    ) -> None:
        if not self.debug_metrics_enabled:
            self.summary.debug_metrics = None
            return
        self.summary.debug_metrics = audio_debug_metrics(
            audio,
            self.sample_rate,
            self.min_segment_seconds,
            self.threshold,
            input_file_duration_sec=input_file_duration_sec,
            full_audio_duration_before_reference_slice=full_audio_duration_before_reference_slice,
            reference_duration_sec=reference_duration_sec
            if reference_duration_sec is not None
            else self.reference_audio_duration_sec,
            reference_source=self.reference_source,
            reference_segment_count=self.reference_segment_count,
            reference_start_sec=self.reference_start_sec,
            reference_end_sec=self.reference_end_sec,
            vad_audio_duration_sec=vad_audio_duration_sec,
            vad_segments=vad_segments,
            fallback_reason=fallback_reason or self.summary.failure_reason,
        )

    def should_keep_segment(self, audio: np.ndarray) -> bool:
        if self.reference_embedding is None:
            self.initialize_reference(audio)
            if self.reference_embedding is None:
                logger.warning("Background voice isolation reference unavailable; keeping segment.")
                return True

        segment_embedding = self.generate_embedding(audio)
        similarity = cosine_similarity_score(self.reference_embedding, segment_embedding)
        self.summary.similarities = self.summary.similarities or []
        self.summary.similarities.append(round(similarity, 6))
        keep = threshold_decision(similarity, self.threshold)
        if keep:
            self.summary.kept_segments += 1
        else:
            self.summary.rejected_segments += 1
        logger.info(
            "Background voice isolation segment decision: similarity=%.6f threshold=%.3f decision=%s",
            similarity,
            self.threshold,
            "keep" if keep else "reject",
        )
        return keep

    def filter_audio_for_prediction(
        self,
        audio: np.ndarray,
        input_file_duration_sec: float | None = None,
    ) -> np.ndarray:
        if not self.enabled:
            self.summary.enabled = False
            return audio

        self.summary.enabled = True
        normalized = normalize_audio(audio, self.sample_rate)
        full_audio_duration_sec = normalized.size / float(self.sample_rate) if normalized.size else 0.0
        if normalized.size == 0:
            self.update_debug_metrics(
                normalized,
                fallback_reason="empty_audio",
                input_file_duration_sec=input_file_duration_sec,
                full_audio_duration_before_reference_slice=full_audio_duration_sec,
                vad_audio_duration_sec=full_audio_duration_sec,
            )
            return audio

        try:
            self.update_debug_metrics(
                normalized,
                input_file_duration_sec=input_file_duration_sec,
                full_audio_duration_before_reference_slice=full_audio_duration_sec,
                vad_audio_duration_sec=full_audio_duration_sec,
            )
            if self.debug_metrics_enabled:
                metrics = self.summary.debug_metrics or {}
                logger.info(
                    "Background voice isolation input metrics: input_file_duration_sec=%s full_audio_duration_before_reference_slice=%s reference_audio_duration_sec=%s vad_audio_duration_sec=%s total_audio_duration_sec=%s sample_rate=%s num_samples=%s audio_rms=%s audio_peak=%s audio_is_silent=%s min_segment_sec=%s threshold=%s",
                    metrics.get("input_file_duration_sec"),
                    metrics.get("full_audio_duration_before_reference_slice"),
                    metrics.get("reference_audio_duration_sec"),
                    metrics.get("vad_audio_duration_sec"),
                    metrics.get("total_audio_duration_sec"),
                    metrics.get("sample_rate"),
                    metrics.get("num_samples"),
                    metrics.get("audio_rms"),
                    metrics.get("audio_peak"),
                    metrics.get("audio_is_silent"),
                    metrics.get("min_segment_sec"),
                    metrics.get("threshold"),
                )
            segments = self.detect_speech_segments(normalized)
            if self.reference_embedding is None and not self.initialize_reference(normalized, vad_segments=segments):
                logger.warning("Background voice isolation reference could not be initialized; using original audio.")
                self.summary.fallback_used = True
                self.summary.failure_reason = "reference_not_ready"
                self.update_debug_metrics(
                    normalized,
                    fallback_reason=self.summary.failure_reason,
                    input_file_duration_sec=input_file_duration_sec,
                    full_audio_duration_before_reference_slice=full_audio_duration_sec,
                    reference_duration_sec=(self.summary.debug_metrics or {}).get("reference_duration_sec")
                    if self.debug_metrics_enabled
                    else None,
                    vad_audio_duration_sec=full_audio_duration_sec,
                )
                return audio

            reference_duration_sec = (
                (self.summary.debug_metrics or {}).get("reference_audio_duration_sec")
                if self.debug_metrics_enabled
                else None
            )
            self.update_debug_metrics(
                normalized,
                vad_segments=segments,
                input_file_duration_sec=input_file_duration_sec,
                full_audio_duration_before_reference_slice=full_audio_duration_sec,
                reference_duration_sec=reference_duration_sec,
                vad_audio_duration_sec=full_audio_duration_sec,
            )
            if not segments:
                self.summary.fallback_used = True
                self.summary.failure_reason = "no_speech_segments"
                self.update_debug_metrics(
                    normalized,
                    vad_segments=segments,
                    fallback_reason=self.summary.failure_reason,
                    input_file_duration_sec=input_file_duration_sec,
                    full_audio_duration_before_reference_slice=full_audio_duration_sec,
                    reference_duration_sec=reference_duration_sec,
                    vad_audio_duration_sec=full_audio_duration_sec,
                )
                metrics = self.summary.debug_metrics or audio_debug_metrics(
                    normalized,
                    self.sample_rate,
                    self.min_segment_seconds,
                    self.threshold,
                    input_file_duration_sec=input_file_duration_sec,
                    full_audio_duration_before_reference_slice=full_audio_duration_sec,
                    reference_duration_sec=reference_duration_sec,
                    vad_audio_duration_sec=full_audio_duration_sec,
                    vad_segments=segments,
                    fallback_reason=self.summary.failure_reason,
                )
                logger.warning(
                    "Background voice isolation found no speech segments; using original audio. total_duration_sec=%s audio_rms=%s audio_peak=%s sample_rate=%s min_segment_sec=%s recommended_next_action=%s",
                    metrics.get("total_audio_duration_sec"),
                    metrics.get("audio_rms"),
                    metrics.get("audio_peak"),
                    metrics.get("sample_rate", self.sample_rate),
                    self.min_segment_seconds,
                    "For Twilio tests, try BACKGROUND_VOICE_MIN_SEGMENT_SEC=0.3 and verify caller_full_audio.wav contains audible caller speech.",
                )
                return audio

            kept_chunks: list[np.ndarray] = []
            for segment in segments:
                segment_audio = normalized[segment.start_sample:segment.end_sample]
                if segment.duration_seconds < self.min_segment_seconds:
                    self.summary.rejected_segments += 1
                    continue
                if self._segment_overlaps_reference(segment):
                    self.summary.kept_segments += 1
                    kept_chunks.append(segment_audio)
                    logger.info(
                        "Background voice isolation segment decision: reference_segment=true threshold=%.3f decision=keep",
                        self.threshold,
                    )
                    continue
                if self.should_keep_segment(segment_audio):
                    kept_chunks.append(segment_audio)

            if not kept_chunks:
                logger.warning("Background voice isolation rejected all segments; using original audio.")
                self.summary.fallback_used = True
                self.summary.failure_reason = "all_segments_rejected"
                self.update_debug_metrics(
                    normalized,
                    vad_segments=segments,
                    fallback_reason=self.summary.failure_reason,
                    input_file_duration_sec=input_file_duration_sec,
                    full_audio_duration_before_reference_slice=full_audio_duration_sec,
                    reference_duration_sec=reference_duration_sec,
                    vad_audio_duration_sec=full_audio_duration_sec,
                )
                return audio

            self.update_debug_metrics(
                normalized,
                vad_segments=segments,
                input_file_duration_sec=input_file_duration_sec,
                full_audio_duration_before_reference_slice=full_audio_duration_sec,
                reference_duration_sec=reference_duration_sec,
                vad_audio_duration_sec=full_audio_duration_sec,
            )
            logger.info(
                "Background voice isolation completed: kept_segments=%s rejected_segments=%s avg_similarity=%s input_file_duration_sec=%s reference_audio_duration_sec=%s vad_audio_duration_sec=%s vad_segment_count=%s vad_speech_duration_sec=%s vad_speech_percentage=%s",
                self.summary.kept_segments,
                self.summary.rejected_segments,
                self.summary.avg_similarity,
                (self.summary.debug_metrics or {}).get("input_file_duration_sec"),
                (self.summary.debug_metrics or {}).get("reference_audio_duration_sec"),
                (self.summary.debug_metrics or {}).get("vad_audio_duration_sec"),
                (self.summary.debug_metrics or {}).get("vad_segment_count"),
                (self.summary.debug_metrics or {}).get("vad_speech_duration_sec"),
                (self.summary.debug_metrics or {}).get("vad_speech_percentage"),
            )
            return np.concatenate(kept_chunks).astype(np.float32)
        except Exception as exc:
            logger.warning("Background voice isolation failed; using original audio: %s", exc)
            self.summary.fallback_used = True
            self.summary.failure_reason = exc.__class__.__name__
            self.update_debug_metrics(
                normalized,
                fallback_reason=self.summary.failure_reason,
                input_file_duration_sec=input_file_duration_sec,
                full_audio_duration_before_reference_slice=full_audio_duration_sec,
                vad_audio_duration_sec=full_audio_duration_sec,
            )
            return audio

    def report_summary(self) -> dict[str, Any]:
        return self.summary.to_report()
