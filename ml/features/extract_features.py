"""
Phase 9: Feature Extraction Pipeline

Extracts acoustic features from processed Common Voice WAV files and writes one
aggregated feature vector per audio file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Some local Python installs expose librosa's joblib cache without a source
# locator. Disabling the cache keeps feature extraction deterministic and avoids
# treating cache setup failures as corrupted audio.
os.environ["LIBROSA_CACHE_LEVEL"] = "0"
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/voiceage_numba_cache")

TARGET_SAMPLE_RATE = 16_000
N_MFCC = 40
N_CHROMA = 12
N_FFT = 2048
HOP_LENGTH = 512
AGGREGATIONS = ("mean", "std", "min", "max")
DEFAULT_METADATA_CHUNK_SIZE = 10_000
DEFAULT_WORKERS = 6
ERROR_FIELDNAMES = [
    "chunk_index",
    "row_number",
    "audio_path",
    "age_group",
    "error_type",
    "error",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Extract Phase 9 audio features.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project_root / "data" / "processed" / "processed_commonvoice_metadata.csv",
        help="Processed metadata CSV.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=project_root / "data" / "processed" / "commonvoice_audio",
        help="Directory containing processed WAV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "features" / "features.parquet",
        help="Deprecated. Feature chunks are written to --chunks-dir.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=project_root / "data" / "features" / "chunks",
        help="Directory for per-chunk parquet files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "data" / "features" / "feature_report.json",
        help="Output feature extraction report.",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=project_root / "data" / "features" / "feature_errors.csv",
        help="CSV file for failed or skipped audio rows.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of multiprocessing workers. Capped at 6.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help="Task chunksize passed to multiprocessing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_METADATA_CHUNK_SIZE,
        help="Number of metadata rows to process per parquet chunk.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke testing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip metadata chunks whose parquet output already exists.",
    )
    return parser.parse_args()


def iter_metadata_chunks(
    metadata_path: Path,
    audio_dir: Path,
    chunk_size: int,
    limit: int | None,
) -> Iterator[tuple[int, list[dict[str, str]]]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")

    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"age_group"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Metadata missing required columns: {', '.join(sorted(missing_columns))}"
            )

        chunk_index = 0
        chunk_tasks: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if limit is not None and ((chunk_index * chunk_size) + len(chunk_tasks)) >= limit:
                break

            audio_path = resolve_audio_path(row, audio_dir)
            chunk_tasks.append(
                {
                    "row_number": str(row_number),
                    "audio_path": str(audio_path),
                    "age_group": row.get("age_group", "").strip(),
                    "age": row.get("age", "").strip(),
                    "source_processed_path": row.get("processed_path", "").strip(),
                    "source_original_path": row.get("original_path", "").strip(),
                }
            )

            if len(chunk_tasks) >= chunk_size:
                yield chunk_index, chunk_tasks
                chunk_index += 1
                chunk_tasks = []

        if chunk_tasks:
            yield chunk_index, chunk_tasks


def resolve_audio_path(row: dict[str, str], audio_dir: Path) -> Path:
    candidates = [
        row.get("processed_path", "").strip(),
        row.get("audio_path", "").strip(),
        row.get("file_name", "").strip(),
    ]

    for raw_path in candidates:
        if not raw_path:
            continue

        path = Path(raw_path)
        if path.is_absolute():
            return path

        direct_path = path
        if direct_path.exists():
            return direct_path

        return audio_dir / path.name

    return audio_dir / ""


def aggregate_feature_matrix(
    values: np.ndarray,
    prefix: str,
    output: dict[str, float],
) -> None:
    matrix = np.atleast_2d(values)

    for index, series in enumerate(matrix, start=1):
        finite_series = series[np.isfinite(series)]
        if finite_series.size == 0:
            stats = {name: math.nan for name in AGGREGATIONS}
        else:
            stats = {
                "mean": float(np.mean(finite_series)),
                "std": float(np.std(finite_series)),
                "min": float(np.min(finite_series)),
                "max": float(np.max(finite_series)),
            }

        for stat_name in AGGREGATIONS:
            output[f"{prefix}_{index:02d}_{stat_name}"] = stats[stat_name]


def extract_audio_features(task: dict[str, str]) -> dict[str, Any]:
    try:
        import librosa
    except ImportError as exc:
        return {
            "ok": False,
            "row_number": task["row_number"],
            "audio_path": task["audio_path"],
            "age_group": task["age_group"],
            "error_type": "missing_dependency",
            "error": str(exc),
        }

    audio_path = Path(task["audio_path"])
    if not audio_path.exists():
        return failed_result(task, "missing_file", "Audio file does not exist")
    if audio_path.suffix.lower() != ".wav":
        return failed_result(task, "invalid_extension", "Expected a WAV file")

    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
        if y.size == 0:
            return failed_result(task, "empty_audio", "Audio file contains no samples")

        y = np.nan_to_num(y.astype(np.float32, copy=False))
        if y.size < N_FFT:
            y = np.pad(y, (0, N_FFT - y.size))

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        delta_mfcc = librosa.feature.delta(mfcc, mode="nearest")
        delta_delta_mfcc = librosa.feature.delta(mfcc, order=2, mode="nearest")
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )
        rms = librosa.feature.rms(
            y=y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_chroma=N_CHROMA,
        )

        features: dict[str, Any] = {
            "metadata_row": int(task["row_number"]),
            "audio_path": str(audio_path),
            "file_name": audio_path.name,
            "label": task["age_group"],
            "age_group": task["age_group"],
            "age": task["age"],
            "duration_seconds": float(librosa.get_duration(y=y, sr=sr)),
            "sample_rate": int(sr),
        }
        aggregate_feature_matrix(mfcc, "mfcc", features)
        aggregate_feature_matrix(delta_mfcc, "delta_mfcc", features)
        aggregate_feature_matrix(delta_delta_mfcc, "delta_delta_mfcc", features)
        aggregate_feature_matrix(spectral_centroid, "spectral_centroid", features)
        aggregate_feature_matrix(spectral_bandwidth, "spectral_bandwidth", features)
        aggregate_feature_matrix(spectral_contrast, "spectral_contrast", features)
        aggregate_feature_matrix(zero_crossing_rate, "zero_crossing_rate", features)
        aggregate_feature_matrix(rms, "rms_energy", features)
        aggregate_feature_matrix(chroma, "chroma", features)

        return {"ok": True, "features": features}
    except Exception as exc:  # noqa: BLE001 - corrupted audio must be skipped safely.
        return failed_result(task, exc.__class__.__name__, str(exc))


def failed_result(task: dict[str, str], error_type: str, error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "row_number": task["row_number"],
        "audio_path": task["audio_path"],
        "age_group": task["age_group"],
        "error_type": error_type,
        "error": error,
    }


def write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write features.parquet. "
            "Install ML dependencies with: pip install -r ml/requirements.txt"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)


def chunk_output_path(chunks_dir: Path, chunk_index: int) -> Path:
    return chunks_dir / f"features_part_{chunk_index:04d}.parquet"


def parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return 0

    return int(pq.ParquetFile(path).metadata.num_rows)


def parquet_feature_column_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return 0

    excluded_columns = {
        "audio_path",
        "file_name",
        "metadata_row",
        "label",
        "age_group",
        "age",
        "duration_seconds",
        "sample_rate",
    }
    schema = pq.ParquetFile(path).schema_arrow
    return sum(1 for name in schema.names if name not in excluded_columns)


def prepare_output_paths(args: argparse.Namespace) -> None:
    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not args.errors.exists():
            with args.errors.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=ERROR_FIELDNAMES).writeheader()
        return

    for chunk_file in args.chunks_dir.glob("features_part_*.parquet"):
        chunk_file.unlink()
    if args.output.exists():
        args.output.unlink()
    if args.errors.exists():
        args.errors.unlink()
    with args.errors.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=ERROR_FIELDNAMES).writeheader()


def append_errors(errors_path: Path, chunk_index: int, errors: list[dict[str, Any]]) -> None:
    if not errors:
        return

    with errors_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDNAMES)
        for error in errors:
            writer.writerow(
                {
                    "chunk_index": chunk_index,
                    "row_number": error.get("row_number", ""),
                    "audio_path": error.get("audio_path", ""),
                    "age_group": error.get("age_group", ""),
                    "error_type": error.get("error_type", ""),
                    "error": error.get("error", ""),
                }
            )


def process_chunk(
    *,
    chunk_index: int,
    tasks: list[dict[str, str]],
    output_path: Path,
    errors_path: Path,
    workers: int,
    task_chunksize: int,
) -> tuple[int, int, Counter[str], list[str], int]:
    features: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped_by_type: Counter[str] = Counter()
    feature_columns: list[str] = []

    worker_count = max(1, min(workers, len(tasks)))
    desc = f"Chunk {chunk_index:04d}"

    if worker_count == 1:
        results = map(extract_audio_features, tasks)
        iterator = tqdm(results, total=len(tasks), desc=desc, unit="file")
        for result in iterator:
            if result["ok"]:
                features.append(result["features"])
            else:
                errors.append(result)
                skipped_by_type[result["error_type"]] += 1
    else:
        with mp.Pool(processes=worker_count) as pool:
            results = pool.imap_unordered(
                extract_audio_features,
                tasks,
                chunksize=max(1, task_chunksize),
            )
            iterator = tqdm(results, total=len(tasks), desc=desc, unit="file")
            for result in iterator:
                if result["ok"]:
                    features.append(result["features"])
                else:
                    errors.append(result)
                    skipped_by_type[result["error_type"]] += 1

    if features:
        if output_path.exists():
            output_path.unlink()
        write_parquet(features, output_path)
        feature_columns = [
            column
            for column in features[0].keys()
            if column
            not in {
                "audio_path",
                "file_name",
                "metadata_row",
                "label",
                "age_group",
                "age",
                "duration_seconds",
                "sample_rate",
            }
        ]

    append_errors(errors_path, chunk_index, errors)
    return len(features), len(errors), skipped_by_type, feature_columns, worker_count


def build_report(
    *,
    started_at: float,
    metadata_rows_seen: int,
    processed_files: int,
    skipped_files: int,
    skipped_by_type: Counter[str],
    chunks_written: int,
    chunks_skipped_existing: int,
    chunk_size: int,
    chunks_dir: Path,
    errors_path: Path,
    metadata_path: Path,
    audio_dir: Path,
    workers: int,
    resume: bool,
    limit: int | None,
    feature_column_count: int,
) -> dict[str, Any]:
    elapsed_seconds = time.time() - started_at

    return {
        "phase": 9,
        "status": "completed",
        "metadata_path": str(metadata_path),
        "audio_dir": str(audio_dir),
        "chunks_dir": str(chunks_dir),
        "errors_path": str(errors_path),
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "workers": workers,
        "chunk_size": chunk_size,
        "resume": resume,
        "limit": limit,
        "total_metadata_rows": metadata_rows_seen,
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "skipped_by_type": dict(sorted(skipped_by_type.items())),
        "chunks_written": chunks_written,
        "chunks_skipped_existing": chunks_skipped_existing,
        "merged_output_written": False,
        "feature_families": {
            "mfcc": N_MFCC,
            "delta_mfcc": N_MFCC,
            "delta_delta_mfcc": N_MFCC,
            "spectral_centroid": 1,
            "spectral_bandwidth": 1,
            "spectral_contrast": 7,
            "zero_crossing_rate": 1,
            "rms_energy": 1,
            "chroma": N_CHROMA,
        },
        "aggregations": list(AGGREGATIONS),
        "feature_column_count": feature_column_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def run_extraction(args: argparse.Namespace) -> int:
    started_at = time.time()
    workers = max(1, min(args.workers, DEFAULT_WORKERS))

    log.info("=" * 60)
    log.info("PHASE 9: Feature Extraction Pipeline")
    log.info("=" * 60)
    log.info("Metadata       : %s", args.metadata)
    log.info("Audio dir      : %s", args.audio_dir)
    log.info("Chunks dir     : %s", args.chunks_dir)
    log.info("Errors CSV     : %s", args.errors)
    log.info("Report         : %s", args.report)
    log.info("Chunk size     : %d", args.chunk_size)
    log.info("Workers        : %d", workers)
    log.info("Resume         : %s", args.resume)
    log.info("=" * 60)

    prepare_output_paths(args)

    metadata_rows_seen = 0
    processed_total = 0
    skipped_total = 0
    chunks_written = 0
    chunks_skipped_existing = 0
    first_feature_columns: list[str] = []
    skipped_by_type: Counter[str] = Counter()

    for chunk_index, tasks in iter_metadata_chunks(
        args.metadata,
        args.audio_dir,
        args.chunk_size,
        args.limit,
    ):
        metadata_rows_seen += len(tasks)
        chunk_path = chunk_output_path(args.chunks_dir, chunk_index)

        if args.resume and chunk_path.exists():
            existing_rows = parquet_row_count(chunk_path)
            processed_total += existing_rows
            chunks_skipped_existing += 1
            if not first_feature_columns:
                first_feature_columns = ["existing"] * parquet_feature_column_count(chunk_path)
            log.info(
                "Chunk %04d: skipping existing %s (%d rows)",
                chunk_index,
                chunk_path,
                existing_rows,
            )
            continue

        log.info(
            "Chunk %04d: processing %d metadata rows -> %s",
            chunk_index,
            len(tasks),
            chunk_path,
        )
        processed_count, error_count, chunk_skipped_by_type, feature_columns, _worker_count = (
            process_chunk(
                chunk_index=chunk_index,
                tasks=tasks,
                output_path=chunk_path,
                errors_path=args.errors,
                workers=workers,
                task_chunksize=args.chunksize,
            )
        )

        processed_total += processed_count
        skipped_total += error_count
        skipped_by_type.update(chunk_skipped_by_type)
        if processed_count > 0:
            chunks_written += 1
        if feature_columns and not first_feature_columns:
            first_feature_columns = feature_columns

        log.info(
            "Chunk %04d complete: features=%d skipped=%d",
            chunk_index,
            processed_count,
            error_count,
        )

    if metadata_rows_seen == 0:
        log.error("No metadata rows found for feature extraction.")
        return 1

    if processed_total == 0:
        log.error("No feature vectors were extracted. Report will be written with failures.")
        report = build_report(
            started_at=started_at,
            metadata_rows_seen=metadata_rows_seen,
            processed_files=0,
            skipped_files=skipped_total,
            skipped_by_type=skipped_by_type,
            chunks_written=chunks_written,
            chunks_skipped_existing=chunks_skipped_existing,
            chunk_size=args.chunk_size,
            chunks_dir=args.chunks_dir,
            errors_path=args.errors,
            metadata_path=args.metadata,
            audio_dir=args.audio_dir,
            workers=workers,
            resume=args.resume,
            limit=args.limit,
            feature_column_count=0,
        )
        write_report(report, args.report)
        return 1

    report = build_report(
        started_at=started_at,
        metadata_rows_seen=metadata_rows_seen,
        processed_files=processed_total,
        skipped_files=skipped_total,
        skipped_by_type=skipped_by_type,
        chunks_written=chunks_written,
        chunks_skipped_existing=chunks_skipped_existing,
        chunk_size=args.chunk_size,
        chunks_dir=args.chunks_dir,
        errors_path=args.errors,
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        workers=workers,
        resume=args.resume,
        limit=args.limit,
        feature_column_count=len(first_feature_columns),
    )
    write_report(report, args.report)

    log.info("=" * 60)
    log.info("PHASE 9 COMPLETE")
    log.info("Metadata rows   : %d", metadata_rows_seen)
    log.info("Feature vectors : %d", processed_total)
    log.info("Skipped files   : %d", skipped_total)
    log.info("Chunks written  : %d", chunks_written)
    log.info("Chunks resumed  : %d", chunks_skipped_existing)
    log.info("Chunks dir      : %s", args.chunks_dir)
    log.info("Errors CSV      : %s", args.errors)
    log.info("Report          : %s", args.report)
    log.info("=" * 60)
    return 0


def main() -> int:
    try:
        return run_extraction(parse_args())
    except Exception as exc:  # noqa: BLE001 - top-level CLI should log cleanly.
        log.error("Feature extraction failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
