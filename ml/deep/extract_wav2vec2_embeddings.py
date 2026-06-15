"""
Phase 11B: Wav2Vec2 Embedding Pipeline

Extracts one mean-pooled facebook/wav2vec2-base embedding vector per processed
Common Voice WAV file. The default run builds the first deep-feature dataset as
a balanced subset of 10,000 samples per age_group class.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_SAMPLES_PER_CLASS = 10_000
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_SEED = 42
CUDA_SAFE_BATCH_SIZE = 4
CPU_SAFE_BATCH_SIZE = 2
ERROR_FIELDNAMES = [
    "chunk_index",
    "metadata_row",
    "audio_path",
    "age_group",
    "error_type",
    "error",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(
        description="Extract Phase 11B Wav2Vec2 embeddings from processed WAV files."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project_root / "data" / "processed" / "processed_commonvoice_metadata.csv",
        help="Processed Common Voice metadata CSV.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=project_root / "data" / "processed" / "commonvoice_audio",
        help="Directory containing processed WAV files.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=project_root / "data" / "embeddings" / "wav2vec2" / "chunks",
        help="Directory for embedding parquet chunks.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root
        / "data"
        / "embeddings"
        / "wav2vec2"
        / "embedding_report.json",
        help="Output embedding extraction report JSON.",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=project_root / "data" / "embeddings" / "wav2vec2" / "embedding_errors.csv",
        help="CSV file for failed or skipped audio rows.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="HuggingFace Wav2Vec2 model name.",
    )
    parser.add_argument(
        "--class-column",
        default="age_group",
        help="Metadata column used to build the balanced subset.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="Number of samples per class for the balanced subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic balanced sampling.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per parquet chunk.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Audio files per model batch. Defaults to 4 on CUDA and 2 on CPU; "
            "CUDA batches are split automatically after OOM."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit after balanced sampling for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunk parquet files that already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing Wav2Vec2 embedding chunks/report/errors before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata and planned chunks without loading the model.",
    )
    return parser.parse_args()


def resolve_audio_path(row: pd.Series, audio_dir: Path) -> Path:
    candidates = [
        str(row.get("processed_path", "") or "").strip(),
        str(row.get("audio_path", "") or "").strip(),
        str(row.get("file_name", "") or "").strip(),
    ]

    for raw_path in candidates:
        if not raw_path or raw_path.lower() == "nan":
            continue

        path = Path(raw_path)
        if path.is_absolute():
            return path
        if path.exists():
            return path
        return audio_dir / path.name

    return audio_dir / ""


def load_balanced_subset(args: argparse.Namespace) -> pd.DataFrame:
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    df = pd.read_csv(args.metadata)
    required_columns = {args.class_column, "processed_path"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata missing required columns: {', '.join(sorted(missing_columns))}"
        )

    df = df.copy()
    df["metadata_row"] = np.arange(2, len(df) + 2)
    df[args.class_column] = df[args.class_column].astype(str).str.strip()
    df = df[df[args.class_column].ne("")]

    class_counts = df[args.class_column].value_counts().sort_index()
    underfilled = class_counts[class_counts < args.samples_per_class]
    if not underfilled.empty:
        details = ", ".join(f"{label}={count}" for label, count in underfilled.items())
        raise ValueError(
            f"Not enough rows for {args.samples_per_class} samples per class: {details}"
        )

    balanced = (
        df.groupby(args.class_column, group_keys=False)
        .sample(n=args.samples_per_class, random_state=args.seed)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )
    if args.limit is not None:
        balanced = balanced.head(args.limit).copy()

    balanced["audio_path"] = balanced.apply(
        lambda row: str(resolve_audio_path(row, args.audio_dir)),
        axis=1,
    )
    return balanced


def planned_chunk_count(row_count: int, chunk_size: int) -> int:
    return (row_count + chunk_size - 1) // chunk_size


def chunk_output_path(chunks_dir: Path, chunk_index: int) -> Path:
    return chunks_dir / f"embeddings_part_{chunk_index:04d}.parquet"


def prepare_output_paths(args: argparse.Namespace) -> None:
    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)

    existing_chunks = sorted(args.chunks_dir.glob("embeddings_part_*.parquet"))
    if args.overwrite:
        for chunk_file in existing_chunks:
            chunk_file.unlink()
        if args.report.exists():
            args.report.unlink()
        if args.errors.exists():
            args.errors.unlink()
    elif existing_chunks and not args.resume:
        raise FileExistsError(
            f"Found {len(existing_chunks)} existing embedding chunks in {args.chunks_dir}. "
            "Use --resume to continue or --overwrite to rebuild."
        )

    if not args.errors.exists():
        with args.errors.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=ERROR_FIELDNAMES).writeheader()


def parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001 - corrupt partial chunks should be recomputed.
        return 0


def parquet_embedding_column_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(path).schema_arrow
        return sum(1 for name in schema.names if name.startswith("embedding_"))
    except Exception:  # noqa: BLE001 - report best-effort details for resume runs.
        return 0


def append_errors(errors_path: Path, chunk_index: int, errors: list[dict[str, Any]]) -> None:
    if not errors:
        return

    with errors_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDNAMES)
        for error in errors:
            writer.writerow(
                {
                    "chunk_index": chunk_index,
                    "metadata_row": error.get("metadata_row", ""),
                    "audio_path": error.get("audio_path", ""),
                    "age_group": error.get("age_group", ""),
                    "error_type": error.get("error_type", ""),
                    "error": error.get("error", ""),
                }
            )


def load_audio_record(row: pd.Series, audio_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    audio_path = resolve_audio_path(row, audio_dir)
    base = {
        "metadata_row": int(row["metadata_row"]),
        "audio_path": str(audio_path),
        "file_name": audio_path.name,
        "label": str(row.get("age_group", "")),
        "age_group": str(row.get("age_group", "")),
        "age": str(row.get("age", "")),
        "duration_seconds": as_optional_float(row.get("duration", None)),
        "sample_rate": TARGET_SAMPLE_RATE,
        "original_path": str(row.get("original_path", "")),
        "processed_path": str(row.get("processed_path", "")),
    }

    if not audio_path.exists():
        return None, failed_result(base, "missing_file", "Audio file does not exist")
    if audio_path.suffix.lower() != ".wav":
        return None, failed_result(base, "invalid_extension", "Expected a WAV file")

    try:
        import librosa
        import soundfile as sf

        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if waveform.size == 0:
            return None, failed_result(base, "empty_audio", "Audio file contains no samples")
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1, dtype=np.float32)
        waveform = np.asarray(waveform, dtype=np.float32)
        waveform = np.nan_to_num(waveform, copy=False)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = librosa.resample(
                waveform,
                orig_sr=int(sample_rate),
                target_sr=TARGET_SAMPLE_RATE,
            ).astype(np.float32, copy=False)

        base["duration_seconds"] = float(waveform.shape[0] / TARGET_SAMPLE_RATE)
        base["sample_rate"] = TARGET_SAMPLE_RATE
        return {"metadata": base, "waveform": waveform}, None
    except Exception as exc:  # noqa: BLE001 - bad audio should not stop the run.
        return None, failed_result(base, exc.__class__.__name__, str(exc))


def as_optional_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def failed_result(base: dict[str, Any], error_type: str, error: str) -> dict[str, Any]:
    return {
        "metadata_row": base.get("metadata_row", ""),
        "audio_path": base.get("audio_path", ""),
        "age_group": base.get("age_group", ""),
        "error_type": error_type,
        "error": error,
    }


def load_model_and_processor(model_name: str) -> tuple[Any, Any, Any, bool]:
    try:
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required. Install ML dependencies with: "
            "pip install -r ml/requirements.txt"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()

    use_amp = device.type == "cuda"
    return model, processor, device, use_amp


def mean_pool_hidden_states(model: Any, hidden_states: Any, attention_mask: Any) -> Any:
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    try:
        feature_mask = model._get_feature_vector_attention_mask(  # noqa: SLF001
            hidden_states.shape[1],
            attention_mask,
        )
    except TypeError:
        feature_mask = model._get_feature_vector_attention_mask(  # noqa: SLF001
            hidden_states.shape[1],
            attention_mask,
            add_adapter=False,
        )

    feature_mask = feature_mask.to(hidden_states.device).unsqueeze(-1)
    feature_mask = feature_mask.to(dtype=hidden_states.dtype)
    summed = (hidden_states * feature_mask).sum(dim=1)
    counts = feature_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def encode_records(
    records: list[dict[str, Any]],
    *,
    model: Any,
    processor: Any,
    device: Any,
    use_amp: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not records:
        return [], [], 0

    try:
        import torch

        waveforms = [record["waveform"] for record in records]
        inputs = processor(
            waveforms,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**inputs)
                pooled = mean_pool_hidden_states(
                    model,
                    outputs.last_hidden_state,
                    inputs.get("attention_mask"),
                )

        embeddings = pooled.detach().float().cpu().numpy()
        rows = [
            build_embedding_row(record["metadata"], embedding)
            for record, embedding in zip(records, embeddings, strict=True)
        ]
        return rows, [], int(embeddings.shape[1])
    except RuntimeError as exc:
        if is_cuda_oom(exc):
            clear_torch_cache(device)
            if len(records) > 1:
                midpoint = len(records) // 2
                left_rows, left_errors, left_dim = encode_records(
                    records[:midpoint],
                    model=model,
                    processor=processor,
                    device=device,
                    use_amp=use_amp,
                )
                right_rows, right_errors, right_dim = encode_records(
                    records[midpoint:],
                    model=model,
                    processor=processor,
                    device=device,
                    use_amp=use_amp,
                )
                return left_rows + right_rows, left_errors + right_errors, max(left_dim, right_dim)

            error = failed_result(
                records[0]["metadata"],
                "cuda_out_of_memory",
                "Single audio file exceeded available CUDA memory",
            )
            return [], [error], 0

        errors = [
            failed_result(record["metadata"], exc.__class__.__name__, str(exc))
            for record in records
        ]
        return [], errors, 0


def is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def clear_torch_cache(device: Any) -> None:
    gc.collect()
    try:
        import torch

        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cleanup should never mask the original failure.
        return


def build_embedding_row(metadata: dict[str, Any], embedding: np.ndarray) -> dict[str, Any]:
    row = dict(metadata)
    for index, value in enumerate(embedding):
        row[f"embedding_{index:04d}"] = np.float32(value)
    return row


def write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write embedding chunks. Install ML dependencies with: "
            "pip install -r ml/requirements.txt"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp_path)
    tmp_path.replace(output_path)


def process_chunk(
    *,
    chunk_index: int,
    chunk_df: pd.DataFrame,
    output_path: Path,
    errors_path: Path,
    audio_dir: Path,
    batch_size: int,
    model: Any,
    processor: Any,
    device: Any,
    use_amp: bool,
) -> tuple[int, int, Counter[str], int]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped_by_type: Counter[str] = Counter()
    embedding_dim = 0

    progress = tqdm(total=len(chunk_df), desc=f"Chunk {chunk_index:04d}", unit="file")
    try:
        for start in range(0, len(chunk_df), batch_size):
            batch_df = chunk_df.iloc[start : start + batch_size]
            records: list[dict[str, Any]] = []

            for _, row in batch_df.iterrows():
                record, error = load_audio_record(row, audio_dir)
                if error is not None:
                    errors.append(error)
                    skipped_by_type[error["error_type"]] += 1
                if record is not None:
                    records.append(record)

            batch_rows, batch_errors, batch_embedding_dim = encode_records(
                records,
                model=model,
                processor=processor,
                device=device,
                use_amp=use_amp,
            )
            rows.extend(batch_rows)
            errors.extend(batch_errors)
            for error in batch_errors:
                skipped_by_type[error["error_type"]] += 1
            embedding_dim = max(embedding_dim, batch_embedding_dim)
            progress.update(len(batch_df))
    finally:
        progress.close()

    if rows:
        write_parquet(rows, output_path)
    append_errors(errors_path, chunk_index, errors)
    return len(rows), len(errors), skipped_by_type, embedding_dim


def class_distribution(df: pd.DataFrame, class_column: str) -> dict[str, int]:
    counts = df[class_column].value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def build_report(
    *,
    started_at: float,
    args: argparse.Namespace,
    selected_df: pd.DataFrame,
    device_name: str,
    batch_size: int,
    processed_files: int,
    skipped_files: int,
    skipped_by_type: Counter[str],
    chunks_written: int,
    chunks_skipped_existing: int,
    embedding_dim: int,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "phase": "11B",
        "status": "completed" if not dry_run else "dry_run",
        "model_name": args.model_name,
        "metadata_path": str(args.metadata),
        "audio_dir": str(args.audio_dir),
        "chunks_dir": str(args.chunks_dir),
        "errors_path": str(args.errors),
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "pooling": "attention_masked_mean_last_hidden_state",
        "device": device_name,
        "batch_size": batch_size,
        "chunk_size": args.chunk_size,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "seed": args.seed,
        "class_column": args.class_column,
        "samples_per_class": args.samples_per_class,
        "limit": args.limit,
        "selected_rows": int(len(selected_df)),
        "selected_class_distribution": class_distribution(selected_df, args.class_column),
        "processed_files": int(processed_files),
        "skipped_files": int(skipped_files),
        "skipped_by_type": dict(sorted(skipped_by_type.items())),
        "chunks_planned": planned_chunk_count(len(selected_df), args.chunk_size),
        "chunks_written": int(chunks_written),
        "chunks_skipped_existing": int(chunks_skipped_existing),
        "embedding_dim": int(embedding_dim),
        "embedding_columns": (
            [f"embedding_{index:04d}" for index in range(embedding_dim)]
            if embedding_dim
            else []
        ),
        "training_started": False,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def run_extraction(args: argparse.Namespace) -> int:
    started_at = time.time()
    prepare_output_paths(args)
    selected_df = load_balanced_subset(args)

    default_batch_size = CUDA_SAFE_BATCH_SIZE
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if args.dry_run:
        batch_size = args.batch_size or CUDA_SAFE_BATCH_SIZE
        report = build_report(
            started_at=started_at,
            args=args,
            selected_df=selected_df,
            device_name="not_loaded",
            batch_size=batch_size,
            processed_files=0,
            skipped_files=0,
            skipped_by_type=Counter(),
            chunks_written=0,
            chunks_skipped_existing=0,
            embedding_dim=0,
            dry_run=True,
        )
        write_report(report, args.report)
        log.info("Dry run complete. Planned rows: %d", len(selected_df))
        return 0

    log.info("=" * 60)
    log.info("PHASE 11B: Wav2Vec2 Embedding Pipeline")
    log.info("=" * 60)
    log.info("Metadata          : %s", args.metadata)
    log.info("Audio dir         : %s", args.audio_dir)
    log.info("Chunks dir        : %s", args.chunks_dir)
    log.info("Report            : %s", args.report)
    log.info("Model             : %s", args.model_name)
    log.info("Selected rows     : %d", len(selected_df))
    log.info("Class distribution: %s", class_distribution(selected_df, args.class_column))
    log.info("Resume            : %s", args.resume)
    log.info("=" * 60)

    model, processor, device, use_amp = load_model_and_processor(args.model_name)
    if args.batch_size is None:
        default_batch_size = CUDA_SAFE_BATCH_SIZE if device.type == "cuda" else CPU_SAFE_BATCH_SIZE
    batch_size = args.batch_size or default_batch_size
    device_name = str(device)

    log.info("Device            : %s", device_name)
    log.info("Batch size        : %d", batch_size)

    processed_total = 0
    skipped_total = 0
    chunks_written = 0
    chunks_skipped_existing = 0
    skipped_by_type: Counter[str] = Counter()
    embedding_dim = 0

    for chunk_index, start in enumerate(range(0, len(selected_df), args.chunk_size)):
        chunk_df = selected_df.iloc[start : start + args.chunk_size]
        chunk_path = chunk_output_path(args.chunks_dir, chunk_index)

        if args.resume and chunk_path.exists():
            existing_rows = parquet_row_count(chunk_path)
            existing_embedding_dim = parquet_embedding_column_count(chunk_path)
            if existing_rows > 0 and existing_embedding_dim > 0:
                processed_total += existing_rows
                chunks_skipped_existing += 1
                embedding_dim = max(embedding_dim, existing_embedding_dim)
                log.info(
                    "Chunk %04d: skipping existing %s (%d rows)",
                    chunk_index,
                    chunk_path,
                    existing_rows,
                )
                continue

            log.warning(
                "Chunk %04d: existing file is empty or unreadable; recomputing %s",
                chunk_index,
                chunk_path,
            )
            chunk_path.unlink()

        log.info(
            "Chunk %04d: processing %d rows -> %s",
            chunk_index,
            len(chunk_df),
            chunk_path,
        )
        processed_count, error_count, chunk_skipped_by_type, chunk_embedding_dim = process_chunk(
            chunk_index=chunk_index,
            chunk_df=chunk_df,
            output_path=chunk_path,
            errors_path=args.errors,
            audio_dir=args.audio_dir,
            batch_size=batch_size,
            model=model,
            processor=processor,
            device=device,
            use_amp=use_amp,
        )

        processed_total += processed_count
        skipped_total += error_count
        skipped_by_type.update(chunk_skipped_by_type)
        embedding_dim = max(embedding_dim, chunk_embedding_dim)
        if processed_count:
            chunks_written += 1
        log.info(
            "Chunk %04d complete: embeddings=%d skipped=%d",
            chunk_index,
            processed_count,
            error_count,
        )

    report = build_report(
        started_at=started_at,
        args=args,
        selected_df=selected_df,
        device_name=device_name,
        batch_size=batch_size,
        processed_files=processed_total,
        skipped_files=skipped_total,
        skipped_by_type=skipped_by_type,
        chunks_written=chunks_written,
        chunks_skipped_existing=chunks_skipped_existing,
        embedding_dim=embedding_dim,
        dry_run=False,
    )
    write_report(report, args.report)

    log.info("=" * 60)
    log.info("PHASE 11B COMPLETE")
    log.info("Embedding vectors : %d", processed_total)
    log.info("Skipped files     : %d", skipped_total)
    log.info("Chunks written    : %d", chunks_written)
    log.info("Chunks resumed    : %d", chunks_skipped_existing)
    log.info("Embedding dim     : %d", embedding_dim)
    log.info("Report            : %s", args.report)
    log.info("=" * 60)
    return 0 if processed_total > 0 else 1


def main() -> int:
    try:
        return run_extraction(parse_args())
    except Exception as exc:  # noqa: BLE001 - top-level CLI should log cleanly.
        log.error("Wav2Vec2 embedding extraction failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
