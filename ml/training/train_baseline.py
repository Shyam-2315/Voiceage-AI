#!/usr/bin/env python3
"""
Phase 10: Baseline Model Training

Loads parquet feature chunks, trains RandomForest and ExtraTrees baselines on
age_group, evaluates both, and saves the best model plus training artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_COLUMN = "age_group"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "features" / "chunks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "baseline"

NON_FEATURE_COLUMNS = {
    "metadata_row",
    "audio_path",
    "file_name",
    "label",
    "age",
    "age_group",
    "duration_seconds",
    "sample_rate",
    "source_processed_path",
    "source_original_path",
    "processed_path",
    "original_path",
    "row_number",
}
NON_FEATURE_NAME_PARTS = (
    "path",
    "text",
    "transcript",
    "sentence",
    "filename",
    "file_name",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    artifact_slug: str
    estimator: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 10 baseline classifiers.")
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing feature parquet chunks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model artifacts and metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Trees per model. Keep this modest on 8GB RAM.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=24,
        help="Tree max_depth. Limits model memory; use 0 for unlimited.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples per leaf. Higher values reduce tree size.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Parallel workers for sklearn. Low default avoids 8GB RAM spikes.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and split without loading features or training.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def list_parquet_files(chunks_dir: Path) -> list[Path]:
    parquet_files = sorted(chunks_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {chunks_dir}")
    return parquet_files


def is_feature_column(field: pa.Field) -> bool:
    name = field.name
    lower_name = name.lower()
    if name in NON_FEATURE_COLUMNS:
        return False
    if any(part in lower_name for part in NON_FEATURE_NAME_PARTS):
        return False
    return pa.types.is_floating(field.type) or pa.types.is_integer(field.type)


def infer_feature_columns(parquet_files: list[Path]) -> tuple[list[str], list[str]]:
    schema = pq.ParquetFile(parquet_files[0]).schema_arrow
    if TARGET_COLUMN not in schema.names:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    feature_columns = [field.name for field in schema if is_feature_column(field)]
    if not feature_columns:
        raise ValueError("No numeric feature columns found after dropping metadata columns")

    dropped_columns = [field.name for field in schema if field.name not in feature_columns]
    feature_set = set(feature_columns)

    for path in parquet_files[1:]:
        current_schema = pq.ParquetFile(path).schema_arrow
        current_names = set(current_schema.names)
        missing_features = sorted(feature_set - current_names)
        if missing_features:
            raise ValueError(
                f"{path} is missing feature columns: {', '.join(missing_features[:10])}"
            )
        if TARGET_COLUMN not in current_names:
            raise ValueError(f"{path} is missing target column: {TARGET_COLUMN}")

    return feature_columns, dropped_columns


def parquet_row_counts(parquet_files: list[Path], max_rows: int | None) -> list[int]:
    counts: list[int] = []
    remaining = max_rows

    for path in parquet_files:
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        if remaining is None:
            counts.append(rows)
            continue

        if remaining <= 0:
            counts.append(0)
            continue

        clipped_rows = min(rows, remaining)
        counts.append(clipped_rows)
        remaining -= clipped_rows

    return counts


def read_targets(
    parquet_files: list[Path],
    row_counts: list[int],
) -> tuple[np.ndarray, np.ndarray, Counter[str]]:
    targets: list[np.ndarray] = []

    for path, rows_to_read in zip(parquet_files, row_counts):
        if rows_to_read <= 0:
            continue

        table = pq.read_table(path, columns=[TARGET_COLUMN])
        if rows_to_read < table.num_rows:
            table = table.slice(0, rows_to_read)

        series = table.column(TARGET_COLUMN).to_pandas()
        targets.append(series.astype("string").str.strip().to_numpy(dtype=object))

    if not targets:
        raise ValueError("No rows available for training")

    raw_targets = np.concatenate(targets)
    valid_mask = pd.notna(raw_targets) & (raw_targets.astype(str) != "")
    label_counts = Counter(raw_targets[valid_mask].astype(str))

    if not label_counts:
        raise ValueError(f"No valid {TARGET_COLUMN} labels found")
    if min(label_counts.values()) < 2:
        raise ValueError(
            "Stratified split requires at least 2 examples per class. "
            f"Class counts: {dict(sorted(label_counts.items()))}"
        )

    return raw_targets, valid_mask, label_counts


def build_split(
    raw_targets: np.ndarray,
    valid_mask: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    valid_indices = np.flatnonzero(valid_mask)
    labels = raw_targets[valid_indices].astype(str)

    label_encoder = LabelEncoder()
    y_valid = label_encoder.fit_transform(labels).astype(np.int16, copy=False)

    train_raw_idx, test_raw_idx, y_train, y_test = train_test_split(
        valid_indices,
        y_valid,
        test_size=test_size,
        random_state=random_state,
        stratify=y_valid,
    )

    assignment = np.full(raw_targets.shape[0], -1, dtype=np.int8)
    y_by_raw = np.full(raw_targets.shape[0], -1, dtype=np.int16)
    assignment[train_raw_idx] = 1
    assignment[test_raw_idx] = 0
    y_by_raw[valid_indices] = y_valid

    return assignment, y_by_raw, y_train, y_test, valid_indices, label_encoder


def load_split_feature_arrays(
    parquet_files: list[Path],
    row_counts: list[int],
    feature_columns: list[str],
    assignment: np.ndarray,
    y_by_raw: np.ndarray,
    train_size: int,
    test_size: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.empty((train_size, len(feature_columns)), dtype=np.float32)
    X_test = np.empty((test_size, len(feature_columns)), dtype=np.float32)
    y_train = np.empty(train_size, dtype=np.int16)
    y_test = np.empty(test_size, dtype=np.int16)

    train_cursor = 0
    test_cursor = 0
    global_start = 0

    for file_number, (path, rows_to_read) in enumerate(zip(parquet_files, row_counts), start=1):
        global_end = global_start + rows_to_read
        if rows_to_read <= 0:
            continue

        chunk_assignment = assignment[global_start:global_end]
        if np.all(chunk_assignment < 0):
            global_start = global_end
            continue

        table = pq.read_table(path, columns=feature_columns)
        if rows_to_read < table.num_rows:
            table = table.slice(0, rows_to_read)

        chunk_matrix = table.to_pandas().to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(chunk_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        train_mask = chunk_assignment == 1
        test_mask = chunk_assignment == 0

        train_count = int(train_mask.sum())
        if train_count:
            train_slice = slice(train_cursor, train_cursor + train_count)
            X_train[train_slice] = chunk_matrix[train_mask]
            y_train[train_slice] = y_by_raw[global_start:global_end][train_mask]
            train_cursor += train_count

        test_count = int(test_mask.sum())
        if test_count:
            test_slice = slice(test_cursor, test_cursor + test_count)
            X_test[test_slice] = chunk_matrix[test_mask]
            y_test[test_slice] = y_by_raw[global_start:global_end][test_mask]
            test_cursor += test_count

        del table, chunk_matrix
        gc.collect()

        logger.info(
            "Loaded projected features from %s/%s chunks; train rows=%s, test rows=%s",
            file_number,
            len(parquet_files),
            train_cursor,
            test_cursor,
        )
        global_start = global_end

    if train_cursor != train_size or test_cursor != test_size:
        raise RuntimeError(
            "Feature loading produced unexpected split sizes: "
            f"train {train_cursor}/{train_size}, test {test_cursor}/{test_size}"
        )

    return X_train, X_test, y_train, y_test


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    max_depth = None if args.max_depth == 0 else args.max_depth
    common_params = {
        "n_estimators": args.n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "class_weight": "balanced",
    }
    return [
        ModelSpec(
            name="RandomForestClassifier",
            artifact_slug="random_forest",
            estimator=RandomForestClassifier(**common_params),
        ),
        ModelSpec(
            name="ExtraTreesClassifier",
            artifact_slug="extra_trees",
            estimator=ExtraTreesClassifier(**common_params),
        ),
    ]


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    labels = np.arange(len(classes))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_and_save_candidates(
    specs: list[ModelSpec],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    metrics: dict[str, dict[str, Any]] = {}
    candidate_paths: dict[str, Path] = {}

    for spec in specs:
        logger.info("=" * 60)
        logger.info("Training %s", spec.name)
        logger.info("=" * 60)

        model = spec.estimator
        model.fit(X_train, y_train)
        model_metrics = evaluate_model(model, X_test, y_test, classes)
        metrics[spec.name] = model_metrics

        candidate_path = output_dir / f"{spec.artifact_slug}.candidate.joblib"
        joblib.dump(model, candidate_path, compress=3)
        candidate_paths[spec.name] = candidate_path

        logger.info(
            "%s: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
            spec.name,
            model_metrics["accuracy"],
            model_metrics["precision_weighted"],
            model_metrics["recall_weighted"],
            model_metrics["f1_weighted"],
        )

        del model
        gc.collect()

    return metrics, candidate_paths


def save_outputs(
    output_dir: Path,
    best_model_name: str,
    candidate_paths: dict[str, Path],
    label_encoder: LabelEncoder,
    feature_columns: list[str],
    metrics_output: dict[str, Any],
    logger: logging.Logger,
) -> None:
    model_path = output_dir / "voiceage_baseline_model.joblib"
    encoder_path = output_dir / "label_encoder.joblib"
    feature_columns_path = output_dir / "feature_columns.json"
    metrics_path = output_dir / "baseline_metrics.json"

    shutil.copy2(candidate_paths[best_model_name], model_path)
    joblib.dump(label_encoder, encoder_path)

    with feature_columns_path.open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    for path in candidate_paths.values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    logger.info("Saved best model: %s", model_path)
    logger.info("Saved label encoder: %s", encoder_path)
    logger.info("Saved feature columns: %s", feature_columns_path)
    logger.info("Saved metrics: %s", metrics_path)


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("PHASE 10: Baseline Model Training")
    logger.info("=" * 60)

    parquet_files = list_parquet_files(args.chunks_dir)
    row_counts = parquet_row_counts(parquet_files, args.max_rows)
    total_rows = int(sum(row_counts))
    if total_rows == 0:
        raise ValueError("No rows selected for training")

    logger.info("Feature chunks: %s", len(parquet_files))
    logger.info("Selected rows: %s", total_rows)

    feature_columns, dropped_columns = infer_feature_columns(parquet_files)
    logger.info("Feature columns retained: %s", len(feature_columns))
    logger.info("Columns dropped: %s", ", ".join(dropped_columns))

    raw_targets, valid_mask, label_counts = read_targets(parquet_files, row_counts)
    invalid_rows = int((~valid_mask).sum())
    logger.info("Valid labeled rows: %s", int(valid_mask.sum()))
    logger.info("Invalid or empty target rows skipped: %s", invalid_rows)
    logger.info("Target distribution: %s", dict(sorted(label_counts.items())))

    assignment, y_by_raw, y_train_split, y_test_split, valid_indices, label_encoder = build_split(
        raw_targets=raw_targets,
        valid_mask=valid_mask,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    del y_train_split, y_test_split

    train_size = int((assignment == 1).sum())
    test_size = int((assignment == 0).sum())
    classes = [str(label) for label in label_encoder.classes_]

    logger.info("Train rows: %s", train_size)
    logger.info("Test rows: %s", test_size)
    logger.info("Classes: %s", classes)

    if args.dry_run:
        logger.info("Dry run complete; no features loaded and no models trained.")
        return 0

    logger.info("Loading projected feature arrays as float32")
    X_train, X_test, y_train, y_test = load_split_feature_arrays(
        parquet_files=parquet_files,
        row_counts=row_counts,
        feature_columns=feature_columns,
        assignment=assignment,
        y_by_raw=y_by_raw,
        train_size=train_size,
        test_size=test_size,
        logger=logger,
    )

    del raw_targets, valid_mask, valid_indices, assignment, y_by_raw
    gc.collect()

    logger.info("Training matrix: %s rows x %s features", X_train.shape[0], X_train.shape[1])
    logger.info("Test matrix: %s rows x %s features", X_test.shape[0], X_test.shape[1])

    specs = model_specs(args)
    metrics, candidate_paths = train_and_save_candidates(
        specs=specs,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classes=classes,
        output_dir=args.output_dir,
        logger=logger,
    )

    best_model_name = max(metrics, key=lambda name: metrics[name]["f1_weighted"])
    best_metrics = metrics[best_model_name]
    metrics_output = {
        "phase": 10,
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "selection_metric": "f1_weighted",
        "best_metrics": best_metrics,
        "training_config": {
            "chunks_dir": str(args.chunks_dir),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "stratified": True,
            "n_estimators": args.n_estimators,
            "max_depth": None if args.max_depth == 0 else args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "n_jobs": args.n_jobs,
            "class_weight": "balanced",
            "dtype": "float32",
            "max_rows": args.max_rows,
        },
        "rows": {
            "selected": total_rows,
            "valid_labeled": int(len(y_train) + len(y_test)),
            "invalid_or_empty_target": invalid_rows,
            "train": int(len(y_train)),
            "test": int(len(y_test)),
        },
        "columns": {
            "feature_count": len(feature_columns),
            "features": feature_columns,
            "dropped": dropped_columns,
        },
        "classes": classes,
        "class_distribution": dict(sorted(label_counts.items())),
        "models": metrics,
    }

    save_outputs(
        output_dir=args.output_dir,
        best_model_name=best_model_name,
        candidate_paths=candidate_paths,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        metrics_output=metrics_output,
        logger=logger,
    )

    logger.info("=" * 60)
    logger.info("PHASE 10 COMPLETE")
    logger.info("Best model: %s", best_model_name)
    logger.info(
        "Best metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        best_metrics["accuracy"],
        best_metrics["precision_weighted"],
        best_metrics["recall_weighted"],
        best_metrics["f1_weighted"],
    )
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line script should log fatal context.
        logging.getLogger(__name__).exception("Training failed: %s", exc)
        raise SystemExit(1)
