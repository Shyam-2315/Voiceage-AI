#!/usr/bin/env python3
"""
Phase 11C: Wav2Vec2 Embedding Classifier

Loads Wav2Vec2 embedding parquet chunks, trains classical classifiers on the
frozen embedding vectors, compares weighted classification metrics, and saves
the best classifier plus label encoder.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_COLUMN = "age_group"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "embeddings" / "wav2vec2" / "chunks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "deep"
MODEL_FILENAME = "wav2vec2_embedding_classifier.joblib"
ENCODER_FILENAME = "label_encoder.joblib"
METRICS_FILENAME = "wav2vec2_embedding_metrics.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    artifact_slug: str
    estimator: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Phase 11C classifiers on frozen Wav2Vec2 embeddings."
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help="Directory containing Wav2Vec2 embedding parquet chunks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model artifacts and metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=24)
    parser.add_argument("--xgb-n-estimators", type=int, default=200)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=5)
    parser.add_argument("--mlp-max-iter", type=int, default=100)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke tests. Defaults to all embedding chunks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate chunks, labels, and split without loading features or training.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
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


def is_embedding_field(field: pa.Field) -> bool:
    return field.name.startswith("embedding_") and (
        pa.types.is_floating(field.type) or pa.types.is_integer(field.type)
    )


def infer_embedding_columns(parquet_files: list[Path]) -> tuple[list[str], list[str]]:
    schema = pq.ParquetFile(parquet_files[0]).schema_arrow
    if TARGET_COLUMN not in schema.names:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    embedding_columns = [field.name for field in schema if is_embedding_field(field)]
    if not embedding_columns:
        raise ValueError("No embedding_* feature columns found")

    embedding_set = set(embedding_columns)
    dropped_columns = [field.name for field in schema if field.name not in embedding_set]

    for path in parquet_files[1:]:
        current_schema = pq.ParquetFile(path).schema_arrow
        current_names = set(current_schema.names)
        missing_features = sorted(embedding_set - current_names)
        if missing_features:
            raise ValueError(
                f"{path} is missing embedding columns: {', '.join(missing_features[:10])}"
            )
        if TARGET_COLUMN not in current_names:
            raise ValueError(f"{path} is missing target column: {TARGET_COLUMN}")

    return embedding_columns, dropped_columns


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
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

    return assignment, y_by_raw, y_train, y_test, label_encoder


def load_split_embedding_arrays(
    parquet_files: list[Path],
    row_counts: list[int],
    embedding_columns: list[str],
    assignment: np.ndarray,
    y_by_raw: np.ndarray,
    train_size: int,
    test_size: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.empty((train_size, len(embedding_columns)), dtype=np.float32)
    X_test = np.empty((test_size, len(embedding_columns)), dtype=np.float32)
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

        table = pq.read_table(path, columns=embedding_columns)
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
            "Loaded embeddings from %s/%s chunks; train rows=%s, test rows=%s",
            file_number,
            len(parquet_files),
            train_cursor,
            test_cursor,
        )
        global_start = global_end

    if train_cursor != train_size or test_cursor != test_size:
        raise RuntimeError(
            "Embedding loading produced unexpected split sizes: "
            f"train {train_cursor}/{train_size}, test {test_cursor}/{test_size}"
        )

    return X_train, X_test, y_train, y_test


def package_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def model_specs(args: argparse.Namespace, class_count: int) -> tuple[list[ModelSpec], dict[str, str]]:
    skipped_models: dict[str, str] = {}
    specs: list[ModelSpec] = [
        ModelSpec(
            name="LogisticRegression",
            artifact_slug="logistic_regression",
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1_000,
                            class_weight="balanced",
                            random_state=args.random_state,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="RandomForestClassifier",
            artifact_slug="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=args.rf_n_estimators,
                max_depth=None if args.rf_max_depth == 0 else args.rf_max_depth,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            ),
        ),
    ]

    if package_available("xgboost"):
        import xgboost as xgb

        specs.append(
            ModelSpec(
                name="XGBoost",
                artifact_slug="xgboost",
                estimator=xgb.XGBClassifier(
                    objective="multi:softprob",
                    num_class=class_count,
                    n_estimators=args.xgb_n_estimators,
                    learning_rate=args.xgb_learning_rate,
                    max_depth=args.xgb_max_depth,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    tree_method="hist",
                    max_bin=128,
                    eval_metric="mlogloss",
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                ),
            )
        )
    else:
        skipped_models["XGBoost"] = "xgboost is not installed"

    specs.append(
        ModelSpec(
            name="MLPClassifier",
            artifact_slug="mlp",
            estimator=Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(256, 128),
                            activation="relu",
                            solver="adam",
                            alpha=1e-4,
                            batch_size=256,
                            learning_rate="adaptive",
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=10,
                            max_iter=args.mlp_max_iter,
                            random_state=args.random_state,
                        ),
                    ),
                ]
            ),
        )
    )

    return specs, skipped_models


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    y_pred = np.asarray(model.predict(X_test)).reshape(-1).astype(np.int16, copy=False)
    labels = np.arange(len(classes))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
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


def train_candidates(
    specs: list[ModelSpec],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], dict[str, str]]:
    metrics: dict[str, dict[str, Any]] = {}
    model_paths: dict[str, Path] = {}
    failed_models: dict[str, str] = {}

    for spec in specs:
        logger.info("=" * 60)
        logger.info("Training %s", spec.name)
        logger.info("=" * 60)

        try:
            model = spec.estimator
            model.fit(X_train, y_train)
            model_metrics = evaluate_model(model, X_test, y_test, classes)
            metrics[spec.name] = model_metrics

            candidate_path = output_dir / f"{spec.artifact_slug}.candidate.joblib"
            joblib.dump(model, candidate_path, compress=3)
            model_paths[spec.name] = candidate_path

            logger.info(
                "%s: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                spec.name,
                model_metrics["accuracy"],
                model_metrics["precision_weighted"],
                model_metrics["recall_weighted"],
                model_metrics["f1_weighted"],
            )
        except Exception as exc:  # noqa: BLE001 - train remaining requested models if one fails.
            failed_models[spec.name] = str(exc)
            logger.exception("%s training failed: %s", spec.name, exc)
        finally:
            try:
                del model
            except UnboundLocalError:
                pass
            gc.collect()

    if not metrics:
        raise RuntimeError(f"All classifier training failed: {failed_models}")

    return metrics, model_paths, failed_models


def save_outputs(
    output_dir: Path,
    best_model_name: str,
    model_paths: dict[str, Path],
    label_encoder: LabelEncoder,
    metrics_output: dict[str, Any],
    logger: logging.Logger,
) -> None:
    model_path = output_dir / MODEL_FILENAME
    encoder_path = output_dir / ENCODER_FILENAME
    metrics_path = output_dir / METRICS_FILENAME

    shutil.copy2(model_paths[best_model_name], model_path)
    joblib.dump(label_encoder, encoder_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    for path in model_paths.values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    logger.info("Saved best model: %s", model_path)
    logger.info("Saved label encoder: %s", encoder_path)
    logger.info("Saved metrics: %s", metrics_path)


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("PHASE 11C: Wav2Vec2 Embedding Classifier")
    logger.info("=" * 60)
    logger.info("Using frozen Wav2Vec2 embeddings only; no Wav2Vec2 fine-tuning is performed.")

    parquet_files = list_parquet_files(args.chunks_dir)
    row_counts = parquet_row_counts(parquet_files, args.max_rows)
    total_rows = int(sum(row_counts))
    if total_rows == 0:
        raise ValueError("No rows selected for training")

    logger.info("Embedding chunks: %s", len(parquet_files))
    logger.info("Selected rows: %s", total_rows)

    embedding_columns, dropped_columns = infer_embedding_columns(parquet_files)
    logger.info("Embedding feature columns retained: %s", len(embedding_columns))
    logger.info("Non-embedding columns dropped: %s", ", ".join(dropped_columns))

    raw_targets, valid_mask, label_counts = read_targets(parquet_files, row_counts)
    invalid_rows = int((~valid_mask).sum())
    assignment, y_by_raw, y_train_split, y_test_split, label_encoder = build_split(
        raw_targets=raw_targets,
        valid_mask=valid_mask,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_size = int((assignment == 1).sum())
    test_size = int((assignment == 0).sum())
    classes = [str(label) for label in label_encoder.classes_]

    logger.info("Valid labeled rows: %s", int(valid_mask.sum()))
    logger.info("Invalid or empty target rows skipped: %s", invalid_rows)
    logger.info("Target distribution: %s", dict(sorted(label_counts.items())))
    logger.info("Train rows: %s", train_size)
    logger.info("Test rows: %s", test_size)
    logger.info("Classes: %s", classes)

    if args.dry_run:
        logger.info("Dry run complete; no embeddings loaded and no models trained.")
        return 0

    logger.info("Loading all selected embedding chunks into train/test arrays as float32")
    X_train, X_test, y_train, y_test = load_split_embedding_arrays(
        parquet_files=parquet_files,
        row_counts=row_counts,
        embedding_columns=embedding_columns,
        assignment=assignment,
        y_by_raw=y_by_raw,
        train_size=train_size,
        test_size=test_size,
        logger=logger,
    )
    del raw_targets, valid_mask, assignment, y_by_raw, y_train_split, y_test_split
    gc.collect()

    specs, skipped_models = model_specs(args, class_count=len(classes))
    for model_name, reason in skipped_models.items():
        logger.warning("Skipping %s: %s", model_name, reason)

    metrics, model_paths, failed_models = train_candidates(
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
        "phase": "11C",
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "selection_metric": "f1_weighted",
        "best_metrics": best_metrics,
        "training_config": {
            "chunks_dir": str(args.chunks_dir),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "stratified": True,
            "target_column": TARGET_COLUMN,
            "feature_prefix": "embedding_",
            "dtype": "float32",
            "max_rows": args.max_rows,
            "n_jobs": args.n_jobs,
            "rf_n_estimators": args.rf_n_estimators,
            "rf_max_depth": args.rf_max_depth,
            "xgb_n_estimators": args.xgb_n_estimators,
            "xgb_learning_rate": args.xgb_learning_rate,
            "xgb_max_depth": args.xgb_max_depth,
            "mlp_max_iter": args.mlp_max_iter,
            "wav2vec2_fine_tuned": False,
        },
        "rows": {
            "selected": total_rows,
            "valid_labeled": int(len(y_train) + len(y_test)),
            "invalid_or_empty_target": invalid_rows,
            "train": int(len(y_train)),
            "test": int(len(y_test)),
        },
        "columns": {
            "feature_count": len(embedding_columns),
            "features": embedding_columns,
            "dropped": dropped_columns,
        },
        "classes": classes,
        "class_distribution": dict(sorted(label_counts.items())),
        "skipped_models": skipped_models,
        "failed_models": failed_models,
        "models": metrics,
    }

    save_outputs(
        output_dir=args.output_dir,
        best_model_name=best_model_name,
        model_paths=model_paths,
        label_encoder=label_encoder,
        metrics_output=metrics_output,
        logger=logger,
    )

    logger.info("=" * 60)
    logger.info("PHASE 11C COMPLETE")
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
