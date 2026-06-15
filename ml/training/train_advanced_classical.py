#!/usr/bin/env python3
"""
Phase 11A: Advanced Classical Model Training

Trains gradient-boosted tabular models on Phase 9 parquet feature chunks using
the same stratified train/test split and feature order as the Phase 10 baseline.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train_baseline import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    build_split,
    infer_feature_columns,
    list_parquet_files,
    load_split_feature_arrays,
    parquet_row_counts,
    read_targets,
)


DEFAULT_BASELINE_DIR = PROJECT_ROOT / "models" / "baseline"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "advanced"
TARGET_COLUMN = "age_group"


@dataclass(frozen=True)
class Candidate:
    name: str
    artifact_slug: str
    train_fn: Callable[..., Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 11A advanced classical models.")
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing feature parquet chunks.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Directory containing Phase 10 feature_columns.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for advanced model artifacts and metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--early-stopping-rounds", type=int, default=15)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Parallel workers. Low default avoids 8GB RAM spikes.",
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


def package_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def load_feature_columns(
    parquet_files: list[Path],
    baseline_dir: Path,
    logger: logging.Logger,
) -> tuple[list[str], list[str], str]:
    inferred_features, dropped_columns = infer_feature_columns(parquet_files)
    feature_path = baseline_dir / "feature_columns.json"

    if not feature_path.exists():
        logger.warning("Baseline feature column artifact not found; using inferred feature order")
        return inferred_features, dropped_columns, "inferred_from_parquet"

    with feature_path.open("r", encoding="utf-8") as f:
        baseline_features = json.load(f)

    if not isinstance(baseline_features, list) or not all(
        isinstance(col, str) for col in baseline_features
    ):
        raise ValueError(f"Invalid feature column artifact: {feature_path}")

    inferred_set = set(inferred_features)
    missing = [col for col in baseline_features if col not in inferred_set]
    if missing:
        raise ValueError(
            f"Baseline feature columns missing from parquet schema: {', '.join(missing[:10])}"
        )

    logger.info("Using Phase 10 baseline feature order from %s", feature_path)
    return baseline_features, dropped_columns, str(feature_path)


def evaluate_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
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


def normalize_predictions(raw_predictions: Any) -> np.ndarray:
    predictions = np.asarray(raw_predictions)
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    return predictions.reshape(-1).astype(np.int16, copy=False)


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Any:
    import lightgbm as lgb

    callbacks = [
        lgb.early_stopping(args.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=10),
    ]
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=31,
        max_bin=127,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        class_weight="balanced",
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )
    logger.info("LightGBM best_iteration=%s", getattr(model, "best_iteration_", None))
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Any:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        max_bin=128,
        eval_metric="mlogloss",
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    logger.info("XGBoost best_iteration=%s", getattr(model, "best_iteration", None))
    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Any:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Weighted",
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.max_depth,
        l2_leaf_reg=3.0,
        random_seed=args.random_state,
        thread_count=args.n_jobs,
        od_type="Iter",
        od_wait=args.early_stopping_rounds,
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    logger.info("CatBoost best_iteration=%s", model.get_best_iteration())
    return model


def available_candidates(logger: logging.Logger) -> tuple[list[Candidate], dict[str, str]]:
    candidates: list[Candidate] = []
    skipped: dict[str, str] = {}

    if package_available("lightgbm"):
        candidates.append(Candidate("LightGBM", "lightgbm", train_lightgbm))
    else:
        skipped["LightGBM"] = "lightgbm is not installed"

    if package_available("xgboost"):
        candidates.append(Candidate("XGBoost", "xgboost", train_xgboost))
    else:
        skipped["XGBoost"] = "xgboost is not installed"

    if package_available("catboost"):
        candidates.append(Candidate("CatBoost", "catboost", train_catboost))
    else:
        skipped["CatBoost"] = "catboost is not installed"

    for model_name, reason in skipped.items():
        logger.warning("Skipping %s: %s", model_name, reason)

    if not candidates:
        raise RuntimeError(
            "No advanced model libraries are installed. Install lightgbm and xgboost "
            "from ml/requirements.txt before running Phase 11A."
        )

    return candidates, skipped


def train_candidates(
    candidates: list[Candidate],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    output_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], dict[str, str]]:
    metrics: dict[str, dict[str, Any]] = {}
    model_paths: dict[str, Path] = {}
    failures: dict[str, str] = {}

    for candidate in candidates:
        logger.info("=" * 60)
        logger.info("Training %s", candidate.name)
        logger.info("=" * 60)
        try:
            model = candidate.train_fn(X_train, y_train, X_test, y_test, classes, args, logger)
            y_pred = normalize_predictions(model.predict(X_test))
            model_metrics = evaluate_predictions(y_test, y_pred, classes)
            metrics[candidate.name] = model_metrics

            candidate_path = output_dir / f"{candidate.artifact_slug}.candidate.joblib"
            joblib.dump(model, candidate_path, compress=3)
            model_paths[candidate.name] = candidate_path

            logger.info(
                "%s: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                candidate.name,
                model_metrics["accuracy"],
                model_metrics["precision_weighted"],
                model_metrics["recall_weighted"],
                model_metrics["f1_weighted"],
            )
        except Exception as exc:  # noqa: BLE001 - keep remaining models trainable.
            failures[candidate.name] = str(exc)
            logger.exception("%s training failed: %s", candidate.name, exc)
        finally:
            try:
                del model
            except UnboundLocalError:
                pass
            gc.collect()

    if not metrics:
        raise RuntimeError(f"All advanced model training failed: {failures}")

    return metrics, model_paths, failures


def save_outputs(
    output_dir: Path,
    best_model_name: str,
    model_paths: dict[str, Path],
    label_encoder: Any,
    feature_columns: list[str],
    metrics_output: dict[str, Any],
    logger: logging.Logger,
) -> None:
    best_model_path = output_dir / "voiceage_advanced_model.joblib"
    encoder_path = output_dir / "label_encoder.joblib"
    feature_columns_path = output_dir / "feature_columns.json"
    metrics_path = output_dir / "advanced_metrics.json"

    shutil.copy2(model_paths[best_model_name], best_model_path)
    joblib.dump(label_encoder, encoder_path)

    with feature_columns_path.open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    for path in model_paths.values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    logger.info("Saved best advanced model: %s", best_model_path)
    logger.info("Saved label encoder: %s", encoder_path)
    logger.info("Saved feature columns: %s", feature_columns_path)
    logger.info("Saved metrics: %s", metrics_path)


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("PHASE 11A: Advanced Classical Model Training")
    logger.info("=" * 60)

    parquet_files = list_parquet_files(args.chunks_dir)
    row_counts = parquet_row_counts(parquet_files, args.max_rows)
    total_rows = int(sum(row_counts))
    if total_rows == 0:
        raise ValueError("No rows selected for training")

    logger.info("Feature chunks: %s", len(parquet_files))
    logger.info("Selected rows: %s", total_rows)

    feature_columns, dropped_columns, feature_source = load_feature_columns(
        parquet_files=parquet_files,
        baseline_dir=args.baseline_dir,
        logger=logger,
    )
    logger.info("Feature columns retained: %s", len(feature_columns))
    logger.info("Columns dropped: %s", ", ".join(dropped_columns))

    raw_targets, valid_mask, label_counts = read_targets(parquet_files, row_counts)
    invalid_rows = int((~valid_mask).sum())
    assignment, y_by_raw, y_train_split, y_test_split, valid_indices, label_encoder = build_split(
        raw_targets=raw_targets,
        valid_mask=valid_mask,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    del y_train_split, y_test_split, valid_indices

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
        logger.info("Dry run complete; no features loaded and no models trained.")
        return 0

    candidates, skipped_models = available_candidates(logger)

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
    del raw_targets, valid_mask, assignment, y_by_raw
    gc.collect()

    metrics, model_paths, failed_models = train_candidates(
        candidates=candidates,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
        output_dir=args.output_dir,
        args=args,
        logger=logger,
    )

    best_model_name = max(metrics, key=lambda name: metrics[name]["f1_weighted"])
    best_metrics = metrics[best_model_name]
    metrics_output = {
        "phase": "11A",
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "selection_metric": "f1_weighted",
        "best_metrics": best_metrics,
        "training_config": {
            "chunks_dir": str(args.chunks_dir),
            "baseline_dir": str(args.baseline_dir),
            "feature_source": feature_source,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "stratified": True,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "early_stopping_rounds": args.early_stopping_rounds,
            "n_jobs": args.n_jobs,
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
        "skipped_models": skipped_models,
        "failed_models": failed_models,
        "models": metrics,
    }

    save_outputs(
        output_dir=args.output_dir,
        best_model_name=best_model_name,
        model_paths=model_paths,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        metrics_output=metrics_output,
        logger=logger,
    )

    logger.info("=" * 60)
    logger.info("PHASE 11A COMPLETE")
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
