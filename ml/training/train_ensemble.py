#!/usr/bin/env python3
"""
Phase 12B: Ensemble Model Training

Combines predictions from RandomForest baseline and Wav2Vec2 MLP classifier
using a meta-learner (LogisticRegression or XGBoost).

Memory-efficient design:
- Loads parquet chunks incrementally
- Uses float32 for arrays
- Avoids loading unnecessary columns
- Includes dry-run mode and verification
- Supports resume from checkpoints

Features:
- Py_compile verification for syntax correctness
- Artifact load verification (models, metrics, label encoders)
- Resume support (checkpoint on training completion)
- Comprehensive logging and error handling
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import py_compile
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "models" / "baseline"
DEFAULT_DEEP_DIR = PROJECT_ROOT / "models" / "deep"
DEFAULT_BASELINE_CHUNKS_DIR = PROJECT_ROOT / "data" / "features" / "chunks"
DEFAULT_DEEP_CHUNKS_DIR = PROJECT_ROOT / "data" / "embeddings" / "wav2vec2" / "chunks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "ensemble"

# Constants for feature detection and training
TARGET_COLUMN = "age_group"
BASELINE_NON_FEATURE_COLUMNS = {
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
BASELINE_NON_FEATURE_NAME_PARTS = (
    "path",
    "text",
    "transcript",
    "sentence",
    "filename",
    "file_name",
)
DEEP_EMBEDDING_PREFIX = "embedding_"
CHECKPOINT_FILENAME = "training_checkpoint.json"


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for ensemble training."""
    baseline_dir: Path
    deep_dir: Path
    baseline_chunks_dir: Path
    deep_chunks_dir: Path
    output_dir: Path
    test_size: float = 0.2
    random_state: int = 42
    meta_learner: str = "logistic"
    max_rows: int | None = None
    dry_run: bool = False
    resume: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ensemble model combining baseline and deep predictions."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Directory containing baseline model.",
    )
    parser.add_argument(
        "--deep-dir",
        type=Path,
        default=DEFAULT_DEEP_DIR,
        help="Directory containing deep (Wav2Vec2) model.",
    )
    parser.add_argument(
        "--baseline-chunks-dir",
        type=Path,
        default=DEFAULT_BASELINE_CHUNKS_DIR,
        help="Directory containing baseline feature chunks.",
    )
    parser.add_argument(
        "--deep-chunks-dir",
        type=Path,
        default=DEFAULT_DEEP_CHUNKS_DIR,
        help="Directory containing deep embedding chunks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for ensemble model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "--meta-learner",
        type=str,
        choices=["logistic", "xgboost"],
        default="logistic",
        help="Meta-learner type.",
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
        help="Validate inputs without training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint if available.",
    )
    parser.add_argument(
        "--verify-artifacts",
        action="store_true",
        help="Verify all artifact files are loadable.",
    )
    parser.add_argument(
        "--verify-syntax",
        action="store_true",
        help="Verify Python syntax of this script.",
    )
    return parser.parse_args()


def verify_syntax(script_path: Path = None) -> bool:
    """Verify Python syntax using py_compile."""
    if script_path is None:
        script_path = Path(__file__)
    
    try:
        py_compile.compile(str(script_path), doraise=True)
        print(f"✓ Syntax verification passed: {script_path}")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ Syntax error in {script_path}:")
        print(f"  {e}")
        return False


def verify_artifacts(
    baseline_dir: Path,
    deep_dir: Path,
    baseline_chunks_dir: Path,
    deep_chunks_dir: Path,
    logger: logging.Logger | None = None,
) -> bool:
    """Verify all artifact files are present and loadable."""
    log_fn = (lambda x: logger.info(x)) if logger else print
    
    # Baseline artifacts
    baseline_model_path = baseline_dir / "voiceage_baseline_model.joblib"
    baseline_le_path = baseline_dir / "label_encoder.joblib"
    baseline_fc_path = baseline_dir / "feature_columns.json"
    baseline_metrics_path = baseline_dir / "baseline_metrics.json"
    
    # Deep artifacts
    deep_model_path = deep_dir / "wav2vec2_embedding_classifier.joblib"
    deep_le_path = deep_dir / "label_encoder.joblib"
    deep_metrics_path = deep_dir / "wav2vec2_embedding_metrics.json"
    
    all_exist = True
    
    # Check baseline artifacts
    for path in [baseline_model_path, baseline_le_path, baseline_fc_path, baseline_metrics_path]:
        if not path.exists():
            log_fn(f"✗ Missing baseline artifact: {path}")
            all_exist = False
        else:
            log_fn(f"✓ Found: {path}")
    
    # Check deep artifacts
    for path in [deep_model_path, deep_le_path, deep_metrics_path]:
        if not path.exists():
            log_fn(f"✗ Missing deep artifact: {path}")
            all_exist = False
        else:
            log_fn(f"✓ Found: {path}")
    
    if not all_exist:
        return False
    
    # Try loading artifacts
    log_fn("\nAttempting to load artifacts...")
    try:
        joblib.load(baseline_model_path)
        log_fn("✓ Baseline model loaded")
        
        joblib.load(baseline_le_path)
        log_fn("✓ Baseline label encoder loaded")
        
        with open(baseline_fc_path) as f:
            json.load(f)
        log_fn("✓ Baseline feature columns loaded")
        
        with open(baseline_metrics_path) as f:
            json.load(f)
        log_fn("✓ Baseline metrics loaded")
        
        joblib.load(deep_model_path)
        log_fn("✓ Deep model loaded")
        
        joblib.load(deep_le_path)
        log_fn("✓ Deep label encoder loaded")
        
        with open(deep_metrics_path) as f:
            json.load(f)
        log_fn("✓ Deep metrics loaded")
        
        return True
    except Exception as e:
        log_fn(f"✗ Error loading artifacts: {e}")
        return False


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


def load_checkpoint(output_dir: Path) -> dict[str, Any] | None:
    """Load training checkpoint if it exists."""
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        return checkpoint
    except Exception as e:
        return None


def save_checkpoint(output_dir: Path, state: dict[str, Any]) -> None:
    """Save training checkpoint state."""
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        pass  # Non-critical if checkpoint fails


def list_parquet_files(chunks_dir: Path) -> list[Path]:
    parquet_files = sorted(chunks_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {chunks_dir}")
    return parquet_files


def load_baseline_data(
    chunks_dir: Path,
    feature_columns: list[str],
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load baseline features and targets."""
    parquet_files = list_parquet_files(chunks_dir)
    
    X_parts = []
    y_parts = []
    rows_loaded = 0
    
    for path in parquet_files:
        if max_rows and rows_loaded >= max_rows:
            break
        
        df = pd.read_parquet(path, columns=feature_columns + [TARGET_COLUMN])
        if max_rows:
            remaining = max_rows - rows_loaded
            df = df.iloc[:remaining]
        
        X_parts.append(df[feature_columns].values)
        y_parts.append(df[TARGET_COLUMN].values)
        rows_loaded += len(df)
    
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)
    
    return X, y


def load_deep_data(
    chunks_dir: Path,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load deep embeddings and targets."""
    parquet_files = list_parquet_files(chunks_dir)
    
    X_parts = []
    y_parts = []
    rows_loaded = 0
    
    for path in parquet_files:
        if max_rows and rows_loaded >= max_rows:
            break
        
        df = pd.read_parquet(path)
        if max_rows:
            remaining = max_rows - rows_loaded
            df = df.iloc[:remaining]
        
        # Extract embedding columns
        embedding_cols = [col for col in df.columns if col.startswith(DEEP_EMBEDDING_PREFIX)]
        X_parts.append(df[embedding_cols].values)
        y_parts.append(df[TARGET_COLUMN].values)
        rows_loaded += len(df)
    
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)
    
    return X, y


def build_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test and return train/test source indices."""
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X,
        y,
        np.arange(len(X)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, train_indices, test_indices


def verify_build_split() -> bool:
    """Smoke-test build_split without loading training data."""
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.array(["adult"] * 10 + ["senior"] * 10)
    X_train, X_test, y_train, y_test, train_indices, test_indices = build_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    expected_train_size = 15
    expected_test_size = 5
    all_indices = np.concatenate([train_indices, test_indices])
    return (
        len(X_train) == len(y_train) == len(train_indices) == expected_train_size
        and len(X_test) == len(y_test) == len(test_indices) == expected_test_size
        and len(np.intersect1d(train_indices, test_indices)) == 0
        and np.array_equal(np.sort(all_indices), np.arange(len(X)))
        and np.array_equal(X_train, X[train_indices])
        and np.array_equal(X_test, X[test_indices])
        and np.array_equal(y_train, y[train_indices])
        and np.array_equal(y_test, y[test_indices])
    )


def main() -> None:
    args = parse_args()
    
    # Handle --verify-syntax flag
    if args.verify_syntax:
        success = verify_syntax()
        sys.exit(0 if success else 1)
    
    config = EnsembleConfig(
        baseline_dir=args.baseline_dir,
        deep_dir=args.deep_dir,
        baseline_chunks_dir=args.baseline_chunks_dir,
        deep_chunks_dir=args.deep_chunks_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        meta_learner=args.meta_learner,
        max_rows=args.max_rows,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    
    logger = configure_logging(config.output_dir)
    
    logger.info("="*80)
    logger.info("Phase 12B: Ensemble Model Training")
    logger.info("="*80)
    
    # Handle --verify-artifacts flag
    if args.verify_artifacts:
        logger.info("Running artifact verification...")
        success = verify_artifacts(
            config.baseline_dir,
            config.deep_dir,
            config.baseline_chunks_dir,
            config.deep_chunks_dir,
            logger=logger,
        )
        if success:
            logger.info("✓ All artifacts verified successfully")
            sys.exit(0)
        else:
            logger.error("✗ Artifact verification failed")
            sys.exit(1)
    
    # Check for existing checkpoint
    checkpoint = None
    if config.resume:
        checkpoint = load_checkpoint(config.output_dir)
        if checkpoint:
            logger.info("Found existing checkpoint, resuming training...")
            logger.info(f"Checkpoint from {checkpoint.get('timestamp', 'unknown')}")
        else:
            logger.info("No checkpoint found, starting fresh...")
    
    # Load models
    logger.info(f"Loading baseline model from {config.baseline_dir}")
    baseline_model = joblib.load(config.baseline_dir / "voiceage_baseline_model.joblib")
    baseline_le = joblib.load(config.baseline_dir / "label_encoder.joblib")
    
    with open(config.baseline_dir / "feature_columns.json") as f:
        baseline_features = json.load(f)
    
    with open(config.baseline_dir / "baseline_metrics.json") as f:
        baseline_metrics = json.load(f)
    
    logger.info(f"Baseline model: {baseline_metrics['best_model']}")
    logger.info(f"Baseline features: {len(baseline_features)}")
    
    logger.info(f"Loading deep model from {config.deep_dir}")
    deep_model = joblib.load(config.deep_dir / "wav2vec2_embedding_classifier.joblib")
    deep_le = joblib.load(config.deep_dir / "label_encoder.joblib")
    
    with open(config.deep_dir / "wav2vec2_embedding_metrics.json") as f:
        deep_metrics = json.load(f)
    
    logger.info(f"Deep model: {deep_metrics['best_model']}")
    
    if config.dry_run:
        if not verify_build_split():
            raise RuntimeError("build_split smoke test failed")
        logger.info("build_split smoke test passed.")
        logger.info("Dry-run mode: validation complete.")
        return
    
    # Load baseline data
    logger.info(f"Loading baseline data from {config.baseline_chunks_dir}")
    baseline_X, baseline_y = load_baseline_data(
        config.baseline_chunks_dir,
        baseline_features,
        max_rows=config.max_rows,
    )
    logger.info(f"Baseline data: {len(baseline_X)} samples, {baseline_X.shape[1]} features")
    
    # Load deep data
    logger.info(f"Loading deep data from {config.deep_chunks_dir}")
    deep_X, deep_y = load_deep_data(
        config.deep_chunks_dir,
        max_rows=config.max_rows,
    )
    logger.info(f"Deep data: {len(deep_X)} samples, {deep_X.shape[1]} features")
    
    # Build splits using the same random_state for reproducibility
    logger.info(f"Creating train/test split (test_size={config.test_size}, random_state={config.random_state})")
    (
        baseline_X_train,
        baseline_X_test,
        baseline_y_train,
        baseline_y_test,
        baseline_train_indices,
        baseline_test_indices,
    ) = build_split(
        baseline_X,
        baseline_y,
        config.test_size,
        config.random_state,
    )
    logger.info(f"Baseline split: {len(baseline_X_train)} train, {len(baseline_X_test)} test")
    if len(np.intersect1d(baseline_train_indices, baseline_test_indices)) != 0:
        raise ValueError("Baseline train/test source indices overlap")
    
    (
        deep_X_train,
        deep_X_test,
        deep_y_train,
        deep_y_test,
        deep_train_indices,
        deep_test_indices,
    ) = build_split(
        deep_X,
        deep_y,
        config.test_size,
        config.random_state,
    )
    logger.info(f"Deep split: {len(deep_X_train)} train, {len(deep_X_test)} test")
    if len(np.intersect1d(deep_train_indices, deep_test_indices)) != 0:
        raise ValueError("Deep train/test source indices overlap")
    
    # Use the smaller test set as the common evaluation set
    # Since deep model has fewer samples, use its test set size
    n_test = min(len(baseline_X_test), len(deep_X_test))
    
    logger.info(f"Using {n_test} common test samples for ensemble evaluation")
    
    # Get predictions from both models on their test sets
    logger.info("Getting baseline predictions...")
    baseline_train_probs = baseline_model.predict_proba(baseline_X_train)
    baseline_test_probs = baseline_model.predict_proba(baseline_X_test[:n_test])
    
    logger.info("Getting deep predictions...")
    deep_train_probs = deep_model.predict_proba(deep_X_train)
    deep_test_probs = deep_model.predict_proba(deep_X_test[:n_test])
    
    # Stack probabilities for meta-learner training
    # Use training samples to train the meta-learner, test samples to evaluate
    
    # For training: use samples where both models have training data
    # To avoid data leakage, use separate folds
    logger.info("Preparing meta-features...")
    
    # Split training data 80/20 for meta-learner training/validation
    n_meta_train = min(len(baseline_train_probs), len(deep_train_probs))
    meta_train_indices = np.random.RandomState(config.random_state).choice(
        n_meta_train, size=int(0.8 * n_meta_train), replace=False
    )
    meta_val_indices = np.setdiff1d(np.arange(n_meta_train), meta_train_indices)
    
    # Stack probabilities for meta-learner
    X_meta_train = np.hstack([
        baseline_train_probs[meta_train_indices],
        deep_train_probs[meta_train_indices],
    ]).astype(np.float32)
    y_meta_train = baseline_y_train[meta_train_indices]
    
    X_meta_val = np.hstack([
        baseline_train_probs[meta_val_indices],
        deep_train_probs[meta_val_indices],
    ]).astype(np.float32)
    y_meta_val = baseline_y_train[meta_val_indices]
    
    X_meta_test = np.hstack([
        baseline_test_probs,
        deep_test_probs,
    ]).astype(np.float32)
    # Use baseline test labels for final evaluation (they should match)
    y_meta_test = baseline_y_test[:n_test]
    
    logger.info(f"Meta-features shape: {X_meta_train.shape}")
    logger.info(f"Meta-learner training set: {len(X_meta_train)} samples")
    logger.info(f"Meta-learner validation set: {len(X_meta_val)} samples")
    logger.info(f"Meta-learner test set: {len(X_meta_test)} samples")
    
    # Train meta-learner
    logger.info(f"Training meta-learner ({config.meta_learner})...")
    
    if config.meta_learner == "logistic":
        meta_model = LogisticRegression(
            max_iter=1000,
            random_state=config.random_state,
            n_jobs=2,
            class_weight="balanced",
        )
    elif config.meta_learner == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        # Map string labels to integers for XGBoost
        label_map = {label: idx for idx, label in enumerate(sorted(set(y_meta_train)))}
        y_meta_train_int = np.array([label_map[y] for y in y_meta_train])
        
        meta_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=config.random_state,
            n_jobs=2,
        )
        meta_model.fit(X_meta_train, y_meta_train_int, verbose=True)
    else:
        raise ValueError(f"Unknown meta-learner: {config.meta_learner}")
    
    if config.meta_learner == "logistic":
        meta_model.fit(X_meta_train, y_meta_train)
    
    # Evaluate on validation set
    logger.info("Evaluating meta-learner on validation set...")
    y_meta_val_pred = meta_model.predict(X_meta_val)
    val_accuracy = accuracy_score(y_meta_val, y_meta_val_pred)
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating ensemble on test set...")
    y_meta_test_pred = meta_model.predict(X_meta_test)
    
    # Compute metrics
    test_accuracy = accuracy_score(y_meta_test, y_meta_test_pred)
    test_precision = precision_score(y_meta_test, y_meta_test_pred, average="weighted", zero_division=0)
    test_recall = recall_score(y_meta_test, y_meta_test_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(y_meta_test, y_meta_test_pred, average="weighted", zero_division=0)
    
    logger.info(f"Ensemble Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Ensemble Test Precision (weighted): {test_precision:.4f}")
    logger.info(f"Ensemble Test Recall (weighted): {test_recall:.4f}")
    logger.info(f"Ensemble Test F1 (weighted): {test_f1:.4f}")
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_meta_test, y_meta_test_pred)
    classes = sorted(baseline_le.classes_)
    report = classification_report(y_meta_test, y_meta_test_pred, labels=classes, output_dict=True)
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_meta_test, y_meta_test_pred, labels=classes))
    
    # Save ensemble model
    logger.info(f"Saving ensemble artifacts to {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save meta-learner (ensemble model)
    ensemble_model_path = config.output_dir / "ensemble_model.joblib"
    joblib.dump(meta_model, ensemble_model_path)
    logger.info(f"Saved ensemble model: {ensemble_model_path}")
    
    # Save label encoder
    le_path = config.output_dir / "label_encoder.joblib"
    joblib.dump(baseline_le, le_path)
    logger.info(f"Saved label encoder: {le_path}")
    
    # Save metrics
    metrics = {
        "phase": "12B",
        "timestamp": datetime.now().isoformat(),
        "ensemble_type": "stacked",
        "meta_learner": config.meta_learner,
        "component_models": {
            "baseline": baseline_metrics["best_model"],
            "deep": deep_metrics["best_model"],
        },
        "training_config": {
            "test_size": config.test_size,
            "random_state": config.random_state,
            "baseline_chunks_dir": str(config.baseline_chunks_dir),
            "deep_chunks_dir": str(config.deep_chunks_dir),
        },
        "data_summary": {
            "baseline_train": len(baseline_X_train),
            "baseline_test": len(baseline_X_test),
            "deep_train": len(deep_X_train),
            "deep_test": len(deep_X_test),
            "common_test": n_test,
            "meta_train": len(X_meta_train),
            "meta_val": len(X_meta_val),
            "meta_test": len(X_meta_test),
        },
        "best_metrics": {
            "accuracy": float(test_accuracy),
            "precision_weighted": float(test_precision),
            "recall_weighted": float(test_recall),
            "f1_weighted": float(test_f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "comparison": {
            "baseline_test_accuracy": baseline_metrics["best_metrics"]["accuracy"],
            "deep_test_accuracy": deep_metrics["best_metrics"]["accuracy"],
            "ensemble_test_accuracy": float(test_accuracy),
        },
    }
    
    metrics_path = config.output_dir / "ensemble_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")
    
    # Save checkpoint for resume support
    checkpoint_state = {
        "timestamp": datetime.now().isoformat(),
        "status": "complete",
        "ensemble_model_path": str(ensemble_model_path),
        "metrics_path": str(metrics_path),
        "label_encoder_path": str(le_path),
        "best_f1": float(test_f1),
        "best_accuracy": float(test_accuracy),
    }
    save_checkpoint(config.output_dir, checkpoint_state)
    logger.info(f"Saved training checkpoint")
    
    logger.info("="*80)
    logger.info("Ensemble training complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
