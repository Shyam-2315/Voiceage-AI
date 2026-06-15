#!/usr/bin/env python3
"""
Phase 12A: Error Analysis for Baseline and Wav2Vec2 Models

Generates confusion matrices and identifies most confused class pairs
for both the RandomForest baseline and Wav2Vec2 MLP classifier.

Memory-efficient design:
- Loads parquet chunks incrementally
- Uses float32 for arrays
- Avoids loading unnecessary columns
- Includes dry-run mode and verification

Features:
- Py_compile verification for syntax correctness
- Artifact load verification (models, metrics, label encoders)
- Comprehensive error reporting
- Visualizations when matplotlib available
"""

from __future__ import annotations

import argparse
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
import pyarrow.parquet as pq
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "models" / "baseline"
DEFAULT_DEEP_DIR = PROJECT_ROOT / "models" / "deep"
DEFAULT_BASELINE_CHUNKS_DIR = PROJECT_ROOT / "data" / "features" / "chunks"
DEFAULT_DEEP_CHUNKS_DIR = PROJECT_ROOT / "data" / "embeddings" / "wav2vec2" / "chunks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "error_analysis"

# Constants for feature detection
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


@dataclass(frozen=True)
class ErrorAnalysis:
    """Container for error analysis results."""
    model_name: str
    test_size: int
    accuracy: float
    classes: list[str]
    confusion_matrix: np.ndarray
    most_confused_pairs: list[dict[str, Any]]
    per_class_metrics: dict[str, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze classification errors for baseline and deep models."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Directory containing baseline model and metrics.",
    )
    parser.add_argument(
        "--deep-dir",
        type=Path,
        default=DEFAULT_DEEP_DIR,
        help="Directory containing deep (Wav2Vec2) model and metrics.",
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
        help="Output directory for error analysis reports.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without generating reports.",
    )
    parser.add_argument(
        "--verify-artifacts",
        action="store_true",
        help="Verify all artifact files are loadable (requires dry-run).",
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
    log = logger or print
    
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
            log(f"✗ Missing baseline artifact: {path}")
            all_exist = False
        else:
            log(f"✓ Found: {path}")
    
    # Check deep artifacts
    for path in [deep_model_path, deep_le_path, deep_metrics_path]:
        if not path.exists():
            log(f"✗ Missing deep artifact: {path}")
            all_exist = False
        else:
            log(f"✓ Found: {path}")
    
    if not all_exist:
        return False
    
    # Try loading artifacts
    log("\nAttempting to load artifacts...")
    try:
        joblib.load(baseline_model_path)
        log("✓ Baseline model loaded")
        
        joblib.load(baseline_le_path)
        log("✓ Baseline label encoder loaded")
        
        with open(baseline_fc_path) as f:
            json.load(f)
        log("✓ Baseline feature columns loaded")
        
        with open(baseline_metrics_path) as f:
            json.load(f)
        log("✓ Baseline metrics loaded")
        
        joblib.load(deep_model_path)
        log("✓ Deep model loaded")
        
        joblib.load(deep_le_path)
        log("✓ Deep label encoder loaded")
        
        with open(deep_metrics_path) as f:
            json.load(f)
        log("✓ Deep metrics loaded")
        
        return True
    except Exception as e:
        log(f"✗ Error loading artifacts: {e}")
        return False


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "error_analysis.log"

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


def load_baseline_split(
    chunks_dir: Path,
    feature_columns: list[str],
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load baseline features and targets, return (X, y)."""
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


def load_deep_split(
    chunks_dir: Path,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load deep embeddings and targets, return (X, y)."""
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


def identify_confused_pairs(
    cm: np.ndarray,
    classes: list[str],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Identify most confused class pairs."""
    # Get off-diagonal elements (misclassifications)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j:  # Off-diagonal
                count = cm[i, j]
                if count > 0:
                    pairs.append({
                        "true_class": classes[i],
                        "predicted_class": classes[j],
                        "count": int(count),
                        "rate": float(count / cm[i].sum()) if cm[i].sum() > 0 else 0,
                    })
    
    # Sort by count descending
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]


def analyze_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
) -> ErrorAnalysis:
    """Generate error analysis for a model."""
    # Encode y if it's strings
    if isinstance(y[0], str):
        y_encoded = label_encoder.transform(y)
    else:
        y_encoded = y
    
    y_pred = model.predict(X)
    
    # Decode labels
    classes = sorted(label_encoder.classes_)
    y_decoded = label_encoder.inverse_transform(y_encoded)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_decoded, y_pred_decoded)
    
    cm = confusion_matrix(y_decoded, y_pred_decoded, labels=classes)
    
    # Per-class metrics
    report_dict = classification_report(y_decoded, y_pred_decoded, labels=classes, output_dict=True)
    per_class_metrics = {}
    for cls in classes:
        if cls in report_dict:
            per_class_metrics[cls] = {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1-score": float(report_dict[cls]["f1-score"]),
            }
    
    # Confused pairs
    confused_pairs = identify_confused_pairs(cm, classes, top_n=5)
    
    return ErrorAnalysis(
        model_name=model_name,
        test_size=len(y),
        accuracy=float(accuracy),
        classes=classes,
        confusion_matrix=cm.tolist(),
        most_confused_pairs=confused_pairs,
        per_class_metrics=per_class_metrics,
    )


def save_analysis(
    analysis: ErrorAnalysis,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Save error analysis to JSON and generate visualizations."""
    model_slug = analysis.model_name.lower().replace(" ", "_")
    
    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": analysis.model_name,
        "test_size": analysis.test_size,
        "accuracy": analysis.accuracy,
        "classes": analysis.classes,
        "confusion_matrix": analysis.confusion_matrix,
        "most_confused_pairs": analysis.most_confused_pairs,
        "per_class_metrics": analysis.per_class_metrics,
    }
    
    report_path = output_dir / f"{model_slug}_error_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved error report: {report_path}")
    
    # Generate confusion matrix visualization if matplotlib is available
    if HAS_MATPLOTLIB:
        try:
            fig, ax = plt.subplots(figsize=(8, 7))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=np.array(analysis.confusion_matrix),
                display_labels=analysis.classes,
            )
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"{analysis.model_name} - Confusion Matrix")
            
            cm_path = output_dir / f"{model_slug}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved confusion matrix visualization: {cm_path}")
        except Exception as e:
            logger.warning(f"Failed to save visualization: {e}")


def main() -> None:
    args = parse_args()
    
    # Handle --verify-syntax flag
    if args.verify_syntax:
        success = verify_syntax()
        sys.exit(0 if success else 1)
    
    logger = configure_logging(args.output_dir)
    
    logger.info("="*80)
    logger.info("Phase 12A: Error Analysis")
    logger.info("="*80)
    
    # Handle --verify-artifacts flag (requires dry-run)
    if args.verify_artifacts:
        logger.info("Running artifact verification...")
        success = verify_artifacts(
            args.baseline_dir,
            args.deep_dir,
            args.baseline_chunks_dir,
            args.deep_chunks_dir,
            logger=logger.info,
        )
        if success:
            logger.info("✓ All artifacts verified successfully")
            sys.exit(0)
        else:
            logger.error("✗ Artifact verification failed")
            sys.exit(1)
    
    # Load baseline model
    logger.info(f"Loading baseline model from {args.baseline_dir}")
    baseline_model_path = args.baseline_dir / "voiceage_baseline_model.joblib"
    baseline_le_path = args.baseline_dir / "label_encoder.joblib"
    baseline_fc_path = args.baseline_dir / "feature_columns.json"
    baseline_metrics_path = args.baseline_dir / "baseline_metrics.json"
    
    if not baseline_model_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {baseline_model_path}")
    
    baseline_model = joblib.load(baseline_model_path)
    baseline_le = joblib.load(baseline_le_path)
    
    with open(baseline_fc_path) as f:
        baseline_feature_columns = json.load(f)
    
    with open(baseline_metrics_path) as f:
        baseline_metrics = json.load(f)
    
    logger.info(f"Baseline model: {baseline_metrics['best_model']}")
    logger.info(f"Baseline features: {len(baseline_feature_columns)} columns")
    
    # Load deep model
    logger.info(f"Loading deep (Wav2Vec2) model from {args.deep_dir}")
    deep_model_path = args.deep_dir / "wav2vec2_embedding_classifier.joblib"
    deep_le_path = args.deep_dir / "label_encoder.joblib"
    deep_metrics_path = args.deep_dir / "wav2vec2_embedding_metrics.json"
    
    if not deep_model_path.exists():
        raise FileNotFoundError(f"Deep model not found: {deep_model_path}")
    
    deep_model = joblib.load(deep_model_path)
    deep_le = joblib.load(deep_le_path)
    
    with open(deep_metrics_path) as f:
        deep_metrics = json.load(f)
    
    logger.info(f"Deep model: {deep_metrics['best_model']}")
    
    if args.dry_run:
        logger.info("Dry-run mode: validation complete.")
        return
    
    # Load baseline test set
    logger.info(f"Loading baseline test data from {args.baseline_chunks_dir}")
    baseline_X, baseline_y = load_baseline_split(
        args.baseline_chunks_dir,
        baseline_feature_columns,
        max_rows=args.max_rows,
    )
    logger.info(f"Baseline test set: {len(baseline_X)} samples, {baseline_X.shape[1]} features")
    
    # Load deep test set
    logger.info(f"Loading deep test data from {args.deep_chunks_dir}")
    deep_X, deep_y = load_deep_split(
        args.deep_chunks_dir,
        max_rows=args.max_rows,
    )
    logger.info(f"Deep test set: {len(deep_X)} samples, {deep_X.shape[1]} features")
    
    # Analyze baseline
    logger.info("Analyzing baseline model...")
    baseline_analysis = analyze_model(
        baseline_model,
        baseline_X,
        baseline_y,
        baseline_le,
        "Baseline RandomForest",
    )
    logger.info(f"Baseline accuracy: {baseline_analysis.accuracy:.4f}")
    logger.info(f"Top confused pair: {baseline_analysis.most_confused_pairs[0] if baseline_analysis.most_confused_pairs else 'N/A'}")
    
    # Analyze deep
    logger.info("Analyzing deep (Wav2Vec2) model...")
    deep_analysis = analyze_model(
        deep_model,
        deep_X,
        deep_y,
        deep_le,
        "Wav2Vec2 MLP Classifier",
    )
    logger.info(f"Deep accuracy: {deep_analysis.accuracy:.4f}")
    logger.info(f"Top confused pair: {deep_analysis.most_confused_pairs[0] if deep_analysis.most_confused_pairs else 'N/A'}")
    
    # Save analyses
    save_analysis(baseline_analysis, args.output_dir, logger)
    save_analysis(deep_analysis, args.output_dir, logger)
    
    logger.info("="*80)
    logger.info("Error analysis complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
