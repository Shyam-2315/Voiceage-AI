#!/usr/bin/env python3
"""
Phase 14: Final Evaluation for Fine-Tuned Wav2Vec2

Runs inference-only evaluation for the final age-group classifier and writes
lead/demo-ready metrics, confusion matrix, confusion pairs, and confidence
analysis reports.
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "wav2vec_50k" / "best"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "processed_commonvoice_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "final_evaluation"

EXPECTED_CLASSES = ["Adult", "Middle_Age", "Senior", "Teen"]
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class EvaluationData:
    frame: pd.DataFrame
    class_counts: dict[str, int]
    excluded_training_pool: dict[str, Any]


class AudioEvaluationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        audio_column: str,
        label_column: str,
        max_duration_seconds: float | None,
    ) -> None:
        self.paths = frame[audio_column].astype(str).tolist()
        self.labels = frame[label_column].astype(int).tolist()
        self.true_labels = frame["true_label"].astype(str).tolist()
        self.source_indices = frame["source_index"].astype(int).tolist()
        self.max_samples = (
            int(max_duration_seconds * TARGET_SAMPLE_RATE)
            if max_duration_seconds is not None and max_duration_seconds > 0
            else None
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        audio_path = self.paths[index]
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=TARGET_SAMPLE_RATE,
            ).astype(np.float32, copy=False)
        if self.max_samples is not None and len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        if len(audio) == 0:
            raise ValueError(f"Empty audio file: {audio_path}")

        return {
            "input_values": audio.astype(np.float32, copy=False),
            "labels": self.labels[index],
            "audio_path": audio_path,
            "true_label": self.true_labels[index],
            "source_index": self.source_indices[index],
        }


class AudioDataCollator:
    def __init__(self, feature_extractor: Any) -> None:
        self.feature_extractor = feature_extractor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.feature_extractor.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        batch["audio_path"] = [feature["audio_path"] for feature in features]
        batch["true_label"] = [feature["true_label"] for feature in features]
        batch["source_index"] = [feature["source_index"] for feature in features]
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the final fine-tuned Wav2Vec2 age-group classifier."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2500,
        help="Balanced test rows per class. Use 0 to evaluate all available rows.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audio-column", default="processed_path")
    parser.add_argument("--target-column", default="age_group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=8.0,
        help="Truncate audio for memory-safe inference. Use 0 to disable.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k-errors", type=int, default=25)
    parser.add_argument(
        "--exclude-known-training-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the reproducible Phase 13 balanced pool when metrics are available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and balanced test selection without loading model or running inference.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "evaluation.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_class < 0:
        raise ValueError("--samples-per-class must be 0 or a positive integer")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be 0 or greater")
    if args.max_duration_seconds < 0:
        raise ValueError("--max-duration-seconds must be 0 or greater")
    if args.top_k_errors < 0:
        raise ValueError("--top-k-errors must be 0 or greater")


def known_training_pool_info(args: argparse.Namespace) -> dict[str, Any] | None:
    metrics_path = args.model_path.parent / "wav2vec2_finetuned_metrics.json"
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    training_config = metrics.get("training_config", {})
    pool_samples_per_class = training_config.get("samples_per_class")
    if pool_samples_per_class is None:
        return None

    if training_config.get("audio_column", args.audio_column) != args.audio_column:
        return None
    if training_config.get("target_column", args.target_column) != args.target_column:
        return None

    return {
        "metrics_path": metrics_path,
        "samples_per_class": int(pool_samples_per_class),
        "seed": int(training_config.get("seed", args.seed)),
    }


def exclude_known_training_pool(
    df: pd.DataFrame,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    empty_info = {
        "enabled": bool(args.exclude_known_training_pool),
        "applied": False,
        "reason": None,
        "rows_excluded": 0,
        "samples_per_class": None,
        "seed": None,
        "metrics_path": None,
    }
    if not args.exclude_known_training_pool:
        empty_info["reason"] = "disabled_by_cli"
        return df, empty_info

    info = known_training_pool_info(args)
    if info is None:
        empty_info["reason"] = "no_compatible_training_metrics_found"
        logger.info("Known training-pool exclusion skipped: %s", empty_info["reason"])
        return df, empty_info

    class_counts = df[args.target_column].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
    underfilled = class_counts[class_counts < info["samples_per_class"]]
    if not underfilled.empty:
        empty_info["reason"] = "metadata_has_too_few_rows_to_reconstruct_training_pool"
        logger.info("Known training-pool exclusion skipped: %s", empty_info["reason"])
        return df, empty_info

    pool = df.groupby(args.target_column, group_keys=False).sample(
        n=info["samples_per_class"],
        random_state=info["seed"],
    )
    filtered = df.drop(index=pool.index).reset_index(drop=True)
    applied_info = {
        "enabled": True,
        "applied": True,
        "reason": None,
        "rows_excluded": int(len(pool)),
        "samples_per_class": int(info["samples_per_class"]),
        "seed": int(info["seed"]),
        "metrics_path": str(info["metrics_path"]),
    }
    logger.info(
        "Excluded known Phase 13 balanced pool: %s rows (%s per class, seed=%s).",
        applied_info["rows_excluded"],
        applied_info["samples_per_class"],
        applied_info["seed"],
    )
    return filtered, applied_info


def load_balanced_metadata(args: argparse.Namespace, logger: logging.Logger) -> EvaluationData:
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata not found: {args.metadata}")

    df = pd.read_csv(args.metadata)
    missing_columns = {args.audio_column, args.target_column} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Metadata missing columns: {', '.join(sorted(missing_columns))}")

    df = df.reset_index(names="source_index")
    df = df[[args.audio_column, args.target_column, "source_index"]].copy()
    df[args.audio_column] = df[args.audio_column].astype(str).str.strip()
    df[args.target_column] = df[args.target_column].astype(str).str.strip()
    df = df[
        df[args.audio_column].ne("")
        & df[args.audio_column].ne("nan")
        & df[args.target_column].isin(EXPECTED_CLASSES)
    ].copy()

    class_counts = df[args.target_column].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)
    missing_classes = class_counts[class_counts <= 0]
    if not missing_classes.empty:
        raise ValueError(f"Missing expected classes: {missing_classes.index.tolist()}")

    df, excluded_training_pool = exclude_known_training_pool(df, args, logger)
    class_counts = df[args.target_column].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)

    if args.samples_per_class == 0:
        sample_count = int(class_counts.min())
    else:
        sample_count = args.samples_per_class

    underfilled = class_counts[class_counts < sample_count]
    if not underfilled.empty:
        raise ValueError(
            "Not enough rows for requested --samples-per-class: "
            f"{underfilled.to_dict()}"
        )

    selected = (
        df.groupby(args.target_column, group_keys=False)
        .sample(n=sample_count, random_state=args.seed)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )
    selected["true_label"] = selected[args.target_column]
    selected["label"] = selected["true_label"].map({name: idx for idx, name in enumerate(EXPECTED_CLASSES)})

    selected_counts = {
        label: int(count)
        for label, count in selected["true_label"].value_counts().reindex(EXPECTED_CLASSES).items()
    }
    return EvaluationData(
        frame=selected,
        class_counts=selected_counts,
        excluded_training_pool=excluded_training_pool,
    )


def validate_audio_paths(frame: pd.DataFrame, audio_column: str, logger: logging.Logger) -> None:
    missing = [path for path in frame[audio_column].astype(str).tolist() if not Path(path).exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing selected audio files: {preview}")
    logger.info("Audio path check passed for %s selected rows.", len(frame))


def model_label_mapping(model: Any) -> tuple[dict[int, str], dict[str, int]]:
    raw_id2label = getattr(model.config, "id2label", None) or {}
    id2label = {int(idx): str(label) for idx, label in raw_id2label.items()}
    if not id2label:
        id2label = {idx: label for idx, label in enumerate(EXPECTED_CLASSES)}

    expected_id2label = {idx: label for idx, label in enumerate(EXPECTED_CLASSES)}
    if id2label != expected_id2label:
        raise ValueError(
            "Model label mapping does not match expected classes. "
            f"Expected {expected_id2label}, found {id2label}"
        )

    return id2label, {label: idx for idx, label in id2label.items()}


def run_inference(
    model: Any,
    feature_extractor: Any,
    data: EvaluationData,
    args: argparse.Namespace,
    device: torch.device,
    use_fp16: bool,
) -> dict[str, Any]:
    dataset = AudioEvaluationDataset(
        data.frame,
        audio_column=args.audio_column,
        label_column="label",
        max_duration_seconds=args.max_duration_seconds or None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=AudioDataCollator(feature_extractor),
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    true_confidences: list[float] = []
    audio_paths: list[str] = []
    source_indices: list[int] = []

    model.eval()
    progress = tqdm(dataloader, desc="Evaluating", unit="batch")

    autocast_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16
        else nullcontext()
    )

    with torch.inference_mode():
        for batch in progress:
            labels = batch.pop("labels")
            paths = batch.pop("audio_path")
            indices = batch.pop("source_index")
            batch.pop("true_label")

            model_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }

            with autocast_context:
                outputs = model(**model_inputs)
                logits = outputs.logits

            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            predictions = probs.argmax(axis=-1)
            batch_confidences = probs.max(axis=-1)
            label_array = labels.numpy()
            true_class_confidences = probs[np.arange(len(label_array)), label_array]

            y_true.extend(label_array.astype(int).tolist())
            y_pred.extend(predictions.astype(int).tolist())
            confidences.extend(batch_confidences.astype(float).tolist())
            true_confidences.extend(true_class_confidences.astype(float).tolist())
            audio_paths.extend(paths)
            source_indices.extend([int(index) for index in indices])

    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "confidence": np.asarray(confidences, dtype=np.float64),
        "true_class_confidence": np.asarray(true_confidences, dtype=np.float64),
        "audio_paths": audio_paths,
        "source_indices": source_indices,
    }


def top_confused_pairs(cm: np.ndarray, classes: list[str]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    row_totals = cm.sum(axis=1)
    for true_idx, true_label in enumerate(classes):
        for pred_idx, pred_label in enumerate(classes):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count <= 0:
                continue
            pairs.append(
                {
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": count,
                    "rate_within_true_class": float(count / row_totals[true_idx])
                    if row_totals[true_idx]
                    else 0.0,
                }
            )
    return sorted(pairs, key=lambda item: item["count"], reverse=True)


def summarize_values(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def confidence_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    true_class_confidence: np.ndarray,
    audio_paths: list[str],
    source_indices: list[int],
    classes: list[str],
    top_k_errors: int,
) -> dict[str, Any]:
    correct_mask = y_true == y_pred
    bins = np.asarray([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001])
    bucket_rows: list[dict[str, Any]] = []

    for start, end in zip(bins[:-1], bins[1:]):
        mask = (confidence >= start) & (confidence < end)
        bucket_rows.append(
            {
                "range": f"{start:.1f}-{min(end, 1.0):.1f}",
                "count": int(mask.sum()),
                "accuracy": float(correct_mask[mask].mean()) if mask.any() else None,
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
            }
        )

    per_true_class = {}
    per_predicted_class = {}
    for idx, label in enumerate(classes):
        true_mask = y_true == idx
        pred_mask = y_pred == idx
        per_true_class[label] = {
            "predicted_confidence": summarize_values(confidence[true_mask]),
            "true_class_confidence": summarize_values(true_class_confidence[true_mask]),
            "accuracy": float(correct_mask[true_mask].mean()) if true_mask.any() else None,
        }
        per_predicted_class[label] = {
            "predicted_confidence": summarize_values(confidence[pred_mask]),
            "precision": float(correct_mask[pred_mask].mean()) if pred_mask.any() else None,
        }

    errors = np.flatnonzero(~correct_mask)
    high_confidence_errors = sorted(errors, key=lambda idx: confidence[idx], reverse=True)[:top_k_errors]
    low_confidence_correct = sorted(
        np.flatnonzero(correct_mask),
        key=lambda idx: confidence[idx],
    )[:top_k_errors]

    def prediction_row(index: int) -> dict[str, Any]:
        return {
            "source_index": int(source_indices[index]),
            "audio_path": audio_paths[index],
            "true_label": classes[int(y_true[index])],
            "predicted_label": classes[int(y_pred[index])],
            "confidence": float(confidence[index]),
            "true_class_confidence": float(true_class_confidence[index]),
        }

    return {
        "overall_confidence": summarize_values(confidence),
        "correct_prediction_confidence": summarize_values(confidence[correct_mask]),
        "incorrect_prediction_confidence": summarize_values(confidence[~correct_mask]),
        "true_class_confidence": summarize_values(true_class_confidence),
        "confidence_buckets": bucket_rows,
        "per_true_class": per_true_class,
        "per_predicted_class": per_predicted_class,
        "high_confidence_errors": [prediction_row(int(idx)) for idx in high_confidence_errors],
        "lowest_confidence_correct_predictions": [
            prediction_row(int(idx)) for idx in low_confidence_correct
        ],
    }


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Final Wav2Vec2 Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def build_reports(
    inference: dict[str, Any],
    data: EvaluationData,
    args: argparse.Namespace,
    device: torch.device,
    use_fp16: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray]:
    y_true = inference["y_true"]
    y_pred = inference["y_pred"]
    labels = np.arange(len(EXPECTED_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=EXPECTED_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    confused_pairs = top_confused_pairs(cm, EXPECTED_CLASSES)
    confidence = confidence_analysis(
        y_true=y_true,
        y_pred=y_pred,
        confidence=inference["confidence"],
        true_class_confidence=inference["true_class_confidence"],
        audio_paths=inference["audio_paths"],
        source_indices=inference["source_indices"],
        classes=EXPECTED_CLASSES,
        top_k_errors=args.top_k_errors,
    )

    final_metrics = {
        "phase": "14",
        "timestamp": datetime.now().isoformat(),
        "task": "final_wav2vec2_age_group_evaluation",
        "model_path": args.model_path,
        "metadata": args.metadata,
        "output_dir": args.output_dir,
        "classes": EXPECTED_CLASSES,
        "test_set": {
            "balanced": True,
            "rows": int(len(y_true)),
            "samples_per_class": data.class_counts,
            "seed": args.seed,
            "audio_column": args.audio_column,
            "target_column": args.target_column,
            "max_duration_seconds": args.max_duration_seconds,
            "excluded_known_training_pool": data.excluded_training_pool,
        },
        "hardware": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "fp16": use_fp16,
            "batch_size": args.batch_size,
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top_confused_class_pairs": confused_pairs,
        "confusion_matrix": cm.tolist(),
    }
    return final_metrics, report, confidence, cm


def main() -> int:
    args = parse_args()
    validate_args(args)
    logger = configure_logging(args.output_dir)

    logger.info("=" * 64)
    logger.info("PHASE 14: Final Wav2Vec2 Evaluation")
    logger.info("=" * 64)
    logger.info("Model path: %s", args.model_path)
    logger.info("Metadata: %s", args.metadata)
    logger.info("Output dir: %s", args.output_dir)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"Missing model config: {args.model_path / 'config.json'}")

    data = load_balanced_metadata(args, logger)
    logger.info("Expected classes: %s", EXPECTED_CLASSES)
    logger.info("Balanced test rows: %s", len(data.frame))
    logger.info("Balanced class counts: %s", data.class_counts)
    validate_audio_paths(data.frame, args.audio_column, logger)

    if args.dry_run:
        logger.info("Dry run complete; model loading and inference skipped.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    logger.info("Using device: %s", device)
    logger.info("Using fp16 on GPU: %s", use_fp16)
    logger.info(
        "Memory-conscious settings: batch_size=%s max_duration_seconds=%s num_workers=%s",
        args.batch_size,
        args.max_duration_seconds,
        args.num_workers,
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()

    logger.info("Loading feature extractor and model.")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_path)
    model = AutoModelForAudioClassification.from_pretrained(args.model_path)
    model_label_mapping(model)
    model.to(device)

    inference = run_inference(
        model=model,
        feature_extractor=feature_extractor,
        data=data,
        args=args,
        device=device,
        use_fp16=use_fp16,
    )
    final_metrics, report, confidence, cm = build_reports(
        inference=inference,
        data=data,
        args=args,
        device=device,
        use_fp16=use_fp16,
    )

    write_json(args.output_dir / "final_metrics.json", final_metrics)
    write_json(args.output_dir / "classification_report.json", report)
    write_json(args.output_dir / "confidence_analysis.json", confidence)
    plot_confusion_matrix(cm, EXPECTED_CLASSES, args.output_dir / "confusion_matrix.png")

    logger.info("Accuracy: %.4f", final_metrics["accuracy"])
    logger.info("Weighted F1: %.4f", final_metrics["f1_weighted"])
    logger.info("Saved final_metrics.json")
    logger.info("Saved classification_report.json")
    logger.info("Saved confidence_analysis.json")
    logger.info("Saved confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
