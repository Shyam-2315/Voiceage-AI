#!/usr/bin/env python3
"""
Phase 13: Wav2Vec2 Fine-Tuning

Fine-tunes a HuggingFace Wav2Vec2 sequence classifier directly on processed
Common Voice WAV files for 4-class age-group classification.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
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
from torch.utils.data import Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "processed_commonvoice_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "wav2vec2_finetuned"
DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"
TARGET_COLUMN = "age_group"
TARGET_SAMPLE_RATE = 16_000
EXPECTED_CLASS_COUNT = 4
METRICS_FILENAME = "wav2vec2_finetuned_metrics.json"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    eval: pd.DataFrame
    label_encoder: LabelEncoder
    class_counts: dict[str, int]


class AudioClassificationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        audio_column: str,
        label_column: str,
        target_sample_rate: int,
        max_duration_seconds: float | None,
    ) -> None:
        self.paths = frame[audio_column].astype(str).tolist()
        self.labels = frame[label_column].astype(int).tolist()
        self.target_sample_rate = target_sample_rate
        self.max_samples = (
            int(max_duration_seconds * target_sample_rate)
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
        if sample_rate != self.target_sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.target_sample_rate,
            ).astype(np.float32, copy=False)
        if self.max_samples is not None and len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        if len(audio) == 0:
            raise ValueError(f"Empty audio file: {audio_path}")

        return {
            "input_values": audio.astype(np.float32, copy=False),
            "labels": self.labels[index],
        }


class AudioDataCollator:
    def __init__(self, feature_extractor: Any) -> None:
        self.feature_extractor = feature_extractor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        inputs = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.feature_extractor.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 13 Wav2Vec2 fine-tuning for 4-class age-group classification."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--audio-column", default="processed_path")
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Optional balanced sample count per class before the train/eval split.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap after class balancing and shuffling for smoke runs.",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=8.0,
        help="Truncate audio to this length for RTX 4050 memory compatibility. Use 0 to disable.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use mixed precision. Defaults to enabled on CUDA and disabled otherwise.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce activation memory during fine-tuning.",
    )
    parser.add_argument(
        "--freeze-feature-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze Wav2Vec2 convolutional feature encoder.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a checkpoint path, or from the latest output-dir checkpoint.",
    )
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument(
        "--audio-check-limit",
        type=int,
        default=2_000,
        help="Number of selected audio paths to existence-check before training. Use 0 to skip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata, labels, split, and training config without model training.",
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


def validate_args(args: argparse.Namespace) -> None:
    if args.test_size <= 0 or args.test_size >= 1:
        raise ValueError("--test-size must be between 0 and 1")
    if args.samples_per_class is not None and args.samples_per_class < 2:
        raise ValueError("--samples-per-class must be at least 2")
    if args.max_rows is not None and args.max_rows < EXPECTED_CLASS_COUNT * 2:
        raise ValueError("--max-rows must leave at least 2 rows per class")
    if args.per_device_train_batch_size < 1 or args.per_device_eval_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.eval_steps < 1 or args.save_steps < 1:
        raise ValueError("--eval-steps and --save-steps must be positive")
    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save-steps must be a multiple of --eval-steps")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be positive")


def load_metadata(args: argparse.Namespace) -> pd.DataFrame:
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    df = pd.read_csv(args.metadata)
    missing_columns = {args.audio_column, args.target_column} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Metadata missing columns: {', '.join(sorted(missing_columns))}")

    df = df[[args.audio_column, args.target_column]].copy()
    df[args.audio_column] = df[args.audio_column].astype(str).str.strip()
    df[args.target_column] = df[args.target_column].astype(str).str.strip()
    df = df[
        df[args.audio_column].ne("")
        & df[args.audio_column].ne("nan")
        & df[args.target_column].ne("")
        & df[args.target_column].ne("nan")
    ].copy()

    if df.empty:
        raise ValueError("No valid metadata rows found")

    class_counts = df[args.target_column].value_counts().sort_index()
    if len(class_counts) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} age-group classes, found {len(class_counts)}: "
            f"{class_counts.to_dict()}"
        )

    if args.samples_per_class is not None:
        underfilled = class_counts[class_counts < args.samples_per_class]
        if not underfilled.empty:
            raise ValueError(
                "Not enough rows for requested --samples-per-class: "
                f"{underfilled.to_dict()}"
            )
        df = (
            df.groupby(args.target_column, group_keys=False)
            .sample(n=args.samples_per_class, random_state=args.seed)
            .reset_index(drop=True)
        )

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    limited_counts = df[args.target_column].value_counts().sort_index()
    if len(limited_counts) != EXPECTED_CLASS_COUNT or limited_counts.min() < 2:
        raise ValueError(
            "Selected rows must contain all 4 classes with at least 2 rows each. "
            f"Class counts: {limited_counts.to_dict()}"
        )

    return df.reset_index(drop=True)


def validate_audio_paths(frame: pd.DataFrame, audio_column: str, limit: int, logger: logging.Logger) -> None:
    if limit <= 0:
        logger.info("Audio path existence check skipped.")
        return

    paths = frame[audio_column].head(limit).astype(str).tolist()
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing audio files in selected rows: {preview}")
    logger.info("Audio path check passed for %s selected rows.", len(paths))


def build_split(frame: pd.DataFrame, args: argparse.Namespace) -> SplitData:
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(frame[args.target_column]).astype(np.int64)
    frame = frame.copy()
    frame["label"] = labels

    train_idx, eval_idx = train_test_split(
        np.arange(len(frame)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    train = frame.iloc[train_idx].reset_index(drop=True)
    eval_frame = frame.iloc[eval_idx].reset_index(drop=True)
    class_counts = {
        str(label): int(count)
        for label, count in frame[args.target_column].value_counts().sort_index().items()
    }
    return SplitData(
        train=train,
        eval=eval_frame,
        label_encoder=label_encoder,
        class_counts=class_counts,
    )


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir():
            try:
                checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
            except (IndexError, ValueError):
                continue
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_resume_checkpoint(args: argparse.Namespace) -> str | None:
    if args.resume_from_checkpoint is None:
        return None
    if args.resume_from_checkpoint != "latest":
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    checkpoint_path = latest_checkpoint(args.output_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint-* directories found in {args.output_dir}")
    return str(checkpoint_path)


def logits_from_predictions(predictions: Any) -> np.ndarray:
    return predictions[0] if isinstance(predictions, tuple) else predictions


def prediction_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits = logits_from_predictions(eval_pred.predictions)
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_weighted": float(
            precision_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def detailed_metrics(logits: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict[str, Any]:
    logits = logits_from_predictions(logits)
    predictions = np.argmax(logits, axis=-1)
    label_ids = np.arange(len(classes))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_weighted": float(
            precision_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=label_ids).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_ids,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
    }


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


def create_training_args(args: argparse.Namespace, use_fp16: bool) -> TrainingArguments:
    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "save_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_weighted",
        "greater_is_better": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.num_train_epochs,
        "fp16": use_fp16,
        "dataloader_num_workers": args.dataloader_num_workers,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "none",
        "seed": args.seed,
    }

    strategy_arg = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_arg] = "steps"
    return TrainingArguments(**training_kwargs)


def save_metrics(
    args: argparse.Namespace,
    split_data: SplitData,
    trainer: Trainer,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    use_fp16: bool,
) -> None:
    classes = [str(label) for label in split_data.label_encoder.classes_]
    payload = {
        "phase": "13",
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "task": "wav2vec2_fine_tuning_age_group_classification",
        "class_count": len(classes),
        "classes": classes,
        "selection_metric": "eval_f1_weighted",
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "class_distribution": split_data.class_counts,
        "rows": {
            "train": int(len(split_data.train)),
            "eval": int(len(split_data.eval)),
            "total": int(len(split_data.train) + len(split_data.eval)),
        },
        "training_config": {
            "metadata": str(args.metadata),
            "audio_column": args.audio_column,
            "target_column": args.target_column,
            "test_size": args.test_size,
            "seed": args.seed,
            "samples_per_class": args.samples_per_class,
            "max_rows": args.max_rows,
            "max_duration_seconds": args.max_duration_seconds,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "early_stopping_patience": args.early_stopping_patience,
            "fp16": use_fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "freeze_feature_encoder": args.freeze_feature_encoder,
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train_metrics": to_jsonable(train_metrics),
        "eval_metrics": to_jsonable(eval_metrics),
        "test_metrics": to_jsonable(test_metrics),
    }

    metrics_path = args.output_dir / METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    args = parse_args()
    validate_args(args)
    logger = configure_logging(args.output_dir)
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("PHASE 13: Wav2Vec2 Fine-Tuning")
    logger.info("=" * 60)

    frame = load_metadata(args)
    split_data = build_split(frame, args)
    validate_audio_paths(frame, args.audio_column, args.audio_check_limit, logger)

    classes = [str(label) for label in split_data.label_encoder.classes_]
    logger.info("Classes: %s", classes)
    logger.info("Selected class distribution: %s", split_data.class_counts)
    logger.info("Train rows: %s", len(split_data.train))
    logger.info("Eval rows: %s", len(split_data.eval))
    logger.info("Model: %s", args.model_name)
    logger.info(
        "RTX-compatible settings: train_batch=%s eval_batch=%s grad_accum=%s max_duration=%s",
        args.per_device_train_batch_size,
        args.per_device_eval_batch_size,
        args.gradient_accumulation_steps,
        args.max_duration_seconds,
    )

    use_fp16 = args.fp16 if args.fp16 is not None else torch.cuda.is_available()
    if use_fp16 and not torch.cuda.is_available():
        raise ValueError("fp16 mixed precision requires CUDA. Re-run with --no-fp16 on CPU.")
    logger.info("Mixed precision fp16: %s", use_fp16)

    if args.dry_run:
        create_training_args(args, use_fp16=use_fp16)
        resolve_resume_checkpoint(args)
        logger.info("HuggingFace TrainingArguments validation passed.")
        logger.info("Dry run complete; no HuggingFace model loaded and no training performed.")
        return 0

    id2label = {idx: label for idx, label in enumerate(classes)}
    label2id = {label: idx for idx, label in id2label.items()}

    logger.info("Loading feature extractor and model.")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name,
        num_labels=len(classes),
        label2id=label2id,
        id2label=id2label,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    if args.freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        logger.info("Frozen Wav2Vec2 feature encoder.")

    train_dataset = AudioClassificationDataset(
        split_data.train,
        audio_column=args.audio_column,
        label_column="label",
        target_sample_rate=TARGET_SAMPLE_RATE,
        max_duration_seconds=args.max_duration_seconds,
    )
    eval_dataset = AudioClassificationDataset(
        split_data.eval,
        audio_column=args.audio_column,
        label_column="label",
        target_sample_rate=TARGET_SAMPLE_RATE,
        max_duration_seconds=args.max_duration_seconds,
    )

    training_args = create_training_args(args, use_fp16=use_fp16)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
        data_collator=AudioDataCollator(feature_extractor),
        compute_metrics=prediction_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    resume_checkpoint = resolve_resume_checkpoint(args)
    if resume_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    train_metrics = {key: float(value) for key, value in train_result.metrics.items()}
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    prediction_output = trainer.predict(eval_dataset)
    test_metrics = detailed_metrics(
        logits=prediction_output.predictions,
        labels=prediction_output.label_ids,
        classes=classes,
    )

    best_dir = args.output_dir / "best"
    trainer.save_model(str(best_dir))
    feature_extractor.save_pretrained(best_dir)
    joblib.dump(split_data.label_encoder, args.output_dir / LABEL_ENCODER_FILENAME)
    save_metrics(
        args=args,
        split_data=split_data,
        trainer=trainer,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        test_metrics=test_metrics,
        use_fp16=use_fp16,
    )

    logger.info("Best checkpoint: %s", trainer.state.best_model_checkpoint)
    logger.info("Saved best model to: %s", best_dir)
    logger.info("Weighted F1: %.4f", test_metrics["f1_weighted"])
    logger.info("=" * 60)
    logger.info("PHASE 13 COMPLETE")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line script should log fatal context.
        logging.getLogger(__name__).exception("Fine-tuning failed: %s", exc)
        raise SystemExit(1)
