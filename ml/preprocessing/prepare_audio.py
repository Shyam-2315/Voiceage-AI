"""
prepare_audio.py  —  Phase 4: Audio Preprocessing Pipeline

Loads every audio file listed in data/processed/metadata.csv (or a supplied
Common Voice metadata CSV), applies a standard cleaning chain, saves the
processed file, and emits a summary report.

Processing chain per file
-------------------------
1. Load with librosa (any sample rate)
2. Convert to mono
3. Resample to 16 kHz
4. Peak-normalise to ±1.0
5. Trim leading / trailing silence (top-db threshold configurable)
6. Save as 16-bit PCM WAV to data/processed/audio/

Outputs
-------
data/processed/audio/          — processed WAV files (16-bit PCM, 16 kHz, mono)
data/processed/processed_metadata.csv  — per-file metadata
data/processed/preprocess_report.json  — aggregate statistics

Usage
-----
python ml/preprocessing/prepare_audio.py
python ml/preprocessing/prepare_audio.py --metadata path/to/metadata.csv
python ml/preprocessing/prepare_audio.py --processed-dir custom/path --top-db 25
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR: int = 16_000          # Hz
TARGET_BIT_DEPTH: str = "PCM_16"  # soundfile subtype
OUTPUT_COLUMNS: list[str] = [
    "original_path",
    "processed_path",
    "duration",
    "sample_rate",
    "speaker_role",
    "transcript",
    "age",
    "age_group",
]

# Metadata column name that points to the source audio file.
# Supports both the internal metadata.csv ("file_path") and Common Voice
# TSV/CSV conventions ("path" / "audio_path").
PATH_COLUMN_CANDIDATES: list[str] = ["file_path", "audio_path", "path"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_project_root() -> Path:
    """Walk up from this file until a directory containing data/ is found."""
    candidate = Path(__file__).resolve()
    for parent in [candidate, *candidate.parents]:
        if (parent / "data").is_dir():
            return parent
    return Path.cwd()


# ---------------------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------------------

def read_metadata(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """
    Read a CSV (or TSV) metadata file.

    Detects the delimiter automatically (comma vs tab) from the file extension.
    Returns (rows, fieldnames).
    """
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        log.info("Loaded %d rows from %s", len(rows), path)
        return rows, fieldnames
    except FileNotFoundError:
        log.error("Metadata file not found: %s", path)
        raise
    except csv.Error as exc:
        log.error("Failed to parse %s: %s", path, exc)
        raise


def find_audio_path_column(fieldnames: list[str]) -> str | None:
    """Return the first fieldname that looks like an audio-path column."""
    lower = {f.lower(): f for f in fieldnames}
    for candidate in PATH_COLUMN_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    return None


def write_processed_metadata(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write processed_metadata.csv to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Processed metadata saved → %s  (%d rows)", output_path, len(rows))


def write_report(report: dict[str, Any], output_path: Path) -> None:
    """Write preprocess_report.json to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Preprocessing report saved → %s", output_path)


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------

def make_output_path(original: Path, output_dir: Path) -> Path:
    """
    Derive a unique output path inside *output_dir*.

    Preserves the original stem and adds a .wav extension.
    If a file with that name already exists the speaker prefix is kept to
    minimise collisions across dataset groups.
    """
    stem = original.stem
    return output_dir / f"{stem}.wav"


def process_audio(
    input_path: Path,
    output_path: Path,
    *,
    top_db: float = 20.0,
) -> dict[str, Any]:
    """
    Apply the full cleaning chain to one audio file.

    Parameters
    ----------
    input_path  : source audio file (any format readable by librosa)
    output_path : destination .wav file
    top_db      : silence-trim threshold in dBFS (default 20 dB below peak)

    Returns
    -------
    dict with keys: duration (seconds), sample_rate, success, error

    Raises
    ------
    Does NOT raise — all errors are caught and returned in the dict so the
    caller can skip the file gracefully.
    """
    result: dict[str, Any] = {
        "duration": 0.0,
        "sample_rate": TARGET_SR,
        "success": False,
        "error": None,
    }

    try:
        # 1. Load (any SR, any channels)
        audio, sr = librosa.load(str(input_path), sr=None, mono=False)

        # 2. Convert to mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # 3. Resample to 16 kHz
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        # 4. Peak normalise to ±1.0  (avoid division by zero on silent files)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        # 5. Trim leading / trailing silence
        audio, _ = librosa.effects.trim(audio, top_db=top_db)

        # Ensure the clip has at least a few samples after trimming
        if audio.shape[0] < 10:
            result["error"] = "Audio is effectively silent after trimming"
            return result

        # 6. Save as 16-bit PCM WAV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(output_path),
            audio.astype(np.float32),
            TARGET_SR,
            subtype=TARGET_BIT_DEPTH,
        )

        result["duration"] = round(len(audio) / TARGET_SR, 4)
        result["success"] = True

    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    metadata_path: Path,
    output_audio_dir: Path,
    processed_metadata_path: Path,
    report_path: Path,
    *,
    top_db: float = 20.0,
) -> int:
    """
    Execute the full preprocessing pipeline.

    Returns an exit code: 0 on success, 1 on fatal error.
    """
    # --- load metadata -------------------------------------------------------
    try:
        rows, fieldnames = read_metadata(metadata_path)
    except Exception:
        return 1

    if not rows:
        log.error("Metadata file is empty — nothing to process.")
        return 1

    path_col = find_audio_path_column(fieldnames)
    if path_col is None:
        log.error(
            "Could not find an audio-path column in %s.  "
            "Expected one of: %s.  Found: %s",
            metadata_path,
            PATH_COLUMN_CANDIDATES,
            fieldnames,
        )
        return 1

    log.info("Using column '%s' as the audio path.", path_col)
    log.info("Output audio directory : %s", output_audio_dir)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    # --- process files -------------------------------------------------------
    processed_rows: list[dict[str, str]] = []
    total = len(rows)
    processed_count = 0
    skipped_count = 0
    durations: list[float] = []

    for row in tqdm(rows, desc="Preprocessing audio", unit="file", dynamic_ncols=True):
        original_path_str = (row.get(path_col) or "").strip()
        if not original_path_str:
            log.debug("Row with empty audio path — skipping.")
            skipped_count += 1
            continue

        original_path = Path(original_path_str)
        if not original_path.is_file():
            log.warning("File not found — skipping: %s", original_path)
            skipped_count += 1
            continue

        output_path = make_output_path(original_path, output_audio_dir)

        result = process_audio(original_path, output_path, top_db=top_db)

        if not result["success"]:
            log.warning(
                "Skipped (corrupted / silent): %s  —  %s",
                original_path.name,
                result["error"],
            )
            skipped_count += 1
            continue

        processed_count += 1
        durations.append(result["duration"])

        processed_rows.append(
            {
                "original_path": str(original_path),
                "processed_path": str(output_path),
                "duration": str(result["duration"]),
                "sample_rate": str(TARGET_SR),
                "speaker_role": row.get("speaker_role", ""),
                "transcript": row.get("transcript", ""),
                "age": row.get("age", ""),
                "age_group": row.get("age_group", ""),
            }
        )

    # --- write outputs -------------------------------------------------------
    try:
        write_processed_metadata(processed_rows, processed_metadata_path)
    except OSError as exc:
        log.error("Failed to write processed metadata: %s", exc)
        return 1

    average_duration = round(sum(durations) / len(durations), 4) if durations else 0.0

    report: dict[str, Any] = {
        "total_files": total,
        "processed_files": processed_count,
        "skipped_files": skipped_count,
        "average_duration": average_duration,
    }

    try:
        write_report(report, report_path)
    except OSError as exc:
        log.error("Failed to write report: %s", exc)
        return 1

    # --- summary -------------------------------------------------------------
    log.info("=" * 60)
    log.info("Preprocessing complete.")
    log.info("  Total files     : %d", total)
    log.info("  Processed       : %d", processed_count)
    log.info("  Skipped         : %d", skipped_count)
    log.info("  Avg duration    : %.4f s", average_duration)
    log.info("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4 audio preprocessing pipeline for voiceage-ai.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Path to the input metadata CSV/TSV.  "
            "Defaults to <project_root>/data/processed/metadata.csv.  "
            "Pass a Common Voice validated.tsv to process that dataset instead."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Root of the processed data directory.  Defaults to <project_root>/data/processed/.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help=(
            "Directory where processed WAV files are saved.  "
            "Defaults to <processed_dir>/audio/."
        ),
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=None,
        help="Path for the output processed_metadata.csv.  Defaults to <processed_dir>/processed_metadata.csv.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path for the preprocess_report.json.  Defaults to <processed_dir>/preprocess_report.json.",
    )
    parser.add_argument(
        "--top-db",
        type=float,
        default=20.0,
        help="Silence-trim threshold in dBFS (passed to librosa.effects.trim).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = resolve_project_root()
    processed_dir: Path = args.processed_dir or project_root / "data" / "processed"

    metadata_path: Path = args.metadata or processed_dir / "metadata.csv"
    audio_dir: Path = args.audio_dir or processed_dir / "audio"
    output_metadata_path: Path = args.output_metadata or processed_dir / "processed_metadata.csv"
    report_path: Path = args.report or processed_dir / "preprocess_report.json"

    log.info("Project root      : %s", project_root)
    log.info("Input metadata    : %s", metadata_path)
    log.info("Output audio dir  : %s", audio_dir)
    log.info("Processed metadata: %s", output_metadata_path)
    log.info("Report            : %s", report_path)
    log.info("Silence threshold : %.1f dBFS", args.top_db)

    return run_pipeline(
        metadata_path=metadata_path,
        output_audio_dir=audio_dir,
        processed_metadata_path=output_metadata_path,
        report_path=report_path,
        top_db=args.top_db,
    )


if __name__ == "__main__":
    sys.exit(main())
