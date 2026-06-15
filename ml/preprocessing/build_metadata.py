"""
Build Phase 3 metadata mapping for voiceage-ai.

Inputs:
    data/processed/audio_inventory.csv
    data/processed/conversation_audio.csv
    data/processed/conversation_text.csv

Output:
    data/processed/metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


OUTPUT_COLUMNS = [
    "file_path",
    "file_name",
    "dataset_group",
    "conversation_id",
    "speaker_role",
    "audio_index",
    "transcript",
    "age",
    "age_group",
]

FILENAME_PATTERN = re.compile(
    r"^(?P<conversation_id>\d+)_(?P<speaker_role>client|speaker|therapist)_audio"
    r"(?:_(?P<audio_index>\d+))?\.[^.]+$",
    re.IGNORECASE,
)

ROLE_BY_GROUP = {
    "audio_client": "client",
    "audio_speaker": "speaker",
    "audio_therapist": "therapist",
}

AGE_COLUMNS = ("age", "client_age", "speaker_age", "therapist_age", "participant_age")


def resolve_project_root() -> Path:
    """Walk up from this file until a project root containing data/ is found."""
    candidate = Path(__file__).resolve()
    for parent in [candidate, *candidate.parents]:
        if (parent / "data").is_dir():
            return parent
    return Path.cwd()


def read_csv(path: Path, required: bool = True) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV into dictionaries with safe error handling."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
            return rows, fieldnames
    except FileNotFoundError:
        message = f"CSV not found: {path}"
        if required:
            log.error(message)
            raise
        log.warning(message)
        return [], []
    except csv.Error as exc:
        log.error("Failed to parse CSV %s: %s", path, exc)
        raise
    except OSError as exc:
        log.error("Failed to read CSV %s: %s", path, exc)
        raise


def parse_audio_filename(name: str) -> dict[str, str]:
    """Extract conversation id, speaker role, and optional audio index from a file name."""
    match = FILENAME_PATTERN.match(Path(name).name)
    if not match:
        return {
            "conversation_id": "unknown",
            "speaker_role": "unknown",
            "audio_index": "unknown",
        }

    values = match.groupdict()
    return {
        "conversation_id": values["conversation_id"],
        "speaker_role": values["speaker_role"].lower(),
        "audio_index": values.get("audio_index") or "unknown",
    }


def first_present(row: dict[str, str], columns: Iterable[str]) -> str:
    """Return the first non-empty value found in row for any candidate column."""
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def normalized_role(row: dict[str, str], parsed_role: str = "") -> str:
    """Return human_type when available, otherwise infer from filename or dataset group."""
    human_type = (row.get("human_type") or "").strip().lower()
    if human_type:
        return human_type
    if parsed_role:
        return parsed_role
    return ROLE_BY_GROUP.get((row.get("dataset_group") or "").strip().lower(), "unknown")


def age_group(age: str) -> str:
    """Map an age value to a coarse age group."""
    if not age or age == "unknown":
        return "unknown"

    try:
        age_value = float(str(age).strip())
    except (TypeError, ValueError):
        return "unknown"

    if age_value < 0:
        return "unknown"
    if age_value <= 12:
        return "Child"
    if age_value <= 19:
        return "Teen"
    if age_value <= 45:
        return "Adult"
    if age_value <= 60:
        return "Middle_Age"
    return "Senior"


def build_audio_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    """Index conversation_audio rows by parsed filename key."""
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    duplicate_count = 0

    for row in rows:
        parsed = parse_audio_filename(row.get("audio_path", ""))
        key = (
            parsed["conversation_id"],
            parsed["speaker_role"],
            parsed["audio_index"],
        )
        if "unknown" in key:
            continue
        if key in lookup:
            duplicate_count += 1
            continue
        lookup[key] = row

    if duplicate_count:
        log.warning("Skipped %d duplicate conversation_audio filename key(s).", duplicate_count)

    return lookup


def build_text_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], str]:
    """Index transcripts by conversation_id, normalized role, and text_id."""
    lookup: dict[tuple[str, str, str], str] = {}
    duplicate_count = 0

    for row in rows:
        conversation_id = (row.get("conversation_id") or "").strip()
        text_id = (row.get("text_id") or "").strip()
        text = (row.get("text") or "").strip()
        if not conversation_id or not text_id or not text:
            continue

        role = normalized_role(row)
        key = (conversation_id, role, text_id)
        if key in lookup:
            duplicate_count += 1
            continue
        lookup[key] = text

    if duplicate_count:
        log.warning("Skipped %d duplicate conversation_text key(s).", duplicate_count)

    return lookup


def get_transcript(
    audio_row: dict[str, str] | None,
    parsed: dict[str, str],
    text_lookup: dict[tuple[str, str, str], str],
) -> str:
    """Find aligned transcript text for an audio row, with a filename-index fallback."""
    if audio_row:
        metadata_conversation_id = (audio_row.get("conversation_id") or "").strip()
        metadata_audio_id = (audio_row.get("audio_id") or "").strip()
        metadata_role = normalized_role(audio_row, parsed["speaker_role"])

        if metadata_conversation_id and metadata_audio_id:
            transcript = text_lookup.get(
                (metadata_conversation_id, metadata_role, metadata_audio_id), ""
            )
            if transcript:
                return transcript

    if parsed["audio_index"] != "unknown":
        return text_lookup.get(
            (parsed["conversation_id"], parsed["speaker_role"], parsed["audio_index"]),
            "",
        )

    return ""


def build_metadata(
    inventory_rows: list[dict[str, str]],
    audio_rows: list[dict[str, str]],
    text_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], Counter[str], int, int, int]:
    """Build output metadata rows and summary counters."""
    audio_lookup = build_audio_lookup(audio_rows)
    text_lookup = build_text_lookup(text_rows)

    output_rows: list[dict[str, str]] = []
    age_group_counts: Counter[str] = Counter()
    matched_conversation_rows = 0
    rows_with_transcript = 0
    rows_with_age = 0

    for inventory_row in inventory_rows:
        parsed = parse_audio_filename(inventory_row.get("file_name", ""))
        key = (
            parsed["conversation_id"],
            parsed["speaker_role"],
            parsed["audio_index"],
        )
        audio_row = audio_lookup.get(key)
        if audio_row:
            matched_conversation_rows += 1

        source_for_age = audio_row or {}
        age = first_present(source_for_age, AGE_COLUMNS) or "unknown"
        group = age_group(age)
        transcript = get_transcript(audio_row, parsed, text_lookup)

        if transcript:
            rows_with_transcript += 1
        if age != "unknown" and group != "unknown":
            rows_with_age += 1

        age_group_counts[group] += 1
        output_rows.append(
            {
                "file_path": inventory_row.get("file_path", ""),
                "file_name": inventory_row.get("file_name", ""),
                "dataset_group": inventory_row.get("dataset_group", ""),
                "conversation_id": parsed["conversation_id"],
                "speaker_role": parsed["speaker_role"],
                "audio_index": parsed["audio_index"],
                "transcript": transcript,
                "age": age,
                "age_group": group,
            }
        )

    return (
        output_rows,
        age_group_counts,
        matched_conversation_rows,
        rows_with_transcript,
        rows_with_age,
    )


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    """Write metadata rows to the output CSV."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        log.error("Failed to write metadata CSV %s: %s", path, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Build audio metadata mapping CSV.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Path to data/processed. Defaults to <project_root>/data/processed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <processed_dir>/metadata.csv.",
    )
    args = parser.parse_args()

    project_root = resolve_project_root()
    processed_dir = args.processed_dir or project_root / "data" / "processed"
    output_path = args.output or processed_dir / "metadata.csv"

    inventory_path = processed_dir / "audio_inventory.csv"
    conversation_audio_path = processed_dir / "conversation_audio.csv"
    conversation_text_path = processed_dir / "conversation_text.csv"

    log.info("Project root       : %s", project_root)
    log.info("Processed data dir : %s", processed_dir)

    try:
        inventory_rows, inventory_columns = read_csv(inventory_path)
        audio_rows, audio_columns = read_csv(conversation_audio_path)
        text_rows, text_columns = read_csv(conversation_text_path)

        print(f"audio_inventory.csv columns: {inventory_columns}")
        print(f"conversation_audio.csv columns: {audio_columns}")
        print(f"conversation_text.csv columns: {text_columns}")

        rows, counts, matched, with_transcript, with_age = build_metadata(
            inventory_rows=inventory_rows,
            audio_rows=audio_rows,
            text_rows=text_rows,
        )
        write_metadata(output_path, rows)

    except Exception as exc:  # noqa: BLE001
        log.error("Metadata build failed: %s", exc)
        return 1

    print()
    print("Metadata summary")
    print(f"total audio rows: {len(inventory_rows)}")
    print(f"matched conversation rows: {matched}")
    print(f"rows with transcript: {with_transcript}")
    print(f"rows with age: {with_age}")
    print("age_group counts:")
    for group in ("Child", "Teen", "Adult", "Middle_Age", "Senior", "unknown"):
        print(f"  {group}: {counts.get(group, 0)}")
    print(f"metadata output: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
