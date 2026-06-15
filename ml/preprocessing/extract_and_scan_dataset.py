"""
extract_and_scan_dataset.py

Handles extraction of split archive groups from data/raw/ and builds an
inventory CSV of all discovered audio files.

Archive groups detected by prefix:
  - audio_client  (audio_client.z01, .z02, .z03)
  - audio_speaker (audio_speaker.z01)
  - audio_therapist (audio_therapist.z01 – .z05)

Extraction target: data/raw/extracted/<group>/
Inventory output:  data/processed/audio_inventory.csv

Usage:
    python ml/preprocessing/extract_and_scan_dataset.py [--raw-dir PATH] [--dry-run]
"""

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
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
ARCHIVE_GROUPS = ["audio_client", "audio_speaker", "audio_therapist"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
METADATA_FILES = ["conversation_audio.csv", "conversation_text.csv"]

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_project_root() -> Path:
    """Walk up from this file until we find the project root (contains data/)."""
    candidate = Path(__file__).resolve()
    for parent in [candidate, *candidate.parents]:
        if (parent / "data").is_dir():
            return parent
    # Fallback: two levels above ml/preprocessing/
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Archive detection
# ---------------------------------------------------------------------------

def find_archive_parts(raw_dir: Path, group: str) -> list[Path]:
    """Return sorted list of split archive parts for a given group prefix."""
    parts = sorted(raw_dir.glob(f"{group}.z*"))
    # Keep only files whose suffix matches .z\d+ (skip Zone.Identifier noise)
    parts = [p for p in parts if p.suffix.lstrip(".").isdigit() or p.suffix == ".zip"]
    return parts


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_group(raw_dir: Path, group: str, dest_dir: Path, dry_run: bool) -> bool:
    """
    Attempt to extract a split-zip archive group using the `7z` command.

    The first part of a 7-Zip split archive (e.g. audio_client.z01) can be
    passed directly to `7z x` and it will reassemble the remaining parts
    automatically, provided they sit in the same directory.

    Returns True on success, False when extraction could not be completed.
    """
    parts = find_archive_parts(raw_dir, group)

    if not parts:
        log.warning("No archive parts found for group '%s' in %s — skipping.", group, raw_dir)
        return False

    # The first part is the entry point for 7z
    first_part = parts[0]
    log.info("Group '%s': found %d part(s): %s", group, len(parts), [p.name for p in parts])

    if dry_run:
        log.info("[DRY RUN] Would extract '%s' → %s", first_part.name, dest_dir)
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check 7z availability
    sevenzip_cmd = _find_7z()
    if sevenzip_cmd is None:
        _print_manual_instructions(raw_dir, group, parts, dest_dir)
        return False

    cmd = [sevenzip_cmd, "x", str(first_part), f"-o{dest_dir}", "-y"]
    log.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute safeguard per group
        )
        if result.returncode == 0:
            log.info("✔  Extracted '%s' → %s", group, dest_dir)
            return True
        else:
            log.error(
                "7z exited with code %d for group '%s'.\nstdout: %s\nstderr: %s",
                result.returncode,
                group,
                result.stdout[-2000:],
                result.stderr[-2000:],
            )
            _print_manual_instructions(raw_dir, group, parts, dest_dir)
            return False

    except FileNotFoundError:
        log.error("7z binary not found at '%s'.", sevenzip_cmd)
        _print_manual_instructions(raw_dir, group, parts, dest_dir)
        return False
    except subprocess.TimeoutExpired:
        log.error("Extraction timed out for group '%s'.", group)
        _print_manual_instructions(raw_dir, group, parts, dest_dir)
        return False
    except Exception as exc:  # noqa: BLE001
        log.error("Unexpected error during extraction of '%s': %s", group, exc)
        _print_manual_instructions(raw_dir, group, parts, dest_dir)
        return False


def _find_7z() -> str | None:
    """Return the 7z executable path, or None if not found."""
    for candidate in ("7z", "7za", "7zr"):
        if shutil.which(candidate):
            return candidate
    return None


def _print_manual_instructions(
    raw_dir: Path, group: str, parts: list[Path], dest_dir: Path
) -> None:
    """Print clear manual extraction instructions when automatic extraction fails."""
    first = parts[0] if parts else f"{group}.z01"
    print("\n" + "=" * 70)
    print(f"  MANUAL EXTRACTION REQUIRED — group: {group}")
    print("=" * 70)
    print("Automatic extraction failed or 7z is not installed.")
    print()
    print("Install 7-Zip on Ubuntu/Debian:")
    print("    sudo apt-get install p7zip-full")
    print()
    print("Then run:")
    print(f"    mkdir -p '{dest_dir}'")
    print(f"    7z x '{first}' -o'{dest_dir}' -y")
    print()
    print("All split parts must remain in the same directory:")
    for p in parts:
        print(f"    {p}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Audio inventory scan
# ---------------------------------------------------------------------------

def scan_audio_files(extracted_root: Path) -> list[dict]:
    """
    Recursively scan extracted_root/<group>/ directories for audio files.

    Returns a list of dicts with keys:
        file_path, file_name, extension, size_mb, dataset_group
    """
    rows: list[dict] = []

    for group in ARCHIVE_GROUPS:
        group_dir = extracted_root / group
        if not group_dir.is_dir():
            log.warning("Extracted directory does not exist: %s — skipping scan.", group_dir)
            continue

        found = 0
        for audio_file in group_dir.rglob("*"):
            if not audio_file.is_file():
                continue
            if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            size_mb = audio_file.stat().st_size / (1024 * 1024)
            rows.append(
                {
                    "file_path": str(audio_file.resolve()),
                    "file_name": audio_file.name,
                    "extension": audio_file.suffix.lower(),
                    "size_mb": round(size_mb, 4),
                    "dataset_group": group,
                }
            )
            found += 1

        log.info("Group '%s': %d audio file(s) found.", group, found)

    return rows


# ---------------------------------------------------------------------------
# Inventory CSV
# ---------------------------------------------------------------------------

def save_inventory(rows: list[dict], output_path: Path) -> None:
    """Write the audio inventory to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_path", "file_name", "extension", "size_mb", "dataset_group"]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Inventory saved → %s  (%d rows)", output_path, len(rows))


# ---------------------------------------------------------------------------
# Metadata CSV copy / verify
# ---------------------------------------------------------------------------

def handle_metadata_files(raw_dir: Path, processed_dir: Path, dry_run: bool) -> None:
    """Copy conversation CSVs from raw/ to processed/ and verify they are readable."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    for filename in METADATA_FILES:
        src = raw_dir / filename
        dst = processed_dir / filename

        if not src.exists():
            log.warning("Metadata file not found: %s", src)
            continue

        # Verify the source is readable and non-empty
        try:
            size = src.stat().st_size
            if size == 0:
                log.warning("Metadata file is empty: %s", src)
            else:
                log.info("Verified metadata file: %s  (%.2f KB)", src.name, size / 1024)
        except OSError as exc:
            log.error("Cannot read metadata file %s: %s", src, exc)
            continue

        if dry_run:
            log.info("[DRY RUN] Would copy %s → %s", src, dst)
            continue

        try:
            shutil.copy2(src, dst)
            log.info("Copied %s → %s", src.name, dst)
        except OSError as exc:
            log.error("Failed to copy %s: %s", src.name, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract split archives and build an audio inventory for voiceage-ai."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Path to data/raw/ directory. Defaults to <project_root>/data/raw/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without actually extracting or writing files.",
    )
    args = parser.parse_args()

    # Resolve directories
    project_root = resolve_project_root()
    raw_dir: Path = args.raw_dir if args.raw_dir else project_root / "data" / "raw"
    extracted_root = raw_dir / "extracted"
    processed_dir = project_root / "data" / "processed"
    inventory_path = processed_dir / "audio_inventory.csv"

    log.info("Project root   : %s", project_root)
    log.info("Raw data dir   : %s", raw_dir)
    log.info("Extracted into : %s", extracted_root)
    log.info("Processed dir  : %s", processed_dir)

    if not raw_dir.exists():
        log.error("Raw data directory not found: %s", raw_dir)
        sys.exit(1)

    if args.dry_run:
        log.info("*** DRY RUN MODE — no files will be written ***")

    # ------------------------------------------------------------------
    # Step 1 – Extract archive groups
    # ------------------------------------------------------------------
    log.info("--- Step 1: Extracting archive groups ---")
    extraction_results: dict[str, bool] = {}
    for group in ARCHIVE_GROUPS:
        dest = extracted_root / group
        success = extract_group(raw_dir, group, dest, dry_run=args.dry_run)
        extraction_results[group] = success

    # Summary
    succeeded = [g for g, ok in extraction_results.items() if ok]
    failed = [g for g, ok in extraction_results.items() if not ok]
    log.info("Extraction summary — success: %s | failed/skipped: %s", succeeded, failed)

    # ------------------------------------------------------------------
    # Step 2 – Scan extracted directories for audio files
    # ------------------------------------------------------------------
    log.info("--- Step 2: Scanning for audio files ---")
    if not args.dry_run:
        inventory_rows = scan_audio_files(extracted_root)
        if not inventory_rows:
            log.warning(
                "No audio files found. This may be expected if extraction failed or "
                "archives are still being extracted manually."
            )
    else:
        log.info("[DRY RUN] Skipping scan — no files extracted yet.")
        inventory_rows = []

    # ------------------------------------------------------------------
    # Step 3 – Save inventory CSV
    # ------------------------------------------------------------------
    log.info("--- Step 3: Saving inventory CSV ---")
    if not args.dry_run:
        save_inventory(inventory_rows, inventory_path)
    else:
        log.info("[DRY RUN] Would save inventory → %s", inventory_path)

    # ------------------------------------------------------------------
    # Step 4 – Copy / verify metadata CSVs
    # ------------------------------------------------------------------
    log.info("--- Step 4: Handling metadata CSV files ---")
    handle_metadata_files(raw_dir, processed_dir, dry_run=args.dry_run)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Dataset preparation complete.")
    log.info("  Audio files inventoried : %d", len(inventory_rows))
    log.info("  Inventory CSV           : %s", inventory_path)
    log.info("  Extracted root          : %s", extracted_root)
    if failed:
        log.warning(
            "  Groups requiring manual extraction: %s  "
            "(see manual instructions printed above)",
            failed,
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
