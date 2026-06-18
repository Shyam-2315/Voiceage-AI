"""Phase 1 validation for the background voice isolation POC."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


MIN_PYTHON = (3, 9)
REQUIRED_FOLDERS = (
    "sample_audio",
    "outputs",
    "reports",
    "src",
    "tests",
)
REQUIRED_IMPORTS = (
    "numpy",
    "scipy",
    "soundfile",
    "librosa",
    "torch",
)


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        required = ".".join(str(part) for part in MIN_PYTHON)
        current = ".".join(str(part) for part in sys.version_info[:3])
        raise RuntimeError(f"Python {required}+ is required; found {current}")

    current = ".".join(str(part) for part in sys.version_info[:3])
    print(f"Python version OK: {current}")


def check_required_folders(base_dir: Path) -> None:
    missing = [
        folder
        for folder in REQUIRED_FOLDERS
        if not (base_dir / folder).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required folders: " + ", ".join(sorted(missing))
        )

    print("Required folders OK")


def check_imports() -> None:
    failures: list[str] = []

    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append(f"{module_name}: {exc}")

    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise ImportError(f"Dependency import checks failed:\n{details}")

    print("Dependency imports OK")


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    check_python_version()
    check_required_folders(base_dir)
    check_imports()

    print("Phase 1 check passed: background voice isolation POC is ready.")


if __name__ == "__main__":
    main()
