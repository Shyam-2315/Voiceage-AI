#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  cat >&2 <<EOF
Project virtualenv not found at .venv.

Create the supported Python 3.11 runtime first:
  scripts/setup_env.sh

If Python 3.11 is not on PATH, run:
  PYTHON_BIN=/path/to/python3.11 scripts/setup_env.sh
EOF
  exit 1
fi

PYTHON_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$PYTHON_VERSION" != "3.11" ]; then
  cat >&2 <<EOF
Unsupported .venv Python version: $PYTHON_VERSION
VoiceAge AI production API is pinned to Python 3.11.

Recreate .venv with:
  scripts/setup_env.sh
EOF
  exit 1
fi

"$PYTHON" - <<'PY'
import importlib.util
import sys

required = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "python-multipart": "multipart",
    "pydantic": "pydantic",
    "numpy": "numpy",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "torch": "torch",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "requests": "requests",
    "twilio": "twilio",
    "websockets": "websockets",
}

missing = [name for name, module in required.items() if importlib.util.find_spec(module) is None]
if missing:
    print("Missing runtime dependencies in .venv: " + ", ".join(missing), file=sys.stderr)
    print("Install them with: scripts/setup_env.sh", file=sys.stderr)
    raise SystemExit(1)

try:
    import app.main  # noqa: F401
except Exception as exc:
    print(f"FastAPI app import failed: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
PY

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
LOG_LEVEL="${LOG_LEVEL:-info}"

exec "$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
