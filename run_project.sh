#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

needs_setup=false

if [ ! -x "$VENV_PYTHON" ]; then
  needs_setup=true
else
  VENV_VERSION="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "$VENV_VERSION" != "3.11" ]; then
    needs_setup=true
  elif ! "$VENV_PYTHON" - <<'PY'
import importlib.util

required = (
    "fastapi",
    "uvicorn",
    "multipart",
    "pydantic",
    "numpy",
    "librosa",
    "soundfile",
    "torch",
    "transformers",
    "safetensors",
    "requests",
    "twilio",
    "websockets",
)

raise SystemExit(0 if all(importlib.util.find_spec(name) for name in required) else 1)
PY
  then
    needs_setup=true
  fi
fi

if [ "$needs_setup" = true ]; then
  echo "Preparing VoiceAge AI Python 3.11 environment..."
  PYTHON_BIN="$PYTHON_BIN" "$ROOT_DIR/scripts/setup_env.sh"
fi

echo "Starting VoiceAge AI API at http://localhost:8765"
exec "$ROOT_DIR/scripts/run_api.sh"
