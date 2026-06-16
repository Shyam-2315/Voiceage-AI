#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  cat >&2 <<EOF
Python 3.11 was not found as '$PYTHON_BIN'.

Install Python 3.11 or point to an existing interpreter:
  PYTHON_BIN=/path/to/python3.11 scripts/setup_env.sh
EOF
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$PYTHON_VERSION" != "3.11" ]; then
  echo "Unsupported interpreter: $PYTHON_BIN reports Python $PYTHON_VERSION; Python 3.11 is required." >&2
  exit 1
fi

if [ -x ".venv/bin/python" ]; then
  VENV_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "$VENV_VERSION" != "3.11" ]; then
    BACKUP_DIR=".venv.py${VENV_VERSION}.backup.$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Existing .venv uses Python $VENV_VERSION; moving it to $BACKUP_DIR"
    mv ".venv" "$BACKUP_DIR"
  fi
fi

if [ ! -x ".venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv ".venv"
fi

".venv/bin/python" -m pip install --upgrade pip setuptools wheel
".venv/bin/python" -m pip install -r requirements.txt

".venv/bin/python" - <<'PY'
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
    print("Missing imports after install: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)

print("VoiceAge API environment ready:", sys.executable)
PY
