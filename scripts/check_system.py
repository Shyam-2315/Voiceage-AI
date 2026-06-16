#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402


REQUIRED_IMPORTS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "python-multipart": "multipart",
    "pydantic": "pydantic",
    "torch": "torch",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "numpy": "numpy",
    "requests": "requests",
    "twilio": "twilio",
    "websockets": "websockets",
}

PRODUCTION_MODEL_PATH = PROJECT_ROOT / "models" / "wav2vec_75k" / "best"
REQUIRED_DIRS = (
    PROJECT_ROOT / "app",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "ml",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "models" / "wav2vec_75k",
    PROJECT_ROOT / "models" / "wav2vec_75k" / "best",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "data" / "realtime_conversations",
    PROJECT_ROOT / "data" / "twilio_predictions",
    PROJECT_ROOT / "reports",
)

TWILIO_ENV = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "PUBLIC_BASE_URL")
AZURE_ENV = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_REALTIME_ENDPOINT",
    "AZURE_OPENAI_REALTIME_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
)


def status_line(ok: bool, label: str, detail: str = "") -> str:
    state = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    return f"[{state}] {label}{suffix}"


def package_ready(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def env_status(names: tuple[str, ...]) -> tuple[bool, list[str]]:
    missing = [name for name in names if not os.getenv(name)]
    return not missing, missing


def main() -> int:
    print("VoiceAge AI readiness report")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"model_version: {settings.model_version}")
    print(f"model_path: {settings.model_path}")

    python_ok = sys.version_info[:2] == (3, 11)
    print(status_line(python_ok, "Python 3.11 runtime", "required for production API"))

    missing_packages = [
        package_name
        for package_name, import_name in REQUIRED_IMPORTS.items()
        if not package_ready(import_name)
    ]
    print(status_line(not missing_packages, "Python packages", ", ".join(missing_packages) or "all installed"))

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_detail = torch.cuda.get_device_name(0) if cuda_available else "CPU fallback"
        print(status_line(True, "torch installed", f"version={torch.__version__}"))
        print(status_line(cuda_available, "CUDA availability", device_detail))
    except Exception as exc:
        print(status_line(False, "torch/CUDA check", str(exc)))

    model_exists = settings.model_path.exists()
    print(status_line(model_exists, "Model path exists", str(settings.model_path)))
    model_default = settings.model_path == PRODUCTION_MODEL_PATH
    detail = str(PRODUCTION_MODEL_PATH)
    if os.getenv("MODEL_PATH"):
        detail = f"MODEL_PATH override active; production default is {PRODUCTION_MODEL_PATH}"
    print(status_line(model_default or bool(os.getenv("MODEL_PATH")), "Production model default", detail))

    missing_dirs = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_DIRS if not path.exists()]
    print(status_line(not missing_dirs, "Required directories", ", ".join(missing_dirs) or "all present"))

    twilio_ready, twilio_missing = env_status(TWILIO_ENV)
    twilio_detail = "present" if twilio_ready else "missing: " + ", ".join(twilio_missing)
    print(status_line(twilio_ready, "Twilio config", twilio_detail))

    azure_ready, azure_missing = env_status(AZURE_ENV)
    azure_detail = "present" if azure_ready else "missing: " + ", ".join(azure_missing)
    print(status_line(azure_ready, "Azure realtime config", azure_detail))

    public_url = os.getenv("PUBLIC_BASE_URL", "")
    public_url_ok = public_url.startswith("https://") if public_url else False
    print(status_line(public_url_ok, "PUBLIC_BASE_URL HTTPS", public_url or "not set"))

    critical_ok = python_ok and not missing_packages and model_exists and not missing_dirs
    detail = "prediction API can start" if critical_ok else "fix package/model warnings above"
    print(status_line(critical_ok, "Core API readiness", detail))
    return 0 if critical_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
