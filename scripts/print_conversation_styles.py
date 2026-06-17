#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.conversation_style_service import AGE_GROUPS, get_conversation_style  # noqa: E402


def main() -> int:
    for age_group in AGE_GROUPS:
        style = get_conversation_style(age_group)
        print(json.dumps(style.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
