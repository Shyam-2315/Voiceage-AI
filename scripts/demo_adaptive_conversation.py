#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.conversation_style_service import (  # noqa: E402
    AGE_GROUPS,
    build_conversation_style_instructions,
    get_conversation_style,
    select_conversation_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo VoiceAge adaptive conversation style selection.")
    parser.add_argument(
        "--age-group",
        default="Senior",
        help="Simulated predicted age group. Unknown values fall back to the default style.",
    )
    parser.add_argument(
        "--default-style",
        default="Adult",
        help="Fallback style used when the simulated age group is unknown or missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Adaptive conversation style mapping:")
    for age_group in AGE_GROUPS:
        style = get_conversation_style(age_group)
        print(json.dumps(style.as_dict(), indent=2))

    selection = select_conversation_style(args.age_group, default_age_group=args.default_style)
    print("\nSimulated prediction:")
    print(json.dumps(selection.log_payload(), indent=2))

    print("\nFinal realtime prompt addon:")
    print(build_conversation_style_instructions(selection.style))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
