from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


AGE_GROUPS = ("Child", "Teen", "Adult", "Middle_Age", "Senior")
DEFAULT_AGE_GROUP = "Adult"


@dataclass(frozen=True)
class ConversationStyle:
    age_group: str
    tone: str
    speaking_speed: str
    pitch_style: str
    vocabulary_level: str
    question_complexity: str
    interruption_delay_ms: int
    system_prompt_addon: str

    def __post_init__(self) -> None:
        required_strings = {
            "age_group": self.age_group,
            "tone": self.tone,
            "speaking_speed": self.speaking_speed,
            "pitch_style": self.pitch_style,
            "vocabulary_level": self.vocabulary_level,
            "question_complexity": self.question_complexity,
            "system_prompt_addon": self.system_prompt_addon,
        }
        for field_name, value in required_strings.items():
            if not value or not value.strip():
                raise ValueError(f"{field_name} is required for ConversationStyle.")
        if self.interruption_delay_ms <= 0:
            raise ValueError("interruption_delay_ms must be positive.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationStyleSelection:
    requested_age_group: str | None
    selected_age_group: str
    style: ConversationStyle
    used_fallback: bool

    def log_payload(self) -> dict[str, Any]:
        return {
            "requested_age_group": self.requested_age_group,
            "selected_age_group": self.selected_age_group,
            "used_fallback": self.used_fallback,
            "conversation_style": self.style.as_dict(),
        }


CONVERSATION_STYLES: dict[str, ConversationStyle] = {
    "Child": ConversationStyle(
        age_group="Child",
        tone="warm, patient, and encouraging",
        speaking_speed="slightly slower than normal",
        pitch_style="bright and gentle",
        vocabulary_level="child-friendly concrete words",
        question_complexity="very simple questions with concrete words",
        interruption_delay_ms=850,
        system_prompt_addon=(
            "Use child-friendly words. Keep answers very short and concrete. "
            "Ask one simple question at a time and give the caller time to answer."
        ),
    ),
    "Teen": ConversationStyle(
        age_group="Teen",
        tone="relaxed, respectful, and direct",
        speaking_speed="natural conversational pace",
        pitch_style="natural and energetic without sounding exaggerated",
        vocabulary_level="casual everyday words",
        question_complexity="moderate questions with casual wording",
        interruption_delay_ms=650,
        system_prompt_addon=(
            "Use a relaxed but respectful style. Avoid sounding parental or overly formal. "
            "Keep questions clear and avoid long explanations."
        ),
    ),
    "Adult": ConversationStyle(
        age_group="Adult",
        tone="friendly, concise, and professional",
        speaking_speed="normal conversational pace",
        pitch_style="neutral and confident",
        vocabulary_level="standard adult vocabulary",
        question_complexity="standard questions with concise context",
        interruption_delay_ms=600,
        system_prompt_addon=(
            "Use a concise professional style. Give enough context to be useful, "
            "then ask a clear next question when needed."
        ),
    ),
    "Middle_Age": ConversationStyle(
        age_group="Middle_Age",
        tone="clear, helpful, and composed",
        speaking_speed="measured normal pace",
        pitch_style="steady and reassuring",
        vocabulary_level="clear practical vocabulary",
        question_complexity="standard questions with practical detail",
        interruption_delay_ms=700,
        system_prompt_addon=(
            "Use a clear and practical style. Keep responses organized, avoid filler, "
            "and ask direct follow-up questions."
        ),
    ),
    "Senior": ConversationStyle(
        age_group="Senior",
        tone="calm",
        speaking_speed="slower speech",
        pitch_style="steady, gentle, and easy to hear",
        vocabulary_level="simple words",
        question_complexity="simple words and one question at a time",
        interruption_delay_ms=1100,
        system_prompt_addon=(
            "Use simple words and a calm tone. Speak more slowly. "
            "Ask only one question at a time. Wait longer before asking the next question. "
            "Avoid jargon and long multi-step explanations."
        ),
    ),
}


def normalize_age_group(age_group: str | None, default_age_group: str = DEFAULT_AGE_GROUP) -> str:
    default = normalized_known_age_group(default_age_group) or DEFAULT_AGE_GROUP
    if not age_group:
        return default

    normalized = normalized_known_age_group(age_group)
    return normalized or default


def normalized_known_age_group(age_group: str | None) -> str | None:
    if not age_group:
        return None
    normalized = str(age_group).strip().replace("-", "_").replace(" ", "_")
    lookup = {group.lower(): group for group in AGE_GROUPS}
    return lookup.get(normalized.lower())


def get_conversation_style(age_group: str | None, default_age_group: str = DEFAULT_AGE_GROUP) -> ConversationStyle:
    return select_conversation_style(age_group, default_age_group).style


def select_conversation_style(
    age_group: str | None,
    default_age_group: str = DEFAULT_AGE_GROUP,
) -> ConversationStyleSelection:
    selected_age_group = normalize_age_group(age_group, default_age_group)
    requested_age_group = str(age_group).strip() if age_group else None
    requested_known_age_group = normalized_known_age_group(age_group)
    used_fallback = requested_known_age_group is None or requested_known_age_group != selected_age_group
    return ConversationStyleSelection(
        requested_age_group=requested_age_group,
        selected_age_group=selected_age_group,
        style=CONVERSATION_STYLES[selected_age_group],
        used_fallback=used_fallback,
    )


def build_conversation_style_instructions(style: ConversationStyle) -> str:
    return (
        "Age-adaptive conversation style is active.\n"
        f"- Selected age group: {style.age_group}\n"
        f"- Tone: {style.tone}\n"
        f"- Speaking speed: {style.speaking_speed}\n"
        f"- Pitch style: {style.pitch_style}\n"
        f"- Vocabulary level: {style.vocabulary_level}\n"
        f"- Question complexity: {style.question_complexity}\n"
        f"- Interruption delay target: wait about {style.interruption_delay_ms} ms after the caller stops.\n"
        f"- Style guidance: {style.system_prompt_addon}\n"
        "Do not mention the predicted age group unless the caller asks directly."
    )
