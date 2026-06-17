from __future__ import annotations

import unittest

from app.services.conversation_style_service import (
    AGE_GROUPS,
    build_conversation_style_instructions,
    get_conversation_style,
    select_conversation_style,
)


class ConversationStyleServiceTest(unittest.TestCase):
    def test_all_age_groups_return_valid_style(self) -> None:
        for age_group in AGE_GROUPS:
            with self.subTest(age_group=age_group):
                style = get_conversation_style(age_group)

                self.assertEqual(style.age_group, age_group)
                self.assertTrue(style.tone)
                self.assertTrue(style.speaking_speed)
                self.assertTrue(style.pitch_style)
                self.assertTrue(style.vocabulary_level)
                self.assertTrue(style.question_complexity)
                self.assertGreater(style.interruption_delay_ms, 0)
                self.assertTrue(style.system_prompt_addon)

    def test_unknown_age_group_falls_back_to_adult(self) -> None:
        selection = select_conversation_style("Unknown")

        self.assertEqual(selection.selected_age_group, "Adult")
        self.assertEqual(selection.style.age_group, "Adult")
        self.assertTrue(selection.used_fallback)

    def test_missing_age_group_falls_back_to_adult(self) -> None:
        selection = select_conversation_style(None)

        self.assertEqual(selection.selected_age_group, "Adult")
        self.assertEqual(selection.style.age_group, "Adult")
        self.assertTrue(selection.used_fallback)

    def test_senior_style_is_slower_simple_and_has_longer_delay(self) -> None:
        senior = get_conversation_style("Senior")
        adult = get_conversation_style("Adult")

        self.assertIn("slow", senior.speaking_speed.lower())
        self.assertIn("simple", senior.question_complexity.lower())
        self.assertGreater(senior.interruption_delay_ms, adult.interruption_delay_ms)

    def test_prompt_addon_is_non_empty_for_every_age_group(self) -> None:
        for age_group in AGE_GROUPS:
            with self.subTest(age_group=age_group):
                style = get_conversation_style(age_group)
                prompt_addon = build_conversation_style_instructions(style)

                self.assertTrue(style.system_prompt_addon.strip())
                self.assertTrue(prompt_addon.strip())
                self.assertIn(style.system_prompt_addon, prompt_addon)


if __name__ == "__main__":
    unittest.main()
