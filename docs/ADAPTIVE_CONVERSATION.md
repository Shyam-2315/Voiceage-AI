# Adaptive Conversation

## Problem

VoiceAge AI predicts an age group from caller-only audio, but a realtime phone assistant should not use the same pacing, vocabulary, and question style for every caller. The adaptive conversation module updates the assistant style after age prediction while preserving the existing Twilio realtime call flow, caller-only audio capture, and Wav2Vec2 model.

## Solution

`app/services/conversation_style_service.py` defines a typed `ConversationStyle` dataclass and a style map for `Child`, `Teen`, `Adult`, `Middle_Age`, and `Senior`. Each style includes:

- `age_group`
- `tone`
- `speaking_speed`
- `pitch_style`
- `vocabulary_level`
- `question_complexity`
- `interruption_delay_ms`
- `system_prompt_addon`

The realtime bridge keeps its initial Twilio and OpenAI/Azure session setup unchanged. After the existing live VoiceAge prediction succeeds, the bridge selects an adaptive style, sends one adaptive Realtime `session.update` for that selected age group, and logs the selected style. If style selection or session update fails, the error is logged and the call continues.

Unknown or missing age groups fall back to `DEFAULT_CONVERSATION_STYLE`, which defaults to `Adult`.

## Flowchart Text

```text
Twilio call starts
  -> Twilio Media Stream connects to /api/realtime/twilio-stream
  -> Bridge opens OpenAI/Azure Realtime WebSocket
  -> Bridge sends normal initial session.update
  -> Caller-only inbound Twilio audio is captured
  -> Enough caller-only audio is available
  -> Existing Wav2Vec2 live age prediction runs
  -> If prediction fails: log skip and continue normal call
  -> If ENABLE_ADAPTIVE_CONVERSATION=false: log skip and continue normal call
  -> Select style for predicted_age_group
  -> Unknown/missing group falls back to DEFAULT_CONVERSATION_STYLE
  -> If selected style already applied: skip duplicate session.update
  -> Send adaptive session.update with prompt addon and VAD delay
  -> Continue the same Twilio realtime call
```

## Age Group Behavior Table

| Age Group | Tone | Speaking Speed | Pitch Style | Vocabulary Level | Question Complexity | Interruption Delay |
| --- | --- | --- | --- | --- | --- | --- |
| Child | warm, patient, and encouraging | slightly slower than normal | bright and gentle | child-friendly concrete words | very simple questions with concrete words | 850 ms |
| Teen | relaxed, respectful, and direct | natural conversational pace | natural and energetic without sounding exaggerated | casual everyday words | moderate questions with casual wording | 650 ms |
| Adult | friendly, concise, and professional | normal conversational pace | neutral and confident | standard adult vocabulary | standard questions with concise context | 600 ms |
| Middle_Age | clear, helpful, and composed | measured normal pace | steady and reassuring | clear practical vocabulary | standard questions with practical detail | 700 ms |
| Senior | calm | slower speech | steady, gentle, and easy to hear | simple words | simple words and one question at a time | 1100 ms |

## Config Flags

```bash
ENABLE_ADAPTIVE_CONVERSATION=true
DEFAULT_CONVERSATION_STYLE=Adult
```

`ENABLE_ADAPTIVE_CONVERSATION=false` disables the adaptive `session.update` while leaving prediction, Twilio streaming, and logging behavior in place.

`DEFAULT_CONVERSATION_STYLE` controls the fallback style for unknown or missing age groups. Invalid fallback values are safely normalized to `Adult`.

## Demo Steps

Print the style map and simulate a Senior prediction:

```bash
python scripts/demo_adaptive_conversation.py
```

Simulate an unknown age group and confirm Adult fallback:

```bash
python scripts/demo_adaptive_conversation.py --age-group Unknown
```

Run the realtime API with Twilio and OpenAI/Azure environment variables set:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Configure the Twilio phone number voice webhook:

```text
POST https://<public-base-url>/api/twilio/realtime-voice
```

Call the Twilio number and speak naturally until live prediction runs. In application logs, confirm:

```text
Adaptive conversation selected: predicted_age_group=<model result> selected_age_group=<style key>
Realtime outbound event=session.update
```

In `data/realtime_conversations/<call_id>/events.jsonl`, confirm:

```text
voiceage.live_prediction.completed
adaptive_conversation.style_selected
session.update.sent
```

For a Senior prediction, the assistant should shift toward calmer tone, slower speech, simple words, one question at a time, and a longer wait before the next question.

## Limitations

- The current Wav2Vec2 model labels are not changed by this module.
- The current production model labels may not include every style group; unsupported labels still have safe fallback behavior.
- Realtime voice pitch and speed are guided through prompt instructions and VAD timing, not low-level audio DSP.
- The adaptive update happens only after enough caller-only audio is captured and age prediction completes.
- The module does not infer exact age and should not disclose or rely on exact age.
