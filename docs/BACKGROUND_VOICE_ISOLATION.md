# Background Voice Isolation

## Problem

The realtime Twilio flow already captures caller-only audio by excluding assistant audio. That does not remove another human speaker, TV audio, or other speech that may be present near the caller. If that background speech is included in the caller-only snapshot, it can affect VoiceAge age prediction.

## Solution

Background voice isolation is an optional, fail-open filter that runs only before VoiceAge age prediction. It combines:

- Silero VAD to detect speech segments.
- SpeechBrain ECAPA-TDNN to create speaker embeddings.
- A caller voiceprint from the first clean caller speech.
- Cosine similarity to decide whether later speech segments match the caller.

The Twilio stream, Realtime transcript flow, Wav2Vec2 model, and Adaptive Conversation module are not replaced.

## Live Call Flow

1. Twilio inbound media is captured as before.
2. Realtime audio and transcript handling continue unchanged.
3. When enough caller-only audio is available for VoiceAge, the live prediction task writes `caller_only_audio.wav`.
4. If `BACKGROUND_VOICE_ISOLATION_ENABLED=false`, the existing prediction path is used.
5. If enabled, `BackgroundVoiceIsolationService` filters the caller-only audio before age prediction.
6. If isolation fails, the system logs a warning and falls back to the original caller-only audio.
7. Reports may include a `background_voice_isolation` summary with safe metadata only.

## Config Flags

```bash
BACKGROUND_VOICE_ISOLATION_ENABLED=false
BACKGROUND_VOICE_ISOLATION_THRESHOLD=0.75
BACKGROUND_VOICE_REFERENCE_SECONDS=5
BACKGROUND_VOICE_MIN_SEGMENT_SEC=1.0
BACKGROUND_VOICE_DEBUG_METRICS=true
```

`BACKGROUND_VOICE_ISOLATION_THRESHOLD` controls the cosine similarity cutoff. Higher values are stricter and may reject true caller speech. Lower values keep more audio but may include background speakers.

## Logging And Reports

Logs include safe metadata only:

- Whether isolation is enabled.
- Whether the reference is ready.
- Segment similarity.
- Keep/reject counts.

Logs must not include raw audio, base64 payloads, API keys, or secrets.

When enabled, VoiceAge reports include:

- `enabled`
- `reference_ready`
- `kept_segments`
- `rejected_segments`
- `avg_similarity`
- `threshold`
- fallback status if isolation failed
- `debug_metrics` when `BACKGROUND_VOICE_DEBUG_METRICS=true`

## Debug Metrics

Debug metrics are enabled by default with:

```bash
BACKGROUND_VOICE_DEBUG_METRICS=true
```

They are safe metadata only. They do not include raw audio, Twilio payloads, API keys, or transcripts.

The `background_voice_isolation.debug_metrics` block can include:

| Metric | Meaning |
| --- | --- |
| `total_audio_duration_sec` | Duration of the audio sent to isolation after normalization. |
| `sample_rate` | Sample rate used by the isolation service, expected to be `16000`. |
| `num_samples` | Number of normalized audio samples. |
| `audio_rms` | Root mean square amplitude. Near-zero values usually mean very quiet audio. |
| `audio_peak` | Highest absolute amplitude. `0.0` means digital silence. |
| `audio_is_silent` | Whether peak amplitude is effectively silent. |
| `reference_duration_sec` | Amount of audio used to build the caller voiceprint. |
| `vad_segment_count` | Number of speech segments returned by Silero VAD. |
| `vad_speech_duration_sec` | Total duration of detected speech. |
| `vad_speech_percentage` | Detected speech duration divided by total audio duration. |
| `min_segment_sec` | Configured minimum VAD segment duration. |
| `threshold` | Speaker similarity threshold. |
| `fallback_reason` | Reason the system used original caller audio instead of isolated audio. |

### Interpreting "No Speech Segments"

If logs show:

```text
Background voice isolation found no speech segments; using original audio.
```

Check these fields:

- `total_audio_duration_sec`: confirms enough audio reached the isolation step.
- `audio_rms` and `audio_peak`: if both are near zero, the captured caller WAV may be silent or too quiet.
- `sample_rate`: should be `16000` after app preprocessing.
- `min_segment_sec`: if this is too high, short Twilio caller turns may be discarded.
- `vad_segment_count`: `0` means Silero did not accept any speech region.

Recommended next actions:

- Confirm `caller_only_audio.wav` is audible in the call folder.
- Lower `BACKGROUND_VOICE_MIN_SEGMENT_SEC` for Twilio tests.
- Use a shorter clean caller reference window if early call audio is clear.

Recommended live Twilio test values:

```bash
BACKGROUND_VOICE_ISOLATION_ENABLED=true
BACKGROUND_VOICE_DEBUG_METRICS=true
BACKGROUND_VOICE_MIN_SEGMENT_SEC=0.3
BACKGROUND_VOICE_ISOLATION_THRESHOLD=0.65
BACKGROUND_VOICE_REFERENCE_SECONDS=3
```

## How To Test

Install the updated dependencies, then run:

```bash
python -m compileall app scripts ml
python scripts/check_system.py
python -m unittest discover -s tests
```

Dry-run with local files:

```bash
python scripts/test_background_isolation.py \
  --input sample_audio/mixed_call.wav \
  --reference sample_audio/caller_reference.wav
```

For a live Twilio test:

1. Set `BACKGROUND_VOICE_ISOLATION_ENABLED=true`.
2. Keep `BACKGROUND_VOICE_ISOLATION_THRESHOLD=0.75` for the first test.
3. Start the API and expose it through the existing Twilio tunnel setup.
4. Place a call with only the primary caller speaking for the first few seconds.
5. Add a second/background speaker later in the call.
6. Check `data/realtime_conversations/<call>/voiceage_live_prediction.json`.
7. Confirm `background_voice_isolation` metadata is present and no raw audio is logged.

## Limitations

- This is a segment filter, not true source separation.
- Overlapping caller and background speech can still be misclassified.
- Very short segments produce weaker speaker embeddings.
- A poor reference clip can cause false accepts or false rejects.
- First-run model downloads may add latency when the feature is enabled.

## Rollback Plan

Set:

```bash
BACKGROUND_VOICE_ISOLATION_ENABLED=false
```

Restart the service. With the flag disabled, the existing caller-only age prediction path is used.
