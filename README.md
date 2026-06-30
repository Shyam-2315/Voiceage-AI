## Phase 15: FastAPI Inference API

Install API dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Smoke test with an audio file:

```bash
python scripts/test_predict_api.py path/to/audio.wav
```

## Phase 16: Twilio Voice Integration

Set local environment variables before running the API. `PUBLIC_BASE_URL` must be the HTTPS URL that Twilio can reach, such as an ngrok forwarding URL during local development.

```bash
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="your_twilio_auth_token"
export PUBLIC_BASE_URL="https://your-public-ngrok-or-host-url"
```

Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Twilio Console setup:

1. In Twilio Console, open **Phone Numbers > Manage > Active numbers**.
2. Select the phone number callers will dial.
3. Under **Voice Configuration**, set **A call comes in** to **Webhook**.
4. Set the webhook URL to:

```text
https://your-public-ngrok-or-host-url/api/twilio/voice
```

5. Set the method to `POST` and save.

The voice webhook returns TwiML that asks the caller to speak for 10 to 20 seconds and records the call. The recording action URL thanks the caller and hangs up. Twilio sends the completed recording callback to:

```text
https://your-public-ngrok-or-host-url/api/twilio/recording-complete
```

That callback downloads the recording from Twilio, runs the existing VoiceAge AI model service, and returns JSON containing the recording URL, predicted age group, confidence, and timestamp.

Prediction logs are saved as JSON files in:

```text
data/twilio_predictions/
```

Signature validation is enabled when `TWILIO_AUTH_TOKEN` is set and the `twilio` package is installed. Keep the Console webhook URL exactly aligned with `PUBLIC_BASE_URL` so Twilio signature validation receives the same public URL that Twilio signed.

## Phase 17: Twilio + OpenAI Realtime Conversation Integration

This adds a separate live phone conversation flow using Twilio Media Streams and the OpenAI Realtime API. The existing recording-based Twilio webhook remains available at `/api/twilio/voice`; no training or model retraining is required.

Set these environment variables before running the API:

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_REALTIME_ENDPOINT="https://admin-mf2e0nkt-eastus2.cognitiveservices.azure.com/openai/realtime"
export AZURE_OPENAI_REALTIME_DEPLOYMENT="gpt-realtime-mini"
export AZURE_OPENAI_API_VERSION="2024-10-01-preview"
export PUBLIC_BASE_URL="https://your-public-ngrok-or-host-url"
export REALTIME_VOICE="alloy"
export REALTIME_VAD_THRESHOLD="0.4"
export REALTIME_VAD_SILENCE_MS="400"
export REALTIME_VAD_PREFIX_MS="200"
```

The API connects to Azure OpenAI Realtime over:

```text
wss://admin-mf2e0nkt-eastus2.cognitiveservices.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-realtime-mini
```

Azure authentication uses the `api-key` WebSocket header from `AZURE_OPENAI_API_KEY`. The legacy `OPENAI_API_KEY` and `OPENAI_REALTIME_MODEL` settings remain available for non-Azure OpenAI Realtime usage when Azure settings are not present.

Twilio Console setup for live realtime conversation:

```text
POST https://<ngrok-url>/api/twilio/realtime-voice
```

The realtime webhook returns TwiML with `<Connect><Stream>` and sends Twilio audio to:

```text
wss://<ngrok-url-without-https>/api/realtime/twilio-stream
```

Twilio inbound caller audio is forwarded as mulaw 8000 Hz audio to OpenAI Realtime, and OpenAI audio deltas are streamed back to Twilio in the same format. Conversation events, transcripts, caller-only audio, and optional assistant audio are saved under:

```text
data/realtime_conversations/
```

Each realtime call folder can include:

```text
caller_only_audio.ulaw
caller_only_audio.wav
assistant_audio.ulaw
latency_metrics.json
```

Realtime latency metrics are saved per call at:

```text
data/realtime_conversations/<call_id>/latency_metrics.json
```

## Why caller-only audio is used for VoiceAge

VoiceAge predicts age from the speaker's voice. In realtime phone calls, the saved conversation can include both the caller and the AI assistant. If assistant speech is included in the audio sent to VoiceAge, the model may classify the assistant voice instead of the caller.

For realtime Twilio calls, VoiceAge therefore uses only `caller_only_audio.wav`, built from inbound Twilio media frames. Azure/OpenAI assistant `response.audio.delta` frames are excluded from this file and may be saved separately as `assistant_audio.ulaw` for debugging. Reports never fall back to mixed conversation audio for VoiceAge prediction.

If `caller_only_audio.wav` is missing or shorter than 3 seconds, the report marks `voiceage_prediction_success=false` with `failure_reason=caller_only_audio_missing_or_too_short`.

## Phase 19: Post-Call Reports

Every realtime Twilio call now generates readable post-call reports after the stream stops or disconnects. Reports are written to:

```text
data/realtime_conversations/<call_id>/reports/
```

Each call folder contains:

```text
voiceage_report.json
voiceage_report.md
conversation_report.json
conversation_report.md
combined_call_report.json
combined_call_report.md
```

The VoiceAge report runs the existing `models/wav2vec_50k/best` model against `caller_only_audio.wav` and includes the predicted age group, confidence, class probabilities, model version, audio source, caller audio duration, and evaluation reference of 81.37% accuracy / 81.29% weighted F1. The report explains that the model predicts an age group from voice, not an exact age, and states that the AI assistant voice was excluded.

The conversation report summarizes transcripts when available, user/assistant turn counts, call duration, latency metrics, and any errors or warnings found in the realtime event log. If transcript or latency data is missing, the reports are still generated and state what was unavailable.

The combined call report is written for non-technical review. It includes an executive summary, system success checks for VoiceAge, Twilio Media Streams, and Azure/OpenAI Realtime, call quality notes, and a recommendation of `Ready for demo`, `Partial success`, or `Needs debugging`.

To regenerate reports for an older call folder:

```bash
python scripts/generate_call_report.py --call-dir data/realtime_conversations/<call_id>
```

The realtime bridge logs this message when report generation finishes:

```text
Post-call reports generated: <path>
```

## Notebook Demo: Background Voice Isolation

A standalone demo notebook is available at:

```text
notebooks/background_voice_isolation_demo.ipynb
```

Run it from the repository root:

```bash
jupyter notebook notebooks/background_voice_isolation_demo.ipynb
```

The notebook loads a saved `caller_full_audio.wav`, runs Silero VAD plus ECAPA-TDNN speaker verification to produce `caller_full_background_isolated.wav`, optionally runs the existing Wav2Vec2 age prediction flow on the isolated audio, and writes `background_voice_isolation_demo_report.json`.
