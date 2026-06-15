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

Twilio caller audio is forwarded as mulaw 8000 Hz audio to OpenAI Realtime, and OpenAI audio deltas are streamed back to Twilio in the same format. Conversation events, transcripts, and the first caller audio capture for later VoiceAge prediction are saved under:

```text
data/realtime_conversations/
```
