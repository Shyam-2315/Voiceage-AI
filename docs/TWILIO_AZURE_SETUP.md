# Twilio and Azure Setup

## Twilio

1. Set the phone number voice webhook to `POST https://<client-domain>/api/twilio/realtime-voice`.
2. Set `PUBLIC_BASE_URL=https://<client-domain>`.
3. Set `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`.
4. Confirm the public URL supports HTTPS and WebSocket upgrade traffic.

For the recording-only flow, use `POST https://<client-domain>/api/twilio/voice`.

## Azure OpenAI Realtime

Set:

```text
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_REALTIME_ENDPOINT=
AZURE_OPENAI_REALTIME_DEPLOYMENT=gpt-realtime-mini
AZURE_OPENAI_API_VERSION=2024-10-01-preview
REALTIME_VOICE=alloy
REALTIME_VAD_THRESHOLD=0.55
REALTIME_VAD_SILENCE_MS=600
REALTIME_VAD_PREFIX_MS=200
```

The application starts without Twilio or Azure environment variables. Prediction endpoints remain available, and Twilio/realtime endpoints return clear errors when required runtime settings are missing.
