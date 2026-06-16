# VoiceAge AI API Contract

Base URL: `https://<client-domain>` in production or `http://localhost:8765` locally.

## GET /health

Returns service and model readiness.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "wav2vec_75k",
  "model_path": "/app/models/wav2vec_75k/best",
  "device": "cuda"
}
```

## POST /api/predict-age

Uploads caller audio and returns an age-group prediction.

- Content type: `multipart/form-data`
- Field: `file`
- Supported extensions: `wav`, `mp3`, `m4a`

```bash
curl -X POST http://localhost:8765/api/predict-age \
  -F "file=@sample.wav"
```

```json
{
  "predicted_age_group": "Adult",
  "confidence": 0.91,
  "confidence_level": "high",
  "class_probabilities": {
    "Adult": 0.91,
    "Middle_Age": 0.04,
    "Senior": 0.03,
    "Teen": 0.02
  },
  "model_version": "wav2vec_75k",
  "processing_time_ms": 184
}
```

## POST /api/twilio/voice

Twilio voice webhook for the recording flow. It returns TwiML that asks the caller to record a short sample and posts the recording callback to `/api/twilio/recording-complete`.

Response content type: `application/xml`.

## POST /api/twilio/realtime-voice

Twilio voice webhook for the realtime assistant flow. It returns TwiML that connects the call to the Media Streams WebSocket.

Requires `PUBLIC_BASE_URL` to be set to the public HTTPS base URL.

Response content type: `application/xml`.

## WebSocket /api/realtime/twilio-stream

Twilio Media Streams connects here. The bridge forwards inbound caller audio to Azure/OpenAI Realtime and sends assistant audio back to Twilio.

VoiceAge prediction uses caller-only inbound audio. Assistant audio is captured separately and excluded from age prediction.

## Report Outputs

Realtime call artifacts are stored under:

```text
data/realtime_conversations/<call_sid>/
```

Expected files include:

- `events.jsonl`
- `metadata.json`
- `latency_metrics.json`
- `caller_only_audio.wav`
- `caller_only_audio.ulaw`
- `assistant_audio.ulaw`
- `voiceage_live_prediction.json`
- `reports/voiceage_report.json`
- `reports/conversation_report.json`
- `reports/combined_call_report.json`

Twilio recording predictions are stored under:

```text
data/twilio_predictions/
```
