# Deployment

## Environment

Copy `.env.example` to the deployment environment and set real values there. Do not commit `.env`.

Required for prediction API:

- `MODEL_PATH=models/wav2vec_75k/best`

Required for Twilio webhooks:

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `PUBLIC_BASE_URL`

Required for realtime assistant:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_REALTIME_ENDPOINT`
- `AZURE_OPENAI_REALTIME_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

## Run Locally

```bash
scripts/run_api.sh
```

Defaults:

- Host: `0.0.0.0`
- Port: `8765`

Override with:

```bash
HOST=127.0.0.1 PORT=8765 scripts/run_api.sh
```

## Readiness Check

```bash
python scripts/check_system.py
```

The check verifies Python packages, CUDA availability, model path, and Twilio/Azure environment presence.
