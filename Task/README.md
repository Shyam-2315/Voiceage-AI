# Task Realtime Transcription

This Task-local copy of `test.py` runs the existing FastAPI/Twilio call workflow and transcribes completed call recordings with Azure OpenAI Realtime using the `gpt-realtime-mini` deployment.

## Setup

```bash
cd ~/Projects/voiceage-ai/Task
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in this `Task` folder or export these variables before running:

```bash
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_REALTIME_DEPLOYMENT=gpt-realtime-mini
AZURE_OPENAI_API_VERSION=your_azure_openai_api_version

TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
PHONE_NUMBER_FROM=your_twilio_phone_number
PHONE_NUMBERS=+15551234567
DOMAIN=your-public-domain.example
```

Optional Realtime tuning:

```bash
REALTIME_AUDIO_SAMPLE_RATE=24000
REALTIME_AUDIO_CHUNK_SIZE=12000
PORT=6060
QUESTIONS_FILE=questions.xlsx
```

## Run

```bash
cd ~/Projects/voiceage-ai/Task
python test.py
```

`test.py` starts the FastAPI app with Uvicorn. When Twilio sends a completed recording callback, the script downloads the recording, saves it as `logs/audio_<call_sid>.wav`, transcribes it with Azure OpenAI Realtime, prints `Transcript: ...`, writes `logs/transcript_<call_sid>.txt`, and then creates the existing structured JSON report.

## Expected Output

Console output keeps the existing style, including messages such as:

```text
Processing recording for call: <call_sid>
Transcript: <transcribed text>
LOG: Processing transcript...
SUCCESS: JSON generated
Final JSON: {...}
```

If `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, or `AZURE_OPENAI_API_VERSION` is missing, the script exits with a clear error telling you what to set in `.env`.

## Limitations

The recording transcription path converts PCM WAV audio to mono PCM16 in memory before sending it to Realtime. If Twilio returns a non-PCM WAV encoding, conversion will fail with an `STT Error` message and the existing retry/failure path will mark the transcript as failed.
