import os
import json
import asyncio
import base64
import io
import wave
from urllib.parse import urlencode
from fastapi import FastAPI, WebSocket, Request
from fastapi.websockets import WebSocketDisconnect
from twilio.rest import Client
import websockets
from dotenv import load_dotenv
import uvicorn
import re
import pandas as pd
import datetime
from openai import AzureOpenAI
import aiosqlite
from contextlib import asynccontextmanager
import httpx
from fastapi.responses import PlainTextResponse


load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PHONE_NUMBER_FROM = os.getenv("PHONE_NUMBER_FROM")
QUESTIONS_FILE = os.getenv("QUESTIONS_FILE", "questions.xlsx")
# PHONE_NUMBER_TO = os.getenv("PHONE_NUMBER_TO")
PHONE_NUMBERS = os.getenv("PHONE_NUMBERS", "")
PHONE_NUMBERS_LIST = [num.strip() for num in PHONE_NUMBERS.split(",") if num.strip()]

raw_domain = os.getenv("DOMAIN", "")
DOMAIN = re.sub(r"(^\w+:|^)\/\/|\/+$", "", raw_domain)
PORT = int(os.getenv("PORT", 6060))

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_REALTIME_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_REALTIME_DEPLOYMENT", "gpt-realtime-mini"
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
REALTIME_AUDIO_SAMPLE_RATE = int(os.getenv("REALTIME_AUDIO_SAMPLE_RATE", 24000))
REALTIME_AUDIO_CHUNK_SIZE = int(os.getenv("REALTIME_AUDIO_CHUNK_SIZE", 12000))

DB_PATH = "call_data.db"

if not AZURE_OPENAI_API_KEY:
    raise ValueError(
        "Missing AZURE_OPENAI_API_KEY. Set AZURE_OPENAI_API_KEY in your .env file before running test.py."
    )

if not AZURE_OPENAI_ENDPOINT:
    raise ValueError(
        "Missing AZURE_OPENAI_ENDPOINT. Set AZURE_OPENAI_ENDPOINT in your .env file before running test.py."
    )

if not AZURE_OPENAI_API_VERSION:
    raise ValueError(
        "Missing AZURE_OPENAI_API_VERSION. Set AZURE_OPENAI_API_VERSION in your .env file before running test.py."
    )

json_azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    await init_db()

    if not PHONE_NUMBERS_LIST:
        print("ERROR: PHONE_NUMBERS is not set.")
    else:
        print(f"Calling {len(PHONE_NUMBERS_LIST)} numbers...")

        asyncio.get_running_loop().call_later(
            2, lambda: asyncio.create_task(make_bulk_calls(PHONE_NUMBERS_LIST))
        )
    yield
    # Shutdown logic
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """CREATE TABLE IF NOT EXISTS calls (
                call_sid TEXT PRIMARY KEY,
                stream_sid TEXT,
                start_time TIMESTAMP,

                recording_url TEXT,
                recording_duration INTEGER,

                transcript TEXT,
                transcript_status TEXT,   -- pending / completed / failed

                structured_data TEXT,
                processing_status TEXT    -- pending / completed / failed
            );"""
        )


async def make_bulk_calls(phone_numbers):
    for number in phone_numbers:
        asyncio.create_task(make_call(number))
        await asyncio.sleep(0.5)


async def mark_transcript_failed(call_sid):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE calls
            SET transcript_status = 'failed'
            WHERE call_sid = ?
            """,
            (call_sid,),
        )
        await db.commit()


async def mark_processing_failed(call_sid):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE calls
            SET processing_status = 'failed'
            WHERE call_sid = ?
            """,
            (call_sid,),
        )
        await db.commit()


async def save_processing_results(call_sid, transcript, structured):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE calls
            SET 
                transcript = ?,
                transcript_status = ?,
                structured_data = ?,
                processing_status = ?
            WHERE call_sid = ?
            """,
            (
                transcript,
                "completed",
                json.dumps(structured),
                "completed",
                call_sid,
            ),
        )
        await db.commit()


async def save_recording(call_sid, recording_url, recording_duration):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE calls
            SET recording_url = ?, recording_duration = ?
            WHERE call_sid = ?
            """,
            (recording_url, recording_duration, call_sid),
        )
        await db.commit()


async def save_call_start(call_sid):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO calls (
                call_sid,
                start_time,
                transcript_status,
                processing_status
            ) VALUES (?, ?, ?, ?)
            """,
            (
                call_sid,
                datetime.datetime.now(),
                "pending",
                "pending",
            ),
        )
        await db.commit()


def load_questions(file_path):
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported File Format, Use CSV or EXCEL")

        questions_list = df.iloc[:, 0].dropna().tolist()

        formatted_questions = "\n".join(
            [f"{i + 1}.{q}" for i, q in enumerate(questions_list)]
        )

        return formatted_questions

    except Exception as e:
        print(f"Error loading Questions' File: {e}")
        return ""


async def process_recording(recording_url, call_sid):
    print(f"Processing recording for call: {call_sid}")

    audio_url = (
        recording_url if recording_url.endswith(".wav") else recording_url + ".wav"
    )

    async with httpx.AsyncClient(
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30.0
    ) as client:
        audio_response = await client.get(audio_url)

    if audio_response.status_code != 200:
        print("Failed to download recording:", audio_response.status_code)
        await mark_transcript_failed(call_sid)
        return

    audio_bytes = audio_response.content
    os.makedirs("logs", exist_ok=True)

    with open(f"logs/audio_{call_sid}.wav", "wb") as f:
        f.write(audio_bytes)
    # transcript = await transcribe_audio(audio_bytes)
    for _ in range(2):
        transcript = await transcribe_audio(audio_bytes)
        if transcript.strip():
            break

    if not transcript.strip():
        print("Empty transcript. Skipping processing.")
        await mark_transcript_failed(call_sid)
        return

    print("Transcript:", transcript)
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/transcript_{call_sid}.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    structured = await generate_structured_json_from_text(transcript)

    if not structured:
        await mark_processing_failed(call_sid)
        return

    await save_processing_results(call_sid, transcript, structured)

    print("Final JSON:", structured)


def _pcm_sample_to_int16(sample_bytes: bytes) -> int:
    sample_width = len(sample_bytes)

    if sample_width == 1:
        return (sample_bytes[0] - 128) << 8

    value = int.from_bytes(sample_bytes, "little", signed=True)
    if sample_width > 2:
        value >>= (sample_width - 2) * 8
    elif sample_width < 2:
        value <<= (2 - sample_width) * 8

    return max(-32768, min(32767, value))


def _pcm_frames_to_mono_samples(
    frames: bytes, channels: int, sample_width: int
) -> list[int]:
    samples = []
    frame_width = channels * sample_width

    for frame_start in range(0, len(frames), frame_width):
        frame = frames[frame_start : frame_start + frame_width]
        if len(frame) < frame_width:
            break

        channel_total = 0
        for channel_index in range(channels):
            sample_start = channel_index * sample_width
            sample_end = sample_start + sample_width
            channel_total += _pcm_sample_to_int16(frame[sample_start:sample_end])

        samples.append(int(channel_total / channels))

    return samples


def _resample_mono_samples(
    samples: list[int], source_rate: int, target_rate: int
) -> list[int]:
    if source_rate == target_rate or not samples:
        return samples

    output_length = max(1, round(len(samples) * target_rate / source_rate))
    resampled = []

    for output_index in range(output_length):
        source_position = output_index * source_rate / target_rate
        lower_index = int(source_position)
        upper_index = min(lower_index + 1, len(samples) - 1)
        fraction = source_position - lower_index
        interpolated = (
            samples[lower_index] * (1 - fraction) + samples[upper_index] * fraction
        )
        resampled.append(int(interpolated))

    return resampled


def _int16_samples_to_bytes(samples: list[int]) -> bytes:
    pcm16 = bytearray()
    for sample in samples:
        clamped = max(-32768, min(32767, int(sample)))
        pcm16.extend(clamped.to_bytes(2, "little", signed=True))
    return bytes(pcm16)


def _wav_bytes_to_pcm16(audio_bytes: bytes) -> bytes:
    """Convert a PCM WAV recording to mono PCM16 for Realtime input."""

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise ValueError(
            "Realtime transcription expects a PCM WAV recording from Twilio. "
            "Received audio could not be read as PCM WAV."
        ) from exc

    samples = _pcm_frames_to_mono_samples(frames, channels, sample_width)
    samples = _resample_mono_samples(
        samples,
        source_rate=frame_rate,
        target_rate=REALTIME_AUDIO_SAMPLE_RATE,
    )

    return _int16_samples_to_bytes(samples)


def _extract_text_from_response_done(response: dict) -> str:
    text_parts = []
    for output_item in response.get("response", {}).get("output", []):
        for content in output_item.get("content", []):
            text = (
                content.get("text")
                or content.get("transcript")
                or content.get("output_text")
            )
            if text:
                text_parts.append(text)
    return " ".join(text_parts).strip()


def _azure_realtime_ws_url() -> str:
    endpoint = AZURE_OPENAI_ENDPOINT.rstrip("/")
    if endpoint.startswith("https://"):
        endpoint = "wss://" + endpoint[len("https://") :]
    elif endpoint.startswith("http://"):
        endpoint = "ws://" + endpoint[len("http://") :]
    elif not endpoint.startswith(("ws://", "wss://")):
        endpoint = "wss://" + endpoint

    query = urlencode(
        {
            "api-version": AZURE_OPENAI_API_VERSION,
            "deployment": AZURE_OPENAI_REALTIME_DEPLOYMENT,
        }
    )
    return f"{endpoint}/openai/realtime?{query}"


async def _transcribe_with_realtime(pcm16_audio: bytes) -> str:
    transcript_parts = []
    completed_transcript = ""
    headers = {"api-key": AZURE_OPENAI_API_KEY}

    async with websockets.connect(
        _azure_realtime_ws_url(),
        additional_headers=headers,
        max_size=None,
    ) as openai_ws:
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_format": "pcm16",
                "instructions": (
                    "Transcribe the user's audio exactly. Return only the transcript text, "
                    "with no markdown, labels, or commentary."
                ),
            },
        }
        await openai_ws.send(json.dumps(session_update))

        for start in range(0, len(pcm16_audio), REALTIME_AUDIO_CHUNK_SIZE):
            audio_chunk = pcm16_audio[start : start + REALTIME_AUDIO_CHUNK_SIZE]
            await openai_ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    }
                )
            )

        await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await openai_ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text"],
                        "instructions": (
                            "Transcribe the committed audio exactly. Return only the transcript text."
                        ),
                    },
                }
            )
        )

        async for message in openai_ws:
            response = json.loads(message)
            event_type = response.get("type")

            if event_type == "error":
                error = response.get("error", {})
                message = error.get("message") or response
                raise RuntimeError(f"Realtime transcription error: {message}")

            if event_type in {
                "response.text.delta",
                "response.output_text.delta",
                "response.audio_transcript.delta",
            }:
                transcript_parts.append(response.get("delta", ""))

            elif event_type in {
                "response.text.done",
                "response.output_text.done",
                "response.audio_transcript.done",
                "conversation.item.input_audio_transcription.completed",
            }:
                transcript = response.get("text") or response.get("transcript")
                if transcript and not transcript_parts:
                    completed_transcript = transcript

            elif event_type == "response.done":
                response_text = _extract_text_from_response_done(response)
                if response_text and not transcript_parts and not completed_transcript:
                    completed_transcript = response_text
                break

        # Fallback option for the future: if Realtime exposes a dedicated completed
        # transcription event for this model, handle it above and keep this return shape.
        return ("".join(transcript_parts) or completed_transcript).strip()


async def transcribe_audio(audio_bytes):
    try:
        pcm16_audio = await asyncio.to_thread(_wav_bytes_to_pcm16, audio_bytes)
        return await _transcribe_with_realtime(pcm16_audio)
    except Exception as e:
        print("STT Error:", e)
        return ""


async def generate_structured_json_from_text(transcript_text: str):
    """Takes transcript text, extracts structured medical survey JSON."""

    prompt = f"""
    You are a high-accuracy medical survey data extractor. 

    CONTEXT: 
    The transcript you are processing is a call recording between an AI Interviewer (who asks the survey questions) and a Human Respondent (who answers the questions).

    TASK:
    Identify the distinct speakers, extract the HUMAN'S answers from the transcript, and map EACH question asked by the AI to the MOST APPROPRIATE option using semantic understanding.

    ---

    CORE PRINCIPLE:
    - ALWAYS try to map an answer to one of the available options.
    - Only return "N/A" if absolutely no usable answer exists.
    - If a human can reasonably interpret an answer → you MUST map it.
    - The options will available in the AI interviewer's question part in the transcript. Use them as reference for mapping but do not rely on exact wording.

    ---

    STEP 1: SPEAKER IDENTIFICATION & QUESTION–ANSWER ALIGNMENT (CRITICAL)

    - The transcript contains a conversation between an AI and a human. Transcripts may lack explicit speaker labels, so you must rely on conversational context to determine who is speaking.
    - You MUST group the human's answers by the AI's questions.

    For EACH question:
    1. Detect when the AI Interviewer asks a question from the survey.
    2. Collect ALL subsequent speech from the HUMAN Respondent until the AI asks the NEXT question.
    3. Treat ALL of the human's speech in that specific window as candidate answers for that question.
    4. Some questions might get repeated or rephrased, but they should all be grouped together and treated as one question for mapping purposes. Use the latestet answer if multiple answers exist for the same question. Always prioritize meaning and intent over exact wording. If an answer can be reasonably interpreted and mapped to one of the valid options, you MUST do so.

    IMPORTANT:
    - Retries, rephrasings, and repeated questions belong to the SAME question.
    - Do NOT mix answers across questions.

    ---

    STEP 2: FILTERING

    From the candidate answers (the human's speech):
    - REMOVE:
    - meaningless text ("asdasd", "yes", "okay")
    - irrelevant replies
    - unrelated conversation
    - non-understandable responses

    - KEEP:
    - any answer with interpretable meaning
    - even weak or informal responses

    ---

    STEP 3: SEMANTIC EVALUATION

    - Interpret meaning, tone, and intent (NOT exact words).
    - Map natural language confidently:

    Examples of Mapping:
    - "amazing", "phenomenal", "perfect" → Excellent
    - "pretty good", "doing well" → Very Good / Good
    - "okay", "average", "middle" → Fair
    - "not bad" → Good or Fair
    - "bad", "struggling" → Poor

    Abilities:
    - "I can do everything easily" → Completely
    - "I manage most things" → Mostly
    - "I struggle sometimes" → Moderately / A Little

    Frequency:
    - "never" / "almost never" → Never / Rarely
    - "sometimes" → Sometimes
    - "always" → Always

    Fatigue:
    - "a bit tired" → Mild
    - "very tired", "exhausted" → Severe / Very Severe

    Pain:
    - Extract number if present
    - "worst imaginable" → 10
    - Clamp strictly between 0–10

    ---

    STEP 4: BEST ANSWER SELECTION

    - A question may have multiple valid answers from the human.
    - Always pick/interpret the last valid answer if multiple answers exist for the same question, as it is likely the most complete or clarified response.

    ---

    STEP 5: FINAL DECISION

    - If at least ONE valid human answer exists → map it
    - If ALL human answers are garbage or empty → return "N/A"

    ---

    STRICT RULES:

    - Prefer mapping over rejecting if answer is interpretable in any way.
    - Do NOT be overly strict
    - Do NOT invent answers
    - Do NOT mix questions
    - Do NOT output explanations

    ---

    OUTPUT FORMAT (STRICT JSON ONLY):

    {{
        "general_health": "",
        "quality_of_life": "",
        "physical_health": "",
        "mental_health": "",
        "social_satisfaction": "",
        "social_roles": "",
        "physical_activities": "",
        "emotional_problems": "",
        "fatigue": "",
        "pain_scale": ""
    }}

    ---

    Transcript:
    {transcript_text}
    """

    try:
        print("LOG: Processing transcript...")

        response = await asyncio.to_thread(
            json_azure_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data extraction assistant. Output ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        structured_json_str = response.choices[0].message.content
        structured_data = json.loads(structured_json_str)

        # Optional: save locally
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/report_{int(datetime.datetime.now().timestamp())}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=4)

        print("SUCCESS: JSON generated")
        return structured_data

    except Exception as e:
        print(f"ERROR during JSON conversion: {e}")
        return None


BASE_SYSTEM_PROMPT = (
    "You are a health survey interviewer conducting a phone interview whose only job is to collect data.\n"
    "Your tone should always feel human and professional — never robotic.\n\n"
    # "- Start EXACTLY with:\n"
    # "  'Hello, thank you for taking this call. I'm conducting a short health survey on behalf of Dr.Steve.'\n"
    "PRIMARY TASK:\n"
    "- Ask exactly these questions in order, one at a time and strictly along with options.\n"
    "- Do not skip or add questions unless the retry or exit rules apply.\n\n"
    "SESSION EXIT DETECTION:\n"
    "- Recognize exit intent if the user says anything like:\n"
    "  * 'Stop', 'end', 'quit', 'I'm done'.\n"
    "  * 'I can't answer right now / today'.\n"
    "  * 'I need to go', 'I'm busy', 'call me later'.\n"
    "  * 'I don't want to continue'.\n"
    "  * Any similar expression of wanting to pause or leave.\n"
    "- When detected, say exactly:\n"
    "  'Of course, I completely understand. Thank you so much for your time today. "
    "Please don't hesitate to reach out when you're ready. "
    "Take care of yourself. Goodbye!'\n"
    "- Then end the session immediately. Do not ask any more questions.\n\n"
    "ANSWER HANDLING (HIGHEST PRIORITY RULE):\n"
    "- Always Try to interpret the user's answer based on meaning and intent, not exact wording.\n"
    "- When accepting, ask yourself:\n"
    "  'Can this answer be approximately mapped to one of the question's options by meaning?'\n"
    "- If YES → ACCEPT and IMMEDIATELY move to the next question.\n"
    "- If NO → REJECT and apply the Retry Rule.\n"
    "- Use common sense. If a human interviewer would understand it, accept it.\n"
    "- Below are examples of how to interpret answers (but do not rely solely on these examples, use them as a guide for semantic understanding):\n"
    "- 'I am doing really well' → maps to Very Good or Good → ACCEPT immediately and ask NEXT question.\n"
    "- 'I feel great' → maps to Excellent → ACCEPT immediately and ask NEXT question.\n"
    "- Likewise, evaluate all types of answers and map them to the appropriate options.\n"
    "- NEVER ask the user to confirm or specify their answer after accepting.\n"
    "- NEVER repeat the options back to the user after they have answered.\n"
    "- Do not accept pure deflections like 'I don't know' or 'whatever'\n"
    "RETRY RULE (Max 3 Attempts Per Question):\n"
    "- Attempt 1: Ask the question normally.\n"
    "- Attempt 2: Say 'I'm sorry, I didn't quite get a valid response. "
    "Let me try again' → then re-ask.\n"
    "- Attempt 3: Say 'I apologize for the trouble. Let me try one last time.' → then re-ask.\n"
    "- After 3 failures: Say 'No problem at all, let's move on.' → skip to next question.\n\n"
    "STRICT BEHAVIOR:\n"
    "- NEVER ASK THE USER TO PICK ANYONE OPTION FOR THE SAME QUESTION JUST IMMEDIATELY MOVE ON TO THE NEXT QUESTION IF USER'S ANSWER IS INTERPRETABLE AND SEMANTICALLY ACCEPTABLE.\n"
    "- NEVER continue questioning if the user signals emotional distress or exit intent.\n"
    "- NEVER ask follow-up questions beyond what's listed.\n"
    "- NEVER be cold, clinical, or dismissive.\n"
    "- Maintain a warm, calm, and professional tone at all times.\n\n"
    "ENDING (Normal Completion):\n"
    "- After the 10th valid answer, say exactly:\n"
    "'Thank you so much for taking the time to complete this survey. Take care and goodbye!'\n\n"
    "THE QUESTIONS:\n"
)

LOADED_QUESTIONS = load_questions(QUESTIONS_FILE)
SYSTEM_MESSAGE = f"{BASE_SYSTEM_PROMPT}{LOADED_QUESTIONS}"
print(SYSTEM_MESSAGE)
VOICE = "alloy"
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
LOG_EVENT_TYPES = [
    "error",
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
]


active_calls = {}
active_calls_lock = asyncio.Lock()

if not (
    TWILIO_ACCOUNT_SID
    and TWILIO_AUTH_TOKEN
    and PHONE_NUMBER_FROM
    and AZURE_OPENAI_API_KEY
):
    raise ValueError(
        "Missing Twilio and/or Azure OpenAI environment variables. Please set them in the .env file."
    )

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# async def log_to_db(sid, speaker, text):
#     """Appends transcript line directly to the calls table."""
#     formatted_line = f"[{speaker}]: {text}\n"

#     async with aiosqlite.connect(DB_PATH) as db:
#         # 1. Insert the call record if it doesn't exist, initializing full_transcript as empty string
#         await db.execute(
#             "INSERT OR IGNORE INTO calls (stream_sid, start_time, transcript) VALUES (?, ?, ?)",
#             (sid, datetime.datetime.now(), ""),
#         )

#         # 2. Append the new message to the existing full_transcript string
#         await db.execute(
#             "UPDATE calls SET transcript = transcript || ? WHERE stream_sid = ?",
#             (formatted_line, sid),
#         )
#         await db.commit()


@app.post("/recording-callback")
async def recording_callback(request: Request):
    form = await request.form()

    recording_url = form.get("RecordingUrl")
    recording_duration = int(form.get("RecordingDuration") or 0)
    call_sid = form.get("CallSid")

    if not recording_url:
        print("No recording URL received")
        return

    print(f"Processing recording for call: {call_sid}")
    print("Recording ready:", recording_url)
    print("Duration:", recording_duration)

    await save_recording(call_sid, recording_url, recording_duration)

    asyncio.create_task(process_recording(recording_url, call_sid))

    return PlainTextResponse("OK", status_code=200)


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        _azure_realtime_ws_url(),
        additional_headers={"api-key": AZURE_OPENAI_API_KEY},
    ) as openai_ws:
        await initialize_session(openai_ws)
        stream_sid = None  # load unique id

        async def receive_from_twilio():
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data["event"] == "media" and openai_ws.state.name == "OPEN":
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data["media"]["payload"],
                        }
                        await openai_ws.send(json.dumps(audio_append))

                    # capture unique id
                    elif data["event"] == "start":
                        stream_sid = data["start"]["streamSid"]

                        print("\n--- START EVENT ---")
                        print("FULL START DATA:", data["start"])
                        print(
                            "RAW customParameters:",
                            data["start"].get("customParameters"),
                        )

                        call_sid = data["start"].get("callSid")

                        print("EXTRACTED call_sid:", call_sid)
                        print("EXTRACTED stream_sid:", stream_sid)

                        if not call_sid:
                            print("❌ ERROR: call_sid missing in stream")
                            return

                        async with active_calls_lock:
                            active_calls[call_sid] = stream_sid

                        print(
                            "DB UPDATE TRY → call_sid:",
                            call_sid,
                            "stream_sid:",
                            stream_sid,
                        )

                        async with aiosqlite.connect(DB_PATH) as db:
                            await db.execute(
                                "UPDATE calls SET stream_sid = ? WHERE call_sid = ?",
                                (stream_sid, call_sid),
                            )
                            await db.commit()

                            # 🔥 THIS IS THE IMPORTANT CHECK
                            cursor = await db.execute(
                                "SELECT stream_sid FROM calls WHERE call_sid = ?",
                                (call_sid,),
                            )
                            row = await cursor.fetchone()
                            print("DB VALUE AFTER UPDATE:", row)

            except WebSocketDisconnect:
                print(f"Client disconnected: {stream_sid}")
                if openai_ws.state.name == "OPEN":
                    await openai_ws.close()

        async def send_to_twilio():
            nonlocal stream_sid
            should_hangup = False  # Flag to trigger hangup after audio finishes

            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)

                    # ====================================================================
                    # NEW: HANDLE INTERRUPTION (BARGE-IN)
                    # When Azure OpenAI's VAD detects the user speaking, tell Twilio to 
                    # instantly stop playing the current audio buffer.
                    # ====================================================================
                    # if response["type"] == "input_audio_buffer.speech_started":
                    #     logger.info("User interrupted! Clearing Twilio audio buffer.")
                    #     clear_event = {
                    #         "event": "clear",
                    #         "streamSid": stream_sid
                    #     }
                    #     await websocket.send_json(clear_event)
                    # # ====================================================================

                    # # ====================================================================
                    # # YOUR PROPOSAL: DROP OLD AUDIO ONLY WHEN NEW AI RESPONSE STARTS
                    # # ====================================================================
                    # if response["type"] == "response.created":
                    #     print("AI generated a new response! Clearing Twilio's old audio buffer.")
                    #     clear_event = {
                    #         "event": "clear",
                    #         "streamSid": stream_sid
                    #     }
                    #     await websocket.send_json(clear_event)
                    # # ====================================================================

                    # 1. LOG THE AI ASSISTANT (And check for goodbye phrases)
                    if response["type"] == "response.audio_transcript.done":
                        ai_text = response.get("transcript", "")
                        print(f"\n[AI Assistant]: {ai_text}")
                        # await log_to_db(stream_sid, "AI Assistant", ai_text)

                        # Check if the AI just said goodbye
                        if any(
                            phrase in ai_text.lower()
                            for phrase in [
                                # "thank you for your time",
                                # "that concludes",
                                "goodbye",
                                "take care",
                            ]
                        ):
                            print("Closing phrase detected. Preparing to hang up...")
                            should_hangup = True

                    # 3. RELAY AUDIO DELTAS TO TWILIO
                    if response["type"] == "response.audio.delta" and response.get(
                        "delta"
                    ):
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": response["delta"]},
                        }
                        await websocket.send_json(audio_delta)

                    # 4. GRACEFUL HANGUP
                    # response.done fires after the AI has finished its current turn
                    if response["type"] == "response.done" and should_hangup:
                        print("Final audio completed. Hanging up call.")
                        await asyncio.sleep(10)  # Buffer time to ensure audio is fully sent
                        if stream_sid:
                            await hang_up_call(stream_sid)
                        break

            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        try:
            receive_task = asyncio.create_task(receive_from_twilio())
            send_task = asyncio.create_task(send_to_twilio())

            await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel whichever task is still running in the background
            for task in [receive_task, send_task]:
                if not task.done():
                    task.cancel()

        except Exception as e:
            print(f"Error Starting the tasks: {e}")


async def hang_up_call(stream_sid: str):
    async with active_calls_lock:
        for call_sid, s_sid in list(active_calls.items()):
            if s_sid == stream_sid:
                try:
                    await asyncio.to_thread(
                        client.calls(call_sid).update, status="completed"
                    )
                    print(f"Call {call_sid} hung up successfully.")
                    del active_calls[call_sid]
                    break
                except Exception as e:
                    print(f"Error hanging up call: {e}")


async def send_initial_conversation_item(openai_ws):
    """Send initial conversation so AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "The user is on the line.Start immediately by asking Question 1 now. Dont say anything else."
                    ),
                }
            ],
        },
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI using the correct 2026 schema."""
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "voice": VOICE,
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "instructions": SYSTEM_MESSAGE,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.90,  # Increase to 0.8 or 0.9 to ignore minor background noises
                "prefix_padding_ms": 300,  # Amount of audio to include before speech is detected
                "silence_duration_ms": 500,  # How long the user must be quiet before AI responds
            },
        },
    }
    print("Sending session update:", json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Have the AI speak first
    await send_initial_conversation_item(openai_ws)


# async def check_number_allowed(to):
#     """Check if a number is allowed to be called."""
#     try:
#         incoming_numbers = client.incoming_phone_numbers.list(phone_number=to)
#         if incoming_numbers:
#             return True

#         outgoing_caller_ids = client.outgoing_caller_ids.list(phone_number=to)
#         if outgoing_caller_ids:
#             return True

#         return False
#     except Exception as e:
#         print(f"Error checking phone number: {e}")
#         return False


async def make_call(phone_number_to_call: str):
    """Make an outbound call."""
    if not phone_number_to_call:
        raise ValueError("Please provide a phone number to call.")

    # is_allowed = await check_number_allowed(phone_number_to_call)
    # if not is_allowed:
    #     raise ValueError(
    #         f"The number {phone_number_to_call} is not recognized as a valid outgoing number or caller ID."
    #     )

    outbound_twiml = f"""
    <Response>
        <Say voice="Google.en-US-Journey-D">Hello, thank you for taking this call. I'm conducting a brief health survey on behalf of Dr. Steve. Please listen to the full question and all answer options before responding. Thank you.</Say>
        <Connect>
            <Stream url="wss://{DOMAIN}/media-stream">
                <Parameter name="call_sid" value="{{{{CallSid}}}}" />
            </Stream>
        </Connect>
    </Response>
    """

    call = client.calls.create(
        from_=PHONE_NUMBER_FROM,
        to=phone_number_to_call,
        twiml=outbound_twiml,
        record=True,
        recording_status_callback=f"https://{DOMAIN}/recording-callback",
        recording_status_callback_event=["completed"],
    )

    print("DB INSERT CALL_SID:", call.sid)

    await save_call_start(call.sid)
    await log_call_sid(call.sid)


async def log_call_sid(call_sid):
    """Log the call SID."""
    print(f"Call started with SID: {call_sid}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
