# Background Voice Isolation POC

## Goal

This standalone proof of concept explores background noise and background speaker isolation for VoiceAge AI without changing the main application pipeline.

The POC is intentionally isolated under `poc/background_voice_isolation` so experiments, generated files, and reports can be developed safely before any production integration is considered.

## Background Voice Problem

Voice recordings can contain multiple overlapping sound sources:

- Primary speaker audio that should be preserved.
- Stationary or changing background noise such as fans, traffic, room tone, or device noise.
- Background speakers whose speech may overlap with or distract from the target speaker.

This POC focuses on evaluating practical ways to separate or suppress non-target audio while preserving intelligibility and speaker characteristics for the foreground voice.

## Planned Phases

1. **Phase 1: Environment and structure check**
   - Create the isolated project structure.
   - Define initial dependencies.
   - Verify required folders and imports.

2. **Phase 2: Sample audio preparation**
   - Add representative sample clips.
   - Detect speech regions with Silero VAD.
   - Export speech-only audio and a JSON segment report.

3. **Phase 3: Baseline noise reduction**
   - Test simple spectral and signal-processing approaches.
   - Generate initial output artifacts and quality notes.

4. **Phase 4: Background speaker isolation experiments**
   - Evaluate source separation or speaker-aware approaches.
   - Compare outputs against baseline noise reduction.

5. **Phase 5: Reporting and recommendation**
   - Summarize results, limitations, and risks.
   - Recommend whether any approach is suitable for integration work.

## Folder Structure

```text
poc/background_voice_isolation/
├── sample_audio/
├── outputs/
├── reports/
├── src/
├── tests/
├── README.md
├── requirements.txt
├── run_phase1_check.py
├── run_phase2_check.py
└── vad_test.py
```

## Running Phase Checks

From the POC folder:

```bash
cd ~/Projects/voiceage-ai/poc/background_voice_isolation
python -m compileall .
python run_phase1_check.py
```

The phase check validates the Python version, confirms the required POC folders exist, imports the initial dependencies, and prints a success message when everything passes.

## Phase 2: Silero VAD Speech Detection

Phase 2 uses Silero Voice Activity Detection (VAD) to find regions that contain speech in an input WAV file. The script loads audio, converts it to mono, resamples it to 16 kHz, runs VAD, merges nearby speech segments, and writes a speech-only WAV made by concatenating the detected speech regions.

This detects speech activity, not speaker identity. If a background speaker is loud enough, Silero VAD can mark that speech as speech too. Later phases should evaluate speaker-aware separation or diarization if the POC needs to distinguish the foreground speaker from other speakers.

### Add Sample Audio

Place a WAV file in:

```text
sample_audio/
```

Example:

```text
sample_audio/test.wav
```

### Run Phase 2

From this folder:

```bash
python -m compileall .
python run_phase2_check.py
```

If no sample WAV exists, the check exits cleanly and prints:

```text
Add a WAV file into sample_audio/ and run: python vad_test.py sample_audio/<file>.wav
```

To run directly on a specific file:

```bash
python vad_test.py sample_audio/test.wav
```

### Expected Outputs

```text
outputs/speech_only.wav
reports/vad_report.json
```

The console output includes input duration, speech duration, number of segments, and speech percentage.

### Limitations

- VAD detects speech versus non-speech; it does not identify which speaker is talking.
- Overlapping speakers may be exported together as speech.
- Very noisy, reverberant, clipped, or low-volume recordings can reduce detection quality.
- The speech-only WAV concatenates detected regions, so original timing gaps are removed.
