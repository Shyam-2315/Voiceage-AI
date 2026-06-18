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
   - Add speaker verification using SpeechBrain ECAPA-TDNN.
   - Compare a caller reference voiceprint against test speech segments.
   - Report cosine similarity and same-speaker decisions.

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
├── run_phase3_check.py
├── speaker_verification.py
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

## Phase 3: Speaker Verification

Phase 3 uses SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) to compare whether two speech clips likely belong to the same speaker. The intended workflow is to keep a clean reference clip for the main caller, then compare speech segments against that caller voiceprint.

### Why VAD Is Not Enough

Phase 2 VAD separates speech from non-speech. It cannot tell whether detected speech is the main caller, a dispatcher, TV audio, a bystander, or another background speaker. Speaker verification adds a second decision layer by comparing voice characteristics between a known caller reference and a test segment.

### Caller Voiceprint

The reference file, usually `sample_audio/caller_reference.wav`, should contain clear speech from the main caller. SpeechBrain converts that reference audio into an embedding, which is a numerical speaker representation. Test segments are converted into embeddings the same way.

### Cosine Similarity

The POC compares the reference embedding and test embedding with cosine similarity:

- Scores closer to `1.0` mean the embeddings point in a similar direction and are more likely to be the same speaker.
- Lower scores indicate weaker speaker similarity.
- The default same-speaker threshold is `0.75`.

### Add Test Audio

Place these WAV files in `sample_audio/`:

```text
sample_audio/caller_reference.wav
sample_audio/test_segment.wav
```

### Run Phase 3

From this folder:

```bash
python -m compileall .
python -m unittest discover -s tests
python run_phase3_check.py
```

If the sample WAV files are missing, the check exits cleanly and prints:

```text
Add caller_reference.wav and test_segment.wav into sample_audio/
```

To run directly:

```bash
python speaker_verification.py --reference sample_audio/caller_reference.wav --test sample_audio/test_segment.wav
```

Optional threshold override:

```bash
python speaker_verification.py --reference sample_audio/caller_reference.wav --test sample_audio/test_segment.wav --threshold 0.8
```

### Expected Output

```text
reports/speaker_verification_report.json
```

The console output includes the similarity score, same-speaker decision, and threshold used.

### Threshold Tuning

The `0.75` threshold is an initial POC default, not a production value. It should be tuned with representative caller, background speaker, device, noise, and call-quality samples before integration decisions are made.

### Limitations

- Speaker verification needs a clean enough reference clip for the main caller.
- Short clips, overlapping speakers, background noise, codecs, and emotional speech can affect embedding quality.
- This verifies similarity to a reference speaker; it does not perform full diarization by itself.
- The model may download runtime files into `outputs/` the first time it loads.
