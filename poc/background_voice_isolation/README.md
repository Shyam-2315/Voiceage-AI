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
   - Define expected input and output formats.
   - Create repeatable preprocessing utilities.

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
└── run_phase1_check.py
```

## Running Phase Checks

From the POC folder:

```bash
cd ~/Projects/voiceage-ai/poc/background_voice_isolation
python -m compileall .
python run_phase1_check.py
```

The phase check validates the Python version, confirms the required POC folders exist, imports the initial dependencies, and prints a success message when everything passes.
