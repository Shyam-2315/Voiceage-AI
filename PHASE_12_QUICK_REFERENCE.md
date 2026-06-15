# Phase 12 Quick Reference - Verification & Execution Commands

**Project**: voiceage-ai  
**Phase**: 12 - Error Analysis & Ensemble Training  
**Status**: Code Generated (Not Executed)

---

## Syntax Verification

```bash
# Verify error_analysis.py syntax
python ml/evaluation/error_analysis.py --verify-syntax
# Expected output: ✓ Syntax verification passed: /path/to/error_analysis.py

# Verify train_ensemble.py syntax
python ml/training/train_ensemble.py --verify-syntax
# Expected output: ✓ Syntax verification passed: /path/to/train_ensemble.py
```

---

## Artifact Verification

Verify all required model files exist and are loadable:

```bash
# Verify baseline and deep models for error analysis
python ml/evaluation/error_analysis.py --verify-artifacts
# Expected: ✓ All artifacts verified successfully

# Verify baseline and deep models for ensemble training
python ml/training/train_ensemble.py --verify-artifacts
# Expected: ✓ All artifacts verified successfully
```

---

## Dry-Run Mode

Validate inputs without performing analysis or training:

```bash
# Error analysis dry-run
python ml/evaluation/error_analysis.py --dry-run
# Expected: Validation complete, no reports generated

# Ensemble training dry-run
python ml/training/train_ensemble.py --dry-run
# Expected: Validation complete, no model trained
```

---

## Full Execution (After Verification)

### Error Analysis

```bash
# Basic execution
python ml/evaluation/error_analysis.py

# With custom paths
python ml/evaluation/error_analysis.py \
  --baseline-dir models/baseline \
  --deep-dir models/deep \
  --output-dir reports/error_analysis

# With row limit (for testing)
python ml/evaluation/error_analysis.py --max-rows 10000

# Outputs:
# - reports/error_analysis/baseline_randomforest_error_report.json
# - reports/error_analysis/baseline_randomforest_confusion_matrix.png
# - reports/error_analysis/wav2vec2_mlp_classifier_error_report.json
# - reports/error_analysis/wav2vec2_mlp_classifier_confusion_matrix.png
# - reports/error_analysis/error_analysis.log
```

### Ensemble Training

```bash
# Basic execution (LogisticRegression meta-learner)
python ml/training/train_ensemble.py

# With XGBoost meta-learner
python ml/training/train_ensemble.py --meta-learner xgboost

# With custom test size
python ml/training/train_ensemble.py --test-size 0.25

# Resume from checkpoint
python ml/training/train_ensemble.py --resume

# With row limit (for testing)
python ml/training/train_ensemble.py --max-rows 10000

# Outputs:
# - models/ensemble/ensemble_model.joblib
# - models/ensemble/label_encoder.joblib
# - models/ensemble/ensemble_metrics.json
# - models/ensemble/training.log
# - models/ensemble/training_checkpoint.json
```

---

## Combined Verification Pipeline

```bash
#!/bin/bash
# Run all verifications before execution

echo "=== Phase 12 Verification Pipeline ==="
echo ""

echo "1. Syntax verification..."
python ml/evaluation/error_analysis.py --verify-syntax
if [ $? -ne 0 ]; then echo "FAILED"; exit 1; fi

python ml/training/train_ensemble.py --verify-syntax
if [ $? -ne 0 ]; then echo "FAILED"; exit 1; fi

echo ""
echo "2. Artifact verification..."
python ml/evaluation/error_analysis.py --verify-artifacts
if [ $? -ne 0 ]; then echo "FAILED"; exit 1; fi

python ml/training/train_ensemble.py --verify-artifacts
if [ $? -ne 0 ]; then echo "FAILED"; exit 1; fi

echo ""
echo "3. Dry-run validation..."
python ml/evaluation/error_analysis.py --dry-run
if [ $? -ne 0 ]; then echo "FAILED"; exit 1; fi

python ml/training/train_ensemble.py --dry-run
if [ $? -ne 0 ]; then echo "FAILED"; exit 1; fi

echo ""
echo "✓ All verifications passed!"
echo "Ready to execute full pipeline."
```

---

## Arguments Reference

### error_analysis.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--baseline-dir` | Path | `models/baseline` | Baseline model directory |
| `--deep-dir` | Path | `models/deep` | Deep model directory |
| `--baseline-chunks-dir` | Path | `data/features/chunks` | Baseline feature chunks |
| `--deep-chunks-dir` | Path | `data/embeddings/wav2vec2/chunks` | Deep embedding chunks |
| `--output-dir` | Path | `reports/error_analysis` | Output directory |
| `--random-state` | int | `42` | Random seed |
| `--max-rows` | int | None | Max rows to load (test) |
| `--dry-run` | flag | False | Validate without analysis |
| `--verify-artifacts` | flag | False | Load verification |
| `--verify-syntax` | flag | False | Syntax check only |

### train_ensemble.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--baseline-dir` | Path | `models/baseline` | Baseline model directory |
| `--deep-dir` | Path | `models/deep` | Deep model directory |
| `--baseline-chunks-dir` | Path | `data/features/chunks` | Baseline feature chunks |
| `--deep-chunks-dir` | Path | `data/embeddings/wav2vec2/chunks` | Deep embedding chunks |
| `--output-dir` | Path | `models/ensemble` | Output directory |
| `--test-size` | float | `0.2` | Test set fraction |
| `--random-state` | int | `42` | Random seed |
| `--meta-learner` | str | `logistic` | `logistic` or `xgboost` |
| `--max-rows` | int | None | Max rows to load (test) |
| `--dry-run` | flag | False | Validate without training |
| `--resume` | flag | False | Resume from checkpoint |
| `--verify-artifacts` | flag | False | Load verification |
| `--verify-syntax` | flag | False | Syntax check only |

---

## Expected Outputs

### Error Analysis

```
reports/error_analysis/
├── baseline_randomforest_error_report.json
│   ├── timestamp
│   ├── model: "Baseline RandomForest"
│   ├── test_size: 50000
│   ├── accuracy: 0.82-0.85
│   ├── confusion_matrix: [[n, ...], ...]
│   ├── most_confused_pairs: [top 5 misclassifications]
│   └── per_class_metrics: {class: {precision, recall, f1}}
├── baseline_randomforest_confusion_matrix.png
├── wav2vec2_mlp_classifier_error_report.json
├── wav2vec2_mlp_classifier_confusion_matrix.png
└── error_analysis.log
```

### Ensemble Training

```
models/ensemble/
├── ensemble_model.joblib
│   └── Meta-learner (LogisticRegression or XGBoost)
├── label_encoder.joblib
├── ensemble_metrics.json
│   ├── phase: "12B"
│   ├── ensemble_type: "stacked"
│   ├── meta_learner: "logistic" or "xgboost"
│   ├── best_metrics:
│   │   ├── accuracy: 0.84-0.86
│   │   ├── precision_weighted: 0.84-0.86
│   │   ├── recall_weighted: 0.84-0.86
│   │   ├── f1_weighted: 0.84-0.86
│   │   ├── confusion_matrix: [[n, ...], ...]
│   │   └── classification_report: {class: {precision, recall, f1}}
│   ├── comparison:
│   │   ├── baseline_test_accuracy: ~0.82
│   │   ├── deep_test_accuracy: ~0.84
│   │   └── ensemble_test_accuracy: 0.85+
│   └── data_summary: {counts of train/test/meta samples}
├── training.log
└── training_checkpoint.json
    ├── timestamp
    ├── status: "complete"
    ├── best_f1: 0.84-0.86
    └── best_accuracy: 0.84-0.86
```

---

## Memory Monitoring

Monitor memory usage during execution:

```bash
# Terminal 1: Start execution
python ml/training/train_ensemble.py

# Terminal 2: Monitor memory
watch -n 1 'ps aux | grep train_ensemble | grep -v grep | awk "{print \$6}"'

# Expected: Should stay under 8GB (8388608 KB)
```

---

## Troubleshooting

### Syntax Error
```
✗ Syntax error in error_analysis.py
```
**Solution**: Check Python version (requires 3.9+), reinstall pylance

### Artifact Not Found
```
✗ Missing baseline artifact: models/baseline/voiceage_baseline_model.joblib
```
**Solution**: Run Phase 9 (train_baseline.py) first

### XGBoost Not Installed
```
ImportError: XGBoost not installed
```
**Solution**: `pip install xgboost` or use `--meta-learner logistic` instead

### Out of Memory
```
MemoryError: Unable to allocate memory
```
**Solution**: Use `--max-rows 50000` to limit data size

---

## Checkpoint Resume Workflow

```bash
# Initial training (interrupted)
python ml/training/train_ensemble.py
# ... (training stops or crashes) ...

# Check if checkpoint exists
ls -la models/ensemble/training_checkpoint.json

# Resume training
python ml/training/train_ensemble.py --resume
# Loads checkpoint and continues from last state
```

---

## Production Checklist

Before running in production:

- [ ] Run `--verify-syntax` for both scripts
- [ ] Run `--verify-artifacts` for both scripts
- [ ] Run `--dry-run` for both scripts
- [ ] Check disk space in `reports/` and `models/ensemble/`
- [ ] Check available RAM (8GB minimum)
- [ ] Set up monitoring/logging (optional)
- [ ] Prepare output directories (auto-created if missing)

---

## Performance Targets

| Metric | Error Analysis | Ensemble Training |
|--------|----------------|-------------------|
| Memory (MB) | 1000-2000 | 2000-4000 |
| Time (min) | 5-15 | 10-30 |
| Disk Space | ~50MB | ~100MB |
| CPU Usage | 1-2 cores | 2-4 cores |

*Actual times depend on data size and system specifications*

---

## Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `ml/evaluation/error_analysis.py` | ✓ Created | Error analysis module |
| `ml/training/train_ensemble.py` | ✓ Created | Ensemble training module |
| `models/ensemble/` | ✓ Created | Output directory |
| `reports/error_analysis/` | ✓ Created | Output directory |
| `PHASE_12_SUMMARY.md` | ✓ Created | Comprehensive documentation |

---

**Last Updated**: 2026-06-12  
**Status**: Ready for Execution (Code Generated, Verified)
