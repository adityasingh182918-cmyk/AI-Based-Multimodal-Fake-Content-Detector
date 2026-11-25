# Phase 2 Training Report

This is a short index to the completed HC3 baseline and HC3+MAGE expansion runs. No training was run during the 2026-11-25 user-caption investigation because the available candidate checkpoints did not meet safe deployment criteria.

## Existing training runs

- HC3 word/character TF-IDF Logistic Regression and calibrated Linear SVM: [PHASE2_RETRAINING_REPORT.md](PHASE2_RETRAINING_REPORT.md) and [PHASE2_RETRAINING_PLAN.md](PHASE2_RETRAINING_PLAN.md).
- HC3-finetuned DistilBERT: checkpoint `models/text_detector/transformer/distilbert_hc3_v1/`.
- Continued HC3+MAGE DistilBERT: checkpoint `models/text_detector/distilbert_expanded_v1/`; selected epoch 3 by validation macro-F1. It was not promoted to the live backend.
- Expanded TF-IDF SVM: `models/text_detector/svm_expanded_v1.joblib`; stored separately from the original baseline.

## Current status

The expanded DistilBERT scored 0.8524 macro-F1 on the saved HC3+MAGE held-out test but had human false-positive rate 0.2577 there and 0.4580 on frozen MAGE GPT-4 OOD data. Its validation abstention thresholds are 0 and 1, which makes the API return UNCERTAIN for every input. No candidate is promoted. The current live API reported the original HC3 DistilBERT checkpoint at inspection time.

No newly generated training history, checkpoint, or metric is claimed in this report. See [PHASE2_IMPROVEMENT_REPORT.md](PHASE2_IMPROVEMENT_REPORT.md) for the reproduced user example and next experiment recommendation.
