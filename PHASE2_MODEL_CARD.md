# Phase 2 Model Card

## Intended use

Educational and experimental comparison of text patterns learned from particular labeled datasets. The detector is not a reliable authorship verifier and must not be used as proof that a person or model wrote a text.

## Models

- Original baseline: word/character TF-IDF with calibrated Linear SVM, trained on selected HC3-derived QA data.
- Original transformer: DistilBERT fine-tuned on HC3-derived data.
- Expanded candidates: separately stored TF-IDF SVM and continued DistilBERT trained on HC3+MAGE data. They do not overwrite the original checkpoints.
- The live local API reported `distilbert_hc3_v1` at the 2026-11-25 investigation. The code default without `TEXT_MODEL_BACKEND` is the baseline SVM; deployments must inspect `/api/v1/status` to establish the actually loaded backend.

## Limitations and evidence

Performance changes materially by source, domain, text length, and AI generator. The expanded DistilBERT's saved test macro-F1 is 0.8524 on the mixed HC3+MAGE test but its human false-positive rate is 0.2577; on frozen MAGE GPT-4 OOD, human FPR is 0.4580. Its validation abstention thresholds collapse to [0,1]. A user-shared AI-generated image description was scored as 99.74% HUMAN_WRITTEN by the active HC3 DistilBERT and 99.88% HUMAN_WRITTEN by the HC3 SVM. Expanded candidates returned UNCERTAIN rather than AI_GENERATED on that example.

Score fields have model-specific semantics: calibrated SVM probability estimates, raw HC3 DistilBERT softmax, or validation-temperature-scaled expanded DistilBERT softmax. Calibration may not transfer across domains. A score is not accuracy or a guaranteed probability of authorship.

## Deployment decision

No expanded candidate was promoted. Further development requires a licensed, diverse, source-aware caption dataset with independently held-out generator/domain tests and human negatives that discuss AI. Existing OOD test data must remain untouched for model fitting and threshold selection. See [PHASE2_IMPROVEMENT_REPORT.md](PHASE2_IMPROVEMENT_REPORT.md) and [PHASE2_RETRAINING_REPORT.md](PHASE2_RETRAINING_REPORT.md).
