# Phase 2 Text Detector Improvement Investigation

Date: 2026-11-25  
Status: **INCOMPLETE — reliability issue reproduced; no safe checkpoint promotion**

## Scope and safeguards

Inspected the Phase 2 inference services and checkpoint metadata, the live unified API status, training and evaluation reports, dataset manifests, and the text result renderer. No training, calibration, threshold fitting, data download, or model-weight change was performed. The Phase 1/2 snapshot verifier returned `hashes_unchanged: true` and `predictions_unchanged: true` before and after the change. The only application code changed was text-only result-card wording in `unified/static/index.html`.

## Findings

The live unified API at `127.0.0.1:8000` reported Phase 2 backend `transformer`, checkpoint `distilbert_hc3_v1`. In `backend/app/main.py`, the code default when `TEXT_MODEL_BACKEND` is absent is `baseline`; therefore the running process has a selector override or was started with a different runtime configuration. Phase 3 calls this same Phase 2 detector and forwards its output; the rule-based Phase 3 combiner does not recalculate the text label.

Label ordering is internally consistent: both DistilBERT metadata files map `HUMAN_WRITTEN` to index 0 and `AI_GENERATED` to index 1, and the transformer service reads index 1 as the AI score. The SVM service uses its fitted class ordering. No preprocessing or checkpoint-load mismatch was found in the inspected paths.

The explicit AI-generated visual-description example previously shared by the user was scored by each installed model:

| Checkpoint | AI score | API label |
|---|---:|---|
| HC3 TF-IDF SVM | 0.0012 | HUMAN_WRITTEN |
| HC3 DistilBERT (active at inspection) | 0.0026 | HUMAN_WRITTEN |
| Expanded TF-IDF SVM | 0.1292 | UNCERTAIN |
| Expanded DistilBERT | 0.1071 | UNCERTAIN |

The input explicitly says it was created using generative AI, yet the trained classifiers' learned scores do not reflect that statement. This is a genuine model/generalization failure; changing the label mapping would be incorrect. The cause is consistent with a source/domain mismatch between the HC3 QA training data and prompt-like image-description prose, but these scores alone do not prove which stylistic or lexical features caused the failure.

Two other user-shared examples were also rechecked as single-input diagnostics, not as accuracy measurements. The generic business/AI paragraph received AI scores 0.9830 from HC3 SVM and 0.9706 from HC3 DistilBERT; the former returned AI_GENERATED, while the latter returned UNCERTAIN because its configured AI threshold is 0.9983. The “Climate change…” sample received HC3 SVM AI score 0.7886 (AI_GENERATED) and HC3 DistilBERT AI score 0.9201 (UNCERTAIN). Thus the current abstention cutoff also turns a strong model-side AI score into UNCERTAIN. The expanded models return UNCERTAIN because their fitted thresholds are `[0, 1]`.

The exact five captions referenced in the attached prompt were not present as five separate records in the workspace, so a five-example retest is not claimed. The three examples above were already shared in the conversation and were used only for diagnostics, not for training, selection, or evaluation metrics.

## Existing aggregate evidence

These are previously saved results; they were not recalculated or refit during this investigation.

| Model | Expanded held-out test accuracy | Macro-F1 | Human FPR |
|---|---:|---:|---:|
| HC3 SVM | 0.5591 | 0.5323 | 0.1098 |
| Expanded SVM | 0.7503 | 0.7502 | 0.1810 |
| HC3 DistilBERT | 0.5986 | 0.5792 | 0.0934 |
| Expanded DistilBERT | 0.8579 | 0.8524 | 0.2577 |

The expanded test comparison uses the same 5,727 HC3+MAGE held-out rows for each checkpoint; it is not a fully source-independent test. On frozen MAGE GPT-4 OOD data, expanded DistilBERT macro-F1 was 0.7476 with human FPR 0.4580. Its 5%-error validation abstention thresholds collapsed to 0 and 1, so its API returns UNCERTAIN for every ordinary score. This does not meet a useful deployment criterion. Promoting it would replace confidently wrong cases with universal abstention, while retaining a large false-positive risk under the fixed 0.5 evaluation rule.

Primary saved result: `training/reports/text_model_comparison_expanded_test.json`. OOD result: `training/reports/text_expanded_ood_diagnostics.json`. Training and dataset detail: `PHASE2_RETRAINING_REPORT.md`.

## Changes made

- Updated the unified UI's text cards to identify the class toward which the model score points, show a top class for UNCERTAIN outputs, explain that UNCERTAIN means neither decision threshold was crossed, and clarify that a model score is not accuracy or proof of authorship.
- The image result-card branch is unchanged. No model/API inference logic, threshold, preprocessing, checkpoint, or Phase 1 file was changed.
- No linguistic observations are asserted because the model does not provide measured linguistic-feature explanations.

## Validation

- `.venv\Scripts\python.exe -m pytest phase3\tests unified\tests backend\tests -q`: **84 passed**, one non-failing PyTorch sparse-invariant warning.
- `.venv\Scripts\python.exe phase3\tools\snapshot_originals.py --verify`: `hashes_unchanged: true`, `predictions_unchanged: true`.
- Exact shared text examples were scored locally against four existing checkpoints. This is targeted reproduction only and is not an estimate of accuracy.
- Browser verification submitted the exact 483-character user-shared AI description. The live UI still returned HUMAN at 99.7%, while the updated card showed “Model score: 99.7% toward HUMAN,” identified `uncalibrated softmax`, and displayed the authorship/accuracy disclaimer. This confirms the presentation change and the unresolved model error. One normal local history record was created; raw input retention is disabled by default. No training was performed.

## Recommended next improvement

Do not lower thresholds or add a keyword rule to force the known caption into AI_GENERATED. The appropriate next experiment is a source-aware, legally usable dataset containing both AI-generated and human-written image descriptions across multiple generators, authors, topics, and styles. Keep document/source groups apart; train only on the training split, select/calibrate on validation, and preserve independent generator/domain test sets. Include human-written descriptions that discuss AI to measure false positives. The current locked OOD evaluation files must remain evaluation-only.

Until that evidence exists, the detector remains experimental and the reported false negative is unresolved. Phase 2 is not complete or generally reliable.
