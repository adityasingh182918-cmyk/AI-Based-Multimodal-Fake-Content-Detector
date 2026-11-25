# Phase 2 changelog

## 2026-11-25 — Dataset expansion and versioned retraining (in progress)

- Audited current HC3 splits, original SVM/DistilBERT checkpoints, API services, overflow-window fix, reports, environment, CUDA GPU, disk, and tests. Existing weights and data remain untouched.
- Added a pinned deterministic streaming MAGE sampler and an expanded-dataset preparer. Selected only official MAGE train/validation/test splits; the GPT-4 OOD/paraphrase, GRADTEX, and Beemo evaluation artifacts remain frozen.
- Prepared a balanced, source-stratified HC3+MAGE dataset and verified labels, exact duplicates, and split leakage with the existing validator. Removed exact overlaps with frozen OOD examples; documented MAGE's missing original document IDs and related leakage uncertainty.
- Added a validation-only expanded SVM candidate trainer (word, character, combined features; Logistic Regression and SVM C search) and same-example model comparison utility. Trained `svm_expanded_v1.joblib` separately; old model files were preserved.
- Extended the existing DistilBERT trainer with an optional local initialization checkpoint; default HC3 behavior and existing DistilBERT checkpoint path are unchanged. New expanded checkpoints use separate versioned paths.
- Added three focused data-tool tests. Continued DistilBERT training separately from the existing HC3 checkpoint; saved only to `models/text_detector/distilbert_expanded_v1/`.
- Evaluated all four frozen checkpoints on the same 5,727 expanded held-out test examples. Expanded SVM reached accuracy/macro-F1 0.7503/0.7502; expanded DistilBERT 0.8579/0.8524, with human false-positive rate 0.2577. Full per-source, per-length, class, score and confusion matrix details are in the test comparison report.
- Evaluated new models on frozen MAGE GPT-4 OOD, GRADTEX Test B/C, and Beemo expert-edited AI-only samples. Expanded DistilBERT has higher AI recall but elevated human false positives; Beemo is a positive-only sensitivity probe and has no two-class metrics.
- Fitted temperature on expanded validation only. DistilBERT T=2.2867 improved Brier, NLL and ECE on that split and is recorded in the new checkpoint metadata; SVM temperature was not deployed. Both models' strict abstention cutoffs collapse to [0,1], so all examples are UNCERTAIN under that rule.
- Added explicit API backend selection for the two expanded candidates and version/score metadata. Original HC3 SVM remains the default, and Phase 1 endpoints/weights are unchanged.
- Complete backend suite passed: 58 tests in 13.48 seconds. Python compile check passed. Browser viewport and accessibility QA remains unverified; Phase 2 remains in progress due generalization and dataset limitations.

## 2026-11-24 — Initial runnable text baseline

- Added pinned, selective HC3 downloader for `reddit_eli5` and `wiki_csai` and dataset/license documentation.
- Added cleaning, equal answer pairing, exact-deduplication, reproducible question-group splits, and a leakage validator.
- Added TF-IDF Logistic Regression and calibrated Linear SVM training/evaluation tools; selected model using validation macro-F1.
- Added a lazy text inference service, text API/status endpoints, request schema, and three-mode image/text/independent combined UI.
- Added an optional DistilBERT fine-tuning/evaluation pipeline and transformer inference backend, selected by explicit environment configuration.
- Added text dataset, service, and API tests. Image route/source remains in place.
- Phase 3 fusion remains unimplemented.

## 2026-11-24 — Independent robustness evidence

- Added explicit, pinned downloader for MAGE's small GPT-4 OOD and optional paraphrase test CSVs; no training data is fetched.
- Added frozen-checkpoint robustness evaluator with correct MAGE label mapping, domain slices, validation-derived abstention reporting, and single-class sensitivity-only handling.
- Ran and recorded separate Reddit-to-Wikipedia HC3 source holdout, MAGE four-domain OOD test, and GPT-4 paraphrased AI sensitivity slice. Metrics and limitations are documented in the experiment log and dataset documentation.
- Added regression tests for MAGE's `0=AI, 1=human` mapping and undefined single-class metrics.
- No model training, checkpoint replacement, test-based tuning, or multimodal fusion was performed in this continuation.

## 2026-11-24 — Robustness completion and final validation pass

- Reconciled the progress tracker with actual checkpoint/report files and corrected the old `42 tests` claim after the final suite expanded.
- Reviewed and downloaded only the 64 MB GRADTEX test parquet and 8.3 MB Beemo parquet, pinned to immutable revisions. Added explicit dataset loaders, source/model licensing filters, and exact normalized overlap audits against HC3 splits and evaluated MAGE examples.
- Evaluated existing SVM and DistilBERT checkpoints on GRADTEX Test C, paired Test B generation/paraphrase/polish scenarios, and Beemo human-expert-edited AI positives. Added per-domain/source/scenario/length slices, class metrics, FPR/FNR, and single-class-safe reporting. No training data or checkpoints changed.
- Added validation-only temperature/reliability analysis and recorded test results from the validation-selected temperatures without deploying them or changing thresholds.
- Added explicit label and keyboard-focus treatment to the text/image UI, plus integration tests for the combined independent API calls and long-text chunking.
- Live API checks passed for Phase 1 image inference, Phase 2 short/long text handling, and concurrent independent image/text responses. The 48-test backend suite and compile check passed.
- Actual browser viewport/accessibility validation remains incomplete: the computer-use helper denied control of Brave, and no headless browser runtime was installed. Pangram's small Grammarly dataset requires gated authentication and returned 401; no credentials or consent were supplied.
- Phase 3 trained fusion remains unimplemented. No model training or checkpoint overwrite occurred.

## 2026-11-24 — Real-world failure investigation

- Confirmed the default API backend is the HC3 TF-IDF SVM; DistilBERT requires explicit selection. Verified both saved model label mappings and frontend request shapes.
- Corrected DistilBERT response metadata from calibrated-probability wording to `uncalibrated_softmax` and clarified the warning. No score/threshold/checkpoint change.
- Existing OOD evidence indicates weak AI sensitivity on MAGE, GRADTEX, and edited Beemo probes. The user's example is not yet in the workspace, so no example-specific result is claimed.
- Added the private sample analysis workflow and text failure report. Preserved all checkpoints and did not train or tune on independent test data.

## 2026-11-24 — Additional text model and length diagnostics

- Added API-matched per-example comparison of frozen SVM and DistilBERT on the same MAGE, GRADTEX, and Beemo rows, with class scores, SVM margins, false-positive/false-negative IDs, length/source/domain/scenario slices, and unchanged validation threshold decisions. Raw text is not persisted.
- Distinguished calibrated SVM probabilities, raw LinearSVC margins, and uncalibrated DistilBERT softmax. Kept Beemo positive-only metrics sensitivity-only.
- Reproduced a DistilBERT overflow-window tensor error on examples with a shorter final window. Added padding to inference and a long-text regression test; attention masks continue to exclude pad tokens from aggregation weights.
- The user has not supplied the AI-generated text example; that specific failure remains unreproduced. No model weights or thresholds changed.
# 2026-11-25 — User-reported AI-caption false negative

- Audited live unified status, Phase 2 model selection, label mappings, checkpoint metadata, preprocessing, and saved same-split/OOD reports. The running app selected the original `transformer` backend (`distilbert_hc3_v1`); the selector in code uses an environment override when present. Phase 3 forwards the same Phase 2 result and does not reclassify text.
- Reproduced the available user-shared text examples against all four existing models. On the explicitly AI-labeled visual description, HC3 SVM scored AI 0.0012 and HC3 DistilBERT 0.0026; expanded SVM and expanded DistilBERT scored AI 0.1292 and 0.1071, with both expanded APIs returning UNCERTAIN under their validation thresholds. This is a real model/data-domain failure, not a label mapping correction.
- No threshold change, retraining, checkpoint promotion, or image inference modification was made. The expanded DistilBERT's same-test macro-F1 0.8524 does not outweigh human FPR 0.2577 and frozen MAGE OOD human FPR 0.4580; its validation abstention interval collapses to [0,1].
- Updated only text result-card wording in `unified/static/index.html`: shows the top class score direction, explains threshold uncertainty, and states scores are not accuracy/authorship proof. Image-result UI is unchanged.
- Validation: full regression suite **84 passed**, one non-failing PyTorch warning; Phase 1/2 snapshot hash and fixed-prediction checks passed. Five-caption retest is incomplete because those five exact texts are not available in the workspace; three previously shared examples were rechecked for diagnostic purposes only.
- Report: `PHASE2_IMPROVEMENT_REPORT.md`.

