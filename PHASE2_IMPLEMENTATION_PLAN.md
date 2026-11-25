# Phase 2 implementation plan

This plan tracks the first runnable text baseline increment. `COMPLETED` means the described deliverable exists and was verified; it does not imply Phase 2 acceptance as a whole.

| Stage | Status | Scope |
|---|---|---|
| 0. Existing project audit | COMPLETED | Current trackers, saved reports/checkpoints, Python/CUDA, free disk, API/UI code, tests, and documentation re-inspected; tracker discrepancies reconciled. |
| 1. Dataset source and acquisition | COMPLETED | Pin public HC3 revision; fetch only documented `reddit_eli5` and `wiki_csai` files; document licenses and limits. |
| 2. Cleaning and splits | COMPLETED | Pair labels per question; normalize whitespace; exact dedupe; reproducible group-held-out 70/15/15 split; validate leakage. |
| 3. Baselines | COMPLETED | Compare word+character TF-IDF Logistic Regression and calibrated Linear SVM; choose by validation macro-F1. |
| 4. Transformer selection | COMPLETED | Pinned DistilBERT base uncased, Apache-2.0, compatible with Python 3.13 and 6 GB RTX 4050. |
| 5. Transformer training | COMPLETED | Dynamic padding, fp16, batch 8, accumulation 2, validation macro-F1 checkpoint selection; epoch 3 saved. |
| 6. Independent robustness | IN_PROGRESS | Original HC3-era robustness complete; expanded checkpoints now evaluated on frozen MAGE GPT-4 OOD, GRADTEX, and Beemo. Human grammar-correction corpus and stronger group-leakage checks remain unavailable. |
| 7. Abstention calibration | COMPLETED | HC3 calibration study and expanded validation-only temperature scaling recorded. Expanded DistilBERT uses T=2.2867; 5%-error abstention cutoffs collapse to [0,1] for both expanded models, so all results are UNCERTAIN under that rule. |
| 8. Text API | COMPLETED | Text endpoint, validation, chunking, lazy loading, model/backend identity and expanded version selection; confidence semantics identify raw or temperature-scaled scores. |
| 9. Frontend | IN_PROGRESS | Image/text/combined modes inspected; accessible labels and keyboard file selection improved; live API smoke passed. Browser-control helper denied Brave and no headless browser is installed, so visual/responsive/accessibility QA remains unverified. |
| 10. Integration | COMPLETED | Live image and text APIs independently returned results concurrently; the UI invokes and renders the two results separately. No trained fusion model. |
| 11. Automated backend tests | COMPLETED | Full suite passed: 58 tests; browser-level rendering checks remain tracked under Stage 9. |
| 12. Docs/finalization | COMPLETED | Dataset sources, actual metrics, calibration findings, commands, limitations, and recovery tracking updated. |

## Stage 6 evidence (2026-11-24)

- Source holdout: SVM trained on Reddit ELI5 and tested only on 263 Wikipedia HC3 rows; macro-F1 0.5327, accuracy 0.6236, human recall 0.20. Report: `training/reports/text_source_holdout_evaluation.json`.
- MAGE GPT-4 OOD: 1,562 records across CNN/DailyMail, DialogSum, IMDb, and PubMedQA. Frozen SVM macro-F1 0.6367 / AI recall 0.4238; frozen DistilBERT macro-F1 0.7144 / AI recall 0.5963. Report: `training/reports/text_mage_ood_evaluation.json`.
- MAGE GPT-4 paraphrased AI slice: 800 positive-only items; SVM AI recall 0.4713 and DistilBERT AI recall 0.6963. No two-class metrics are valid for this slice. Report: `training/reports/text_mage_paraphrase_ai_evaluation.json`.
- The MAGE files were not used for training, model selection, or threshold selection. Their dataset card declares Apache-2.0; this project downloads only the two small test CSVs on explicit command.
- GRADTEX Test C (3,529 HWT/MIX rows; held-out transformation scenarios): SVM accuracy 0.5271, macro-F1 0.4652, FPR 0.1322, FNR 0.8131; DistilBERT accuracy 0.5004, macro-F1 0.4124, FPR 0.1117, FNR 0.8867. MIX is machine-involved text, not necessarily fully AI-authored.
- GRADTEX Test B paired scenarios (2,642 rows; `generate_topic`, `paraphrase`, `polish_token`): SVM accuracy 0.5223 / macro-F1 0.4561; DistilBERT 0.5151 / 0.4361. The three scenario-level AI recalls range 0.1136–0.2585 for SVM and 0.1227–0.1655 for DistilBERT.
- Beemo: 656 expert-edited AI-origin samples after retaining only documented MIT/Apache source-model families. AI recall-only: SVM 0.3003, DistilBERT 0.3476; all two-class metrics are not applicable.
- Exact casefolded/whitespace-normalized overlap checks found zero matches for GRADTEX Test B/C and Beemo evaluation texts against HC3 train/validation/test and prior MAGE evaluation CSVs. Semantic near-duplicate matching remains unimplemented.
- Calibration investigation: validation-selected temperatures SVM 0.678 and DistilBERT 1.591; applied to HC3 test as an evaluation only. The temperature was not deployed; original thresholds/checkpoints remain unchanged.
- Live API checks: image `/health` and model status ready; synthetic image inference HTTP 200 on CUDA; short text HTTP 422; long text HTTP 200 in 9 chunks; simultaneous independent image/text API calls both HTTP 200. Full suite now passes 48 tests.
- Pangram Grammarly 1,768-row (~3.95 MB) dataset is gated; file access returned 401 and requires user authentication/acceptance. It was not downloaded. No true human-only grammar-correction corpus has been evaluated.

## Gates / constraints

- Do not download gated or unknown-license data without a manual decision.
- Do not redistribute HC3 source-derived data without observing source-specific terms.
- Use validation only for model selection and threshold choice; keep `test.jsonl` untouched until evaluation.
- Do not train automatically at install/startup.

## Dataset expansion continuation (2026-11-25)

The pinned MAGE streaming subset, combined HC3+MAGE prepared splits, and separate expanded SVM/DistilBERT checkpoints are documented in `PHASE2_RETRAINING_PLAN.md` and `PHASE2_EXPERIMENT_LOG.md`. The expanded test comparison report is `training/reports/text_model_comparison_expanded_test.json`; the frozen independent-source report is `training/reports/text_expanded_ood_diagnostics.json`. Expanded DistilBERT reached test macro-F1 0.8524 but human FPR 0.2577; MAGE GPT-4 OOD macro-F1 0.7476 with human FPR 0.4580. These errors prevent a general reliability claim. The latest complete suite has 58 passing tests. The SVM HC3 baseline remains the default; no Phase 1 checkpoint or Phase 3 fusion is changed.

## Real-world failure investigation (2026-11-24)

- Verified default text backend is the HC3 TF-IDF SVM; DistilBERT requires explicit `TEXT_MODEL_BACKEND=transformer`. Saved label ordering matches inference for both.
- Corrected DistilBERT response metadata that had mislabeled raw softmax as calibrated probabilities. No threshold, model, or checkpoint changed.
- Existing MAGE/GRADTEX/Beemo independent reports show poor out-of-domain AI sensitivity; user-specific AI text is not yet present, so the observed failure has not been reproduced.
- Awaiting user-provided text example for private error analysis. No independent data was used for training or calibration.
- Do not imply model scores establish authorship or generalize outside HC3.
