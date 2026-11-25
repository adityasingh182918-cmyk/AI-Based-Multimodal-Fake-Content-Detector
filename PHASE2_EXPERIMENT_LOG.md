# Phase 2 experiment log

## Expansion dataset 001 — source-stratified MAGE sample (2026-11-25)

- Source: `yaful/MAGE@342663f0a2b775455c023f5d36a1341ff0ec5402`; Hugging Face dataset card declares Apache-2.0. Official splits are retained; labels map `0=AI_GENERATED`, `1=HUMAN_WRITTEN` and provenance field is `src`.
- Preflight: 26.33 GiB free on C:. Official source reports 319,071 train, 56,792 validation, and 60,743 test rows. The full raw partitions were streamed; only deterministic capped samples were saved (about 53 MB combined).
- Sampling: seed 42, SHA-256 bottom-k per `src` and label; train caps are 40 AI/1,200 human per source; validation/test caps are 10 AI/300 human per source. Selected raw counts: 24,480 train, 6,120 validation, 7,002 test.
- Combined with unchanged HC3 split files, exact duplicate checks removed 1,809 MAGE examples overlapping frozen MAGE GPT-4 OOD/paraphrase, GRADTEX, or Beemo evaluations; zero exact overlaps with HC3 and zero exact duplicates across the resulting splits. Existing validator reports no errors.
- Final combined counts: train 26,954 (13,730 AI / 13,224 human); validation 6,650 (3,398 / 3,252); test 5,727 (3,158 / 2,569).
- Limitation: MAGE does not expose original document IDs. Official partition and exact text checks are used, but same-document cross-generator/semantic overlap cannot be guaranteed. Dataset-card license is recorded; original-source redistribution terms are not independently audited. Keep data local and do not redistribute.
- Data files: `data/text/raw/mage_expansion/` and `data/text/processed/expanded_v1/` (ignored locally). Validation report: `training/reports/text_expanded_dataset_validation.json`.
- Status: acquisition, preparation, training, held-out test evaluation, external OOD evaluation, calibration investigation, API selection, and regression testing are complete for this experiment. Phase 2 as a whole remains incomplete (see blockers below).

## Validation comparison 001 — existing checkpoints on expanded validation

- Dataset: `data/text/processed/expanded_v1/validation.jsonl`; exactly 6,650 rows (3,398 AI / 3,252 human). This split is used for model selection and is not a final test.
- Existing HC3 SVM: accuracy 0.5838; macro-F1 0.5483; AI false-negative rate 0.7031; human false-positive rate 0.1165.
- Existing HC3 DistilBERT: accuracy 0.6417; macro-F1 0.6182; AI false-negative rate 0.6148; human false-positive rate 0.0904.
- Scores use API-like chunk/window aggregation. Existing thresholds were read for descriptive tri-state coverage only; they were not refit here. These results are validation evidence for the new experiment, not the final held-out evaluation.
- Report: `training/reports/text_model_comparison_expanded_validation.json`.
- Expanded SVM candidate training has started; DistilBERT will train only after the CPU SVM experiment finishes.

## Expanded SVM 001 — validation-selected TF-IDF

- Training used only `data/text/processed/expanded_v1/train.jsonl` (26,954 rows). Candidate features were word-only TF-IDF, character-only TF-IDF, and combined word (1–2 grams) + char_wb (3–5 grams); Logistic Regression and Linear SVM regularization C values 0.5, 1.0, and 2.0 were compared on expanded validation.
- Selected candidate: combined word+character calibrated Linear SVM, C=0.5. Validation accuracy 0.7692 and macro-F1 0.7689. Existing SVM scored 0.5838 accuracy / 0.5483 macro-F1 on the same 6,650 validation rows.
- Validation-derived abstention thresholds are written beside `models/text_detector/svm_expanded_v1.joblib`. Existing SVM checkpoint remains unchanged.
- Detailed candidate results: ignored local report `training/reports/text_svm_expanded_validation.json`; metadata sidecar `models/text_detector/svm_expanded_v1.json`.
- The independent expanded test comparison across all four frozen checkpoints is in `training/reports/text_model_comparison_expanded_test.json` (5,727 same examples; class supports 2,569 human and 3,158 AI). Old SVM: accuracy 0.5591, macro-F1 0.5323, human FPR 0.1098, AI FNR 0.7103. Expanded SVM: 0.7503, 0.7502, 0.1810, 0.3056. Old DistilBERT: 0.5986, 0.5792, 0.0934, 0.6520. Expanded DistilBERT: 0.8579, 0.8524, 0.2577, 0.0481. Confusion matrices, per-source/per-length slices, class-wise precision/recall, predictions and scores are in the report.
- This is an official MAGE test + HC3 test concatenation after exact overlap filtering, not a wholly independent-source test; MAGE lacks original group IDs. Use the separate OOD report for independent robustness evidence.

## Expanded transformer 001 — continued DistilBERT fine-tune

- Initialization: existing `models/text_detector/transformer/distilbert_hc3_v1`; no from-scratch initialization. New output: `models/text_detector/distilbert_expanded_v1/`.
- Device/runtime: RTX 4050 CUDA, PyTorch 2.11.0+cu128, batch 8, gradient accumulation 2, max 256 training tokens, fp16 autocast, AdamW at 1e-5, seed 42. No second transformer was trained concurrently.
- Epoch 1 validation accuracy/macro-F1: 0.7627 / 0.7516. Epoch 2: 0.8074 / 0.8014. Epoch 3: 0.8522 / 0.8502; best checkpoint selected at epoch 3.
- Threshold derivation on expanded validation with API chunk aggregation selected extremes (SVM REAL max=0, AI min=1; DistilBERT same), which label all examples UNCERTAIN. These thresholds were retained as evidence that a 5% per-class error target is not achievable with useful coverage; do not represent the tri-state decision as operationally useful.
- Training truncates each training example at 256 tokens; inference uses overlapping windows (stride 32), so long examples can expose later text at inference, but training/inference context differs.
- History: `training/reports/text_transformer_expanded_history.json` (ignored locally). Best epoch 3; test/OOD results below.

## Frozen robustness and calibration 001 — expanded checkpoints

- New candidates were scored without fitting or threshold tuning on the evaluation datasets. Report: `training/reports/text_expanded_ood_diagnostics.json`; it contains per-example labels/scores, source/domain/scenario/length slices and confusion matrices, without storing raw text.
- MAGE GPT-4 OOD (1,562; two-class): expanded SVM accuracy 0.6677, macro-F1 0.6673, human FPR 0.2795, AI FNR 0.3825; expanded DistilBERT 0.7612, 0.7476, 0.4580, 0.0300. DistilBERT detects more AI but falsely flags more human text.
- GRADTEX Test C (3,529; machine-involved MIX vs human source): SVM 0.5285 accuracy / 0.4812 macro-F1 / 0.1690 human FPR / 0.7735 AI-involvement FNR; DistilBERT 0.5551 / 0.5369 / 0.2462 / 0.6433.
- Selected paired GRADTEX Test B probes (2,642): SVM 0.5443 / 0.4998 / 0.1575 / 0.7540; DistilBERT 0.6219 / 0.6154 / 0.2483 / 0.5079. GRADTEX derives from MAGE, so it is not a fully independent corpus family.
- Beemo expert-edited AI-origin only (656; positive-only sensitivity probe): SVM AI recall 0.5534; DistilBERT 0.8354. No two-class metrics are valid.
- Expanded validation temperature diagnostics: SVM T=0.9696 did not improve all Brier/NLL/ECE measures, so it remains T=1; DistilBERT T=2.2867 improved all three (raw validation Brier/NLL/ECE .11605/.48195/.11240; scaled .10685/.34975/.09806). It is stored in the new checkpoint metadata and applied in service inference. It was fit on validation only; the untouched test was not used to choose it.
- Thresholds for a 5% conditional error target collapsed to [0,1] for both new models: all validation and OOD rows are UNCERTAIN. Calibration probabilities and useful selective decisions are not interchangeable; probability temperature scaling did not fix the abstention coverage problem.
- OOD hard metrics use a 0.5 score cutoff, not the deployed validation abstention rule. Beemo is reported as positive-only. The detector is not a reliable general authorship classifier.

## API and regression validation — expanded version

- Added selectable backends `baseline`, `svm_expanded`, `transformer`, and `transformer_expanded`; the default remains the original HC3 SVM. Model name/version and confidence semantics are returned by status and inference. Image routes/checkpoints are independent.
- API integration test exercised the expanded DistilBERT local checkpoint, temperature-scaled score metadata, and multi-window text inference. `python -m pytest backend\tests -q`: 58 passed in 13.54 seconds. `compileall` passed for the changed Python modules.
- Browser-rendered responsive/accessibility checks have not been performed. Combined UI mode invokes image and text APIs independently; no Phase 3 fusion model exists.

## Baseline 001 — HC3 word/character TF-IDF

- Dataset revision: `Hello-SimpleAI/HC3@4d0ff18143b5a7e1b1e79beb540c04549d1e59d3`.
- Sources: `reddit_eli5`, `wiki_csai`; seed 42; up to 1,000 question groups/source; paired equal answers; group-held-out train/validation/test.
- Current retained row counts: train 2,476; validation 531; test 531. Class counts are in ignored `data/text/processed/hc3/metadata.json`.
- Python 3.13.17, scikit-learn 1.9.1; baseline CPU models.
- Validation: Logistic Regression accuracy 0.9642 / macro-F1 0.9641; calibrated Linear SVM accuracy 0.9793 / macro-F1 0.9792. Linear SVM selected by validation macro-F1.
- Untouched test (Linear SVM): 531 rows; accuracy 0.9680; macro-F1 0.9679; weighted-F1 0.9680; AI ROC-AUC 0.9951; confusion matrix in label order `[HUMAN_WRITTEN, AI_GENERATED]`: `[[242,11],[6,272]]`.
- The test results are in-domain HC3 results, not independent-generator robustness evidence. They must not be interpreted as general authorship accuracy.
- Validation abstention band is stored beside the SVM checkpoint. Its empirical test coverage is recorded in `training/reports/text_evaluation.json` after evaluation.
- Source slices (held-out questions, but same sources as training): `reddit_eli5` 268 rows, macro-F1 0.9851; `wiki_csai` 263 rows, macro-F1 0.9501. This gap is a warning that source/domain artifacts may matter; neither slice is a genuinely unseen-source test.

## Transformer 001 — DistilBERT HC3 fine-tune

- Base model: `distilbert/distilbert-base-uncased@12040accade4e8a0f71eabdb258fecc2e7e948be` (Apache-2.0).
- Environment: Python 3.13.17; PyTorch 2.11.0+cu128; Transformers 4.57.6; RTX 4050 Laptop GPU, 6 GiB.
- Seed 42; max 256 tokens, dynamic batch padding, batch 8, gradient accumulation 2, fp16 autocast, AdamW at 2e-5, 3 epochs; best validation macro-F1 checkpoint.
- Validation macro-F1: epoch 1 0.9449, epoch 2 0.9216, epoch 3 0.9640; selected epoch 3.
- Untouched test: 531 rows; accuracy 0.9435, macro-F1 0.9431, weighted-F1 0.9433, ROC-AUC 0.9945, confusion matrix `[[227,26],[4,274]]`; thresholded coverage 93.4%, 35 uncertain, covered accuracy 98.6%.
- Source slices: `reddit_eli5` macro-F1 0.9739; `wiki_csai` macro-F1 0.9103, also showing source/domain sensitivity.
- The transformer did not beat TF-IDF Linear SVM on validation or test macro-F1, so the baseline remains the default. Neither score establishes cross-generator robustness.

## Robustness 001 — HC3 source holdout

- Training source: `reddit_eli5`; independent test source: `wiki_csai`; checkpoint: `models/text_detector/baseline/hc3_reddit_to_wiki/hc3_tfidf_linear_svm.joblib`.
- Test: 263 rows; accuracy 0.6236; macro-F1 0.5327; weighted-F1 0.5507; human recall 0.2000; AI recall 0.9790; ROC-AUC 0.8507. Confusion matrix in `[HUMAN_WRITTEN, AI_GENERATED]` order: `[[24,96],[3,140]]`.
- The result demonstrates substantial cross-source failure: most human Wikipedia answers were classified as AI. The validation-derived abstention band covered 65.8%; accuracy on covered rows was 0.7399. No threshold was tuned on this source-held-out test.
- Full report: `training/reports/text_source_holdout_evaluation.json`.

## Robustness 002 — MAGE GPT-4 OOD and paraphrase probe

- Source: MAGE pinned at `yaful/MAGE@342663f0a2b775455c023f5d36a1341ff0ec5402`; test files only, declared Apache-2.0. No training examples used. Main file has 1,562 records (800 AI, 762 human) spanning CNN/DailyMail, DialogSum, IMDb, and PubMedQA.
- Frozen TF-IDF SVM: accuracy 0.6543; macro-F1 0.6367; AI precision 0.8110; AI recall 0.4238; human recall 0.8963; ROC-AUC 0.7924; Brier 0.2413. Validation-band coverage 82.4% and covered accuracy 0.6651.
- Frozen DistilBERT: accuracy 0.7177; macro-F1 0.7144; AI precision 0.8017; AI recall 0.5963; human recall 0.8451; ROC-AUC 0.8182; Brier 0.2502. Validation-band coverage 73.0% and covered accuracy 0.6994.
- The optional paraphrase CSV includes human-origin paraphrases labeled as machine by the dataset authors. To avoid treating it as a human-vs-AI authorship test, this probe evaluates only `_gpt4_para` rows (800 known AI paraphrase examples). SVM AI recall was 0.4713; DistilBERT AI recall was 0.6963. No human negatives exist in this slice, so two-class F1, precision, ROC-AUC and calibration are intentionally omitted. Validation-band coverage was 71.4% and 48.0%, respectively.
- These results are robustness evidence only, not new model-selection evidence. Checkpoint, threshold, and preprocessing remain unchanged. Reports: `training/reports/text_mage_ood_evaluation.json` and `training/reports/text_mage_paraphrase_ai_evaluation.json`.

## Robustness 003 — GRADTEX held-out scenarios

- Source/revision: `elisabeth-pl-pl/GRADTEX@553d859da0255d75a39c385c208f7522a2007f53`; dataset card declares CC BY 4.0. Only the 64,081,445-byte `test.parquet` was downloaded. GRADTEX is derived from MAGE, so its transformation test is held out but underlying corpus family is not independent of MAGE.
- Test C (`unseen_scenarios_evaluation`) includes 1,763 HWT and 1,766 MIX rows across four transformation scenarios. Treat MIX as machine-involved writing, not fully machine-authored text. At the unchanged 0.5 binary score threshold: SVM accuracy 0.5271, macro-F1 0.4652, AI-involvement precision 0.5861/recall 0.1869, human FPR 0.1322, AI-involvement FNR 0.8131, confusion `[[330,1436],[233,1530]]` in `[MIX,HWT]` order. DistilBERT accuracy 0.5004, macro-F1 0.4124, precision 0.5038/recall 0.1133, FPR 0.1117, FNR 0.8867, confusion `[[200,1566],[197,1566]]`.
- Test B paired external scenarios (`generate_topic`, `paraphrase`, `polish_token`) yielded 2,642 rows and 1,321 source pairs. Combined machine-involved vs human-source-control scores: SVM accuracy 0.5223, macro-F1 0.4561, precision 0.5739/recall 0.1734, FPR 0.1287, FNR 0.8266, confusion `[[229,1092],[170,1151]]`. DistilBERT accuracy 0.5151, macro-F1 0.4361, precision 0.5602/recall 0.1408, FPR 0.1105, FNR 0.8592, confusion `[[186,1135],[146,1175]]`.
- Test B scenario-specific AI-involvement recall: `generate_topic` SVM 0.2585 / DistilBERT 0.1655; `paraphrase` 0.1136 / 0.1341; `polish_token` 0.1477 / 0.1227. Test C per-scenario values and domain/source/length slices are in its JSON report.
- Validation-derived abstention on C had coverage 86.2% (SVM) / 90.8% (DistilBERT), but covered accuracy 0.5270 / 0.4991. These thresholds do not provide reliable out-of-domain decisions.

## Robustness 004 — Beemo expert-edited AI text

- Source/revision: `Toloka/Beemo@9c014107fe9b85c4c784c1ce3a43b0b7b0a6d162`; 8,345,056-byte parquet. The dataset terms vary by fields and generating model. Evaluation selected 656 `human_edits` outputs produced from documented MIT (`zephyr-7b-beta`) and Apache-2.0 (`Mistral-7B-Instruct-v0.1`, `Mixtral-8x7B-Instruct-v0.1`) source models. The No Robots human-output and LLM-edit fields were not scored.
- This is AI-origin positive-only sensitivity evidence; do not report two-class accuracy, precision, macro-F1, FPR, ROC-AUC, or Brier. SVM detected 30.03% (AI recall; FNR 69.97%); DistilBERT detected 34.76% (FNR 65.24%). Human source controls were intentionally omitted from the slice due their separate NC source license.
- HC3 and prior MAGE exact normalized overlap counts were zero. Report: `training/reports/text_beemo_expert_edit_sensitivity.json`.

## Calibration 001 — Validation-only temperature study

- Fit scalar temperature on HC3 validation logits only: SVM T=0.6778; DistilBERT T=1.5912. No checkpoint, service probability, or abstention threshold was changed.
- SVM validation Brier/NLL/ECE(10 bins): 0.02005/0.07333/0.02309 raw; 0.01918/0.06626/0.01948 temperature-scaled. HC3 test: 0.02640/0.09387/0.02229 raw; 0.02570/0.09180/0.01565 using the validation-fitted T.
- DistilBERT validation: 0.02835/0.14582/0.03589 raw; 0.02789/0.11860/0.03947 scaled (ECE worsened). HC3 test: 0.04975/0.23352/0.05104 raw; 0.04615/0.17456/0.04551 scaled using validation T.
- HC3 test improvement is a single-domain check, not proof of robust calibration. The distinct external reports show low scores and selective accuracy; the `UNCERTAIN` tri-state is implemented but not reliable outside HC3. Full report: `training/reports/text_calibration_analysis.json`.

## Integration validation 001 — Local API/UI smoke checks

- Uvicorn started successfully; `/health`, `/model-status`, and `/text-model-status` returned HTTP 200. Phase 1 ResNet-18 checkpoint loaded on CUDA; the local synthetic image returned HTTP 200. Invalid image returned HTTP 400.
- Short text returned HTTP 422; 15,900-character/2,500-word test text returned HTTP 200 with 9 chunks. Simultaneous independent image and text API requests both returned HTTP 200 and separate responses.
- Static UI test verifies all three modes, explicit text label, keyboard-focusable file chooser, independent API calls, and no fusion claim. Actual browser viewport rendering and assistive-technology checks were not performed: Brave control was denied and no headless browser runtime was installed.
- Final compile check succeeded and the full backend suite passed: 48 tests in 8.87 seconds.
- Exact normalized overlap audit: zero matches for GRADTEX B/C and Beemo examples against HC3 train/validation/test and prior MAGE OOD/paraphrase CSVs. Near-duplicate and semantic overlap analysis is not available.
# Diagnostic follow-up — 2026-11-25

No new training or threshold fitting was performed. The user-provided AI visual-description example was scored by the four existing checkpoints using each service's deployed preprocessing: AI scores were HC3 SVM 0.0012, HC3 DistilBERT 0.0026, expanded SVM 0.1292, expanded DistilBERT 0.1071. The deployed label was HUMAN_WRITTEN for the two HC3 models and UNCERTAIN for both expanded models (expanded abstention limits are 0 and 1). These are single-example diagnostic observations, not evaluation metrics. Expanded checkpoint selection was rejected for deployment because of documented false-positive/abstention limitations. See `PHASE2_IMPROVEMENT_REPORT.md`.

