# Project Progress

Updated: 2026-11-25

## Text false-negative investigation (2026-11-25)

- Status: **IN_PROGRESS; no safe model promotion**. Audited the live unified app, selector, checkpoints, mappings, preprocessing, existing reports, and available user-shared examples. The API reported `transformer` / `distilbert_hc3_v1`; Phase 3 calls the same Phase 2 detector as the standalone endpoint.
- Root cause evidence: no label-order or checkpoint-load mismatch was found. The active HC3 DistilBERT gave the explicit AI-generated visual-description example an AI score of 0.0026 (HUMAN_WRITTEN); the HC3 SVM gave 0.0012. This is a genuine distribution/generalization failure. The exact five-caption set referenced in the request is not present in accessible project files, so it could not be retested as five examples.
- Candidate check: existing expanded SVM/DistilBERT both return UNCERTAIN on that caption (AI scores 0.1292 / 0.1071). On the same expanded held-out test, expanded DistilBERT reached accuracy 0.8579 / macro-F1 0.8524, but human FPR was 0.2577; frozen MAGE GPT-4 OOD human FPR was 0.4580. Its validation uncertainty thresholds are [0,1], making all API predictions UNCERTAIN. No checkpoint or threshold was changed.
- Protected image path: `phase3/tools/snapshot_originals.py --verify` passed before/after work; Phase 1/2 baseline source/model/data hashes and fixed predictions were unchanged. No Phase 1 files, image weights, preprocessing, threshold, or response logic were edited.
- UI: text result cards now identify which class a score points toward, explain the threshold-based UNCERTAIN state, and state that a model score is not accuracy or proof. Image-result rendering and all inference paths are unchanged.
- Tests: full suite `84 passed, 1 warning` (PyTorch sparse-invariant warning); inline JavaScript syntax and three renderer states passed. Browser check submitted the user's exact 483-character caption: the current live model still returned HUMAN with a 99.7% HUMAN score, while the new card correctly labeled it a model score and explained the limitations. This confirms the copy fix does not solve the classifier's false negative. One local analysis result was saved by normal app behavior; raw input retention is disabled by default. No training ran. No Phase 2 model is promoted; general reliability remains unresolved pending suitable source-diverse caption examples and evaluation data.
- Exact next action: obtain a legally usable, independently labeled dataset of AI and human descriptions across sources/domains; source/group split it before training, then fit and evaluate a candidate without reusing any locked OOD test data. Do not add a rule that merely matches the single caption or lower thresholds to force the desired output.

## Phase 2 expansion/retraining update (2026-11-25)

- Status: **IN_PROGRESS**. Expanded dataset acquisition/preparation, new SVM and DistilBERT training, held-out evaluation, frozen OOD evaluation, validation-only calibration, API integration, documentation, compile check, and 58-test backend regression are complete. Phase 2 acceptance remains incomplete due weak cross-domain performance, no suitable licensed human-edit/grammar-correction benchmark, MAGE missing original group IDs, and pending visual/accessibility QA. Local uvicorn startup succeeded; desktop browser inspection was blocked when computer-use approval for Chrome was denied. No original text checkpoints or Phase 1 code/weights were overwritten.
- Audit: existing HC3 SVM and DistilBERT checkpoints, reports, text API and overflow-window implementation are present. Python 3.13.17, PyTorch 2.11.0+cu128, CUDA RTX 4050, `datasets` 4.8.5, and 26.33 GiB free disk were verified.
- Dataset choice: MAGE official Hugging Face revision `342663f0a2b775455c023f5d36a1341ff0ec5402`, card-declared Apache-2.0, separate official train/validation/test streams. External GPT-4 OOD/paraphrase, GRADTEX, and Beemo files remain evaluation-only. The user-excluded noncommercial/uncertain sources remain excluded.
- MAGE deterministic subset: train 24,480 (12,480 AI / 12,000 human), validation 6,120 (3,120 / 3,000), test 7,002 (3,240 / 3,762) before overlap filtering. Subset files use about 53 MB; the stream is capped per `src` and class. HC3 splits were not modified.
- Prepared combined splits: train 26,954 (13,730 AI / 13,224 human), validation 6,650 (3,398 / 3,252), test 5,727 (3,158 / 2,569). Removed 1,809 exact MAGE overlaps with frozen evaluation artifacts; no exact MAGE/HC3, within/across MAGE split duplicates remained. Existing validator reports zero errors. Semantic/group overlap cannot be fully ruled out because MAGE does not expose original document IDs.
- New code: `scripts/download_text_expansion.py`, `scripts/prepare_expanded_text_dataset.py`, `training/train_expanded_svm.py`, `training/compare_text_models.py`, `training/calibrate_expanded_models.py`, and `backend/tests/test_text_expansion_data.py`. Updated DistilBERT trainer for optional local initialization, inference temperature metadata, selector/status schemas/API, tests, and OOD diagnostic report semantics; old HC3 defaults are unchanged.
- Existing checkpoints on the identical expanded validation split: old SVM accuracy 0.5838, macro-F1 0.5483, AI FNR 0.7031, human FPR 0.1165; old DistilBERT accuracy 0.6417, macro-F1 0.6182, AI FNR 0.6148, human FPR 0.0904. Report: `training/reports/text_model_comparison_expanded_validation.json`.
- New focused data-tool tests passed (3). Full backend suite passes: 58 tests in 13.54s. `compileall` passed for changed services, schemas and training/report scripts.
- Expanded SVM: selected word+character calibrated Linear SVM (C=0.5); validation accuracy 0.7692 and macro-F1 0.7689, versus old SVM 0.5838/0.5483 on the same 6,650 validation examples. Checkpoint `models/text_detector/svm_expanded_v1.joblib`; no existing model was overwritten.
- Expanded DistilBERT: continued from the original checkpoint; best epoch 3 (validation accuracy 0.8522, macro-F1 0.8502). New checkpoint `models/text_detector/distilbert_expanded_v1/`; training history `training/reports/text_transformer_expanded_history.json`. New SVM validation macro-F1 0.7689. No original text checkpoint was overwritten.
- Expanded test comparison on identical 5,727 rows: old SVM 0.5591 accuracy/0.5323 macro-F1; new SVM 0.7503/0.7502; old DistilBERT 0.5986/0.5792; new DistilBERT 0.8579/0.8524. New DistilBERT human FPR is 0.2577; new SVM AI FNR is 0.3056. Report contains per-source, per-length, class-wise and confusion-matrix results.
- Frozen OOD results: new SVM and DistilBERT macro-F1 on MAGE GPT-4 OOD 0.6673 and 0.7476; on GRADTEX Test C 0.4812 and 0.5369; selected Test B 0.4998 and 0.6154. On Beemo's 656 AI-origin expert edits, AI recall only is 0.5534 and 0.8354; no two-class metrics are valid. OOD human FPR remains high, especially DistilBERT (0.4580 on MAGE GPT-4).
- Calibration: expanded DistilBERT validation-only temperature T=2.2867 improved Brier/NLL/ECE; SVM temperature was not deployed. Both 5%-error abstention bands collapsed to [0,1], so every validation/OOD response is UNCERTAIN. No reliable operational three-way policy is established.
- API: backends now select via `TEXT_MODEL_BACKEND=baseline|svm_expanded|transformer|transformer_expanded`; default remains original HC3 SVM. Text metadata reports model version and score semantics; independent combined UI remains without fusion.
- Documentation/files updated: `README.md`, `docs/phase2.md`, `PHASE2_IMPLEMENTATION_PLAN.md`, `PHASE2_RETRAINING_PLAN.md`, `PHASE2_EXPERIMENT_LOG.md`, `PHASE2_CHANGELOG.md`, this tracker; added `PHASE2_RETRAINING_REPORT.md`. Local data, model weights, and generated reports remain in ignored data/model/report locations.
- Exact next action: none for the completed expansion experiment. Phase 2 remains `IN_PROGRESS`; if continuing, first obtain an allowed human-edited/grammar-correction evaluation corpus and complete browser/device/accessibility QA. Do not tune thresholds or claim general authorship detection from the current reports.

## Phase 1 retraining update (2026-11-25)

- Status: **IN_PROGRESS** as an improvement investigation; the candidate experiment is complete, but independent camera-photo evidence is blocked and no candidate is promoted.
- Dataset: original CIFAKE data and train/validation/test splits preserved. Four user-labelled REAL photos are stored under ignored `data/image_detection/user_real/train/`, with metadata, hashes, EXIF orientation, and exact/near-duplicate checks. They were included in train only.
- Dataset source decision: OpenFake core is too large for local free disk. CommunityForensics-Small is CC BY-NC-SA 4.0/non-commercial research only; the user answered commercial use may be needed or is uncertain, so it was excluded. No external dataset was downloaded for training.
- Model: ResNet-18 initialized from `resnet18_cifake_aug_v1_calibrated_strict.pt`; 224px pad preprocessing; CUDA RTX 4050; AdamW 1e-5; batch 8; AMP; seed 42; early stopping after epoch 3.
- Candidate checkpoint: `models/image_detector/resnet18_realworld_v1.pt`; best validation-loss epoch 1, validation accuracy 96.5%. The later epochs reached 96% and 95% validation accuracy.
- Held-out CIFAKE test (n=200): candidate 97.5% accuracy, 4% REAL false-positive rate, 1% AI false-negative rate, ROC-AUC 0.9958. The active strict checkpoint scored the same accuracy/FPR/FNR and ROC-AUC 0.9955. Candidate not promoted.
- User photos: candidate predicts REAL for all four after training on them. This is in-sample behavior and does not establish generalization. Their labels are user-provided and not independently verified.
- Backend: still defaults to `resnet18_cifake_aug_v1_calibrated_strict.pt`; candidate loads through `IMAGE_MODEL_CHECKPOINT`; no deployed threshold changed. Phase 2 code untouched.
- Tests: new focused dataset-manifest tests 3 passed; full backend suite 51 passed in 55.28 seconds; `compileall -q backend training scripts` passed.
- Files changed: `training/train_image_model.py`, `scripts/import_user_real_images.py`, `scripts/prepare_realworld_manifest.py`, `backend/tests/test_realworld_dataset.py`, Phase 1 plan/changelog, README, this tracker, and `PHASE1_RETRAINING_PLAN.md` / `PHASE1_RETRAINING_REPORT.md`.
- Exact next action: obtain commercially compatible source-aware high-resolution image data with an independent locked test set before another training run or checkpoint promotion. The strict checkpoint remains active.

## Current phase and stage

- Current project phase: Phase 1 + Phase 2 real-world failure investigation.
- Current stage: Additional Phase 1/Phase 2 diagnostics and user-failure reproduction.
- Overall Phase 2 status: **IN_PROGRESS**. Robustness evidence exists for accessible datasets, but a gated human grammar-correction corpus, visual browser QA, and the user-specific text failure reproduction remain incomplete.
- Stage statuses are recorded in [PHASE2_IMPLEMENTATION_PLAN.md](PHASE2_IMPLEMENTATION_PLAN.md).

## Completed tasks

- Audited existing Phase 1 API, service, training/data code, static UI, docs, tests, dependencies, Python/CUDA, disk, and local dataset/checkpoint availability.
- Reviewed HC3 sources and selected only `reddit_eli5` (source ELI5 listed BSD) and `wiki_csai` (Wikipedia CC BY-SA); pinned dataset revision `4d0ff18143b5a7e1b1e79beb540c04549d1e59d3`.
- Downloaded the two reviewed raw files (55.4 MB + 2.2 MB) into ignored `data/text/raw/hc3/`; no login required.
- Prepared equal human/generated answer pairs, exact de-duplication, deterministic 70/15/15 question-group splits, plus validation report. Current counts: train 2,476; validation 531; test 531 (the per-class details are in ignored `data/text/processed/hc3/metadata.json`). Validation found no errors.
- Trained word+character TF-IDF Logistic Regression and calibrated Linear SVM; validation selected Linear SVM.
- Fine-tuned DistilBERT (`distilbert/distilbert-base-uncased`, pinned model revision) on CUDA, 3 epochs, dynamic padding, batch 8, accumulation 2, max length 256, fp16 autocast; selected best checkpoint by validation macro-F1.
- Evaluated both baselines and DistilBERT on the held-out HC3 test split. Results and settings are saved in `training/reports/` locally.
- Implemented optional DistilBERT inference backend (`TEXT_MODEL_BACKEND=transformer`) and the default TF-IDF baseline backend.
- Preserved Phase 1 image routes and added independent text plus image/text UI modes. Combined mode runs detectors separately; no fusion probabilities/model are claimed.
- Full repository tests after robustness changes: 48 passed. `compileall` also passed.
- Ran frozen-checkpoint robustness checks: Reddit-trained SVM to held-out Wikipedia HC3 (macro-F1 0.5327), MAGE GPT-4 four-domain OOD (SVM 0.6367, DistilBERT 0.7144 macro-F1), and an 800-row AI-paraphrase sensitivity probe (SVM AI recall 0.4713, DistilBERT 0.6963). The MAGE data was not used for training or tuning.
- Evaluated GRADTEX held-out Test C (SVM macro-F1 0.4652; DistilBERT 0.4124), paired Test B generation/paraphrase/polish scenarios (0.4561; 0.4361), and 656 Beemo expert-edited AI-origin positives (recall-only 0.3003; 0.3476). All used frozen checkpoints; none were used for fitting or calibration.
- Exact normalized overlap audits for GRADTEX and Beemo found no matches with HC3 train/validation/test or the previously evaluated MAGE files.
- Ran HC3 validation-only temperature/reliability analysis. Temperatures were recorded but not deployed; thresholds and checkpoints were unchanged.
- Live API smoke checks verified Phase 1 image readiness/inference, Phase 2 short-text validation and multi-chunk long text, and simultaneous independent image/text responses.
- Audited both detector pipelines for reported field failures. Verified default image checkpoint, class/preprocessing match, image dataset resolution, selected text backend, label ordering, chunking, thresholds, and frontend request bodies. No misclassification example was present locally.
- Fixed DistilBERT API metadata that called its raw softmax calibrated; it now reports `uncalibrated_softmax`. No score, threshold, or checkpoint changed.
- Added `training/analyze_real_world_failures.py` and separate image/text failure reports. The image report is populated with four supplied examples; the text report remains `WAITING_FOR_USER_EXAMPLE`. No raw user content or paths were written.
- Re-ran compile and full backend regression suite after the DistilBERT overflow-window fix and image diagnostic runner update: 48 passed in 10.81 seconds.
- Compared all four valid image checkpoints on the same 200 CIFAKE validation images. FastAPI's default strict checkpoint is loadable and was confirmed as the active default.
- Reproduced the user's four REAL-labeled image errors: all four checkpoints classified all four supplied JPEGs as AI_GENERATED. Strict-checkpoint AI softmax scores were 0.995656–0.999918; expected labels reflect the user's statement, not independent provenance verification.
- Tested validation-only image perturbations. Native 32x32 inference changed 65/200 decisions; mild blur changed 33/200; JPEG quality 70 and brightness changes altered two each. Results are diagnostic correlated variants, not real-world accuracy evidence.
- Re-evaluated both frozen text models on identical MAGE, GRADTEX, and Beemo examples using API-style chunk/window aggregation; wrote per-example score/error/source/length records without text content.
- Fixed a DistilBERT long-input edge case where a shorter final overflow window could fail tensor construction; padding now allows all windows to be scored, with the attention mask excluding padding from aggregation weights.
- No AI-text example was supplied with the four images; the specific text failure remains unreproduced.

## Dataset and preparation status

- Download status: **COMPLETED** for HC3 `reddit_eli5` + `wiki_csai`, pinned revision above.
- Preparation status: **COMPLETED** for 1,000-question-per-source cap, seed 42; group held out; empty/short/exact duplicate/pair handling recorded.
- Robustness download status: **COMPLETED** for MAGE's 2.19 MB GPT-4 OOD file and optional 3.61 MB paraphrase file, pinned to `342663f0a2b775455c023f5d36a1341ff0ec5402`; only evaluation CSVs fetched.
- Additional robustness data: **COMPLETED** for GRADTEX Test parquet (64,081,445 bytes) and Beemo parquet (8,345,056 bytes), pinned revisions and saved under ignored `data/`.
- Pangram Grammarly source: **BLOCKED**; Hugging Face file access returned HTTP 401 and requires authenticated gated access. No authentication or download occurred.
- Access/licensing: public download, no authentication. HC3 authors state stricter original source terms govern; selected sources are documented in `docs/phase2_datasets.md`. Review attribution/share-alike obligations before redistributing derived material. Other unknown/custom HC3 sources excluded.
- Data is under ignored `/data/`; no data files should be tracked.

## Models, metrics, and artifacts

- Baseline architecture: TF-IDF word (1–2 grams) + char (3–5 grams); validation comparison between Logistic Regression and calibrated Linear SVM.
- Baseline validation: logistic macro-F1 0.9641; Linear SVM macro-F1 0.9792.
- Baseline untouched test: accuracy 0.9680; macro-F1 0.9679; weighted-F1 0.9680; ROC-AUC 0.9951; confusion matrix `[[242,11],[6,272]]` in `[HUMAN_WRITTEN, AI_GENERATED]` order. Validation-derived abstention test coverage 96.0%, 21 UNCERTAIN, covered accuracy 98.0%.
- Baseline source slices: `reddit_eli5` macro-F1 0.9851; `wiki_csai` 0.9501, indicating source sensitivity.
- HC3 test FPR/FNR (human false AI / AI false human): SVM 0.0435 / 0.0216; DistilBERT 0.1028 / 0.0144. Regenerated reports include explicit rates and test length slices.
- Baseline checkpoint: `models/text_detector/baseline/hc3_tfidf_linear_svm.joblib`, with adjacent validation threshold JSON.
- Transformer architecture: DistilBERT sequence classification; pretrained base model Apache-2.0; fine-tuned checkpoint at `models/text_detector/transformer/distilbert_hc3_v1/`.
- Transformer validation macro-F1 by epoch: 0.9449, 0.9216, 0.9640; best epoch 3.
- Transformer untouched test: accuracy 0.9435; macro-F1 0.9431; weighted-F1 0.9433; ROC-AUC 0.9945; confusion matrix `[[227,26],[4,274]]`; validation-derived abstention coverage 93.4%, 35 UNCERTAIN, covered accuracy 98.6%.
- Transformer source slices: `reddit_eli5` macro-F1 0.9739; `wiki_csai` 0.9103, showing a larger source gap.
- The Linear SVM remains the default because validation macro-F1 was stronger. Both test results are in-domain HC3 QA only; no cross-generator robustness claim.
- Text calibration method: validation score quantiles at 5% directional error target. Test split was not used to select thresholds.
- Calibration study: validation-only temperature T=0.6778 (SVM), 1.5912 (DistilBERT). Not deployed. OOD selective results show `UNCERTAIN` thresholds are not dependable beyond HC3.
- Training status: baseline and DistilBERT run completed; no other hyperparameter sweep performed.
- Reports: `training/reports/text_baseline_validation.json`, `text_evaluation.json`, `text_transformer_history.json`, `text_transformer_evaluation.json` (ignored locally); configuration in tracked `training/configs/text_training_config.json`.

## Backend, frontend, integration

- Backend status: text service, request/response schema, `POST /api/v1/predict/text`, and `GET /text-model-status` implemented. Lazy checkpoint loading; bounds are 5 words/40 characters minimum and 50,000 characters maximum. Long text uses ordered chunks/windows and warns aggregation is unvalidated.
- Image backend endpoints remain present: `/`, `/health`, `/model-status`, `/api/v1/predict/image`.
- Frontend status: `/ui` includes image, text, and image+text modes, separate results/errors, explicit text label, keyboard-focusable image chooser, loading messages, and responsive CSS. Local Chrome browser smoke check confirmed the page renders all three modes and clearly labels combined outputs as independent, not trained fusion. Full visual, mobile, and assistive-technology QA remains outstanding.
- Integration status: live image and text endpoints both returned HTTP 200 concurrently. Combined UI compares independent labels only and is marked **PRELIMINARY COMBINED ASSESSMENT – NOT A TRAINED FUSION MODEL**.
- API text default selects TF-IDF baseline. Optional transformer selection uses `TEXT_MODEL_BACKEND=transformer`, `TEXT_MODEL_CHECKPOINT`, and `TEXT_MODEL_DEVICE`.

## Files created or modified in this continuation

- New data tooling: `scripts/download_text_datasets.py`, `scripts/prepare_text_dataset.py`, `scripts/validate_text_dataset.py`.
- New model/training: `training/text_baseline.py`, `training/train_text_baseline.py`, `training/evaluate_text_baseline.py`, `training/train_text_model.py`, `training/evaluate_text_model.py`, `training/configs/text_training_config.json`.
- New backend: `backend/app/services/text_detector.py`, `backend/app/services/transformer_text_detector.py`; extended `backend/app/main.py`, `backend/app/schemas/prediction.py`.
- Modified UI: `backend/app/static/index.html`.
- New tests: `backend/tests/test_text_detector.py`, `backend/tests/test_text_api.py`.
- New docs/tracking: `docs/phase2.md`, `docs/phase2_datasets.md`, `docs/phase2_architecture.md`, `PHASE2_IMPLEMENTATION_PLAN.md`, `PHASE2_CHANGELOG.md`, `PHASE2_EXPERIMENT_LOG.md`; updated `README.md`, `requirements-ml.txt`, `models/README.md`.
- New optional dependency file: `requirements-text-transformer.txt`.

## Tests and known issues

- Latest full repository test run after all changes: 48 passed in 10.81 seconds; compile check passed.
- Dataset validation: passed, no group leakage/exact duplicate or label errors.
- Baseline and transformer test evaluation commands completed and wrote reports, including per-source and text-length slices.
- `.\.venv\Scripts\python.exe -m compileall -q backend training scripts`: passed after transformer files were added.
- Known limitations: small source-specific QA dataset; low source-holdout/GRADTEX results and weak Beemo edit sensitivity; no semantic near-duplicate check; transformer truncates training answers to 256 tokens and aggregates overlapping inference windows; no direct human Grammarly-correction test; no live browser viewport/assistive-tech QA; no trained Phase 3 fusion.
- Real-world failure investigation: four user-provided camera photos have been reproduced as false positives, but independent camera-set evaluation is still needed; the specific AI-text example is still missing. CIFAKE's full local 2,000-image manifest contains only 32x32 JPEGs, so high-resolution camera generalization remains unmeasured.
- Phase 1 candidate retraining: `resnet18_realworld_v1.pt` trained only on CIFAKE train rows plus four user-labelled REAL training photos; it did not improve same-split CIFAKE test accuracy versus the active strict checkpoint and remains unpromoted. Local API smoke check returned `/health` status `ok`; `/model-status` reports active strict checkpoint valid, loaded and ready on CUDA; `/text-model-status` reports baseline checkpoint available. CommunityForensics-Small was excluded after the user indicated commercial use or licensing uncertainty.
- User image examples: four supplied 1200×1600 / 900×1600 JPEGs were evaluated and all produced AI_GENERATED on all four checkpoints. The current strict model's pad transform adds approximately 25.0–43.8% black area for these portrait aspect ratios; this is a plausible OOD factor, not a proven cause.
- Text diagnostic results: MAGE OOD API-matched AI FNR 0.5913 (SVM) / 0.4063 (DistilBERT); GRADTEX Test C 0.8194 / 0.8969; GRADTEX Test B 0.8388 / 0.8698. Beemo positive-only AI recall 0.2973 / 0.3476; binary metrics remain inapplicable.
- Python/CUDA and local API checks passed. Pangram gated access and blocked browser control are outstanding validation blockers. Existing GenImage archive remains incomplete (12 missing parts); Phase 1 CIFAKE results are small benchmark evidence only, not camera-photo validation.

## Continuation files

- Created `scripts/download_text_robustness.py`, `training/evaluate_text_robustness.py`, and `backend/tests/test_text_robustness_metrics.py`.
- Modified `docs/phase2_datasets.md`, `docs/phase2.md`, `README.md`, `PHASE2_IMPLEMENTATION_PLAN.md`, `PHASE2_CHANGELOG.md`, and `PHASE2_EXPERIMENT_LOG.md` with sources, commands, measured evidence, and caveats.
- Local ignored reports: `training/reports/text_source_holdout_evaluation.json`, `text_mage_ood_evaluation.json`, and `text_mage_paraphrase_ai_evaluation.json`.
- Created `scripts/download_text_robustness_extra.py`, `training/evaluate_text_additional_robustness.py`, and `training/analyze_text_calibration.py`; updated the three evaluators, image/text UI, and API/robustness tests.
- Local reports: `text_gradtex_test_c_evaluation.json`, `text_gradtex_test_b_selected_evaluation.json`, `text_beemo_expert_edit_sensitivity.json`, and `text_calibration_analysis.json`.
- Created `training/run_image_additional_diagnostics.py` and `training/run_text_additional_diagnostics.py`; reports: `image_additional_diagnostics.json`, `image_real_world_errors.json`, and `text_additional_diagnostics.json`. `text_real_world_errors.json` remains waiting for the actual AI-text example.
- Created `docs/phase1_additional_diagnostics.md` and `docs/phase2_additional_diagnostics.md`; updated Phase 1/Phase 2 plans and changelogs and this tracker.
- Modified `backend/app/services/transformer_text_detector.py` to pad overflow windows and extended its API regression test.
- Created `training/analyze_real_world_failures.py`, `docs/real_world_failure_analysis.md`, `PHASE1_IMPLEMENTATION_PLAN.md`, and `PHASE1_CHANGELOG.md`; added awaiting-example reports for Phase 1 and Phase 2.
- Modified `backend/app/services/transformer_text_detector.py`, `backend/app/schemas/prediction.py`, its API test, `PHASE2_IMPLEMENTATION_PLAN.md`, `PHASE2_CHANGELOG.md`, and this tracker.

## Exact next action

1. Start the backend with the baseline default: `.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`.
2. Open `http://127.0.0.1:8000/ui` and try image, text, and image+text modes; automated tests covered backend responses but not browser rendering.
3. To use the transformer backend, stop the server and run `$env:TEXT_MODEL_BACKEND='transformer'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`.
4. Obtain the user's specific AI-generated text and expected label to reproduce the reported Phase 2 error using the private workflow in `docs/real_world_failure_analysis.md`.
5. For human-text grammar corrections and full product validation, obtain a licensed corpus and complete visual desktop/mobile/assistive-technology checks in an allowed browser. Phase 2 remains `IN_PROGRESS`; do not present this detector as a general authorship verifier.

## Phase 3 — multimodal fusion status (2026-11-25)

- Current phase/stage: Phase 3, final regression and freeze verification; dataset eligibility is BLOCKED.
- Completed: audited active Phase 1/2 models/configuration and environment; recorded original source/model/data hashes and fixed input predictions; made hash-verified model copies and a source/model backup; implemented isolated frozen encoders, four-way gated fusion architecture, pair-manifest feature extraction, explicit fusion-only training, held-out evaluator, separate API, and separate UI.
- Files created: `phase3/` package files, model copies, reports/baseline, backups, data schema note, and Phase 3 plan/changelog/experiment/evaluation docs.
- Files modified: appended only this Phase 3 section to the existing progress tracker and added a Phase 3 section to README. No Phase 1/2 code, model, dataset, endpoint, threshold, or frontend was edited.
- Dataset download/preparation: NOT_STARTED. MiRAGeNews and MFND were inspected but neither presently verifies eligible terms plus the required independent origin labels for genuinely paired examples.
- Architecture: frozen copied ResNet-18 512-dimensional image feature extractor + frozen copied HC3 TF-IDF sparse text feature extractor; trainable gated 4-class fusion head and auxiliary heads.
- Training/evaluation: NOT_RUN; no Phase 3 checkpoint or metrics exist. No training, test-set tuning, or data download occurred.
- Backend/UI: separate Phase 3 API and UI implemented; copied unimodal predictions are isolated; fusion endpoint reports 503 until a trained checkpoint exists. No trained multimodal fusion model exists.
- Regression: pending final execution of Phase 3 tests, full Phase 1/2 backend tests, and frozen hash/prediction comparison.
- Known blocker: dataset rights/provenance and label semantics do not yet support the four required joint origin classes; no fusion evaluation claim can be made.
- Exact next action: run the three regression commands in the Phase 3 section of README, record actual results, then obtain a suitable paired dataset with verified labels and rights before preparing or training.

### Phase 3 regression result update

- Regression/freeze verification: COMPLETED on 2026-11-25. Phase 3 suite: 6 passed; original `backend/tests`: 58 passed; `compileall phase3`: passed; original source/model/data hashes and fixed Phase 1/2 predictions: unchanged. PyTorch emitted one non-failing sparse invariant warning in an architecture unit test.
- Phase 3 training/evaluation remain NOT_RUN and BLOCKED by absence of eligible paired data; no fusion checkpoint or model metrics exist.
- Next action remains obtaining rights-cleared paired examples with independently documented image/text origin labels before preparing manifests.

## Phase 3 and unified application — final validation update (2026-11-25)

- Current stage: Phase 3 final validation is COMPLETED; data eligibility remains BLOCKED. The unified local frontend/API implementation is available.
- Completed: audited the nine named multimodal dataset candidates and license/provenance notes; added manifest preflight; implemented a unified three-mode UI, SQLite history, filtering, detail, CSV export, deletion, privacy defaults, and model-status view; validated independent modality outputs with no fake fusion result.
- Files added/changed in this continuation: `unified/` application package, UI, tests, and ignore rules; `DATABASE_SCHEMA.md`; `UNIFIED_FRONTEND_IMPLEMENTATION.md`; `FINAL_APPLICATION_TEST_REPORT.md`; `phase3/tests/test_manifest.py`; Phase 3 data comparison/license reports, training/inference default changes, docs, and tests; appended README and this tracker.
- Data: none downloaded; no split manifests/features created. The best structural lead, RU-AI, remains unapproved because the dataset and underlying source rights are unresolved; all candidate decisions and requirements are recorded under `phase3/reports/`.
- Training/evaluation: no fusion training or test evaluation; no fusion checkpoint and no Phase 3 metrics. Existing Phase 1/2 checkpoints were not changed.
- Model architecture: isolated copied ResNet-18 image encoder plus HC3 TF-IDF/SVM text encoder, frozen; trainable gated fusion heads are implemented only. The saved checkpoint default is `phase3/models/fusion/best_model.pt`; separate last/optimizer checkpoint path is `phase3/models/fusion/last_checkpoint.pt`.
- Backend/frontend: unified app on port 8000 exposes image, text, and image+text modes, records exact results, and keeps fusion null/NOT_TRAINED until a checkpoint exists. Raw input retention is disabled unless opted in. Browser UI checked at desktop and 390x844; no horizontal overflow. No screen-reader/full keyboard audit.
- Tests: final combined test result 71 passed, one non-failing sparse tensor warning; compileall passed; real-checkpoint in-process image/text/multimodal API smoke passed. All 97 frozen original source/model backup hashes match; original Phase 1/2 hash and prediction verification stayed unchanged.
- Known blockers: clear license and origin/pairing evidence for a four-combination dataset; separate human review of that evidence. Until then no Phase 3 training, evaluation, calibrated fusion output, or multimodal quality claim.
- Exact next action: obtain written confirmation of dataset/model-training/redistribution terms and documented source-grouped labels for all four modality-origin combinations. Then create reviewed `train.jsonl`, `validation.jsonl`, and `test.jsonl` per `phase3/data/README.md`, run the explicit encode/train/evaluate commands in `PHASE3_IMPLEMENTATION_PLAN.md`, and record actual results. To run the current combined app: `.\.venv\Scripts\python.exe -m uvicorn unified.app:app --host 127.0.0.1 --port 8000`.

## Detector error-analysis continuation — 2026-11-25

- Added `training/analyze_detector_predictions.py`: inference-only pass over every labeled image/text row in validation or test; records expected/predicted labels, scores, class metrics, per-source text metrics, and failures/uncertain rows without raw text or retraining.
- Ran on the configured FastAPI models using the held-out test splits. Image strict ResNet-18: 200 rows; 85.00% accuracy when uncertain counts as wrong; 85.00% coverage; 100% accuracy among covered predictions; 30 uncertain (8 REAL, 22 AI) and zero confident class swaps on this CIFAKE split. Text HC3 SVM: 531 rows; 94.16% accuracy, 96.17% macro-F1, 95.86% coverage, 98.23% accuracy on covered; 31 non-correct (22 uncertain, 4 human→AI, 5 AI→human). Source accuracy: reddit_eli5 96.64%, wiki_csai 91.63%.
- Existing user-photo analysis still shows the four user-confirmed REAL camera images labeled AI_GENERATED by the image model. This is an out-of-domain failure; the CIFAKE 32×32 benchmark does not validate high-resolution camera images. No causal factor was established here.
- Reports created under ignored `training/reports/`: `detector_error_analysis.json` and `.md`. No dataset files, checkpoints, thresholds, or raw text were changed. No training or tests were run; the new script was byte-compiled and executed over all 200 image and 531 text test rows.
- Next action: use a rights-cleared, camera-image benchmark and source-diverse text evaluation set before making any performance/generalization claim or deciding retraining/calibration changes.

## Follow-up: combined-mode model consistency fix — 2026-11-25

- Reproduced the reported mismatch: the selected `resnet18_realworld_v1.pt` checkpoint labeled all four supplied real photos REAL, while the frozen strict Phase 3 image copy labeled all four AI_GENERATED. The candidate had been trained with those four photos, so its outputs on them are in-sample only. The live and Phase 3-copy text SVM outputs matched on the same HC3 passage.
- Root cause in the unified UI/API: while fusion was untrained, `/api/v1/predict/multimodal` used the Phase 3 frozen copies, while image/text standalone modes used live FastAPI detector objects and their configured checkpoints.
- Fixed unified combined mode to call the same live Phase 1/2 detector instances as the standalone modes until a trained fusion checkpoint exists. Fusion remains null/NOT_TRAINED. History now records the actual independent model versions instead of saying the frozen copies were used.
- Verified with the candidate selected: the supplied real photo returned REAL in standalone and combined mode; the text sample returned HUMAN_WRITTEN in both; image/text model versions matched; fusion remained null. Unified tests: 3 passed; py_compile passed.
- Restart the unified server to load this code change. The standalone experimental `phase3.app` on port 8001 intentionally retains frozen copies; use the unified app on port 8000 for consistent current Phase 1/2 results.


## Phase 3 continuation audit — 2026-11-25

- Current phase/stage: Phase 3 multimodal fusion, dataset eligibility; status
  **BLOCKED** for training. The fusion architecture, feature encoder, trainer,
  evaluator, isolated API, and unified UI already exist.
- Dataset review: screened the previously documented nine candidates and
  added MS COCOAI/Defactify (96,000 rows, 7.51 GB) as a research lead. It has
  real/AI image labels and human COCO captions, but no AI-text labels. Its HF
  card has no declared license and sends users to the authors; COCO image
  rights are per-image Flickr terms. No suitable source is approved, so no
  dataset download, preparation, feature extraction, or training occurred.
- Model/training: only frozen Phase 1/2 copies are in `phase3/models/`; no
  fusion checkpoint exists. No Phase 1/2 retraining or checkpoint changes.
- Verification: Python 3.13.17; PyTorch 2.11.0+cu128; CUDA RTX 4050 available;
  free disk 28,180,164,608 bytes. Original source/model/dataset hashes and
  fixed predictions match the freeze baseline; all 97 backup hashes match. One
  later-added root diagnostic script is reported separately as an addition.
- Remaining blocker: obtain usable written dataset rights and source/group
  metadata for a genuine dataset with all four independent image/text origin
  combinations. No fusion performance metrics or trained checkpoint can be
  reported until that is satisfied.
- Next action: resolve rights and label provenance for RU-AI or MS COCOAI (plus
  a valid AI-text component); otherwise identify a compatible cleared dataset.
  Then create leakage-safe source-group splits before feature extraction.

- Final validation for this continuation: `.venv\Scripts\python.exe -m pytest phase3\tests unified\tests backend\tests -q` passed **71 tests** (one non-failing PyTorch sparse-invariant warning); `compileall -q phase3 unified` passed. The final freeze recheck reports `hashes_unchanged=true`, `predictions_unchanged=true`, and all 97/97 frozen backup files verified.

## Phase 3 rule-based integration and regression — 2026-11-25

- Current phase/stage: Phase 3 application integration is COMPLETED as a
  rule-based combination; trained neural fusion remains BLOCKED by eligible
  paired data and rights/provenance review.
- Implemented: deterministic combination of the current image/text labels in
  the unified app. Each original detector result is retained, no separate
  combined confidence is invented, and any detector failure produces partial
  output without a combined verdict. Either UNCERTAIN maps to INCONCLUSIVE.
- UI: clearly labels the mode RULE-BASED MULTIMODAL ANALYSIS, shows separate
  image/text results and a concise combined label, and explains that it does
  not check semantic consistency. Status/history identify it as rule-based.
- Files created: `phase3/fusion/rule_based_analysis.py`,
  `unified/tests/test_multimodal_rule_engine.py`,
  `PHASE3_IMPLEMENTATION_REPORT.md`, `API_DOCUMENTATION.md`.
- Files modified: `unified/app.py`, `unified/static/index.html`,
  `unified/tests/test_app.py`, `README.md`, `PHASE3_IMPLEMENTATION_PLAN.md`,
  `PHASE3_CHANGELOG.md`, `PHASE3_EXPERIMENT_LOG.md`,
  `PHASE3_EVALUATION_REPORT.md`, `FINAL_APPLICATION_TEST_REPORT.md`, and this
  tracker.
- Dataset/training/checkpoint status: no dataset downloaded, no model trained,
  no fusion checkpoint created or overwritten, and no Phase 1/2 checkpoint or
  threshold changed. No fusion metrics exist.
- Tests: `.venv\Scripts\python.exe -m pytest phase3\tests unified\tests backend\tests -q` — **84 passed**, one non-failing PyTorch sparse-invariant warning. `compileall -q phase3 unified` passed.
- UI smoke: temporary local app loaded; Image + Text and Text modes rendered;
  a 79-character/14-word text input completed and displayed a result. This is
  an interface check only, not model evaluation. No image upload was submitted
  during this browser smoke; API integration is covered by automated tests.
- Freeze: `snapshot_originals.py --verify` reports `hashes_unchanged=true`,
  `predictions_unchanged=true`; only the separately reported later-added
  diagnostic is an additive source file.
- Exact next action: resolve appropriate paired dataset rights and labels
  before neural fusion work. For current rule-based app, restart with
  `\.venv\Scripts\python.exe -m uvicorn unified.app:app --host 127.0.0.1 --port 8000`.
