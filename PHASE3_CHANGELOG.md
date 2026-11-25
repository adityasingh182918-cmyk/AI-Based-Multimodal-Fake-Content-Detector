# Phase 3 changelog

## Initial isolated implementation

- Audited active Phase 1 image and Phase 2 text model configurations.
- Saved original source/model/data hashes and fixed prediction baselines.
- Copied both active model artifacts into Phase 3 and made a hash-verified
  backup of original source/model files. No Phase 1/2 implementation, data,
  checkpoint, threshold, or endpoint was edited.
- Added frozen feature encoders, sparse gated four-way fusion architecture,
  split validation/feature extraction, explicit fusion-only training, and
  held-out test evaluation code.
- Added standalone Phase 3 FastAPI routes and a separate UI. Multimodal route
  remains unavailable until a trained checkpoint is explicitly produced.
- Dataset research did not verify a suitable licensed, genuine-pair dataset
  with independent four-way labels. No dataset download, feature extraction,
  training, or evaluation has occurred.
- Regression completed: Phase 3 tests passed (6); original backend tests passed
  (58); compile check passed. Frozen-state verification returned
  `hashes_unchanged: true` and `predictions_unchanged: true` and saved
  `phase3/reports/original_state_postcheck.json`.

## Continuation: dataset audit, unified app, and final checks — 2026-11-25

- Screened MiRAGeNews, OpenFake, DGM4, MMFakeBench, Fakeddit, M4FC, GenImage,
  VisualNews, and RU-AI. Decisions are normalized to APPROVED FOR TRAINING,
  APPROVED FOR EVALUATION ONLY, RESEARCH LEAD, or REJECTED in the comparison;
  none is approved. No data was downloaded. RU-AI remains blocked on data
  rights and provenance clarification.
- Added preflight checks for four joint classes per split, group leakage,
  cross-split exact image/text duplicates, row provenance, path containment,
  and image integrity before feature extraction.
- Aligned Phase 3 checkpoint defaults to `best_model.pt`; the explicit trainer
  saves optimizer/scheduler state to a separate `last_checkpoint.pt`, records
  learning-rate history, uses validation-loss LR scheduling, and refuses to
  overwrite either checkpoint.
- Created the single local `unified.app` application and UI. Phase 1 and Phase
  2 code/models remain unchanged. History uses a new local SQLite store and
  raw inputs are off by default. Image and multimodal selections are kept
  separately so changing modes cannot silently swap the selected image.
- Final checks: 71 tests pass across Phase 3, unified, and original backend;
  compileall passes; production in-process API smoke returned HTTP 200 for
  image/text/combined requests; combined output carried independent modality
  results and no fusion output. Browser checked desktop and 390px mobile layout;
  no horizontal overflow. All 97 source/model backup hashes match, and the
  original hash/prediction postcheck remains unchanged.
- No Phase 3 model training or test-set evaluation occurred. Dataset rights
  remain the training blocker; no fusion metrics or checkpoint exist.

## Follow-up: unified mode consistency correction

The no-fusion image+text route now reuses the same currently configured live
Phase 1 and Phase 2 detectors as the individual modes, so a custom checkpoint
selection is consistent across all three modes. It still does not create a
fusion prediction. The candidate-vs-frozen-copy mismatch was reproduced on
four user-confirmed REAL photos; because those photos were used to train the
candidate checkpoint, that comparison is an in-sample debugging check only.
After the code change, the unified tests passed (3) and a candidate-configured
API smoke produced matching standalone/combined modality predictions with
`fusion_status=NOT_TRAINED`. Restart the unified server before using it.

## Continuation: source recheck and integrity reconciliation — 2026-11-25

- Added the official MS COCOAI/Defactify source to the candidate audit. It
  provides real/AI images with human COCO captions but no AI-text labels, no
  declared dataset license, and unresolved per-image COCO/Flickr terms. It is
  a research lead only; no data was downloaded.
- Corrected `phase3/tools/snapshot_originals.py --verify` so paths in the freeze
  baseline are compared exactly and later additive files are reported
  separately. A pre-existing diagnostic script added after the freeze had
  caused a false `hashes_unchanged: false`; no baseline file was altered.
- Reverification: baseline Phase 1/2 source, checkpoint, and dataset hashes
  match; frozen fixed-input predictions match; all 97 backup file hashes
  match. No model, dataset, or original Phase 1/2 code was changed.
- No Phase 3 model trained; dataset clearance and complete four-way labels
  remain the blockers.

- Final validation after the audit/tooling adjustment: 71 tests passed across
  Phase 3, unified UI/API, and backend suites; `compileall -q phase3 unified`
  passed. The only warning is PyTorch's non-failing sparse-invariant warning.

## Rule-based multimodal integration — 2026-11-25

- Added `phase3/fusion/rule_based_analysis.py` with a transparent decision
  table. The unified image+text API now runs the same live Phase 1/2 detector
  instances used by the individual API routes, preserves both original result
  objects, and emits no combined verdict when either detector fails.
- Unified UI now clearly says RULE-BASED MULTIMODAL ANALYSIS, displays a short
  combined verdict and separate modality results, and keeps diagnostic JSON in
  optional technical details. It explicitly says semantic consistency is not
  checked and no trained fusion network is available.
- Status and history identify the rule-based result accurately. History write
  failures are reported without failing otherwise successful inference.
- Added coverage of the decision table, detector partial failures, request
  edge cases, original modality result preservation, history/CSV, and history
  database errors.
- Regression: 84 tests passed; `compileall` passed; browser text-mode smoke
  succeeded; frozen Phase 1/2 hashes and fixed predictions remain unchanged.
- No training, threshold adjustment, checkpoint replacement, or dataset
  download occurred. Neural fusion remains blocked by eligible paired data and
  rights/provenance review.
