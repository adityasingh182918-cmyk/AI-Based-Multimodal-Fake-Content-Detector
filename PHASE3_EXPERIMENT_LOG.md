# Phase 3 experiment log

## 2026-11-25 — implementation and data feasibility

- Active image model: copied CIFAKE ResNet-18 strict checkpoint; preprocessing
  and thresholds were read from its checkpoint/metadata.
- Active text model: copied HC3 TF-IDF calibrated linear SVM with the existing
  chunking and threshold policy.
- Runtime audit: Python 3.13.17, PyTorch 2.11.0+cu128, CUDA enabled on NVIDIA
  GeForce RTX 4050 Laptop GPU; pre-copy free disk was approximately 26.89 GiB.
- Baseline fixed inputs and SHA-256 hashes are in
  `phase3/reports/original_state_baseline.json`. Model copies match original
  artifact hashes. A separate hash-verified source/model backup exists.
- MiRAGeNews: HF card lists 15,000 paired records, 3.37 GB, train/validation
  plus five tests; summary distinguishes NYT/Midjourney in train/validation
  and held-out publishers/generators in tests. Visible task label is `real` /
  `fake`, license not listed on card, and the labels do not independently
  attest both modalities' origin. Not downloaded or used.
- MFND: IJCAI paper/repository describe multimodal manipulation/fake-news
  detection and four combinations; license and equivalence to AI-origin labels
  were not verified. Not downloaded or used.
- No training, tuning, checkpoint creation, or test-set use occurred.
- Fusion evaluation: NOT RUN — missing eligible paired manifests and trained
  fusion checkpoint. No metrics are available.

## Final implementation validation — 2026-11-25

- `python -m compileall -q phase3`: passed.
- `python -m pytest phase3/tests -q`: 6 passed; one non-failing PyTorch sparse
  invariant warning from the architecture test.
- `python -m pytest backend/tests -q`: 58 passed.
- `python phase3/tools/snapshot_originals.py --verify`: hashes unchanged and
  fixed image/text predictions unchanged; details saved in
  `phase3/reports/original_state_postcheck.json`.
- The new API integration test exercises copied image and text predictions,
  invalid-image HTTP 400, and untrained-fusion HTTP 503. These checks validate
  the isolated path, not detector generalization or fusion quality.

## Experiment protocol once data is approved

Use group-aware, source-aware train/validation/test manifests, with no shared
document/prompt/group across splits. Fit only fusion layers against training
features, select best epoch by validation macro-F1, and report once on held-out
test. Existing encoder test outputs on exactly the same paired rows provide
the comparison baseline. The test split is never used for thresholds, model
selection, or training.

## 2026-11-25 — final dataset and application validation

- Audited official dataset cards/repos for MiRAGeNews, OpenFake, DGM4,
  MMFakeBench, Fakeddit, M4FC, GenImage, VisualNews, and RU-AI. Detailed
  provenance, size, access and license findings are in
  `phase3/reports/dataset_comparison.md` and
  `phase3/reports/dataset_license_audit.md`. No source passes all gates;
  RU-AI is a research lead only. No download, extraction, or training occurred.
- Added manifest preflight checks. The small synthetic fixture tests validate
  the guards; they are not research data and produce no model results.
- Regression: `.venv\Scripts\python.exe -m pytest phase3\tests unified\tests backend\tests -q`
  passed 71 tests. One non-failing PyTorch sparse-invariant warning remains.
  `compileall -q phase3 unified` passed.
- In-process smoke with actual project checkpoints and existing held-out
  examples: UI/status/image/text/multimodal routes returned HTTP 200; image
  sample output was REAL, human text sample output was HUMAN_WRITTEN, and
  paired route returned both independent predictions with fusion `NOT_TRAINED`.
  This is a route smoke test, not a performance metric. It used a temporary
  database that was removed after the test.
- Browser validation: desktop dashboard rendered; mobile viewport 390×844
  rendered without horizontal overflow; text and image+text panels and the
  honest no-fusion notice were visible. Browser viewport reset after check.
  No screen reader or full keyboard audit was performed.
- Frozen project verification: baseline comparison reports both hashes and
  predictions unchanged. All 97 files in the original source/model backup
  match the baseline SHA-256 entries.
- There are still no eligible fusion training/evaluation examples, features,
  checkpoint, or fusion test metrics. Do not train until data rights and
  independent paired origin labels are resolved.

## 2026-11-25 — continuation audit and freeze verification

- Rechecked official dataset cards and papers for current access, label scope,
  and terms. Added MS COCOAI/Defactify as a research lead: 96,000 rows (7.51
  GB), real and five-generator image labels, and human-written MS COCO captions
  shared between real and generated-image examples. It has no AI-authored text
  labels. The HF card provides no license and directs questions to the authors;
  COCO annotations are CC BY 4.0 but image rights remain under per-image Flickr
  terms. No download or contact was made.
- No candidate now passes all paired-label and intended-use license gates. No
  feature extraction, training, checkpoint creation, or test evaluation ran.
- Freeze verification now compares baseline paths and reports additive files
  separately. Results: all baseline source/model/dataset hashes unchanged; fixed
  Phase 1/2 predictions unchanged. One later-added diagnostic script is
  reported as an addition, not a Phase 1/2 edit. All 97 frozen backup files
  match their SHA-256 baselines.
- Runtime remains Python 3.13.17, PyTorch 2.11.0+cu128, CUDA available on the
  RTX 4050. Disk free at audit: 28,180,164,608 bytes (about 26.25 GiB).

## Rule-based application validation — 2026-11-25

- No experiment/training was run. Implemented the deterministic decision table
  in the unified app only; it operates on live Phase 1/2 labels and retains the
  respective detector output objects/scores. It does not produce a learned
  fused score or check image/text semantic agreement.
- Browser smoke in a temporary local unified app: Image + Text page displayed
  the rule-based notice; Text mode accepted a 79-character, 14-word sample and
  rendered a HUMAN result with estimated score. This is a UI plumbing check,
  not an independent evaluation result.
- Regression suite: 84 passed, 1 non-failing sparse-invariant warning;
  compileall passed. Freeze check: original Phase 1/2 hashes and fixed
  predictions unchanged.
- Neural fusion data, training, and evaluation remain blocked by dataset
  licensing and four-way independent label/provenance requirements.
