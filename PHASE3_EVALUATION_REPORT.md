# Phase 3 evaluation report

**Status: RULE-BASED API VALIDATED; NEURAL FUSION EVALUATION BLOCKED.**

No accuracy, precision, recall, F1, confusion matrix, or generalization claim
is reported. The fusion network and evaluation pipeline exist, but the
evaluation script has not been run. The baseline Phase 1/2 fixed predictions
are recorded only for freeze/regression comparison in
`phase3/reports/original_state_baseline.json`; they are not Phase 3 fusion
metrics.

The primary blocker is evidence-quality and rights suitability: candidates
found either lack a verified license for the user's intended use, lack
independent image/text origin labels, or measure image manipulation/fake-news
instead of the specified AI-origin task. No data was downloaded. Resolve the
dataset terms and label semantics before preparing splits or training.

No neural fusion model is trained, deployed, calibrated, or evaluated. The
standalone `phase3.app` checkpoint-based fusion endpoint returns HTTP 503 while
no checkpoint exists. The unified app separately offers a rule-based verdict;
it is validated as software behavior only, not as a predictive model. No
accuracy or other model metric is applicable to that fixed decision table.

## Implementation regression (not model evaluation)

- Phase 3 tests: 6 passed.
- Existing backend regression suite: 58 passed.
- Phase 3 compile check: passed.
- Frozen Phase 1/2 hash and fixed prediction comparison: unchanged, as recorded
  in `phase3/reports/original_state_postcheck.json`.

These results establish code-path and preservation checks only. They are not
fusion accuracy metrics and do not resolve the dataset blocker.

## Latest implementation regression (2026-11-25)

- Combined Phase 3, unified, and existing backend regression suites: 71 passed.
- Phase 3 and unified compile check: passed.
- Unified API smoke with actual frozen Phase 1/2 checkpoints: image, text, and
  multimodal endpoints returned HTTP 200; fused prediction remained null.
- Unified UI rendered at desktop and 390×844 mobile viewport without horizontal
  overflow. This is a visual layout check, not an assistive-technology audit.
- All 97 frozen source/model backup file hashes matched the baseline.
- Frozen Phase 1/2 hash and fixed prediction comparison remained unchanged,
  as recorded in `phase3/reports/original_state_postcheck.json`.

## Continuation audit (2026-11-25)

The additional MS COCOAI/Defactify source was screened but not downloaded or
evaluated. It provides image-origin labels and human-written captions only;
it does not label AI-authored text. The dataset card directs license questions
to its authors, and COCO image permissions remain source-specific. No Phase 3
training, held-out evaluation, or fusion metric exists.

Freeze check: baseline hashes unchanged, fixed Phase 1/2 predictions
unchanged, and 97/97 frozen backup files match. Regression tests are recorded
after this audit in `PROJECT_PROGRESS.md`.

## Rule-based integration validation (2026-11-25)

- The unified endpoint applies the documented decision table to current
  Phase 1/2 labels, preserves both raw modality response objects, and returns
  no combined verdict when either detector fails. Either UNCERTAIN label maps
  to INCONCLUSIVE.
- This is a deterministic application rule, not a learned or calibrated
  multimodal model. There are no model-quality metrics for this behavior.
- Regression: 84 tests passed across Phase 3, unified, and backend suites;
  compile check passed; browser text-mode smoke returned a result. The freeze
  verifier reports Phase 1/2 hashes and fixed predictions unchanged.
- The paired-data blocker remains open; neural fusion training and evaluation
  were not run.
