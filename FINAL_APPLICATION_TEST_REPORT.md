# Final application validation report

Date: 2026-11-25

## Implemented

- A unified local FastAPI app and frontend at `unified.app:app` for Phase 1 image, Phase 2 text, and image+text analysis.
- Existing Phase 1/2 endpoints and detector code remain untouched. The unified app uses the live existing Phase 1/2 detectors in both individual and combined modes.
- Neural Phase 3 fusion remains explicitly untrained. Combined mode returns the original image/text result objects plus a rule-based label; it does not return a neural fusion score.
- New local SQLite history stores inference response, phase/model versions, time, score metadata, and input hashes. Raw inputs are off by default. History supports filter, pagination, details, CSV export, and deletion.

## Validation actually run

- `.\.venv\Scripts\python.exe -m pytest phase3\tests unified\tests backend\tests -q`: **71 passed**, one non-failing PyTorch sparse-invariant warning.
- `.\.venv\Scripts\python.exe -m compileall -q phase3 unified`: passed.
- Real-checkpoint in-process smoke: root/status/image/text/multimodal APIs returned HTTP 200; the existing held-out CIFAKE real image predicted `REAL`, the HC3 human text sample predicted `HUMAN_WRITTEN`, and image+text returned both independent outputs with fusion untrained. The test used a temporary SQLite store and did not calculate performance metrics.
- Browser: dashboard rendered at 1280×820 and 390×844; mobile document width was 375 px at a 390 px viewport (no horizontal overflow); text and image+text panels appeared; no-fusion notice was visible. Viewport reset after check. No screen-reader or complete keyboard audit was run.
- `.\.venv\Scripts\python.exe phase3\tools\snapshot_originals.py --verify`: `hashes_unchanged: true`, `predictions_unchanged: true`.
- Separate verification: all **97** original source/model backup files matched their baseline SHA-256 hashes.

## Not completed / blocked

- No eligible licensed, genuinely paired dataset with independently documented image origin and text authorship for all four combinations has been approved. RU-AI remains a research lead pending rights clarification. No data was downloaded.
- No Phase 3 features, training run, fusion checkpoint, independent fusion evaluation, or fusion metrics exist. The fusion architecture and explicit scripts are implemented, but training must wait for data approval.
- The unified UI has no screen-reader/full keyboard accessibility audit. Browser layout and API path checks do not replace that audit.

The Phase 3 implementation is **not complete as a trained detector**. The unified frontend now has an operational rule-based combined assessment; dataset eligibility blocks neural fusion training and evaluation.

## Regression follow-up

A candidate-vs-frozen-copy checkpoint mismatch was reproduced on the four
user-confirmed REAL images. The unified no-fusion combined route was changed to
call the same live Phase 1/2 detector instances as the individual modes. With
`resnet18_realworld_v1.pt` selected, image and text outputs matched between
standalone and combined requests; fusion remained null. The candidate's four
photo results are in-sample because those photos were used during its training.
Unified tests after this fix: 3 passed. The separate experimental `phase3.app`
continues to use its frozen copies.

## Rule-based multimodal update — 2026-11-25

- Regression rerun: `.venv\Scripts\python.exe -m pytest phase3\tests unified\tests backend\tests -q` — **84 passed**, one non-failing PyTorch sparse-invariant warning.
- Compile check: `.venv\Scripts\python.exe -m compileall -q phase3 unified` — passed.
- Freeze verifier: `.venv\Scripts\python.exe phase3\tools\snapshot_originals.py --verify` — baseline hashes and fixed Phase 1/2 predictions unchanged.
- Browser smoke on a temporary local instance: combined mode displayed the rule-based notice; text mode accepted a short sample and rendered a result. No image was uploaded in this browser smoke. These are UI checks, not model metrics.
- Automated tests cover rule mapping, preserving both detector outputs, partial failure behavior, form edge cases, and safe history failures. No trained fusion checkpoint or neural fusion metric exists.

## Text result presentation follow-up — 2026-11-25

- Updated only the unified UI text result card. It now identifies the class to which a score points, explains that UNCERTAIN means neither configured threshold was crossed, and labels the score as a model score rather than accuracy. The Phase 1 image result card and detector code are unchanged.
- Regression rerun: `.venv\Scripts\python.exe -m pytest phase3\tests unified\tests backend\tests -q` — **84 passed**, one non-failing PyTorch sparse-invariant warning.
- Phase 1/2 protection rerun: `.venv\Scripts\python.exe phase3\tools\snapshot_originals.py --verify` — baseline hashes and fixed predictions unchanged.
- Browser follow-up submitted the exact user-shared 483-character AI-description example to the live unified app. It still returned HUMAN with a 99.7% HUMAN score; the new text card showed “Model score: 99.7% toward HUMAN,” its score semantics, and the no-proof disclaimer. This verifies the display change and confirms the model failure remains. The analysis was saved by normal local history behavior; raw input retention is disabled by default.
- Inline JavaScript syntax validation passed; a focused renderer check passed for HUMAN, AI-leaning UNCERTAIN, and the unchanged REAL image-card branch. No checkpoint or threshold was changed. Details are in `PHASE2_IMPROVEMENT_REPORT.md`.
