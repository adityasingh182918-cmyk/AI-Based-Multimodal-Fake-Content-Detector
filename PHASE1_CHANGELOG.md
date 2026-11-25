# Phase 1 changelog

## 2026-11-25 — Supplemental real-photo fine-tuning

- Added a license-aware dataset decision record. OpenFake core exceeds available disk capacity; CommunityForensics-Small was excluded because its CC BY-NC-SA 4.0 license is non-commercial research only and the user indicated commercial use may be needed or is uncertain.
- Added `scripts/import_user_real_images.py` to preserve and inventory the four user-labelled REAL photos and `scripts/prepare_realworld_manifest.py` to combine them with the unchanged CIFAKE manifest as train-only samples.
- Added compatible `--init-checkpoint` support and atomic per-epoch last checkpoints with optimizer/scheduler/scaler states to the image trainer. Existing checkpoints are protected from overwrite.
- Fine-tuned ResNet-18 into `models/image_detector/resnet18_realworld_v1.pt` from the active strict checkpoint. Run ID `20261124T184355405329Z`; 3 epochs before early stopping; best validation-loss epoch 1.
- Evaluated baseline, augmented, strict, and candidate checkpoints on the same untouched CIFAKE test set. Candidate ties strict at 97.5% accuracy, 4% REAL FPR, 1% AI FNR; ROC-AUC 0.9958 vs 0.9955. No meaningful improvement was demonstrated, so the FastAPI default and thresholds remain unchanged.
- Candidate predicts REAL for the four user photos after they were used in training; this in-sample result is explicitly not treated as evaluation. Independent camera-photo validation remains blocked.
- Added three unit tests for photo import/deduplication, split preservation, and output overwrite refusal.
- Final verification: full pytest suite 51 passed in 55.28 seconds; `python -m compileall -q backend training scripts` passed. No Phase 2 files were changed.

## 2026-11-24 — Real-world failure investigation

- Audited image checkpoint selection, label order, EXIF/RGB handling, resize mode, normalization, training data resolution, API selection, and existing CIFAKE reports.
- Added the private-sample analyzer and an image failure report that records the missing real-world example without fabricating a prediction.
- Verified no class-order or preprocessing mismatch in the inspected image code. Camera-domain mismatch remains a well-supported hypothesis, not a reproduced cause.
- Preserved all existing image checkpoints; no training or threshold changes.
- Backend compile check and full regression suite passed after the investigation changes: 48 tests.

## 2026-11-24 — Additional checkpoint and preprocessing diagnostics

- Compared all four valid image checkpoints on the same 200-image CIFAKE validation split. The FastAPI default strict checkpoint was confirmed present and loadable.
- Evaluated four user-provided photos labeled REAL by the user: all four checkpoints predicted AI_GENERATED for all four photos. Strict-checkpoint AI softmax scores were 0.995656–0.999918. Inputs and paths were not copied into reports.
- Measured validation-only sensitivity to 32×32 native resolution, pad/stretch, JPEG recompression, blur, and brightness. Results are documented as correlated perturbations, not independent accuracy estimates.
- No checkpoint or deployed threshold was changed; no training was run.
