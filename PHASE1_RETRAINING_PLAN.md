# Phase 1 Real-World Retraining Plan

Updated: 2026-11-25

## Stage 0 — Audit (completed)

- Project root: `C:/Users/Lenovo/Desktop/imad_project`.
- Image model: ResNet-18, RGB, ImageNet mean/std, 224 px, aspect-preserving pad resize in the active preprocessing path.
- Existing image dataset: local CIFAKE subset with 1,600 train, 200 validation, and 200 held-out test images; source images are 32x32 JPEG.
- Existing checkpoints (all preserved): `resnet18_best.pt`, `resnet18_cifake_aug_v1.pt`, `resnet18_cifake_aug_v1_calibrated.pt`, `resnet18_cifake_aug_v1_calibrated_strict.pt`.
- FastAPI default: `resnet18_cifake_aug_v1_calibrated_strict.pt`, configurable with `IMAGE_MODEL_CHECKPOINT`.
- Four user-provided photos are present under `C:/Users/Lenovo/Downloads/`; the user labels them REAL. The current checkpoints predict AI_GENERATED for all four. This is a reproduced error for the user-provided labels, not independently verified image provenance.
- Environment: Python 3.13, PyTorch 2.11.0+cu128, CUDA available, RTX 4050 Laptop GPU, about 5.8 GiB reported free VRAM at audit time.
- Disk: about 27.9 GB free on C: at audit time. This rules out huge corpora and requires checking space again before any download.
- Current trainer always initializes from torchvision pretrained weights and refuses to overwrite its target checkpoint; it does not yet accept an existing checkpoint as initialization.

## Dataset investigation

- Original CIFAKE train/validation/test split will remain unchanged and will be retained in comparisons.
- User photos may be added to training only. They will not be counted as independent test evidence after use in training.
- OpenFake was inspected as a candidate. Its core training shards total roughly 204 GB and its test shards roughly 67.7 GB; it is infeasible on the available disk and is rejected without download.
- OpenFake's core split is not feasible on this disk (about 204 GB train and 67.7 GB test, by the current Hub file listing).
- CommunityForensics-Small is a useful paired dataset but declares CC BY-NC-SA 4.0 and non-commercial research-only use. The user answered that commercial use is possible or uncertain, so it was not used. A temporary inspection download was removed after this decision.
- Other investigated candidates either lacked a complete public mirror/metadata or carried image/model licensing uncertainty. No external multi-generator dataset has been added.

## Planned implementation and training

1. Preserve the current CIFAKE data and every checkpoint.
2. Four user-labelled REAL examples were copied into ignored `data/image_detection/user_real/train/`; raw files are unchanged, dimensions/EXIF/file and pixel hashes are recorded, and duplicates are checked. All four are training-only.
3. Added `scripts/prepare_realworld_manifest.py`; it preserves the original CIFAKE 1,600/200/200 split rows and adds the four user photos only to train. It checks corrupt files, hashes, exact duplicates, and near duplicates among the supplemental photos.
4. Added compatible-checkpoint initialization and per-epoch last/optimizer state saving to the trainer. It refuses to overwrite best or last checkpoint targets.
5. Fine-tuned from `resnet18_cifake_aug_v1_calibrated_strict.pt` on CUDA with 224px pad preprocessing, AdamW 1e-5, batch 8, AMP, seed 42, and early stopping. It stopped after 3 epochs; best validation-loss checkpoint is epoch 1.
6. Evaluated baseline, augmented, active strict, and candidate models on the same unchanged 200-image CIFAKE test split. Results are in `training/reports/eval_realworld_compare_*.pt.json`.
7. Compared candidate vs previous checkpoints on the four user photos only as an explicitly in-sample check. The candidate predicts REAL for each, but this is not evaluation evidence because those photos were included in training.
8. Kept the active FastAPI default and its thresholds unchanged. The candidate is separately selectable with `IMAGE_MODEL_CHECKPOINT` for local inspection.

## Current status

- Stage 0 audit: COMPLETED.
- User-photo copying and metadata: COMPLETED.
- External subset/license/disk verification: BLOCKED for a commercially safe, complete multi-generator dataset within the reviewed sources; the NC-SA corpus was declined by the user.
- Loader and trainer changes: COMPLETED.
- Training experiment: COMPLETED; candidate saved separately and not promoted.
- CIFAKE held-out evaluation: COMPLETED; no measurable test accuracy/FPR/FNR improvement over active strict model.
- Independent camera-photo evaluation: BLOCKED; user photos are training examples and no licensed, independent camera set was obtained.
- Phase 1 report/docs: COMPLETED (`PHASE1_RETRAINING_REPORT.md`, README, progress, plan, and changelog updated).
- Regression: COMPLETED (51 tests passed; compileall passed).

## Constraints

- No existing checkpoint or dataset may be deleted or overwritten.
- The user's photos are supplemental training data only if used; user-provided labels are retained as such.
- Existing CIFAKE test remains untouched. Any external test split remains locked.
- No Phase 2 training, Phase 3 fusion, or edits to Phase 2 implementation.
- No claim of improvement without actual held-out measurements.
