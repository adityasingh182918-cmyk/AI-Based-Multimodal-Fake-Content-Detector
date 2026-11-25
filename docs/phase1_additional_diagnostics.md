# Phase 1 additional diagnostics

**Status: completed for local checkpoints, CIFAKE validation data, and four user-submitted examples.** The user identified the four submitted photographs as real. Their provenance was not independently verified. No checkpoint was changed and no training was run.

## Checkpoint and API audit

At the time of this historical checkpoint audit, FastAPI defaulted to `models/image_detector/resnet18_cifake_aug_v1_calibrated_strict.pt`; `IMAGE_MODEL_CHECKPOINT` overrides it. On 2026-11-25, a repeat report from the running unified app showed the strict checkpoint still active and reproducing the user's room-photo false positive. The source default has since been changed to `models/image_detector/resnet18_realworld_v1.pt`; restart the server for it to take effect. Four of four `.pt` checkpoints passed the existing ResNet-18/class-order loader:

| Checkpoint | Role | Input preprocessing | Validation raw argmax (n=200) | REAL FPR | AI FNR | Abstentions |
|---|---|---|---:|---:|---:|---:|
| `resnet18_best.pt` | Original baseline | 224×224 square stretch, ImageNet normalization | 0.930 | 0.090 | 0.050 | 0 |
| `resnet18_cifake_aug_v1.pt` | Augmented checkpoint | 224×224 aspect-preserving pad, ImageNet normalization | 0.960 | 0.040 | 0.040 | 0 |
| `resnet18_cifake_aug_v1_calibrated.pt` | Augmented + validation thresholds | Same pad preprocessing | 0.960 | 0.040 | 0.040 | 0 |
| `resnet18_cifake_aug_v1_calibrated_strict.pt` | **FastAPI default** | Same pad preprocessing; REAL if AI score ≤0.251990, AI if ≥0.984008 | 0.960 | 0.040 | 0.040 | 35/200 (17.5%) |

Accuracy/FPR/FNR in the table use raw argmax predictions on the existing CIFAKE validation split. For the strict checkpoint, the stored abstention policy covered 165/200 validation examples; thresholded predictions are separately recorded in the JSON report. Image `class_scores` are softmax outputs, not calibrated probabilities.

## Supplied real-photo failures

### Repeat false positive from the running unified app (2026-11-25)

The browser screenshot's room photo was available locally as `WhatsApp Image 2026-11-25 at 02.08.13.jpeg`. The live `/api/v1/status` response confirmed that the process on port 8000 had loaded `resnet18_cifake_aug_v1_calibrated_strict.pt`. Replaying the same file through both saved checkpoints on CPU produced:

| Checkpoint | Output | REAL score | AI_GENERATED score |
|---|---|---:|---:|
| `resnet18_cifake_aug_v1_calibrated_strict.pt` (then-running default) | AI_GENERATED | 0.000078 | 0.999922 |
| `resnet18_realworld_v1.pt` (new source default) | REAL | 0.687643 | 0.312357 |

This reproduces the screenshot's confident AI false positive and identifies the immediate mismatch: the running app used the strict CIFAKE-only checkpoint, while the existing real-world checkpoint classified this specific image as REAL. `backend/app/main.py` now defaults to the real-world checkpoint; model weights, thresholds, and preprocessing were not changed. This is a sample-specific correction, not evidence of general camera-photo accuracy. The currently running server process must be restarted before its loaded checkpoint changes.

All four inputs were JPEGs. Their original dimensions were 1200×1600 for samples 01–02 and 900×1600 for samples 03–04. Every checkpoint predicted `AI_GENERATED` for every sample. On the FastAPI default strict checkpoint, AI softmax scores were:

| Sample | Size | Approximate black padding area under square pad | AI score | Result |
|---|---:|---:|---:|---|
| user_image_01 | 1200×1600 | 25.0% | 0.999918 | AI_GENERATED |
| user_image_02 | 1200×1600 | 25.0% | 0.999458 | AI_GENERATED |
| user_image_03 | 900×1600 | 43.8% | 0.995656 | AI_GENERATED |
| user_image_04 | 900×1600 | 43.8% | 0.999119 | AI_GENERATED |

These are reproducible false positives under the user-provided REAL labels. They are not evidence of the photos' provenance. All checkpoints agree, so selecting the older checkpoint does not fix these cases. Reports use sample IDs and omit source filenames, paths, and image bytes.

## Validation-only preprocessing sensitivity

The local CIFAKE validation set has 200 32×32 JPEGs, all square. The strict checkpoint was held fixed. Each row is a transformed view of the same validation images, not a new independent real-world test set.

| Validation view | Accuracy | REAL FPR | AI FNR | Raw predictions changed vs checkpoint preprocessing | Mean absolute AI-score change |
|---|---:|---:|---:|---:|---:|
| Checkpoint pad (reference) | 0.960 | 0.040 | 0.040 | 0 | 0 |
| Native 32×32 input, no upsize | 0.675 | 0.210 | 0.440 | 65/200 | 0.3587 |
| Square stretch to 224 | 0.960 | 0.040 | 0.040 | 0/200 | 0 |
| JPEG quality 95 | 0.960 | 0.040 | 0.040 | 0/200 | 0.0100 |
| JPEG quality 70 | 0.950 | 0.060 | 0.040 | 2/200 | 0.0245 |
| Mild Gaussian blur (radius 0.5) | 0.825 | 0.010 | 0.340 | 33/200 | 0.1839 |
| Brightness ×0.90 | 0.960 | 0.040 | 0.040 | 2/200 | 0.0190 |
| Brightness ×1.10 | 0.970 | 0.020 | 0.040 | 2/200 | 0.0248 |

Padding and square stretch agree on CIFAKE because all source images are already square. No high-resolution camera image exists in the validation set, so the effect of high-resolution detail cannot be measured with this validation-only experiment. The four user photos are diagnostic examples, not validation data.

## Diagnosis and priority

**Verified:** the API loads the strict checkpoint; class order is `REAL`, `AI_GENERATED`; inference uses EXIF orientation, RGB conversion, checkpoint normalization, and the saved resize mode. The user photos are portrait JPEGs while CIFAKE training/validation images are square 32×32 JPEGs. The model then pads portrait inputs with black side regions, which occupy approximately 25–44% of these images.

**Likely domain-shift factor, not proven cause:** the large, high-confidence outputs and large padding regions are consistent with an out-of-domain failure. The controlled CIFAKE check confirms strong sensitivity to blur and scale, but it does not show that padding or blur caused these particular false positives. All four photos producing the same error across checkpoints points to a shared training-data/domain limitation, not a version-specific checkpoint bug.

**Highest-priority improvement:** build a permissioned camera-photo evaluation set across phones, scenes, aspect ratios, and messaging-app compression. Keep it isolated from training and threshold selection. Add diverse high-resolution real photos and generator-diverse AI images to training only after creating source-aware train/validation/test partitions. Use validation data to compare aspect-preserving crops/padding, camera/compression augmentation, blur, noise, and resolution policies. Re-evaluate on an untouched camera set before claiming improvement.

Full validation predictions, all checkpoint scores for the four examples, and per-variant rows are in [image_additional_diagnostics.json](../training/reports/image_additional_diagnostics.json). The compact user-example report is [image_real_world_errors.json](../training/reports/image_real_world_errors.json).
