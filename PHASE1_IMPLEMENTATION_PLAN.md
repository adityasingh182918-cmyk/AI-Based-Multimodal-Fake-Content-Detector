# Phase 1 implementation plan

This plan tracks the CIFAKE image detector and the real-world camera-photo failure investigation. `COMPLETED` describes verified repository work, not proof of general image-authenticity detection.

| Stage | Status | Scope |
|---|---|---|
| CIFAKE pipeline and ResNet-18 | COMPLETED | 2,000-image reproducible CIFAKE sample, train/validation/test manifests, versioned baseline and augmented checkpoints, validation-only abstention calibration. |
| FastAPI image inference | COMPLETED | EXIF orientation, RGB conversion, checkpoint-specific 224px preprocessing, class mapping, strict calibrated checkpoint by default. |
| CIFAKE evaluation | COMPLETED | Baseline and augmented checkpoints evaluated on the same 200-image held-out CIFAKE test split; results are benchmark-specific. |
| Camera-photo reproduction | COMPLETED | Four user-provided JPEGs labeled REAL by the user were evaluated on all four valid checkpoints; all were predicted AI_GENERATED. Source provenance was not independently verified. |
| Checkpoint comparison and validation sensitivity | COMPLETED | Four frozen checkpoints compared on the same 200-image validation set; strict default tested under scale, resize, JPEG, blur, and brightness variants. |
| Independent camera-photo validation | BLOCKED | The four user-labelled REAL photos were added to training and cannot serve as evaluation. A commercially suitable, complete representative camera-photo evaluation source was not available in the reviewed sources. |
| Real-world model improvement experiment | COMPLETED (not promoted) | Fine-tuned a separate candidate from the strict checkpoint using unchanged CIFAKE train/validation plus four user-labelled REAL photos. Candidate ties strict on CIFAKE test; this is not evidence of camera-domain improvement. |

## Verified checkpoint comparison

On the same 200 CIFAKE test samples, baseline `resnet18_best.pt` had REAL false-positive rate 0.08 and AI false-negative rate 0.01. The augmented strict checkpoint had REAL false-positive rate 0.04 and AI false-negative rate 0.01. Both checkpoints were trained/evaluated on 32×32 CIFAKE content. These numbers do not establish phone-camera performance.

## 2026-11-25 supplemental fine-tuning experiment

- The four user-labelled REAL photos were copied to ignored `data/image_detection/user_real/train/` and included only in training. They are not independent evidence and their labels were not independently verified.
- `scripts/prepare_realworld_manifest.py` retained original CIFAKE train/validation/test assignments and added the user photos only to train.
- The separate candidate `models/image_detector/resnet18_realworld_v1.pt` was initialized from the current strict checkpoint. ResNet-18, 224px aspect-preserving pad preprocessing, ImageNet normalization, AdamW, learning rate 1e-5, batch size 8, mixed precision, seed 42, and early stopping were used. CUDA RTX 4050 was available.
- Early stopping ended after epoch 3; best validation-loss epoch was 1 (validation accuracy 96.5%, loss 0.1303). Epoch 2/3 validation accuracies were 96%/95%.
- On the unchanged CIFAKE test (n=200), strict and candidate both had 97.5% accuracy, 4% REAL false-positive rate, 1% AI false-negative rate, and AI F1 0.9754. ROC-AUC was 0.9955 for strict and 0.9958 for candidate. The tiny AUC change does not justify promotion.
- The candidate predicted all four user photos as REAL after training on them. This is in-sample behavior, not a test result.
- No external training/evaluation image dataset was added. OpenFake was too large for the verified disk space. CommunityForensics-Small was not used after the user indicated commercial use is possible or uncertain and the source's CC BY-NC-SA 4.0 terms restrict it to non-commercial research.
- FastAPI continues to default to `resnet18_cifake_aug_v1_calibrated_strict.pt`; the candidate can be selected locally using `IMAGE_MODEL_CHECKPOINT`. No deployed thresholds were changed. Candidate scores are uncalibrated softmax.

## Exact next action

Obtain a commercially usable, provenance-documented, high-resolution real/synthetic dataset that can remain independent for testing; build a locked source-aware camera evaluation before any further model promotion.
