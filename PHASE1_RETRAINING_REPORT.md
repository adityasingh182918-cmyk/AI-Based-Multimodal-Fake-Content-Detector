# Phase 1 Real-Photo Fine-Tuning Report

Updated: 2026-11-25

## Outcome

A separate ResNet-18 candidate was trained from the existing strict CIFAKE checkpoint using the original CIFAKE training set plus four photos that the user labelled REAL. The original CIFAKE split and all old checkpoints remain unchanged. The candidate ties the strict checkpoint on the unchanged CIFAKE test set; this experiment does **not** establish improved real-world generalization. The FastAPI default remains the existing strict checkpoint.

## Project and data audit

- Architecture: ResNet-18, RGB, 224px, aspect-preserving pad resize, ImageNet mean/std.
- Existing image data: CIFAKE subset with 1,600 train, 200 validation, 200 test; all source images are 32×32 JPEGs.
- Existing checkpoint comparison and active default are recorded in `PHASE1_IMPLEMENTATION_PLAN.md`.
- The four user photos were copied unchanged to ignored `data/image_detection/user_real/train/`. Inventory stores image dimensions, format, EXIF orientation, file and oriented-pixel SHA-256, and user-provided label provenance. All four are train-only. Labels are not independently verified.
- `scripts/prepare_realworld_manifest.py` keeps every CIFAKE row in its original split and appends the four supplemental samples only to train. Exact file duplicates and near-duplicates among those user samples are checked.
- Data and checkpoints remain outside Git tracking.

## Additional source investigation

- [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake): core train shards total about 204 GB and test about 67.7 GB by the Hub listing, so the corpus is too large for the available disk space.
- [CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small): paired real/AI data with broad generator coverage, but the declared CC BY-NC-SA 4.0 license restricts it to non-commercial research. The user indicated commercial use may be needed or uncertain, so it was excluded and not used for training.
- [AIGI-Detection-Quality-Paradox](https://huggingface.co/datasets/Coxy7/AIGI-Detection-Quality-Paradox): the repository has a generated-image split and real-image metadata only; its card says real images are not redistributed because of source licenses. It was not selected as a balanced training source.
- [WasItAIGenerated frontier benchmark](https://huggingface.co/datasets/wasitaigeneratedcom/frontier-image-video-bench-2026-10): the public mirror currently lists only 40 image assets and no `metadata.csv`, while its card describes hundreds of public images. It is incomplete for a reliable source-aware experiment.
- No suitable complete, license-compatible multi-generator training/evaluation dataset was obtained. The resulting experiment is explicitly limited to CIFAKE plus the four user-labelled photos.

## Training configuration and artifacts

- Initialization: `models/image_detector/resnet18_cifake_aug_v1_calibrated_strict.pt`.
- Best candidate: `models/image_detector/resnet18_realworld_v1.pt`.
- Last-epoch/optimizer state: `models/image_detector/resnet18_realworld_v1_last.pt`.
- History: `training/reports/image_model_history_20261124T184355405329Z.json`.
- Run ID: `20261124T184355405329Z`.
- CUDA device: NVIDIA GeForce RTX 4050 Laptop GPU; PyTorch 2.11.0+cu128; about 5,073 MiB free GPU memory at preflight; 26.06 GiB free disk at preflight.
- Seed 42; batch size 8; AdamW learning rate 1e-5; AMP enabled; 4 epochs requested; early stopping after epoch 3; best validation-loss checkpoint was epoch 1.
- Candidate validation accuracy by epoch: 96.5%, 96.0%, 95.0%. Best validation loss: 0.1303 at epoch 1.
- Candidate preprocessing matches the image service; existing strict checkpoint metadata is compatible. The candidate can be loaded through `IMAGE_MODEL_CHECKPOINT`, but is not the default.

## Same-split CIFAKE test comparison

All models below were evaluated on the same untouched 200-image CIFAKE test split. Confusion matrix rows are true `[REAL, AI_GENERATED]`; columns are predicted `[REAL, AI_GENERATED]`. Scores are raw uncalibrated softmax; thresholded selective outputs are not represented by these argmax metrics.

| Checkpoint | Accuracy | Macro F1 | REAL FPR | AI FNR | ROC-AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| `resnet18_best.pt` | 0.955 | 0.9549 | 0.080 | 0.010 | 0.9877 | `[[92, 8], [1, 99]]` |
| `resnet18_cifake_aug_v1.pt` | 0.975 | 0.9750 | 0.040 | 0.010 | 0.9955 | `[[96, 4], [1, 99]]` |
| `resnet18_cifake_aug_v1_calibrated.pt` | 0.975 | 0.9750 | 0.040 | 0.010 | 0.9955 | `[[96, 4], [1, 99]]` |
| `resnet18_cifake_aug_v1_calibrated_strict.pt` | 0.975 | 0.9750 | 0.040 | 0.010 | 0.9955 | `[[96, 4], [1, 99]]` |
| `resnet18_realworld_v1.pt` | 0.975 | 0.9750 | 0.040 | 0.010 | 0.9958 | `[[96, 4], [1, 99]]` |

The candidate has no test-set accuracy/FPR/FNR improvement over the active strict checkpoint. Its ROC-AUC increase of 0.0003 is too small to justify promotion and does not establish real-world improvement.

## User-photo diagnostic (in-sample only)

All four prior checkpoints predicted AI_GENERATED for all four user-labelled REAL photos. After the candidate was trained on those same photos, it predicted REAL for each. This is expected in-sample behavior and must not be interpreted as validation accuracy or evidence that other phone photos will be handled correctly. Per-photo raw scores are in the ignored `training/reports/image_realworld_user_photo_results.json`.

## Calibration, API, and deployment decision

- The active strict checkpoint retains its validation-derived abstention thresholds. No deployed threshold was changed.
- Candidate AI scores are `uncalibrated_softmax`. No candidate probability calibration or candidate-specific uncertain threshold was selected.
- FastAPI still defaults to `resnet18_cifake_aug_v1_calibrated_strict.pt`. The candidate is selectable with `IMAGE_MODEL_CHECKPOINT` for local testing; it is not recommended as the production default based on the available evidence.
- Phase 2 code and Phase 3 fusion were not changed.

## Tests and limitations

- New dataset/import unit tests: 3 passed.
- Full backend test suite: 51 passed in 55.28 seconds.
- `python -m compileall -q backend training scripts`: completed successfully.
- The four user photos are too few to characterize real-world performance; the CIFAKE test is low-resolution and dataset-specific. No independent high-resolution camera-photo test set was obtained. The original false-positive failure is therefore not shown to generalize better.

## Next action

Obtain a complete, source-aware multi-generator dataset with licensing compatible with the intended use, and reserve an untouched camera-photo test set before another training run or checkpoint promotion.
