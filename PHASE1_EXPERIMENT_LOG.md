# Phase 1 Experiment Log

## Baseline (existing artifact)

- Dataset: pinned `dragonintelligence/CIFAKE-image-dataset`, prepared locally at `data/image/cifake_ready`.
- Counts: 1,600 train / 200 validation / 200 test; balanced classes; source test split isolated.
- Model: pretrained ResNet-18, 5 epochs, CUDA / NVIDIA RTX 4050.
- Test: 200 CIFAKE images; accuracy 0.955; AI_GENERATED precision 0.9252, recall 0.9900, F1 0.9565; confusion matrix `[[92, 8], [1, 99]]` (true rows, predicted columns); REAL false-positive rate 0.08.
- Artifact: `models/image_detector/resnet18_best.pt`; report `training/reports/cifake_evaluation.json`.
- Interpretation: CIFAKE benchmark only; no camera-photo false-positive rate was measured.

## Augmented CIFAKE experiment (completed)

- Run started: 2026-11-24 UTC; PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA RTX 4050 6 GiB.
- Dataset: same existing CIFAKE splits; no downloads. Train-only horizontal flip, mild color jitter and low-probability blur; validation uses deterministic aspect-preserving padding.
- Model: pretrained ResNet-18; seed 42; requested 8 epochs; batch 16; early-stopping patience 3; output `models/image_detector/resnet18_cifake_aug_v1.pt`.
- Actual training: 8 epochs completed; selected epoch 6 by validation loss 0.1350; validation accuracy 0.96; training CUDA RTX 4050, peak allocated GPU memory about 0.37 GiB.
- Test (200 CIFAKE images): accuracy 0.975; AI_GENERATED precision 0.9612, recall 0.9900, F1 0.9754; confusion matrix `[[96, 4], [1, 99]]`; REAL FPR 0.04; AI_GENERATED FNR 0.01; ROC-AUC 0.9955.
- Baseline re-evaluated with same evaluator and checkpoint's legacy stretch preprocessing: accuracy 0.955; confusion matrix `[[92, 8], [1, 99]]`; REAL FPR 0.08; AI_GENERATED FNR 0.01; ROC-AUC 0.9877.
- Model choice: selected the new checkpoint as API default based on its lower validation loss (0.1350 vs baseline 0.2255) and higher validation accuracy (0.96 vs 0.93). Test data did not determine the selection. Reports: `training/reports/cifake_aug_v1_evaluation.json` and `training/reports/cifake_baseline_recheck.json`.
- Comparison is restricted to CIFAKE. It is not evidence that real phone/camera images have a lower false-positive rate; CIFAKE real examples are 32x32 CIFAR-10 images.

## Abstention calibration (validation-only)

- Source model: `models/image_detector/resnet18_cifake_aug_v1.pt`; validation manifest split only; 100 REAL / 100 AI_GENERATED.
- Rule: maximize validation coverage while requiring each class's validation error rate to be no greater than 1%; softmax scores remain uncalibrated and are not probabilities of authenticity.
- Selected thresholds: REAL when AI score `<= 0.25198999`; AI_GENERATED when AI score `>= 0.98400819`; otherwise UNCERTAIN.
- Validation: REAL FPR 1/100 (1%); AI false-negative rate 1/100 (1%); 165/200 coverage (82.5%); 35 uncertain.
- Checkpoint: `models/image_detector/resnet18_cifake_aug_v1_calibrated_strict.pt`; API default uses this checkpoint.
- Held-out test report: `training/reports/cifake_aug_v1_calibrated_strict_evaluation.json`. On 200 test examples, thresholded coverage was 170/200 (85%), 30 uncertain, REAL FPR 0/100 and AI FNR 0/100. These results are descriptive of CIFAKE only; they do not establish real-world performance.
