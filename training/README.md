# Training

Phase 1 includes a manifest-backed ResNet-18 training command and held-out test evaluator. See [the root README](../README.md) and [Phase 1 guide](../docs/phase1.md) for setup and commands.

The workspace has a preserved baseline, an augmented CIFAKE checkpoint, and its validation-calibrated abstention checkpoint. On the same held-out 200-image CIFAKE split, the baseline measured 95.5% accuracy and 8% REAL FPR; the augmented run measured 97.5% accuracy and 4% REAL FPR. Thresholds were selected using only the validation split; the calibrated checkpoint reports UNCERTAIN for scores between its saved boundaries. These figures apply to CIFAKE only, not camera photos. New training commands require validated manifests with both labels in train and validation, select checkpoints using validation metrics only, and refuse to overwrite existing checkpoint paths. Evaluation reads only `split=test` rows and uses checkpoint-matched preprocessing.
