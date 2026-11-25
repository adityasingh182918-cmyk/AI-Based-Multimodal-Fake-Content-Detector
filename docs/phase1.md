# Phase 1: image detector workflow

## CIFAKE subset

The initial experiment uses the documented CIFAKE labels: `REAL` images come from CIFAR-10; `FAKE` images were produced using Stable Diffusion 1.4. The preparation code uses the verified Hugging Face Parquet copy `dragonintelligence/CIFAKE-image-dataset`, pinned to commit `6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31`, whose card describes `image` and `label` features and source `train`/`test` splits (`0=FAKE`, `1=REAL`). Mapping is `REAL` → `REAL`, `FAKE` → `AI_GENERATED`. See the [original CIFAKE project](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) and its [compatible Hugging Face dataset card](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset).

The default script selects 800 train, 100 validation, and 100 test images per class. Training and validation come from source `train`; the held-out test examples come only from source `test`. The source Parquet copy has one shard for each split, about 50.5 MB compressed combined. Streaming avoids expanding all 120,000 images but can read the source shards. The script pins the revision, uses seeded streaming shuffle, decodes selected images, hashes written image bytes, rejects duplicate selected files across all splits/classes, and produces `manifest.csv`, split CSVs, `rejected.csv`, and `metadata.json` (including skipped corrupt count, source revision, and seed). Nonempty output paths are never overwritten.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ml.txt
python scripts\prepare_cifake.py --output-root data\image\cifake_ready --train-per-class 800 --validation-per-class 100 --test-per-class 100 --seed 42
```

Do not claim preparation has succeeded until the command completes and reports its counts. Hugging Face/network or `datasets` errors stop preparation; there is no automatic alternate dataset fallback. Local data is Git-ignored.

## Legacy GenImage support

The official [GenImage README](https://github.com/GenImage-Dataset/GenImage) describes `ai/` as AI-generated and `nature/` as images collected from ImageNet. The ADM subset maps `ai` → `AI_GENERATED` and `nature` → `REAL`; these are source benchmark labels, not proof that an image is unaltered. Archive metadata inspection, integrity validation, extracted-folder checks, and manifest generation remain available in `scripts/`.

The uploaded archive is disk 12 of a 13-volume set; `.z01` through `.z12` are absent. Metadata lists six zero-byte images under `train/ai`, and the listing represents about 38.9 GB uncompressed. Do not extract or train on it unless the full set is recovered and validated. The scripts never extract it automatically.

## GPU training

Training defaults to `--device cuda` and exits with a CUDA diagnostic if CUDA is unavailable; it does not silently use CPU. An explicitly requested CPU run must pass `--device cpu`. Startup prints PyTorch/CUDA/GPU details and memory. Tqdm progress shows batches, running loss/accuracy, epoch percentage, learning rate, and elapsed time. Epoch summaries include train and validation loss/accuracy, validation AI_GENERATED precision/recall/F1, duration, and peak allocated GPU memory.

The defaults are 10 epochs, batch size 16, image size 224, seed 42, zero data-loader workers, CUDA mixed precision, and early-stopping patience 3. New runs apply EXIF orientation, aspect-preserving square padding, and mild training-only color/blur augmentation. Validation, test, and inference use the same deterministic resize and normalization. `--no-amp` disables mixed precision. CUDA out-of-memory restores the current epoch-start state and retries with a smaller batch. Pretrained initialization uses `ResNet18_Weights.DEFAULT`; a weight-download failure stops instead of silently training from random weights. Add `--no-pretrained` only to request random initialization explicitly.

The trainer requires both classes in every split, uses only train and validation rows, and chooses checkpoints using validation loss. The held-out test split is never involved in selection. Checkpoints record architecture, class ordering, resize/normalization, pretrained initialization, dataset identifier/revision, seed, run timestamp, best validation metric, completed epochs, and run ID. The trainer refuses to overwrite an existing checkpoint path; choose a unique versioned filename. Interruptions leave prior checkpoints untouched. Persistent log and per-run JSON history are written below `training/reports/`.

```powershell
python -c "import torch, torchvision; print('PyTorch:',torch.__version__); print('torchvision:',torchvision.__version__); print('CUDA available:',torch.cuda.is_available()); print('CUDA runtime:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'); print('GPU memory GiB:',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2) if torch.cuda.is_available() else 'unavailable')"
python training\train_image_model.py --dataset-root data\image\cifake_ready --manifest data\image\cifake_ready\manifests\manifest.csv --checkpoint models\image_detector\resnet18_cifake_aug_v2.pt --epochs 8 --batch-size 16 --image-size 224 --seed 42 --num-workers 0 --early-stopping-patience 3
```

The current environment has Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and an NVIDIA GeForce RTX 4050 Laptop GPU. PyTorch and torchvision must be installed using compatible CUDA wheels from the [official selector](https://pytorch.org/get-started/locally/); the project requirements do not replace them.

## Evaluation

Evaluate after training on manifest rows marked `test`. Evaluation loads the exact selected checkpoint and uses its stored preprocessing. The JSON report includes accuracy, AI_GENERATED precision/recall/F1, per-class metrics, classification report, confusion matrix, ROC-AUC, REAL false-positive rate, and AI_GENERATED false-negative rate when both classes are present. No score is fabricated or promised. Results describe this small CIFAKE subset and may not generalize to newer image generators, edits, or real-world imagery.

```powershell
python training\calibrate_image_model.py --dataset-root data\image\cifake_ready --manifest data\image\cifake_ready\manifests\manifest.csv --checkpoint models\image_detector\resnet18_cifake_aug_v2.pt --output-checkpoint models\image_detector\resnet18_cifake_aug_v2_calibrated.pt --max-class-error-rate 0.01 --min-coverage 0.50 --batch-size 16
python training\evaluate_image_model.py --dataset-root data\image\cifake_ready --manifest data\image\cifake_ready\manifests\manifest.csv --checkpoint models\image_detector\resnet18_cifake_aug_v2_calibrated.pt --report training\reports\cifake_aug_v2_calibrated_evaluation.json --batch-size 16
```

## FastAPI and browser UI

FastAPI retains `GET /`, `GET /health`, `GET /model-status`, and `POST /api/v1/predict/image`. It serves the local upload UI at `/ui` and docs at `/docs`. A valid checkpoint loads once on application startup; if no valid checkpoint exists, the UI explains that the model is not ready and prediction requests return HTTP 503. Image content is EXIF-oriented, decoded through Pillow, converted to RGB, and normalized using checkpoint-matched preprocessing. The calibrated model returns REAL/AI_GENERATED/UNCERTAIN according to validation-selected thresholds, alongside both uncalibrated softmax scores, model/version/device, processing time, and a warning that scores are not proof of authenticity.

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/ui` or `http://127.0.0.1:8000/docs`. The API defaults to a configurable 10 MiB upload limit (`IMAGE_MAX_UPLOAD_BYTES`). It returns 400 for unreadable images, 415 for non-image media types, and 413 for files over the configured limit. `IMAGE_MODEL_CHECKPOINT` and `IMAGE_MODEL_DEVICE` (`auto`, `cuda`, `cpu`) can override checkpoint/device settings. Explicit CUDA selection never falls back to CPU.

## Tests and troubleshooting

```powershell
python -m pytest
```

Tests use synthetic images and a synthetic checkpoint; they do not download CIFAKE, train a model, or claim its accuracy. For missing packages, activate `.venv` then run `python -m pip install -r backend\requirements.txt` and `python -m pip install -r requirements-ml.txt`. For CUDA unavailable, check `torch.cuda.is_available()` and install the matching CUDA PyTorch build/driver. For out-of-memory, lower `--batch-size`, such as 8. For Hugging Face errors, verify internet access and the pinned dataset revision; retry deliberately, since the prep script does not switch sources. For HTTP 503, prepare and train a model, or correct `IMAGE_MODEL_CHECKPOINT`. For port conflicts, use a free `--port` and open the matching `/ui` URL.

Current workspace data is the pinned 2,000-image CIFAKE sample (1,600 train, 200 validation, 200 held-out test; balanced). The original checkpoint remains preserved at `models/image_detector/resnet18_best.pt`. The augmented versioned checkpoint `models/image_detector/resnet18_cifake_aug_v1.pt` was trained for 8 CUDA epochs, with best validation loss at epoch 6 (0.1350; validation accuracy 0.96). On the raw two-class output it measured 0.975 accuracy, 0.04 REAL FPR, 0.01 AI FNR, confusion matrix `[[96, 4], [1, 99]]`, and ROC-AUC 0.9955 on the 200-row test split. A validation-only policy with a 1% maximum error target selected thresholds REAL `<=0.25198999` AI score, AI_GENERATED `>=0.98400819`, and UNCERTAIN between. Validation coverage was 82.5% with 1% class-specific error rates; test coverage was 85% (30 uncertain) with 0/100 false positives and 0/100 false negatives. The API now defaults to `models/image_detector/resnet18_realworld_v1.pt`; inference returns the 3-way label, uncalibrated softmax scores, device, and processing time. This checkpoint includes four user-labelled REAL supplemental photos, so their predictions are not independent evaluation evidence. The strict CIFAKE checkpoint remains preserved and can be selected with `IMAGE_MODEL_CHECKPOINT`. This remains a small CIFAKE test, not evidence of performance on real camera photos. The original GenImage archive is incomplete and was not extracted.
