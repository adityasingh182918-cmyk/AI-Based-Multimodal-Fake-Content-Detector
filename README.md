# AI-Based Multimodal Fake Content Detector

A modular college project to study whether digital content may be authentic, AI-generated, or manipulated. Outputs from future detectors are estimates, not proof of authenticity.

## Objectives and phases

1. **Image (Phase 1):** GenImage inspection/preparation, reproducible CIFAKE subset preparation, ResNet-18 training/evaluation, and FastAPI plus browser UI are implemented. A 2,000-image CIFAKE subset, baseline checkpoint, and evaluation report are present locally; camera-photo generalization is not established.
2. **Text (Phase 2):** HC3 baseline plus a versioned HC3+MAGE expanded SVM/DistilBERT experiment, FastAPI endpoint and UI mode are implemented. Cross-domain performance and useful uncertainty calibration remain limited; this is not a general authorship verifier.
3. **Multimodal:** assess aligned image and text inputs with explicit labels.
4. **Audio and video:** investigate temporal and media-specific signals.

## Architecture and stack

- FastAPI API, schema, and lazy inference service in `backend/`.
- Dataset archive inspection and path-only manifest generation in `scripts/`.
- ResNet-18 transfer-learning training and held-out evaluation in `training/`.
- The lightweight browser UI is served by FastAPI at `/ui`; no frontend build is required.
- Python, PyTorch, torchvision, Pillow, NumPy, tqdm, Hugging Face `datasets`, and FastAPI. PyTorch is intentionally a separate install because its Windows/CUDA wheel must match the machine.

## Project layout

```text
backend/app/       FastAPI endpoints, schemas, and inference service
backend/tests/     Endpoint, detector, and dataset-tool tests
scripts/           Archive inspection and dataset manifest preparation
training/          Dataset adapter, training and evaluation commands
data/image/        Ignored local reports and generated manifests
models/            Ignored trained checkpoints
docs/phase1.md     Phase 1 workflow and limitations
backend/app/static/ Browser image upload UI served by FastAPI
frontend/          Setup notes; no separate build is needed
```

## Python environment and dependencies (Windows PowerShell)

Use the existing Windows virtual environment and verify the CUDA-enabled PyTorch build before preparing data or training. The project environment reports Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and an RTX 4050 Laptop GPU. Install torch/torchvision from the official selector if recreating the environment, matching the local NVIDIA driver and CUDA option. The trainer uses torchvision's current `weights=ResNet18_Weights.DEFAULT` API and PyTorch's `torch.amp.autocast` / `torch.amp.GradScaler` APIs.

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip install -r requirements-ml.txt
```

If PowerShell blocks activation, use a temporary process-only policy in that terminal: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. Install PyTorch from the official [PyTorch selector](https://pytorch.org/get-started/locally/) only when setting up or repairing the environment. Verify its runtime, device, and memory:

```powershell
python -c "import torch, torchvision; print('PyTorch:',torch.__version__); print('torchvision:',torchvision.__version__); print('CUDA available:',torch.cuda.is_available()); print('CUDA runtime:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'); print('GPU memory GiB:',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2) if torch.cuda.is_available() else 'unavailable')"
```

`requirements-ml.txt` installs NumPy, the Hugging Face `datasets` reader, and tqdm; it does not replace the selected torch/torchvision CUDA wheels.

## CIFAKE small-subset experiment (recommended)

The original CIFAKE project describes two classes, REAL images from CIFAR-10 and FAKE images generated with Stable Diffusion 1.4, with official 100,000-image train and 20,000-image test partitions. Its documentation identifies the dataset license as MIT. The compatible Hugging Face Parquet copy used here is `dragonintelligence/CIFAKE-image-dataset`; its verified card documents `image` and `label` features, `train` and `test` splits, and label IDs `0=FAKE`, `1=REAL`. The preparation script pins the inspected dataset commit and maps `REAL` → `REAL`, `FAKE` → `AI_GENERATED`. See the [original CIFAKE source](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) and [verified Hugging Face dataset card](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset).

The default selection is reproducible and contains 800 train, 100 validation, and 100 test images per class (2,000 total). Train and validation are sampled only from the source `train` split; test is sampled only from the source `test` split. The Hugging Face copy has one Parquet shard for each source split (about 50.5 MB compressed combined), so streaming avoids unpacking/retaining all images but may read the shard for each split. These commands fetch only the requested rows into the ignored `data/` tree; they do not run when installing dependencies.

```powershell
python -m pip install -r requirements-ml.txt
python scripts\prepare_cifake.py --output-root data\image\cifake_ready --train-per-class 800 --validation-per-class 100 --test-per-class 100 --seed 42
```

The command creates `data\image\cifake_ready\images\{train,validation,test}\{REAL,AI_GENERATED}` and compatible `manifests\manifest.csv` plus per-split manifests and `rejected.csv`. It records the source revision, seed, split counts, and corrupt images skipped while selecting rows. It refuses to write into a nonempty output directory. Duplicate decoded images are skipped while sampling so each requested split remains unique. Use a new `--output-root` to create another sample. The `datasets` dependency and network access to Hugging Face are required. The legacy GenImage archive inspection and preparation commands below remain available.

## Inspect and prepare the dataset

The previously inspected GenImage ADM archive was incomplete: it listed `train/ai`, `train/nature`, `val/ai`, and `val/nature`, but its 13-volume set was missing `.z01` through `.z12`; its metadata also listed six zero-byte files under `train/ai`. Do not extract or prepare it until all parts are obtained and entries are validated. Set `$Archive` to the location of a complete copy before using these optional legacy commands. The official [GenImage README](https://github.com/GenImage-Dataset/GenImage) defines `ai` as AI-generated images and `nature` as images collected from ImageNet; this code maps `ai` to `AI_GENERATED` and `nature` to `REAL`.

Inspect metadata without extracting files. This command exits non-zero while required volumes are missing; the printed report still includes the missing volume names and central-directory counts:

```powershell
$Archive = Read-Host "Enter the full path to the complete GenImage ADM ZIP"
python scripts\inspect_dataset.py $Archive --report-json data\image\archive-report.json
```

For a standalone, readable ZIP, decode every listed image directly without extracting by adding `--validate-images`. It may take a long time because this archive's listing represents about 38.9 GB uncompressed. The current split archive cannot be decoded from its final volume alone; the inspector reports this and skips payload validation. After all parts are available and manually extracted, use the directory validation command below:

```powershell
python scripts\inspect_dataset.py $Archive --validate-images --report-json data\image\archive-validation.json
```

The inspection tool reports adjacent `.z01`, `.z02`, etc. parts and sequence gaps. No adjacent parts does not prove the archive is complete: compare against the original download source's expected part list. Do not extract until all required parts are present. With all parts beside the `.zip`, use 7-Zip's test command (if installed) to verify compressed payload CRCs without extracting, then extract to a location with at least 40 GB free, preferably more:

```powershell
$Extracted = Join-Path (Split-Path -Parent $Archive) "imagenet_ai_0508_adm_extracted"
7z t $Archive
7z x $Archive "-o$Extracted"
```

Then inspect the extracted structure and decode images:

```powershell
python scripts\inspect_dataset.py $Extracted --validate-images --report-json data\image\extracted-validation.json
```

Only after extraction and validation, create reproducible manifests. They reference source images and do not copy them. The archive's `val/` partition is kept untouched as the held-out `test` split; `train/` is split into train/validation. Exact duplicate bytes crossing train and test are rejected.

```powershell
python scripts\prepare_dataset.py (Join-Path $Extracted "imagenet_ai_0508_adm") --output-dir data\image\manifests --seed 42 --validation-fraction 0.1 --skip-corrupt
```

Optional cap for a small pipeline smoke run (at least 3 images per class across the two source partitions):

```powershell
python scripts\prepare_dataset.py (Join-Path $Extracted "imagenet_ai_0508_adm") --output-dir data\image\smoke-manifests --seed 42 --validation-fraction 0.1 --max-images-per-class 30 --skip-corrupt
```

The output contains `manifest.csv` with all split assignments, `train.csv`, `validation.csv`, `test.csv`, and `rejected.csv`. Unreadable files stop preparation by default; `--skip-corrupt` explicitly excludes them and records their paths/reasons in `rejected.csv`. Existing manifests are preserved; choose another output directory or add `--force` to replace them.

## Train and evaluate (manual; not run automatically)

Training uses pretrained ResNet-18 weights from torchvision by default; torchvision may download these weights the first time training starts. This is a model-weight download, not a dataset download. The training command does not start unless you run it:

For the recommended CIFAKE subset, run these from the repository root after preparing data and installing the selected PyTorch/torchvision build:

```powershell
python training\train_image_model.py --dataset-root data\image\cifake_ready --manifest data\image\cifake_ready\manifests\manifest.csv --checkpoint models\image_detector\resnet18_cifake_aug_v2.pt --epochs 8 --batch-size 16 --image-size 224 --seed 42 --num-workers 0 --early-stopping-patience 3
python training\calibrate_image_model.py --dataset-root data\image\cifake_ready --manifest data\image\cifake_ready\manifests\manifest.csv --checkpoint models\image_detector\resnet18_cifake_aug_v2.pt --output-checkpoint models\image_detector\resnet18_cifake_aug_v2_calibrated.pt --max-class-error-rate 0.01 --min-coverage 0.50 --batch-size 16
python training\evaluate_image_model.py --dataset-root data\image\cifake_ready --manifest data\image\cifake_ready\manifests\manifest.csv --checkpoint models\image_detector\resnet18_cifake_aug_v2_calibrated.pt --report training\reports\cifake_aug_v2_calibrated_evaluation.json --batch-size 16
```

The trainer requires CUDA by default and stops with a diagnostic instead of silently switching to CPU. It prints GPU details, per-batch progress, running loss/accuracy, percent complete, learning rate, elapsed time, and per-epoch validation metrics. Mixed precision is enabled by default on CUDA (`--no-amp` disables it); CUDA out-of-memory retries the current epoch with a reduced batch size and restores its starting state. CPU training is possible only by explicitly passing `--device cpu`. Training writes run logs, history, and metadata. It refuses to overwrite any existing checkpoint path; use another versioned filename for later runs. New runs use EXIF-corrected RGB, aspect-preserving square padding and mild training-only color/blur augmentation; evaluation and inference share deterministic checkpoint-matched preprocessing. Early stopping and best-checkpoint selection use validation loss only. Calibration uses validation only to maximize coverage under the specified per-class error target; the softmax scores remain uncalibrated. The evaluator reports raw two-class metrics and, for calibrated checkpoints, three-way coverage/UNCERTAIN metrics. The FastAPI service loads the configured checkpoint once at startup.

The legacy GenImage training command is retained for when its archive volumes become available:

```powershell
python training\train_image_model.py --dataset-root (Join-Path $Extracted "imagenet_ai_0508_adm") --manifest data\image\manifests\manifest.csv --checkpoint models\image_detector\resnet18_genimage_adm_v1.pt --epochs 10 --batch-size 32 --image-size 224 --seed 42 --num-workers 0
```

To disable pretrained weights, add `--no-pretrained` (the model then trains from random initialization). For an explicitly requested CPU run, also pass `--device cpu`; the default GPU command never silently falls back. Use a smaller batch if CUDA memory is insufficient. Best checkpoint selection uses validation loss only.

Evaluate only the held-out `test` rows:

```powershell
python training\evaluate_image_model.py --dataset-root (Join-Path $Extracted "imagenet_ai_0508_adm") --manifest data\image\manifests\manifest.csv --checkpoint models\image_detector\resnet18_genimage_adm_v1.pt --report training\reports\genimage_adm_v1_evaluation.json
```

The report includes accuracy, AI_GENERATED precision/recall/F1, a REAL/AI_GENERATED confusion matrix, and sample count. A score on this ADM subset does not establish generalization to unseen generators or real-world images.

## Phase 2: text detection

An initial text detector is implemented. It uses HC3 `reddit_eli5` and `wiki_csai` sources, group-held-out splits, and word/character TF-IDF Logistic Regression and calibrated Linear SVM baselines. The validation-selected Linear SVM checkpoint is `models/text_detector/baseline/hc3_tfidf_linear_svm.joblib`. The API exposes `POST /api/v1/predict/text` and `GET /text-model-status`; long text is chunked and returns HUMAN_WRITTEN, AI_GENERATED, or UNCERTAIN. The UI offers independent image, text, and image-plus-text modes. The combined mode runs the two detectors independently and is explicitly not a trained fusion model.

Dataset source terms vary: the selected ELI5-derived source is listed BSD and the Wikipedia-derived source CC BY-SA; other HC3 sources with unknown/custom licenses are excluded. Read [docs/phase2_datasets.md](docs/phase2_datasets.md) before redistribution. Earlier frozen HC3 checkpoints scored 0.5327 macro-F1 on Reddit-to-Wikipedia source holdout, 0.6367/0.7144 on MAGE GPT-4 OOD, and 0.4652/0.4124 on GRADTEX Test C. The expanded checkpoints improve same-source test metrics, but expanded DistilBERT still has 0.4580 human FPR on MAGE GPT-4 OOD and 0.2462 on GRADTEX Test C. Beemo's 656 positive-only edited-AI probe reports recall only. Expanded DistilBERT has validation-fitted temperature scaling; the validation abstention thresholds collapse to all-UNCERTAIN for both expanded models. Browser-level responsive/accessibility checks and direct human grammar-correction testing remain incomplete. Phase 3 offers a rule-based combination of the independent Phase 1/2 labels, but no trained neural fusion model or fusion metrics exist. See [docs/phase2.md](docs/phase2.md), [PHASE2_RETRAINING_REPORT.md](PHASE2_RETRAINING_REPORT.md), and [docs/phase2_datasets.md](docs/phase2_datasets.md).

To prepare, train and evaluate the baseline (after activating `.venv` and installing `requirements-ml.txt`):

```powershell
python scripts\download_text_datasets.py --output-dir data\text\raw\hc3
python scripts\prepare_text_dataset.py --raw-dir data\text\raw\hc3 --output-dir data\text\processed\hc3 --max-groups-per-source 1000 --seed 42
python scripts\validate_text_dataset.py data\text\processed\hc3 --report training\reports\text_dataset_validation.json
python training\train_text_baseline.py --dataset-dir data\text\processed\hc3
python training\evaluate_text_baseline.py --dataset-dir data\text\processed\hc3 --checkpoint models\text_detector\baseline\hc3_tfidf_linear_svm.joblib --report training\reports\text_evaluation.json
python -m pip install -r requirements-text-transformer.txt
python training\train_text_model.py --dataset-dir data\text\processed\hc3 --output-dir models\text_detector\transformer\distilbert_hc3_v1 --epochs 3 --batch-size 8 --gradient-accumulation 2 --max-length 256 --seed 42
python training\evaluate_text_model.py --dataset-dir data\text\processed\hc3 --checkpoint models\text_detector\transformer\distilbert_hc3_v1 --report training\reports\text_transformer_evaluation.json --device cuda --batch-size 8
```

Optional external evaluation (downloads only small test/evaluation artifacts; never trains, selects models, or tunes thresholds on them):

```powershell
python scripts\download_text_robustness.py
python scripts\download_text_robustness.py --include-paraphrase
python training\evaluate_text_robustness.py --dataset data\text\robustness\mage\test_ood_set_gpt.csv --report training\reports\text_mage_ood_evaluation.json
python training\evaluate_text_robustness.py --dataset data\text\robustness\mage\test_ood_set_gpt_para.csv --paraphrased-ai-only --report training\reports\text_mage_paraphrase_ai_evaluation.json
python scripts\download_text_robustness_extra.py --dataset all
python training\evaluate_text_additional_robustness.py --dataset gradtex
python training\evaluate_text_additional_robustness.py --dataset gradtex_b
python training\evaluate_text_additional_robustness.py --dataset beemo
python training\analyze_text_calibration.py --device cuda --batch-size 8
```

The additional downloads are approximately 64 MB for GRADTEX Test and 8.3 MB for Beemo. GRADTEX reports AI involvement (MIX) separately from full authorship; its Test C file is derived from MAGE. Beemo output rights vary by generating model, so the script restricts scoring to documented MIT/Apache-2.0 source models. The human-edit and MAGE paraphrase probes contain no human negatives; their reports intentionally omit accuracy/F1/precision/AUC. The Pangram Grammarly benchmark is gated and was not downloaded.

### Reproducible HC3 + MAGE expansion experiment

MAGE is downloaded only when you run the explicit sampler below. It streams the official train/validation/test splits at a pinned revision and saves deterministic, capped examples, not the complete dataset. The MAGE dataset card declares Apache-2.0. Existing frozen MAGE GPT-4 OOD, GRADTEX, and Beemo evaluation artifacts are excluded from training and calibration. See [the experiment plan](PHASE2_RETRAINING_PLAN.md) for provenance limits and results.

```powershell
# From the repository root. Dependencies must already be installed in .venv.
.\.venv\Scripts\python.exe scripts\download_text_expansion.py --output-dir data\text\raw\mage_expansion --seed 42
.\.venv\Scripts\python.exe scripts\prepare_expanded_text_dataset.py --hc3-dir data\text\processed\hc3 --mage-dir data\text\raw\mage_expansion --output-dir data\text\processed\expanded_v1 --seed 42
.\.venv\Scripts\python.exe scripts\validate_text_dataset.py data\text\processed\expanded_v1 --report training\reports\text_expanded_dataset_validation.json

# Training is explicit. SVM and DistilBERT write to new versioned locations.
.\.venv\Scripts\python.exe training\train_expanded_svm.py --dataset-dir data\text\processed\expanded_v1 --checkpoint models\text_detector\svm_expanded_v1.joblib --report training\reports\text_svm_expanded_validation.json --seed 42
.\.venv\Scripts\python.exe training\train_text_model.py --dataset-dir data\text\processed\expanded_v1 --output-dir models\text_detector\distilbert_expanded_v1 --init-checkpoint models\text_detector\transformer\distilbert_hc3_v1 --history training\reports\text_transformer_expanded_history.json --epochs 3 --batch-size 8 --gradient-accumulation 2 --max-length 256 --learning-rate 0.00001 --seed 42 --device cuda

# Compare four frozen checkpoints on the same held-out test split.
.\.venv\Scripts\python.exe training\compare_text_models.py --dataset-dir data\text\processed\expanded_v1 --split test --device cuda --report training\reports\text_model_comparison_expanded_test.json

# Run backend/API/image/text regression tests.
.\.venv\Scripts\python.exe -m pytest backend\tests -q

# Optional: rerun validation-only calibration (does not train or overwrite weights).
.\.venv\Scripts\python.exe training\calibrate_expanded_models.py --dataset-dir data\text\processed\expanded_v1 --device cuda --batch-size 8 --report training\reports\text_expanded_calibration.json

# Optional frozen OOD scoring only; the held-out datasets are never used for fitting.
.\.venv\Scripts\python.exe training\run_text_additional_diagnostics.py --datasets all --baseline-checkpoint models\text_detector\svm_expanded_v1.joblib --transformer-checkpoint models\text_detector\distilbert_expanded_v1 --device cuda --output training\reports\text_expanded_ood_diagnostics.json
```

The API keeps the original HC3 SVM as its default. Select a version explicitly in PowerShell before startup: `$env:TEXT_MODEL_BACKEND='baseline'`, `svm_expanded`, `transformer`, or `transformer_expanded`. For example:

```powershell
$env:TEXT_MODEL_BACKEND = 'transformer_expanded'
$env:TEXT_MODEL_DEVICE = 'cuda'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Expanded DistilBERT inference uses a scalar temperature fitted only on the expanded validation split. The 5%-error abstention policy selected on validation has endpoints 0 and 1 for both expanded models, which yields UNCERTAIN for every input; this is preserved as an honest report of the calibration limitation, not a useful production policy. The separate experiment report records hard metrics at the 0.5 threshold. The unified Phase 3 mode now has a transparent rule-based label combination; a neural fusion model is not trained.

**Current text reliability note (2026-11-25):** a user-shared AI-generated visual description was classified HUMAN_WRITTEN by both the frozen HC3 SVM and HC3 DistilBERT. The live unified API was using `distilbert_hc3_v1`; this is an out-of-domain model failure, not evidence of a Phase 3 label-mapping defect. The expanded checkpoints also did not classify that example AI_GENERATED, and their validation-derived three-way thresholds return UNCERTAIN for every input. They are not promoted. Do not describe detector scores as accuracy or proof of authorship. See [PHASE2_IMPROVEMENT_REPORT.md](PHASE2_IMPROVEMENT_REPORT.md) for the exact diagnostic and limitations.

## Run the API

From the repository root, after backend and PyTorch dependencies are installed:

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open the upload interface at [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui) and API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Existing routes `GET /`, `GET /health`, `GET /model-status`, and `POST /api/v1/predict/image` are preserved. The upload endpoint accepts multipart field `file`, returns HTTP 503 until a valid trained checkpoint is loaded, HTTP 400 for corrupt image contents, HTTP 415 for non-image media types, and HTTP 413 above the configurable 10 MiB default. Override the maximum with `IMAGE_MAX_UPLOAD_BYTES`; set `IMAGE_MODEL_CHECKPOINT` or `IMAGE_MODEL_DEVICE` (`auto`, `cuda`, `cpu`) when needed. Predictions return REAL, AI_GENERATED, or UNCERTAIN for checkpoints calibrated with validation thresholds, both uncalibrated softmax class scores, model/version/device, and elapsed inference time. The UI runs locally and sends selected images only to this local API.

## Tests

Software tests use tiny synthetic image fixtures and never use them for training:

```powershell
python -m pytest
```

The tests cover endpoints, model readiness, upload validation, a synthetic image checkpoint prediction, text validation/chunking/inference, EXIF/aspect-preserving preprocessing, CIFAKE manifests/label mapping, missing/corrupt images, deterministic GenImage manifests, and duplicate leakage checks. Tests do not download datasets or train models.

## Troubleshooting

- **CUDA unavailable:** check `torch.cuda.is_available()` and `torch.version.cuda`; install the matching NVIDIA driver and CUDA-enabled PyTorch/torchvision wheels from the official selector. The default trainer exits instead of falling back to CPU.
- **GPU out of memory:** rerun with `--batch-size 8` or lower; image size can also be reduced with `--image-size`.
- **CIFAKE download/dependency error:** activate `.venv`, run `python -m pip install -r requirements-ml.txt`, verify Hugging Face network access, then rerun preparation in a new empty output directory.
- **Missing or invalid checkpoint / API HTTP 503:** complete the documented training command and verify `GET /model-status`; check `IMAGE_MODEL_CHECKPOINT` if using a custom path.
- **Missing Python dependency:** run `python -m pip install -r backend\requirements.txt` and `python -m pip install -r requirements-ml.txt`; install torch/torchvision separately so the CUDA build remains compatible.
- **Port conflict:** select an unused port with uvicorn's `--port` argument and open `/ui` on that same port.

## Git and data handling

Large datasets, `.env` files, checkpoints, generated reports, and `data/` contents are ignored by Git. Do not commit archive parts or model weights. This workspace currently has no Git repository or remote configured. The 2,000 selected CIFAKE images, preserved baseline, versioned augmented checkpoint, run logs, and actual evaluation reports are present locally.

## Phase 1 supplemental real-photo fine-tuning

Four photos you labelled REAL were added as supplemental **training-only** examples. The original CIFAKE dataset and its 1,600/200/200 train/validation/test split remain unchanged. The candidate tied the current strict checkpoint on the untouched CIFAKE test, so FastAPI still defaults to the existing strict checkpoint. See [PHASE1_RETRAINING_REPORT.md](PHASE1_RETRAINING_REPORT.md) for results and data-source decisions.

CommunityForensics-Small was not used because its CC BY-NC-SA 4.0 terms restrict use to non-commercial research and commercial use may be needed or uncertain. No external multi-generator data was added. The candidate is selectable for local inspection via `IMAGE_MODEL_CHECKPOINT`, but is not promoted.

### Rebuild the supplemental manifest (Windows PowerShell)

Run from the project root. These commands copy the supplied photos into ignored `data/` storage, write metadata, and combine them with CIFAKE while preserving all existing splits:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\import_user_real_images.py `
  "C:\Users\Lenovo\Downloads\WhatsApp Image 2026-11-24 at 22.43.39.jpeg" `
  "C:\Users\Lenovo\Downloads\WhatsApp Image 2026-11-24 at 22.43.44.jpeg" `
  "C:\Users\Lenovo\Downloads\WhatsApp Image 2026-11-24 at 22.40.19.jpeg" `
  "C:\Users\Lenovo\Downloads\WhatsApp Image 2026-11-24 at 22.38.39.jpeg"
python scripts\prepare_realworld_manifest.py
```

The import command preserves originals and refuses to overwrite mismatched existing files. If the photos move, substitute their current paths.

### Fine-tune a separate candidate

Training is explicit and does not run during API startup. It requires the CUDA-enabled PyTorch environment and the prepared manifest:

```powershell
python training\train_image_model.py `
  --dataset-root . `
  --manifest data\image_detection\combined\manifest.csv `
  --checkpoint models\image_detector\resnet18_realworld_v1.pt `
  --init-checkpoint models\image_detector\resnet18_cifake_aug_v1_calibrated_strict.pt `
  --device cuda --amp --batch-size 8 --epochs 4 `
  --early-stopping-patience 2 --learning-rate 0.00001 --seed 42
```

The trainer refuses to overwrite the best checkpoint path or the derived `_last.pt` file. It selects the best checkpoint by validation loss and does not use the test split for model selection.

### Evaluate the candidate on the unchanged CIFAKE test split

```powershell
python training\evaluate_image_model.py `
  --dataset-root data\image\cifake_ready `
  --manifest data\image\cifake_ready\manifests\manifest.csv `
  --checkpoint models\image_detector\resnet18_realworld_v1.pt `
  --report training\reports\realworld_candidate_cifake_test.json `
  --device cuda --batch-size 16
```

For a fair comparison, evaluate `resnet18_best.pt`, `resnet18_cifake_aug_v1.pt`, `resnet18_cifake_aug_v1_calibrated.pt`, and `resnet18_cifake_aug_v1_calibrated_strict.pt` with the same dataset root and manifest. The four supplemental photos were used for training and must not be reported as independent test data.

### Run the API

The default API command continues to load the existing strict checkpoint:

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

To inspect the candidate locally without changing the project default:

```powershell
$env:IMAGE_MODEL_CHECKPOINT = (Resolve-Path "models\image_detector\resnet18_realworld_v1.pt").Path
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Candidate scores are uncalibrated softmax; it has no validation-selected abstention threshold. Open a new PowerShell window to restore the default, or run `Remove-Item Env:IMAGE_MODEL_CHECKPOINT` before restarting in the same window.

## Current status and limitations

The preserved baseline is `models/image_detector/resnet18_best.pt`; the validation-selected, calibrated checkpoint `models/image_detector/resnet18_cifake_aug_v1_calibrated_strict.pt` remains the API default. The augmented strict checkpoint scored 97.5% accuracy and 4% REAL FPR on the untouched CIFAKE test; the new candidate ties these metrics and was not promoted. The four user photos are in the candidate's training data, so its REAL predictions on those photos are not evaluation results. All CIFAKE results are dataset-specific: images are 32x32 and do not establish camera-photo generalization. OpenFake's core is too large for the available disk; the non-commercial CommunityForensics-Small dataset was excluded because commercial use may be needed or is uncertain. Most recent full backend suite: 58 passed; `compileall` succeeded. The Phase 1 report documents the remaining limits. See `PROJECT_PROGRESS.md` for project-wide status.
# Phase 3 — multimodal fusion (isolated, experimental)

Phase 3 includes a separate experimental `phase3/` package and an operational
rule-based multimodal analysis in the unified app. No suitable paired dataset
with verified image-origin and text-origin labels and acceptable licensing has
been established, so no neural fusion model is trained and no fusion metrics
exist. The unified app on port 8000 combines the two detector labels using a
documented decision table; it does not analyze whether image and text describe
the same content. The standalone experimental `phase3.app` below remains a
checkpoint-based neural-fusion API and may return HTTP 503 until a checkpoint
is trained. Do not describe the rule-based result as a trained fusion model.

Run the isolated Phase 3 API from the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn phase3.app:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/ui` or inspect `http://127.0.0.1:8001/phase3/status`.
The Phase 3 image/text endpoints use frozen copies of the active Phase 1/2
models. They are separate from the production Phase 1/2 backend.

Data schema, eligibility caveats, and the exact explicit preparation/training/
evaluation commands are in `phase3/data/README.md` and
`PHASE3_IMPLEMENTATION_PLAN.md`. Do not begin training until dataset rights,
provenance, paired relationships, labels, and group-aware splits are reviewed.

Phase 3 isolated checks and freeze regression commands:

```powershell
.\.venv\Scripts\python.exe -m pytest phase3\tests -q
.\.venv\Scripts\python.exe -m pytest backend\tests -q
.\.venv\Scripts\python.exe phase3\tools\snapshot_originals.py --verify
```

## Unified application: Phase 1 + Phase 2 + Phase 3 status

Use the unified local app for one dashboard with Image Detection, Text Detection,
and Image + Text Analysis. It calls the existing Phase 1 and Phase 2 detectors
without editing their code, and records prediction history in local SQLite.
Raw text and uploaded image bytes are not retained by default. Combined mode
shows each original detector result and score, then applies a transparent
rule-based label table. Either uncertain output yields an inconclusive result;
if one detector fails, the route returns partial results without a combined
verdict. This is not a trained neural fusion model and does not check
image-caption consistency. See [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn unified.app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/`. The existing standalone backend remains
available with its earlier command, but do not run both apps on port 8000 at
the same time. To set a private history database outside the project, before
starting the unified server:

```powershell
$env:UNIFIED_DB_PATH = 'D:\imad-local\history.sqlite3'
```

To explicitly retain original analysis inputs (off by default):

```powershell
$env:UNIFIED_RETAIN_INPUTS = 'true'
```

The full UI/API implementation, privacy notes, and schema are documented in
[`UNIFIED_FRONTEND_IMPLEMENTATION.md`](UNIFIED_FRONTEND_IMPLEMENTATION.md) and
[`DATABASE_SCHEMA.md`](DATABASE_SCHEMA.md).

Run regression tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest phase3\tests unified\tests backend\tests -q
.\.venv\Scripts\python.exe -m compileall -q phase3 unified
.\.venv\Scripts\python.exe phase3\tools\snapshot_originals.py --verify
```

Neural Phase 3 training remains blocked pending written data-rights clarification and
verified, source-grouped paired image/text labels. The additional MS COCOAI/Defactify
lead provides real/AI images with human captions only; its dataset license is not
declared and COCO image rights vary by source. It was not downloaded. Dataset screening is in
[`phase3/reports/dataset_comparison.md`](phase3/reports/dataset_comparison.md);
no dataset was downloaded and no training was run.

## Analyze all labeled image/text predictions

To score every image and text item in the existing held-out test splits using
the currently configured FastAPI detectors, and save per-item predictions plus
an error summary:

```powershell
.\.venv\Scripts\python.exe training\analyze_detector_predictions.py `
  --split test `
  --device auto `
  --output training\reports\detector_error_analysis.json
```

The command does not train or tune models. It does not store raw text in the
report and refuses to overwrite a report unless `--overwrite` is specified.
See [`docs/detector_prediction_analysis.md`](docs/detector_prediction_analysis.md)
for the latest measured results and their limitations.

The unified image+text mode uses the same currently configured Phase 1/2
checkpoints as the individual modes. Its current combined label is rule-based;
no trained Phase 3 fusion checkpoint is available. Restart the unified server
after changing `IMAGE_MODEL_CHECKPOINT` or updating code.
