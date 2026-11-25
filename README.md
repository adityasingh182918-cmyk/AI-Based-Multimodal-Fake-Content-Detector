# AI-Based Multimodal Fake Content Detector

An experimental research and software project for analyzing whether image and text inputs resemble content labeled as real, AI-generated, human-written, or AI-generated. The project includes image and text inference services and a unified interface that can analyze the two modalities together.

> **Research use:** Detector outputs are estimates learned from particular datasets. They are not proof of authenticity, authorship, factuality, or intent. Read the limitations and evaluation scope before interpreting a result.

## Problem statement

AI-generated and edited media can be difficult to distinguish from human-created content. Classifiers can help study this problem, but their performance depends heavily on the training data, source, domain, generator, and preprocessing. This project builds reproducible experimental pipelines and makes their current limits visible rather than treating a model score as ground truth.

## Objectives

- Build image and text dataset preparation, training, inference, and evaluation workflows.
- Keep checkpoints and experiment reports versioned or otherwise traceable.
- Compare performance on held-out and source-shifted data where available.
- Provide local APIs and a browser interface for image, text, and paired image-text analysis.
- Distinguish validated software behavior from evidence of predictive performance.

## System architecture

```text
Browser UI ──> Unified FastAPI app ──> Phase 1 image detector
                                  ├──> Phase 2 text detector
                                  └──> Phase 3 rule-based combination

Separate experimental Phase 3 app ──> fusion network (checkpoint required)
```

The unified app reuses the Phase 1 and Phase 2 detectors. Its current multimodal result is a deterministic rule over their predicted labels; it is not the output of a trained fusion model. The separate `phase3.app` is a checkpoint-based experimental service and is unavailable for prediction until a suitable trained fusion checkpoint exists.

## Implementation status

| Phase | Status | What is present |
|---|---|---|
| **Phase 1 – Image detection** | Implemented; predictive evidence is dataset-limited | ResNet-18 training/evaluation, checkpoint-aware preprocessing, calibrated abstention metadata, FastAPI inference, and image UI. |
| **Phase 2 – Text detection** | Implemented; experimental | TF-IDF/SVM and DistilBERT pipelines, multiple checkpoint choices, evaluation scripts, API inference, and text UI mode. Cross-domain reliability remains weak. |
| **Phase 3 – Multimodal fusion** | Rule-based unified analysis implemented; learned fusion experimental/incomplete | Unified image-plus-text endpoint combines detector labels by a documented decision table. Fusion network, encoders, and training/evaluation code exist, but no trained/deployed fusion checkpoint or fusion quality metrics are available. |
| **Phase 4 – Advanced media** | Planned | Potential future audio/video and temporal analysis. No completed Phase 4 detector is claimed. |

## Phase 1 – Image detection

The image pipeline uses a ResNet-18 classifier with RGB input, 224-pixel preprocessing, aspect-preserving padding, and ImageNet normalization. Training supports pretrained initialization, augmentation, validation-loss checkpoint selection, and a held-out test workflow. Inference reads the selected local checkpoint and its metadata; the current default is `models/image_detector/resnet18_realworld_v1.pt`. Set `IMAGE_MODEL_CHECKPOINT` to select a different checkpoint explicitly. The real-world checkpoint includes four user-labelled REAL supplemental photos, so its predictions on those examples are not independent evaluation evidence.

The main reproducible dataset sample is a balanced 2,000-image CIFAKE subset: 1,600 training, 200 validation, and 200 test images. On the 200-image CIFAKE test split, the augmented checkpoint recorded 0.975 accuracy, 0.975 macro-F1, 0.04 REAL false-positive rate, and 0.01 AI false-negative rate. These are benchmark-specific results on low-resolution CIFAKE images, not camera-photo performance. A later candidate trained with four user-labeled REAL photos tied the strict checkpoint's test accuracy and was not promoted.

See [Phase 1 workflow and evidence](docs/phase1.md), [implementation plan](PHASE1_IMPLEMENTATION_PLAN.md), and [real-photo fine-tuning report](PHASE1_RETRAINING_REPORT.md).

## Phase 2 – Text detection

Text options include word/character TF-IDF with a calibrated Linear SVM and fine-tuned DistilBERT checkpoints. Original HC3-derived checkpoints and expanded HC3+MAGE candidates are kept separately. The service can be selected with `TEXT_MODEL_BACKEND` (`baseline`, `svm_expanded`, `transformer`, or `transformer_expanded`) and can report its active backend through the status API.

On the same 5,727-row expanded HC3+MAGE test split, expanded SVM achieved 0.7503 accuracy / 0.7502 macro-F1 and expanded DistilBERT 0.8579 / 0.8524. These mixed-split scores do not establish source-independent performance. On frozen MAGE GPT-4 out-of-domain data, expanded DistilBERT macro-F1 was 0.7476 with human false-positive rate 0.4580. GRADTEX results are weaker, and some evaluated labels represent machine involvement rather than wholly AI-authored text. No expanded candidate was promoted as a generally reliable authorship detector.

See [Phase 2 workflow and results](docs/phase2.md), [dataset notes](docs/phase2_datasets.md), [evaluation report](PHASE2_EVALUATION_REPORT.md), and [model card](PHASE2_MODEL_CARD.md).

## Phase 3 – Multimodal fusion

The unified app accepts an image and related text and runs the existing image and text detectors independently. Its current combination rule maps their labels to `AUTHENTIC CONTENT`, `MIXED CONTENT`, `AI-GENERATED CONTENT`, or `INCONCLUSIVE`; if a detector fails, the response can be partial and no combined verdict is made. There is no separate combined confidence score, semantic image-caption consistency check, or factuality check.

This rule-based behavior has software regression evidence, but no predictive metric. A learned fusion architecture and training/evaluation scripts are present under `phase3/`, but a suitable paired dataset and its usage rights remain unresolved. No neural fusion model has been trained, deployed, or evaluated, and the separate checkpoint-based Phase 3 app returns unavailable while its checkpoint is absent.

See [Phase 3 implementation](PHASE3_IMPLEMENTATION_REPORT.md), [evaluation status](PHASE3_EVALUATION_REPORT.md), and [API details](API_DOCUMENTATION.md).

## Phase 4 – Future scope

Phase 4 is a roadmap, not an implemented module. Possible work includes audio and video detection, temporal and provenance signals, better source-aware evaluation, and integration of independently validated models. Any future model should use legally suitable data, preserve source-separated test sets, and report uncertainty and false-positive behavior.

## Technologies

- Python, FastAPI, Uvicorn, Pydantic, and `python-multipart`
- PyTorch and torchvision for image modeling
- scikit-learn for TF-IDF/SVM text modeling
- Hugging Face Transformers for DistilBERT experiments
- Pillow, NumPy, `datasets`, and tqdm for image/data workflows
- HTML, CSS, and JavaScript for the browser UI; no frontend build is required
- pytest and HTTPX for automated checks

## Repository structure

```text
backend/                 Phase 1/2 API services, schemas, static UI, and tests
data/                    Local datasets/manifests; most contents are Git-ignored
docs/                    Workflow, architecture, dataset, and diagnostic guides
frontend/                Browser UI usage notes (UI assets are served by FastAPI)
models/                  Local image/text checkpoints and metadata
notebooks/               Notebook guidance and exploratory work
phase3/                  Isolated fusion research code, tools, tests, and reports
research_paper_output/   Generated paper, analysis, figures, and workbook
scripts/                 Dataset inspection, preparation, and validation tools
training/                Model training, calibration, evaluation, and diagnostics
unified/                 Combined API, browser UI, local history, and tests
*.md                     Phase plans, changelogs, experiment logs, and reports
```

Source files have intentionally not been reorganized: current imports and runtime paths rely on this layout.

## Dataset information

- **CIFAKE:** The checked-in documentation describes a pinned Hugging Face Parquet source used to prepare a reproducible 2,000-image subset. Local dataset contents and manifests are ignored by Git. The sample derives from CIFAKE's REAL CIFAR-10 and Stable Diffusion-generated classes; see the source and license details in [docs/phase1.md](docs/phase1.md) and the preparation instructions in the existing reports.
- **GenImage:** Inspection/preparation scripts are available, but the previously inspected multi-part archive was incomplete and was not extracted for training.
- **HC3 and MAGE:** Used for text baselines and expanded experiments. Additional frozen data slices such as MAGE GPT-4, GRADTEX, and Beemo support specific diagnostics; their label meanings and licensing scopes differ. See [dataset notes](docs/phase2_datasets.md).
- **Phase 3 paired data:** No approved paired dataset was prepared for fusion training. Dataset rights and label suitability remain a blocker.

Dataset downloads are not part of installation. Run preparation/download scripts only when intentionally acquiring data and after checking their documented source and terms.

## Model architectures and artifacts

- **Image:** ResNet-18; see checkpoint metadata for preprocessing, labels, and run details.
- **Text:** TF-IDF with Linear SVM, or fine-tuned DistilBERT transformer.
- **Multimodal:** the unified endpoint currently uses a fixed label decision rule. A gated fusion network and modality encoders exist as experimental Phase 3 code, without a trained fusion checkpoint.

Model weights are local artifacts and are not automatically included in a Git checkout under the current ignore policy. Inference requires the corresponding checkpoint files; see [models/README.md](models/README.md). Do not treat a metadata file alone as a model. Large artifacts should be distributed through Git LFS or an appropriate model host if publication is intended; see the final report before changing tracking policy.

## Experimental results and evaluation

Evaluation scripts keep validation and test roles separate, and reports document dataset-specific results. Image metrics above use the same 200-image CIFAKE test split. Text metrics above distinguish the mixed expanded test from source-shifted diagnostics. Phase 3 has no fusion accuracy, precision, recall, F1, or generalization result because no trained fusion checkpoint was evaluated. Software test passes validate code behavior and do not establish model quality.

For detailed methodology and caveats, consult the phase reports linked above, [Phase 2 experiment log](PHASE2_EXPERIMENT_LOG.md), and [project progress](PROJECT_PROGRESS.md).

## Installation and environment

The commands below use Windows PowerShell, matching the checked-in project guidance. Create a local virtual environment and install the backend runtime first:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

For Phase 1 training/data workflows, install the selected PyTorch/torchvision build for your operating system and CUDA hardware using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install:

```powershell
python -m pip install -r requirements-ml.txt
python -m pip install -r requirements-dev.txt
```

For transformer text training/inference, install:

```powershell
python -m pip install -r requirements-text-transformer.txt
```

`requirements-ml.txt` does not install PyTorch; select that separately. Dataset and pretrained-weight downloads occur only when explicitly running relevant preparation or training workflows. Configure local overrides in an untracked `.env` or shell environment; `.env.example` is a template and contains no real credentials.

## Backend execution

Run the Phase 1/2 API from the repository root:

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Run the unified interface/API instead:

```powershell
python -m uvicorn unified.app:app --host 127.0.0.1 --port 8000
```

Use one application per port. The unified app serves its browser interface at `/` and API documentation at `/docs`. The Phase 1/2 app serves the image interface at `/ui` and its OpenAPI documentation at `/docs`. If a model checkpoint is missing or invalid, the associated inference endpoint may report that it is not ready.

Common local settings include `IMAGE_MODEL_CHECKPOINT`, `IMAGE_MODEL_DEVICE`, `TEXT_MODEL_BACKEND`, `TEXT_MODEL_CHECKPOINT`, `TEXT_MODEL_DEVICE`, and `PHASE3_FUSION_CHECKPOINT`. See [`.env.example`](.env.example), [API documentation](API_DOCUMENTATION.md), and status routes for active model readiness.

## Frontend execution

There is no separate Node/Vite build. The browser pages are static assets served by FastAPI:

- Phase 1 image UI: start the Phase 1/2 API and open `http://127.0.0.1:8000/ui`.
- Unified image, text, and rule-based multimodal UI: start `unified.app` and open `http://127.0.0.1:8000/`.

The browser sends requests to the same local API origin. See [frontend usage notes](frontend/README.md) and [unified UI implementation notes](UNIFIED_FRONTEND_IMPLEMENTATION.md).

## API documentation

FastAPI's interactive OpenAPI page is available at `/docs` for either app. Existing routes include health/model status and image/text prediction; the unified app also provides multimodal prediction, dashboard, privacy, and local history routes. Request and response details are documented in [API_DOCUMENTATION.md](API_DOCUMENTATION.md). The current endpoint definitions are authoritative.

## Training, inference, and evaluation

Training and dataset preparation are explicit commands; installing requirements does not run them. Start with the focused guides:

- Image workflow: [docs/phase1.md](docs/phase1.md), scripts under `training/` and `scripts/`.
- Text workflow: [docs/phase2.md](docs/phase2.md), [architecture](docs/phase2_architecture.md), and scripts under `training/`.
- Fusion experiment: source and tests under `phase3/`, its reports under `phase3/reports/`, and the root Phase 3 reports; current unified multimodal inference is rule-based.
- Evaluation: phase evaluation reports and commands in each workflow guide. Keep held-out test data untouched during model selection.

For inference, install the dependencies for the chosen model, ensure its local checkpoint is present, configure any checkpoint override, start the appropriate app, and inspect `/api/v1/status` (unified) or `/model-status` (Phase 1/2) before submitting input.

## Team

The project team comprises:

- **Aditya Singh** — S24CSEU2200
- **Ronit Jaiswal** — S24CSEU2198
- **Yuvraj Singh** — S24CSEU2182
- **Harsh Singh Chauhan** — S24CSEU2218

The project work covers the Phase 1 image detector, Phase 2 text detector, Phase 3 rule-based multimodal analysis, unified FastAPI/frontend integration, and evaluation/documentation. Individual task assignments were not provided, so this README does not attribute specific components to individual members.

## Limitations

- Image results are based primarily on a small, low-resolution CIFAKE test set; representative camera-photo performance has not been established.
- Text model results vary substantially across sources and domains. A reproduced AI-description example was classified as human-written by active checkpoints; the failure remains unresolved.
- Model score fields have model-specific meanings and are not guaranteed probabilities of authenticity or authorship.
- The unified multimodal rule does not verify whether text describes the image, detect factual inconsistency, or produce a learned joint confidence.
- No trained or evaluated neural fusion checkpoint is available; Phase 4 audio/video work is planned only.
- Checkpoints and datasets are local and ignored by Git under existing rules, so a fresh clone may require separately obtaining compatible artifacts.

## Future improvements

- Build source-aware, license-compatible image and text datasets with independent generator/domain test sets.
- Obtain a suitable paired image-caption dataset with verified rights and labels before training or evaluating fusion.
- Evaluate calibration, abstention, false-positive rates, robustness, and demographic/domain variation on independent data.
- Document reproducible artifact distribution and provide model cards for any checkpoint intended for use.
- Explore audio/video analysis only after defining evaluation data, intended use, and measurable criteria.

## Documentation index

The root contains detailed phase plans, experiment logs, implementation/evaluation reports, and application reports. Additional workflow documents are in `docs/`; preserve these records when preparing a publication release.
