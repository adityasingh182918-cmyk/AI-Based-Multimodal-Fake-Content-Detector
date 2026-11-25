# Phase 3 implementation plan

Phase 3 adds an isolated image-text fusion experiment. Phase 1 and Phase 2 code,
models, thresholds, datasets, endpoints, and UI are frozen. The only permitted
root-level tracking edit is the appended Phase 3 section in `PROJECT_PROGRESS.md`.

## Status

| Stage | Status | Evidence / next action |
|---|---|---|
| 0. Audit and freeze baseline | COMPLETED | `phase3/reports/original_state_baseline.json`; copied baseline checks and all 97 original source/model backup files verified. |
| 1. Paired data and provenance | BLOCKED | Ten named candidates screened. None is approved: no verified four-way paired labels with intended-use rights. RU-AI and MS COCOAI/Defactify are research leads only; do not download pending rights/provenance resolution. |
| 2. Isolated architecture | COMPLETED | Frozen Phase 1/2 encoders, sparse text projection, gated fusion network, explicit four-way targets. |
| 3. Feature encoding and training code | COMPLETED, NOT RUN | Preflight checks provenance fields, all four combinations in each split, group leakage, exact cross-split image/text duplicates, and image integrity. No data/features exist. |
| 4. Fusion evaluation | IMPLEMENTED, NOT RUN | Held-out paired test evaluation script; no checkpoint or test features. |
| 5. API/UI | RULE-BASED ANALYSIS IMPLEMENTED; NEURAL FUSION BLOCKED | `unified.app` presents independent image/text results and applies the documented rule table; `phase3.app` remains the separate checkpoint-based experimental API. |
| 6. Regression and freeze verification | COMPLETED | 84 combined tests pass; browser text UI smoke succeeds; compile check passes; frozen hashes/predictions unchanged. |

## Data decision

The required dataset screening and license notes for MiRAGeNews, OpenFake,
DGM4, MMFakeBench, Fakeddit, M4FC, GenImage, VisualNews, RU-AI, and
MS COCOAI/Defactify are in
[`phase3/reports/dataset_comparison.md`](phase3/reports/dataset_comparison.md)
and [`phase3/reports/dataset_license_audit.md`](phase3/reports/dataset_license_audit.md).
All candidates have a categorical decision; none is approved for training.
RU-AI's Flickr8K slice has the strongest four-way schema, but its precise
dataset/upstream rights remain unresolved. MS COCOAI/Defactify offers paired
real/generated images and human captions (96,000 rows, 7.51 GB), but has no
AI-text class, its HF card directs license questions to the authors, and COCO
image rights are per-image. Neither is cleared, so no subset is downloaded.

The candidate MiRAGeNews dataset has 15,000 image-caption pairs, a 10,000/2,500
train/validation split and five 500-item tests. Its published summary reports
real or AI-generated *multimodal news* with source/generator splits, but the
visible dataset metadata exposes a binary `real`/`fake` label and no clear
license. It does not establish the required independent four-way image/text
origin labels. It is therefore a research lead, not eligible training data as
currently documented. See [dataset card](https://huggingface.co/datasets/anson-huang/mirage-news)
and [paper](https://aclanthology.org/2024.findings-emnlp.959/).

MFND advertises four modality combinations and 11 manipulation types, but the
paper concerns fake news and image manipulation; that is not automatically
equivalent to AI-generated image/text authorship, and no acceptable dataset
license was verified on the paper/repository pages. See [IJCAI paper](https://ijcai.org/proceedings/2025/891)
and [code/data repository](https://github.com/yunan-wang33/sdml). No candidate
was downloaded. Obtain written license and label/provenance confirmation or
select a documented dataset whose definitions match this task before training.

## Architecture and commands

Frozen ResNet-18 features (512 dimensions) and the Phase 2 TF-IDF feature
pipeline feed separate learned projections, a learned gate, four-class output,
and auxiliary modality-origin heads. Only the fusion projections/heads train;
the copied encoders remain frozen. Text TF-IDF stays sparse through projection.

From the project root in PowerShell, after placing a reviewed dataset in the
schema described in `phase3/data/README.md`:

```powershell
.\.venv\Scripts\python.exe -m phase3.training.encode_pairs --data-dir phase3/data/paired --image-root phase3/data --output-dir phase3/data/features --device auto
.\.venv\Scripts\python.exe -m phase3.training.train_fusion --features-dir phase3/data/features --checkpoint phase3/models/fusion/best_model.pt --last-checkpoint phase3/models/fusion/last_checkpoint.pt --history phase3/reports/fusion_training_history.json --device cuda
.\.venv\Scripts\python.exe -m phase3.evaluation.evaluate_fusion --features-dir phase3/data/features --checkpoint phase3/models/fusion/best_model.pt --report phase3/reports/fusion_test_evaluation.json --device cuda
.\.venv\Scripts\python.exe -m uvicorn phase3.app:app --host 127.0.0.1 --port 8001
```

Do not execute these training/evaluation commands until data eligibility is
resolved and manifests are populated. Use `unified.app:app` on port 8000 for
the single-app frontend. It returns independent Phase 1/2 results together
with a transparent label-only rule-based assessment; this is not a learned
fusion model and has no trained-model metrics. The standalone `phase3.app`
multimodal route remains checkpoint-based and returns 503 while untrained.
