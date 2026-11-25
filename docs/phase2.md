# Phase 2: text detection

Phase 2 now includes the HC3 baseline and a separate expanded HC3+MAGE experiment. The MAGE subset is fetched by an explicit pinned streaming command, prepared with separate train/validation/test splits, and used to create versioned expanded SVM and DistilBERT checkpoints. Existing HC3 checkpoints remain available; the API still defaults to the original HC3 SVM. Select a checkpoint with `TEXT_MODEL_BACKEND` as documented in the root README.

On the combined expanded test split, expanded DistilBERT scored 0.8579 accuracy / 0.8524 macro-F1 (human false-positive rate 0.2577); expanded SVM scored 0.7503 / 0.7502 (human FPR 0.1810). These are not independent-source generalization estimates. On MAGE GPT-4 OOD, expanded DistilBERT macro-F1 was 0.7476 with 0.4580 human FPR; expanded SVM macro-F1 was 0.6673 with 0.2795 human FPR. GRADTEX results are also weak, and its source corpus derives from MAGE. See `PHASE2_RETRAINING_PLAN.md` and `PHASE2_EXPERIMENT_LOG.md` for per-source, per-domain, per-length reports and limitations.

Expanded DistilBERT uses temperature-scaled softmax with temperature fitted only on validation; this improves Brier, NLL and ECE on validation. Both expanded model abstention thresholds collapse to [0,1] under the chosen 5% per-class error target, making every prediction UNCERTAIN. This means useful three-way calibration is not solved. The model outputs remain experimental and cannot establish authorship.

## Setup and run

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m pip install -r requirements-ml.txt
python scripts\download_text_datasets.py
python scripts\prepare_text_dataset.py --max-groups-per-source 1000 --seed 42
python scripts\validate_text_dataset.py --report training\reports\text_dataset_validation.json
python training\train_text_baseline.py
python training\evaluate_text_baseline.py --checkpoint models\text_detector\baseline\hc3_tfidf_linear_svm.joblib --report training\reports\text_evaluation.json
python -m pip install -r requirements-text-transformer.txt
python training\train_text_model.py --dataset-dir data\text\processed\hc3 --output-dir models\text_detector\transformer\distilbert_hc3_v1 --epochs 3 --batch-size 8 --gradient-accumulation 2 --max-length 256 --seed 42
python training\evaluate_text_model.py --dataset-dir data\text\processed\hc3 --checkpoint models\text_detector\transformer\distilbert_hc3_v1 --report training\reports\text_transformer_evaluation.json --device cuda --batch-size 8
python scripts\download_text_robustness.py --include-paraphrase
python scripts\download_text_robustness_extra.py --dataset all
python training\evaluate_text_robustness.py --dataset data\text\robustness\mage\test_ood_set_gpt.csv --report training\reports\text_mage_ood_evaluation.json
python training\evaluate_text_robustness.py --dataset data\text\robustness\mage\test_ood_set_gpt_para.csv --paraphrased-ai-only --report training\reports\text_mage_paraphrase_ai_evaluation.json
python training\evaluate_text_additional_robustness.py --dataset gradtex
python training\evaluate_text_additional_robustness.py --dataset gradtex_b
python training\evaluate_text_additional_robustness.py --dataset beemo
python training\analyze_text_calibration.py --device cuda --batch-size 8
python -m pytest backend\tests -q
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Training is a manual command. The downloader and preparation commands do not train. Baseline inference is the default. To select DistilBERT in PowerShell, set `$env:TEXT_MODEL_BACKEND='transformer'` before uvicorn; optional settings are `$env:TEXT_MODEL_CHECKPOINT` and `$env:TEXT_MODEL_DEVICE`. For API testing: `POST /api/v1/predict/text` with `{"text":"..."}`. See [dataset details](phase2_datasets.md) and [architecture](phase2_architecture.md).

## Scope and limitations

The HC3 test split is in-domain, and its high scores should not be generalized to other genres or generators. Independent evaluations were run on frozen checkpoints. Reddit-to-Wikipedia HC3 source holdout: SVM macro-F1 0.5327. MAGE's 1,562-example GPT-4 OOD test: SVM macro-F1 0.6367 and DistilBERT 0.7144. GRADTEX Test C's held-out transformation scenarios: SVM macro-F1 0.4652 and DistilBERT 0.4124. GRADTEX Test B's paired full-generation/AI-edit scenarios: SVM 0.4561 and DistilBERT 0.4361. These machine-involvement labels are not equivalent to full AI authorship. Beemo's 656 human-expert-edited AI-origin items are a positive-only probe: AI recall was 0.3003 (SVM) and 0.3476 (DistilBERT); accuracy, precision, F1, and FPR are not applicable. MAGE's 800-row AI-paraphrase slice is also positive-only and reports recall only. Exact normalized text overlap checks for GRADTEX/Beemo against HC3 splits and the previously evaluated MAGE files found zero matches; semantic near-duplicate checks are not implemented.

For the original HC3 checkpoints, validation-only temperature scaling was investigated but not deployed. The SVM temperature selected on HC3 validation was 0.678; DistilBERT's was 1.591. Test log loss improved on the HC3 test split for both (SVM 0.0939 to 0.0918; DistilBERT 0.2335 to 0.1746), but this small in-domain result did not justify changing those original services. For the separate expanded DistilBERT, a later validation-only temperature of 2.2867 improved Brier, NLL and ECE and is deployed only in that versioned checkpoint. Its score calibration does not solve the abstention problem: expanded model validation thresholds collapse to 0/1 and return UNCERTAIN for all examples. Three-way outputs are not demonstrated reliable out of domain.

The live API smoke checks passed for the image route, text short-input error, multi-chunk text, and simultaneous independent image/text requests. The image model loaded on CUDA and returned a response for a local synthetic fixture. Static UI/API tests and responsive CSS/viewport declarations were inspected; actual desktop/mobile browser rendering and assistive-technology checks remain unverified because the available browser-control runtime denied Brave and no headless browser runtime is installed. The text input now has an explicit label, and the image file input is keyboard-focusable with a visible focus style. No trained Phase 3 fusion model exists.
