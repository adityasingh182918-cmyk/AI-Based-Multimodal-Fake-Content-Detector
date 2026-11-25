# Phase 1 and Phase 2 Protection Report

**Captured before this Phase 3 change:** 2026-11-25

## Verification result

- Original source, model, and dataset manifests unchanged: **True**
- Fixed input predictions unchanged: **True**
- Frozen backup files verified: **97/97 SHA-256 matches**
- Source files in baseline manifest: **61**
- Model artifacts in baseline manifest: **36**
- Dataset files in baseline manifest: **3559**

No Phase 1/2 source, model, checkpoint, dataset, preprocessing, threshold, or prediction logic is to be modified for this task. The full pre-change SHA-256 manifests and fixed-input predictions are recorded in [`original_state_baseline.json`](original_state_baseline.json); the latest independent recheck is [`original_state_postcheck.json`](original_state_postcheck.json).

## Critical inference source hashes

| File | SHA-256 |
|---|---|
| `backend/app/main.py` | `3a61921b59979736b392bc463c456ebf705ff469830cdc39446842e183cbb7e3` |
| `backend/app/services/image_detector.py` | `f2b319292008bbcf728798a2f722b92dc5b4fce7cc926d559765b9f5697ebb0b` |
| `backend/app/services/text_detector.py` | `0fdb8429f8a89ff7800115f7ed6f34c9c601c81f1c0ee07fff3f773f3f7101fc` |
| `backend/app/services/transformer_text_detector.py` | `c168e3eaef94db4fb2d939a6e2ecad09580c9d7f96c1e08537751710ecf6a38c` |
| `backend/app/schemas/prediction.py` | `dbbf2a4847b348ec5957f4ad681def7dc4d2d140e992768fdf98d30998ceae66` |
| `training/train_image_model.py` | `de85da5d31559d1631596d8ebd1704791dbf46c5af35cfbd0892f9bec38bd626` |
| `training/train_text_baseline.py` | `fe9bb641fd6365065ba55c00db83c36255f9d1358ae58837e47bfce3d52594f4` |
| `training/train_text_model.py` | `8f17769445fd8be96278e974d5e913282cc775021e70898ee247d8f9cff331f6` |

## Fixed baseline predictions (redacted inputs)

| Modality | Input reference | Expected | Original output | Original score metadata |
|---|---|---|---|---|
| Image | `fcffaabaf1dcc519…` | REAL | REAL | `{"REAL":0.96178025,"AI_GENERATED":0.038219739}`; 20261124T122547712607Z-validation-abstention |
| Image | `8a75387153f26f44…` | REAL | REAL | `{"REAL":0.999997854,"AI_GENERATED":2.204e-06}`; 20261124T122547712607Z-validation-abstention |
| Image | `d3e1b3b8d52084d6…` | AI_GENERATED | AI_GENERATED | `{"REAL":1.324e-06,"AI_GENERATED":0.999998689}`; 20261124T122547712607Z-validation-abstention |
| Image | `74e38ec9093bdf20…` | AI_GENERATED | UNCERTAIN | `{"REAL":0.093624219,"AI_GENERATED":0.906375825}`; 20261124T122547712607Z-validation-abstention |
| Text | `a72a7906b142c4ac…` | HUMAN_WRITTEN | HUMAN_WRITTEN | `{"HUMAN_WRITTEN":0.999988857,"AI_GENERATED":1.1143e-05}`; calibrated_tfidf_probability |
| Text | `30f9b59b22c58f7b…` | HUMAN_WRITTEN | HUMAN_WRITTEN | `{"HUMAN_WRITTEN":0.99974846,"AI_GENERATED":0.00025154}`; calibrated_tfidf_probability |
| Text | `d880ba729de59086…` | AI_GENERATED | AI_GENERATED | `{"HUMAN_WRITTEN":0.000657161,"AI_GENERATED":0.999342839}`; calibrated_tfidf_probability |
| Text | `94b1c7685840bc17…` | AI_GENERATED | HUMAN_WRITTEN | `{"HUMAN_WRITTEN":0.906037436,"AI_GENERATED":0.093962564}`; calibrated_tfidf_probability |

## Protected file inventory

The full inventories are hashed in the baseline manifest; these paths were checked before implementation:

- `backend/app/__init__.py`
- `backend/app/api/__init__.py`
- `backend/app/core/__init__.py`
- `backend/app/main.py`
- `backend/app/schemas/__init__.py`
- `backend/app/schemas/prediction.py`
- `backend/app/schemas/status.py`
- `backend/app/services/__init__.py`
- `backend/app/services/image_detector.py`
- `backend/app/services/text_detector.py`
- `backend/app/services/transformer_text_detector.py`
- `backend/app/static/index.html`
- `backend/app/utils/__init__.py`
- `backend/tests/__init__.py`
- `backend/tests/http_client.py`
- `backend/tests/test_cifake.py`
- `backend/tests/test_dataset_tools.py`
- `backend/tests/test_image_calibration.py`
- `backend/tests/test_image_detector.py`
- `backend/tests/test_main.py`
- `backend/tests/test_prediction_api.py`
- `backend/tests/test_realworld_dataset.py`
- `backend/tests/test_text_api.py`
- `backend/tests/test_text_detector.py`
- `backend/tests/test_text_expansion_data.py`
- `backend/tests/test_text_robustness_metrics.py`
- `requirements-dev.txt`
- `requirements-ml.txt`
- `requirements-text-transformer.txt`
- `scripts/__init__.py`
- `scripts/download_text_datasets.py`
- `scripts/download_text_expansion.py`
- `scripts/download_text_robustness.py`
- `scripts/download_text_robustness_extra.py`
- `scripts/import_user_real_images.py`
- `scripts/inspect_dataset.py`
- `scripts/prepare_cifake.py`
- `scripts/prepare_dataset.py`
- `scripts/prepare_expanded_text_dataset.py`
- `scripts/prepare_realworld_manifest.py`
- `scripts/prepare_text_dataset.py`
- `scripts/validate_text_dataset.py`
- `training/__init__.py`
- `training/analyze_real_world_failures.py`
- `training/analyze_text_calibration.py`
- `training/calibrate_expanded_models.py`
- `training/calibrate_image_model.py`
- `training/compare_text_models.py`
- `training/data.py`
- `training/evaluate_image_model.py`
- `training/evaluate_text_additional_robustness.py`
- `training/evaluate_text_baseline.py`
- `training/evaluate_text_model.py`
- `training/evaluate_text_robustness.py`
- `training/run_image_additional_diagnostics.py`
- `training/run_text_additional_diagnostics.py`
- `training/text_baseline.py`
- `training/train_expanded_svm.py`
- `training/train_image_model.py`
- `training/train_text_baseline.py`
- `training/train_text_model.py`

The full model and dataset inventories remain in the baseline JSON; all baseline paths matched before this change.
