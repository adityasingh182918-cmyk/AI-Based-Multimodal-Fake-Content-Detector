# Real-world failure investigation

**Status: IN_PROGRESS — awaiting the user's camera photo and AI-generated text.** No substitute examples were created. The reports under `training/reports/image_real_world_errors.json` and `training/reports/text_real_world_errors.json` explicitly record that no example-level prediction or probability has been measured.

## What was verified

### Image detector

- FastAPI defaults to `models/image_detector/resnet18_cifake_aug_v1_calibrated_strict.pt`; `IMAGE_MODEL_CHECKPOINT` can override it. The checkpoint identifies ResNet-18, `REAL` index 0, `AI_GENERATED` index 1, 224-pixel input, ImageNet mean/std, and aspect-preserving square padding.
- Inference applies EXIF orientation, converts to RGB, and runs the checkpoint-matched transform. Training also applies EXIF orientation and RGB conversion. I found no class-order or missing-orientation bug.
- The local CIFAKE manifest has 2,000 samples, and inspection confirmed all 2,000 files are 32×32 JPEGs; the model resizes/pads these small images to 224×224. CIFAKE's real class comes from CIFAR-10, while its generated class comes from Stable Diffusion 1.4. This is a verified resolution/source mismatch with phone photographs, not proof of the exact failure cause.
- On the same 200-image CIFAKE test split, the strict augmented checkpoint had REAL FPR 0.04 and AI FNR 0.01; the earlier baseline had REAL FPR 0.08 and AI FNR 0.01. Neither test includes camera photographs. Lowering thresholds alone cannot establish camera-domain performance.
- The real-example analysis workflow records source and EXIF-oriented dimensions, format/mode, checkpoint preprocessing and normalized tensor summary, label/scores, model version, and inference time. It stores neither image bytes nor source paths in the report.

### Text detector

- With no environment override, FastAPI selects the HC3 TF-IDF + calibrated Linear SVM checkpoint. DistilBERT is selected only with `TEXT_MODEL_BACKEND=transformer`; `TEXT_MODEL_CHECKPOINT` can override its path. Therefore, the running environment must be checked before attributing the reported failure to one model.
- Baseline inference reads the fitted estimator's `classes_` to find `AI_GENERATED`; DistilBERT config maps class 0 to `HUMAN_WRITTEN` and class 1 to `AI_GENERATED`, matching its inference index. I found no label-order bug in the saved models.
- The SVM splits long text near paragraph/sentence boundaries at about 2,000 characters and averages chunk AI scores weighted by word count. DistilBERT uses its checkpoint tokenizer with overlapping 256-token windows and stride 32, then aggregates window softmax scores weighted by attended token count. This is not a validated document-level classifier.
- Verified distribution-shift evidence predates this investigation: MAGE GPT-4 OOD AI FNR is 0.5763 for SVM and 0.4038 for DistilBERT; GRADTEX Test C machine-involvement FNR is 0.8131 and 0.8867; Beemo expert-edited AI positive-only recall is 0.3003 and 0.3476 (no two-class metrics apply). These results support weak out-of-domain sensitivity; they do not reproduce the user's example.
- **Confirmed output metadata bug fixed:** DistilBERT returned `score_type=calibrated_transformer_probability` although raw softmax was used and the temperature study was not deployed. It now reports `uncalibrated_softmax`, and its warning states that softmax probabilities, validation-selected thresholds, and chunk aggregation have no demonstrated OOD/document-level reliability. No model score, decision threshold, or checkpoint changed.
- The frontend sends image bytes to `/api/v1/predict/image` and JSON `{text: ...}` to `/api/v1/predict/text`. Combined mode invokes both APIs independently with `Promise.all`; no fusion model is involved.
- The sample-analysis workflow reports expected/predicted label, class scores, word/character/chunk counts, inference time, selected backend, and checkpoint. It writes no raw text or input path.

## Verified causes versus hypotheses

| Status | Finding |
|---|---|
| Verified implementation issue, fixed | DistilBERT described uncalibrated softmax as calibrated probability. |
| Verified limitation | Image model was trained/evaluated on 32×32 CIFAKE images; no camera-photo evaluation exists. |
| Verified limitation | Text checkpoints were trained on a small HC3 QA corpus and perform poorly on several independent sources/transforms. |
| No bug found in inspected code | Image class order, image EXIF/RGB handling, SVM class lookup, DistilBERT class IDs, and frontend request shapes match saved configuration/routes. |
| Hypothesis, needs supplied examples/experiments | Camera sensor processing, JPEG compression, sharpening/denoising, aspect/padding, color response, or model reliance on CIFAKE artifacts causes the image error. |
| Hypothesis, needs supplied text | Unseen generator, topic/style/source cues, human editing, text length, or chunk aggregation causes the particular text error. |

## Reproduce the supplied examples privately

Place each sample in a private local folder outside public datasets. Create separate CSVs with columns `path,expected_label[,sample_id]`. Image labels are `REAL`/`AI_GENERATED`; text labels are `HUMAN_WRITTEN`/`AI_GENERATED`. Text files should be UTF-8. Relative paths are resolved from the CSV folder first. The analyzer emits no text or image path into JSON.

```powershell
python training\analyze_real_world_failures.py --kind image --manifest data\private_failures\image_manifest.csv --output training\reports\image_real_world_errors.json
python training\analyze_real_world_failures.py --kind text --manifest data\private_failures\text_manifest.csv --text-backend baseline --output training\reports\text_real_world_errors.json
python training\analyze_real_world_failures.py --kind text --manifest data\private_failures\text_manifest.csv --text-backend transformer --output training\reports\text_real_world_errors_distilbert.json
```

Text examples should be scored with the backend actually used in the running app. The additional DistilBERT run is a diagnostic comparison only. Keep user samples out of training and public benchmark corpora unless the user explicitly provides suitable rights and requests that use.

## Improvement plan

### Phase 1

1. Collect a licensed, diverse camera-photo evaluation set covering multiple phones, lighting, scenes, formats, and resolutions; keep it independent from training.
2. Capture source metadata (device/format/resolution) and measure REAL false positives before changing preprocessing or thresholds.
3. Compare versioned image checkpoints on identical held-out CIFAKE and camera sets. Add high-resolution real photographs and diverse generated sources to training only after defining training/validation/test boundaries.
4. Evaluate resolution-preserving resize, crop, JPEG/compression, blur, noise, color, and camera-processing augmentation using validation data. Recalibrate only on representative validation data; keep independent camera evaluation untouched.

### Phase 2

1. Acquire appropriately licensed human and AI text from multiple domains/generator families. Use source/prompt/document grouping to prevent leakage.
2. Include separately labeled paraphrased AI, human-edited AI, and grammar-corrected human examples; keep independent source and generator sets for evaluation.
3. On supplied/private examples, compare SVM and DistilBERT chunk/window scores and predictions before any training change.
4. Train only on training data; select and calibrate on validation data. Compare against the existing MAGE, GRADTEX, and Beemo reports without incorporating those test examples into training or threshold selection.
5. Treat `UNCERTAIN` coverage and errors as empirical trade-offs, not confidence guarantees. Consider document-level training/evaluation before changing chunk aggregation.

No retraining, threshold tuning, or independent-test contamination was performed during this investigation. Existing Phase 1/2 checkpoints are preserved. Phase 3 fusion is not implemented.
