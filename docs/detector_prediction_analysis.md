# Checking every image and text prediction

Use `training/analyze_detector_predictions.py` to run the currently configured
FastAPI image and text detector over every labeled row in the local test split.
It is inference-only: it does not train, tune thresholds, or edit checkpoints.
It writes one JSON file with every sample's expected/predicted class and scores,
plus a Markdown summary with confusion matrices, metrics, text-source slices,
and the first 25 error/uncertain rows. Raw text is not copied into either file.

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe training\analyze_detector_predictions.py `
  --split test `
  --device auto `
  --output training\reports\detector_error_analysis.json
```

The default output is ignored by Git. The script refuses to overwrite an
existing JSON or Markdown report unless `--overwrite` is supplied. It uses the
image manifest at `data/image/cifake_ready/manifests/manifest.csv`, the image
root `data/image/cifake_ready`, and text file
`data/text/processed/hc3/test.jsonl`. Use `--image-manifest`, `--image-root`,
and `--text-file` to analyze other labeled splits. Select `--split validation`
only with corresponding validation manifests/files. `--skip-image` or
`--skip-text` runs just one detector.

## Earlier strict-checkpoint result

The Phase 1 strict ResNet-18 checkpoint was evaluated on 200 CIFAKE test
images: 85.00% accuracy when `UNCERTAIN` counts as incorrect, 85.00% coverage,
and 100% accuracy on covered predictions. It returned `UNCERTAIN` on 30 items
(8 REAL and 22 AI_GENERATED) and made no confident REAL/AI class swaps on this
small test split. This does not contradict the user-reported camera-photo
false positives: those four real photos were previously all labeled
`AI_GENERATED` in `training/reports/image_real_world_errors.json`. The uploaded
photos are high-resolution camera images while the CIFAKE files are 32×32, so
the test score is not evidence that the detector works on camera photos.

## Latest candidate-checkpoint result and mode mismatch

The subsequent run selected `models/image_detector/resnet18_realworld_v1.pt`
(shown by the JSON report), not the default strict checkpoint. On the same 200
CIFAKE test images it predicted 195 correctly and 5 incorrectly: four
REAL→AI_GENERATED and one AI_GENERATED→REAL. Coverage was 100%; REAL false-
positive rate was 4%, and AI false-negative rate was 1%. The candidate's scores
are uncalibrated softmax.

The candidate was trained using the four REAL user photos. A side-by-side
check predicted those four photos REAL with the candidate and AI_GENERATED
with the frozen strict Phase 3 image copy. The candidate result on those same
training photos is in-sample and must not be treated as independent test
evidence.

Before the fusion checkpoint existed, the unified image+text route used the
Phase 3 frozen copies while the standalone image and text modes used the live
configured FastAPI detectors. That caused the modes to disagree. The unified
route now uses the same live Phase 1/2 detectors until fusion is trained.
Restart `unified.app:app` to load the fix. The separate legacy `phase3.app:app`
on port 8001 continues to use its frozen copies, so its unimodal outputs can
differ by design.

The latest ignored JSON report reflects whichever checkpoint the process
selected. Check its `checkpoint` field before comparing it with older results.

## Text result

The Phase 2 HC3 SVM was evaluated on 531 HC3 test passages: accuracy
94.16% with `UNCERTAIN` counted incorrect, macro-F1 96.17%, coverage 95.86%,
and accuracy 98.23% on covered predictions. There were 31 non-correct outcomes:
22 UNCERTAIN, 4 HUMAN_WRITTEN labeled AI_GENERATED, and 5 AI_GENERATED labeled
HUMAN_WRITTEN. Source accuracy was 96.64% on `reddit_eli5` and 91.63% on
`wiki_csai`. This is in-domain HC3 QA evidence, not general text-authorship
reliability.

Latest generated reports (local, ignored outputs):

- `training/reports/detector_error_analysis.json`
- `training/reports/detector_error_analysis.md`

The model output for a particular sample can still be wrong. Treat these
figures as diagnostics of the supplied test splits, not proof of provenance or
performance on user photos or unseen text sources.
