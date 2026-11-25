# Phase 2 expansion and retraining report

Updated: 2026-11-25

## Scope and safeguards

This experiment expands the existing text detector using the pinned official `yaful/MAGE` dataset release. MAGE is streamed and deterministically sampled by source/class; the full source files are not saved. Official MAGE train, validation and test partitions were preserved. Existing HC3 files and checkpoints remain unchanged. Frozen MAGE GPT-4 OOD, GRADTEX and Beemo evaluation artifacts were excluded from training and calibration. No Phase 1 model or Phase 3 fusion was modified.

The dataset card states Apache-2.0. The MAGE source rows do not provide original document/group IDs, so exact normalized-text duplicates could be removed but semantic and cross-generator document leakage cannot be ruled out. Treat the combined test results as a same-source held-out benchmark, not proof of broad generalization.

## Dataset and models

Prepared HC3+MAGE counts:

| Split | Total | AI generated | Human written |
|---|---:|---:|---:|
| Train | 26,954 | 13,730 | 13,224 |
| Validation | 6,650 | 3,398 | 3,252 |
| Test | 5,727 | 3,158 | 2,569 |

The expanded SVM uses combined word and character TF-IDF with calibrated Linear SVM C=0.5. Expanded DistilBERT continued from the existing HC3-finetuned checkpoint and selected epoch 3 using validation macro-F1. They are saved separately at `models/text_detector/svm_expanded_v1.joblib` and `models/text_detector/distilbert_expanded_v1/`.

## Same-example held-out test comparison

| Checkpoint | Accuracy | Macro-F1 | Human FPR | AI FNR |
|---|---:|---:|---:|---:|
| Existing HC3 SVM | 0.5591 | 0.5323 | 0.1098 | 0.7103 |
| Expanded SVM | 0.7503 | 0.7502 | 0.1810 | 0.3056 |
| Existing HC3 DistilBERT | 0.5986 | 0.5792 | 0.0934 | 0.6520 |
| Expanded DistilBERT | 0.8579 | 0.8524 | 0.2577 | 0.0481 |

All four models were scored on exactly the same 5,727 prepared test examples. Expanded DistilBERT has the best benchmark macro-F1 and AI recall, while producing a higher human false-positive rate. The detailed report includes class-wise precision/recall, confusion matrices, source and length slices, and per-example predictions/scores: `training/reports/text_model_comparison_expanded_test.json`.

Metric source is explicitly the API-matched comparison (character chunking for SVM; overlapping token windows for DistilBERT). The standalone legacy evaluators process rows differently (single full SVM row and DistilBERT truncation), so their local reports differ slightly: SVM 0.7524 accuracy / 0.7523 macro-F1; DistilBERT 0.8537 / 0.8479. Do not mix those values with the API-matched comparison; long-text scoring parity between standalone evaluators and service remains a maintenance limitation.

## Frozen independent robustness evidence

These datasets were not used for training, selection or calibration. Hard metrics use score cutoff 0.5.

| Dataset | Kind | Expanded SVM accuracy / macro-F1 | Expanded DistilBERT accuracy / macro-F1 |
|---|---|---:|---:|
| MAGE GPT-4 OOD (n=1,562) | Two class | 0.6677 / 0.6673 | 0.7612 / 0.7476 |
| GRADTEX Test C (n=3,529) | HWT vs machine-involved MIX | 0.5285 / 0.4812 | 0.5551 / 0.5369 |
| Selected paired GRADTEX Test B (n=2,642) | Source controls vs transformation probes | 0.5443 / 0.4998 | 0.6219 / 0.6154 |
| Beemo expert-edited AI (n=656) | AI-origin positive-only | Not applicable | Not applicable |

MAGE GPT-4 OOD human FPR / AI FNR: SVM 0.2795 / 0.3825; DistilBERT 0.4580 / 0.0300. GRADTEX Test C: SVM 0.1690 / 0.7735; DistilBERT 0.2462 / 0.6433. GRADTEX Test B: SVM 0.1575 / 0.7540; DistilBERT 0.2483 / 0.5079. On Beemo, AI recall only is 0.5534 for SVM and 0.8354 for DistilBERT; no negatives exist, so accuracy, F1, precision and FPR are not reported.

Full report `training/reports/text_expanded_ood_diagnostics.json` contains per-source/domain/scenario/length metrics, confusion matrices and scores. GRADTEX derives from MAGE, so it is not an independent corpus family. Results show material domain sensitivity and do not support a universal AI-text detector claim.

## Calibration and API

Scalar temperature was fit using only `data/text/processed/expanded_v1/validation.jsonl`. Expanded DistilBERT temperature 2.2867 improved validation Brier score (0.11605 to 0.10685), log loss (0.48195 to 0.34975), and 10-bin ECE (0.11240 to 0.09806), so it is recorded in new checkpoint metadata and used at inference. Expanded SVM temperature scaling did not improve all three, so it remains at 1.0. A calibrated score is still not a guarantee of domain-independent confidence.

The validation quantile abstention policy for a 5% per-class error target produced thresholds REAL max 0 and AI min 1 for both expanded models. Therefore all responses are `UNCERTAIN` under this policy. This is honest but not practically useful; no robust three-way threshold solution is claimed. The current text API supports selecting `baseline`, `svm_expanded`, `transformer`, or `transformer_expanded`; the original HC3 SVM remains default. Combined UI results still run image and text independently and are not a trained fusion model.

## Validation and remaining work

- Dataset validator: no errors on the prepared expanded splits.
- Changed Python modules: `compileall` passed.
- Backend suite: `58 passed in 13.54s`.
- API integration tests exercise expanded DistilBERT scoring, temperature metadata, long text windows, and image route regression.
- Browser viewport, responsive, keyboard/screen-reader validation remains unverified.
- No accessible, appropriately licensed human grammar-correction/edited-AI dataset was added. The previously gated corpus remains excluded; do not use noncommercial/uncertain-license data for a project with possible commercial use.
- Phase 2 remains `IN_PROGRESS`, not complete. Priority next work: resolve dataset license/access for human-edited text, test on a truly independent source with document/group IDs, improve human false-positive behavior, and complete browser accessibility QA. Do not retrain or tune on the independent evaluation sets.
