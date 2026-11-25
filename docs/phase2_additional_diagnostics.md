# Phase 2 additional diagnostics

**Status: evaluation complete for stored independent datasets; the user's specific AI-text failure remains unreproduced because no text example was supplied.** Both checkpoints remained frozen. No independent evaluation data was used for training, model selection, or threshold tuning.

## Same-example comparison

The SVM and DistilBERT were scored on the same rows within each dataset. The new diagnostic uses API-like aggregation: the SVM splits long text at its configured character boundary and word-weights chunk probabilities; DistilBERT uses the local tokenizer with 256-token sliding windows, stride 32, attention-mask token weights, and padding for the final partial window. Older saved evaluator reports truncate DistilBERT inputs to one 256-token sequence; their results are therefore not directly interchangeable with these API-matched metrics.

Binary metrics use each model's AI score at 0.5. FPR is human predicted AI; FNR is AI predicted human. Dataset label meanings remain specific: GRADTEX `MIX` means machine-involved text, not necessarily wholly AI-authored text. Beemo has positive AI-origin samples only, so two-class accuracy, macro-F1 and FPR are inapplicable.

| Dataset (n) | Model | Accuracy | Macro-F1 | Human recall | AI/machine recall | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| MAGE GPT-4 OOD (1,562) | SVM | 0.6466 | 0.6272 | 0.8963 | 0.4088 | 0.1037 | 0.5913 |
|  | DistilBERT | 0.7234 | 0.7197 | 0.8596 | 0.5938 | 0.1404 | 0.4063 |
| GRADTEX Test C (3,529) | SVM | 0.5288 | 0.4638 | 0.8775 | 0.1806 | 0.1225 | 0.8194 |
|  | DistilBERT | 0.5030 | 0.4084 | 0.9036 | 0.1031 | 0.0964 | 0.8969 |
| GRADTEX Test B selected (2,642) | SVM | 0.5201 | 0.4491 | 0.8789 | 0.1612 | 0.1211 | 0.8388 |
|  | DistilBERT | 0.5136 | 0.4298 | 0.8970 | 0.1302 | 0.1030 | 0.8698 |
| Beemo expert-edited AI only (656) | SVM | — | — | — | 0.2973 | — | 0.7027 |
|  | DistilBERT | — | — | — | 0.3476 | — | 0.6524 |

Per-example records include expected label, predicted label, class scores, SVM margin where available, validation-threshold decision, length, chunk/window count, source/domain/scenario, inference time, and FP/FN sample IDs. They contain no text. See [text_additional_diagnostics.json](../training/reports/text_additional_diagnostics.json).

## Length and source patterns

Length bins are very short (1–19 words), short (20–99), medium (100–299), and long (300+). The MAGE bins show:

| MAGE length | n | SVM AI recall | DistilBERT AI recall | SVM human recall | DistilBERT human recall |
|---|---:|---:|---:|---:|---:|
| Very short | 7 | 0.000* | 0.000* | not estimable* | not estimable* |
| Short | 127 | 0.3465 | 0.4257 | 0.9615 | 1.0000 |
| Medium | 998 | 0.4311 | 0.6594 | 0.8612 | 0.7980 |
| Long | 430 | 0.3967 | 0.5272 | 0.9593 | 0.9675 |

`*` The very-short MAGE group contains seven AI examples and no human examples. Both models missed all seven; accuracy and macro-F1 are not reported because this is a single-class slice. GRADTEX AI-involvement recall also stays low across its length bins: approximately 0.137–0.204 for SVM and 0.062–0.122 for DistilBERT on Test C. On Beemo's positive-only probe, AI recall rises from 0/31 very-short examples to 0.355/0.398 on 300+ word examples (SVM/DistilBERT); this is not two-class accuracy.

Performance also varies by source/domain. For MAGE, AI recall ranges 0.18–0.565 (SVM) and 0.235–0.845 (DistilBERT) across IMDb, CNN, DialogSum, and PubMed. On GRADTEX Test C it ranges 0.087–0.354 and 0.029–0.253 across domains. This is evidence of association with domain/source in these samples, not proof that source cues cause each error. The human-edited Beemo probe has only AI-origin samples.

The API rejects inputs shorter than 40 characters or 5 words and longer than 50,000 characters. The direct model-length analysis includes very-short external rows and separately reports whether the API would accept them. It reports three Beemo rows below the API minimum. For accepted long inputs, the fixed DistilBERT service now pads the final partial overflow window; all windows are included and overlap is attention-weighted. The API's 50,000-character cap remains explicit. Training examples were truncated to 256 tokens during fine-tuning, a possible training limitation rather than evidence that inference silently ignores document tails.

## Score and threshold handling

- **SVM:** the persisted pipeline is `CalibratedClassifierCV` over LinearSVC. `predict_proba` gives calibration-model probabilities, but calibration remains domain-dependent. The diagnostic also records the mean AI-oriented raw LinearSVC margin across calibration folds; this margin is not a probability.
- **DistilBERT:** output scores are raw softmax and uncalibrated. No temperature scaling is applied.
- **Thresholds:** the existing SVM thresholds remain 0.41430 (human-side maximum AI score) and 0.77327 (AI-side minimum). DistilBERT thresholds remain 0.65304 and 0.99830. They were not changed or refit.
- The independent dataset summaries above use 0.5 binary decisions; each per-example record separately stores the unchanged three-way validation-threshold result, including `UNCERTAIN`.

## Specific failure and highest-priority improvement

No AI-generated text example was supplied with the image uploads, so the reported AI→HUMAN error is not reproduced. The existing external comparisons demonstrate weak AI recall, especially for held-out transformations and edited AI text. The highest-priority improvement is a licensed, source-aware corpus that includes several unseen LLM families, multiple domains, human-edited AI, human grammar correction, and paraphrase, with prompt/document groups held out by source. Train only on training groups, calibrate/select on validation, and keep generator/source holdouts untouched for final evaluation. Do not select thresholds from MAGE, GRADTEX Test B/C, or Beemo.

The diagnostic found a confirmed long-input error: tokenizer overflow windows with a shorter final chunk could not form a PyTorch tensor without padding. Inference now requests padding; attention masks ensure padding tokens do not affect the aggregation weights. A regression test covers multi-window input with a partial last window. This changes no model weights or thresholds.
