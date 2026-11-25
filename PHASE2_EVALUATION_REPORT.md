# Phase 2 Evaluation Report

No test-set evaluation or threshold tuning was run during the 2026-11-25 targeted user-caption investigation. This report points to the saved evaluation artifacts and separates their dataset scopes.

| Evaluation | Scope | Key result |
|---|---|---|
| HC3 original test | In-domain held-out QA | SVM accuracy 0.9680, macro-F1 0.9679; DistilBERT accuracy 0.9435, macro-F1 0.9431 |
| HC3+MAGE expansion test | Mixed held-out split, n=5,727 | Expanded SVM accuracy/macro-F1 0.7503/0.7502; expanded DistilBERT 0.8579/0.8524 |
| MAGE GPT-4 OOD | Frozen two-class external set | Expanded SVM macro-F1 0.6673; expanded DistilBERT 0.7476; DistilBERT human FPR 0.4580 |
| GRADTEX Test C | Human-written vs machine-involved mix | Expanded SVM macro-F1 0.4812; expanded DistilBERT 0.5369 |
| Beemo edited AI probe | AI-origin positives only | Expanded SVM AI recall 0.5534; expanded DistilBERT 0.8354; no accuracy/F1 reported |

Sources: `training/reports/text_evaluation.json`, `training/reports/text_transformer_evaluation.json`, `training/reports/text_model_comparison_expanded_test.json`, and `training/reports/text_expanded_ood_diagnostics.json`. These datasets have different scopes and should not be compared as if they were one test set. The MAGE paraphrase and Beemo probes contain AI-positive samples only and do not support two-class accuracy or macro-F1.

The 2026-11-25 user-caption scores are in [PHASE2_IMPROVEMENT_REPORT.md](PHASE2_IMPROVEMENT_REPORT.md); they are qualitative single-example diagnostics, not evaluation metrics. The user-caption error remains unresolved.
