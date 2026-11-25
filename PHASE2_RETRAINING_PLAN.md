# Phase 2 retraining plan

Updated: 2026-11-25

This continuation expands and retrains Phase 2 only. Existing checkpoints, HC3 files/splits, Phase 1 image detection, and all independent robustness datasets are preserved.

## Stage 0 — audit (COMPLETED)

- Existing baseline: calibrated word/character TF-IDF Linear SVM, `models/text_detector/baseline/hc3_tfidf_linear_svm.joblib`; HC3 validation-derived abstention thresholds live beside the checkpoint.
- Existing transformer: fine-tuned DistilBERT at `models/text_detector/transformer/distilbert_hc3_v1/`; it uses the fixed overflow-window path, dynamic inference padding, attention-mask-weighted aggregation, and local tokenizer files.
- API: `POST /api/v1/predict/text`; environment selects `TEXT_MODEL_BACKEND=baseline|transformer`, and `TEXT_MODEL_CHECKPOINT` selects a checkpoint. Image endpoints are separate and remain untouched.
- Existing HC3 processed counts: 2,476 train, 531 validation, 531 test. Keep all three original JSONL files unchanged.
- Existing test/robustness evidence: HC3 test SVM macro-F1 0.9679 and DistilBERT macro-F1 0.9431; MAGE GPT-4 OOD macro-F1 0.6367/0.7144; GRADTEX Test C macro-F1 0.4652/0.4124; Beemo edited-AI sensitivity 0.3003/0.3476. External datasets are evaluation-only.
- Existing test suite most recently recorded as 51 passing after Phase 1 additions. Re-run it after this implementation.
- Runtime preflight: Python 3.13.17, PyTorch 2.11.0+cu128, CUDA available on NVIDIA RTX 4050 Laptop GPU, scikit-learn 1.9.1, `datasets` 4.8.5, 26.33 GiB free on C:.

## Dataset choice and guardrails

Candidate expansion source: official `yaful/MAGE` Hugging Face dataset at immutable revision `342663f0a2b775455c023f5d36a1341ff0ec5402`. Its card declares Apache-2.0; the authors describe 447,674 English examples from human sources and 27 LLMs, with `label=0` for machine text and `label=1` for human text. Its official `train`, `validation`, and `test` splits are kept separate. The raw train file is about 403.7 MB; validation and test are about 72 MB each. A deterministic streaming sampler will retain capped examples per `src` and label so the training subset remains manageable.

`MAGE test_ood_set_gpt.csv` and `test_ood_set_gpt_para.csv` are separate, previously evaluated OOD artifacts and must never enter training or threshold selection. GRADTEX and Beemo also remain frozen evaluation-only. HC3 `test.jsonl` remains untouched. Exact normalized-text duplicates will be audited across the prepared splits and known held-out artifacts; MAGE has no original document/group ID, so semantic or cross-generator group leakage cannot be ruled out. This limitation must be stated in reports.

The user rejected a noncommercial-only dataset because commercial use may be needed or is uncertain. Do not use CommunityForensics-Small, gated Pangram data, Beemo human-source controls, or sources with unclear terms for training. MAGE's dataset card declares Apache-2.0; preserve attribution and keep derived sample files local/ignored. No third-party generated text will be fabricated.

## Planned stages

| Stage | Status | Work |
|---|---|---|
| 0. Existing project audit | COMPLETED | Inspect checkpoints, split files, reports, service/API, training paths, tests, environment, GPU, and disk. |
| 1. Source selection and acquisition | COMPLETED | Pinned MAGE streaming sampler created and run on all official splits; selected rows retain `src`, label and split provenance. |
| 2. Cleaning and split validation | COMPLETED | Combined new MAGE samples with matching HC3 splits only; 1,809 exact duplicates with frozen evaluation rows were excluded; validator reports no errors or cross-split exact duplicates. |
| 3. Old-model baseline on new validation | COMPLETED | Old SVM and DistilBERT scored on identical 6,650 validation rows. |
| 4. Expanded SVM | COMPLETED | Word/character ablations and SVM C values evaluated on training/validation only; selected `word_char_svm_c05`, saved to the separate versioned checkpoint. |
| 5. Expanded DistilBERT | COMPLETED | Continued fine-tuning from the original DistilBERT checkpoint; best validation macro-F1 at epoch 3, saved in the separate expanded directory. |
| 6. Frozen robustness evaluation | COMPLETED | All four checkpoints compared on the same expanded held-out test; expanded candidates evaluated on frozen MAGE GPT-4 OOD, GRADTEX, and Beemo, without fitting or threshold selection. |
| 7. Calibration investigation | COMPLETED | Validation-only temperature fit; deploy T=2.2867 for expanded DistilBERT after all three validation calibration metrics improved; SVM remains unscaled. Existing score cutoff policy yields only UNCERTAIN, so useful abstention remains unresolved. |
| 8. API integration | COMPLETED | Added versioned backend selection and model identity/score semantics; kept original HC3 SVM as default and Phase 1 independent. API integration and overflow tests pass. |
| 9. Regression tests and documentation | COMPLETED | Backend suite: 58 passed; compile check passed; trackers and README updated. Visual browser/responsive/assistive-technology QA remains unverified. |

## Experiment outcome and remaining limitations (2026-11-25)

- Dataset files: train 26,954 (13,730 AI/13,224 human), validation 6,650 (3,398/3,252), test 5,727 (3,158/2,569). Existing HC3 split files were not modified.
- Expanded DistilBERT achieved 0.8579 accuracy and 0.8524 macro-F1 on the combined prepared test, with human false-positive rate 0.2577. Expanded SVM achieved 0.7503/0.7502, with human FPR 0.1810 and AI FNR 0.3056. These are same-source held-out results, not broad generalization claims.
- On MAGE GPT-4 OOD, expanded DistilBERT macro-F1 was 0.7476 (human FPR 0.4580, AI FNR 0.0300); expanded SVM macro-F1 was 0.6673 (FPR 0.2795, FNR 0.3825). GRADTEX and Beemo results and all class/source/length slices are in `training/reports/text_expanded_ood_diagnostics.json`.
- Validation-fitted scalar temperature for expanded DistilBERT is stored in the new checkpoint metadata and used by inference. Both expanded models' 5%-error validation abstention thresholds collapse to 0/1, yielding all-UNCERTAIN outputs. A practical, validated three-way decision policy is not established.
- Phase 2 remains `IN_PROGRESS`: a licensed and representative human grammar-correction/edited-AI dataset is unavailable under current license constraints; MAGE lacks document/group IDs; cross-domain false positives remain high; visual frontend/accessibility QA is pending. Do not call the detector reliable for arbitrary text.

## Non-overwrite policy

New outputs use new paths only. Training must refuse existing nonempty output files/directories unless a new versioned path is chosen. No old checkpoint, HC3 split, or independent evaluation file may be modified.
