# Phase 2 text dataset

## Selected source

The first experiment uses the public English [HC3 dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3), pinned to revision `4d0ff18143b5a7e1b1e79beb540c04549d1e59d3`. Its records contain an `id`, `question`, `human_answers`, `chatgpt_answers`, and `source`. A human answer is mapped to `HUMAN_WRITTEN`; a `chatgpt_answers` entry is mapped to `AI_GENERATED`. The labels express dataset provenance, not a general truth about authorship.

Only `reddit_eli5` and `wiki_csai` are selected. The HC3 authors list ELI5 as BSD licensed and Wikipedia as CC BY-SA; they warn that source licenses stricter than CC BY-SA take precedence. `open_qa` is listed with a custom/unclear source license, and medicine and finance are marked unknown, so those files are excluded. The Wikipedia-derived material requires attribution and share-alike compliance. Review the source terms before redistributing processed examples or a derived dataset. The project stores data under ignored `data/` and does not track it in source control.

The two selected raw files are approximately 57.6 MB combined. At preparation time the script deterministically selects up to 1,000 question groups per source, pairs equal counts of human and generated answers per question, normalizes only whitespace/NULs, rejects short and exact duplicate answers, then creates group-held-out 70/15/15 train/validation/test splits. The test set is not used for baseline selection. Resulting counts and rejection totals are recorded in `data/text/processed/hc3/metadata.json`.

## Caveats

HC3 is mainly English question-answer text and ChatGPT-era generation. It does not represent all genres, current model families, paraphrased or edited generation, or human writing. Question-answer formatting, source, and style can become shortcuts. The current test split is held out by question group but shares the two source datasets and generator provenance with training, so it is an in-domain test and not independent cross-generator robustness evidence. Exact duplicate checks are performed; semantic near-duplicate analysis is not implemented.

The held-out source slices are not uniform: baseline Linear SVM macro-F1 was 0.985 on `reddit_eli5` and 0.950 on `wiki_csai`; DistilBERT scored 0.974 and 0.910 respectively. These differences strengthen the warning that source/domain artifacts affect results; they are not evidence of unseen-source robustness.

The optional pretrained base model is [DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased), pinned to revision `12040accade4e8a0f71eabdb258fecc2e7e948be`; its model card declares Apache-2.0. It is a general pretrained encoder, not an AI-text detector until fine-tuned. Transformers 4.57.6 works with the installed Python 3.13/PyTorch CUDA stack; the RTX 4050 has 6 GiB VRAM. Model weights are fetched only by the explicit training command.

## Independent robustness sources reviewed

## Expansion source: MAGE official train/validation/test

The expansion uses [yaful/MAGE](https://huggingface.co/datasets/yaful/MAGE) at immutable revision `342663f0a2b775455c023f5d36a1341ff0ec5402`; the [authors' repository](https://github.com/yafuly/MAGE) describes 447,674 human and machine-generated English examples sourced from ten human-writing datasets and 27 LLMs across seven provider/model families. The dataset card declares Apache-2.0 and reports `0=machine-generated`, `1=human-written`, with `text`, `label`, and `src` fields. This adds broad generator/domain coverage, but not Gemini/Claude specifically, direct human grammar correction, or human-edited AI labels.

The official Hugging Face split files are about 403.7 MB train, 72.3 MB validation, and 71.7 MB test; preflight showed 26.33 GiB free on C:. The project streams each split at the pinned revision and stores a reproducible capped sample only. Each source/label group is sampled by stable SHA-256 rank with seed 42: up to 40 AI and 1,200 human rows per `src` for training, and up to 10 AI / 300 human per `src` for validation and test. The stream reported 319,071 train, 56,792 validation, and 60,743 test rows; the selected raw samples contain 24,480 / 6,120 / 7,002 rows before duplicate filtering. The 1809 exact sample overlaps with HC3 or frozen external evaluation text are removed before merge. Final combined HC3+MAGE split counts are recorded in `data/text/processed/expanded_v1/metadata.json`; HC3 originals remain untouched.

MAGE's schema does not provide an original document/prompt group ID. The preparation tool preserves its official split assignments, source field and labels, removes exact normalized duplicates across splits and frozen evaluation artifacts, and reports the limitation. It cannot guarantee that independently generated versions of a source document never cross the authors' official splits. Underlying original-text source terms and generator output terms have not been independently re-audited for downstream redistribution; MAGE's Apache-2.0 card is the declared dataset license, but only local training/evaluation copies are kept and not redistributed.

The source dataset is a direct-generation dataset; MAGE's separate GPT-4 paraphrase benchmark remains evaluation-only. Human-edited AI and human grammar correction remain evaluation gaps. Beemo and GRADTEX stay frozen; no rows from those evaluations are used in training or threshold tuning.

Reproduce acquisition and preparation:

```powershell
.\.venv\Scripts\python.exe scripts\download_text_expansion.py --output-dir data\text\raw\mage_expansion --seed 42
.\.venv\Scripts\python.exe scripts\prepare_expanded_text_dataset.py --hc3-dir data\text\processed\hc3 --mage-dir data\text\raw\mage_expansion --output-dir data\text\processed\expanded_v1 --seed 42
.\.venv\Scripts\python.exe scripts\validate_text_dataset.py data\text\processed\expanded_v1 --report training\reports\text_expanded_dataset_validation.json
```

The downloader uses `datasets` already included in `requirements-ml.txt`, no authentication, and no large raw file persistence. Existing HC3 MAGE-GPT-4 OOD/paraphrase, GRADTEX, and Beemo held-out artifacts remain evaluation-only.

- **MAGE GPT-4 OOD/paraphrase files** remain held-out robustness evaluation only. The separate official MAGE `train`/`validation`/`test` splits described above supply the expansion sample. The GPT-4 OOD files cover CNN/DailyMail, DialogSum, IMDb, and PubMedQA; the project stores only these small evaluation CSVs. The documented mapping is `0=machine-generated`, `1=human`.
- The MAGE ordinary GPT-4 OOD file has 1,562 usable rows: 800 AI and 762 human across four domains. This project evaluated the already-frozen HC3 checkpoints on it as a separate source. Both models had materially lower metrics than on HC3: SVM macro-F1 0.6367 and AI recall 0.4238; DistilBERT macro-F1 0.7144 and AI recall 0.5963. These are external benchmark results, not deployment guarantees.
- The optional MAGE paraphrase file is not a conventional authorship benchmark: its authors label paraphrased outputs as machine-generated even when the underlying source is human. Therefore, only rows whose `src` ends in `_gpt4_para` are used for the explicitly labeled sensitivity-only probe; it has no human negative examples, so precision, ROC-AUC, and two-class F1 are not reported. On 800 such AI paraphrase examples, SVM AI recall was 0.4713 and DistilBERT AI recall 0.6963 at the ordinary 0.5 decision point. Validation-derived abstention coverage was 71.4% and 48.0%, respectively. Do not interpret this as an authorship test of paraphrased human writing.
- A source-held-out HC3 experiment trained on `reddit_eli5` and evaluated on the untouched `wiki_csai` source yielded SVM macro-F1 0.5327, accuracy 0.6236, and human recall 0.20. This is strong evidence that HC3 in-domain performance was source-sensitive.
- **RAID** is a valuable future benchmark: its paper and repository describe many generators, domains, and attacks, but the official repository documents large files (hundreds of MB to multiple GB) and notes the main test labels are unavailable. It was not downloaded.
- **M4** is a useful multilingual/multi-domain benchmark. Its official repository states that licensing varies with the contributing source data and generated content follows underlying model terms. That component-by-component review is not completed, so it was not downloaded.

## Additional frozen-checkpoint robustness data

- **GRADTEX** is a small, version-pinned addition for held-out human/AI-involvement probes. Its dataset card reports 142,214 rows and 226 MB total across splits; only `test.parquet` (64,081,445 bytes) was downloaded. The pinned revision is `553d859da0255d75a39c385c208f7522a2007f53`; the card declares CC BY 4.0. The benchmark extends MAGE, so this is a new held-out evaluation artifact, but not an independent underlying human-source collection from MAGE.
- GRADTEX Test C (`unseen_scenarios_evaluation`) contains 3,529 rows: 1,763 HWT and 1,766 MIX, from the four test-only scenarios `polish_sentence`, `complete_middle`, `rewrite_style_skeptical`, and `rewrite_style_maximal`. The binary test is interpreted as machine-involved MIX versus purely human HWT; MIX does not mean wholly AI-authored. On this split, SVM accuracy/macro-F1 were 0.5271/0.4652 (AI-involvement recall 0.1869; human false-positive rate 0.1322; AI-involvement false-negative rate 0.8131). DistilBERT was 0.5004/0.4124 (recall 0.1133; FPR 0.1117; FNR 0.8867).
- GRADTEX Test B was also evaluated only for the paired `generate_topic`, `paraphrase`, and `polish_token` scenarios against their `human_source_text` originals. This yielded 2,642 rows (1,321 paired generated/edited items plus their original human controls; the same source controls may recur across models/scenarios). Combined accuracy/macro-F1 were 0.5223/0.4561 for SVM and 0.5151/0.4361 for DistilBERT. This remains machine-involvement evaluation, not an authorship claim. Per-scenario/source/domain/length metrics are in the local JSON report.
- Exact normalized-text overlap audits found zero overlap for the GRADTEX C/B evaluated texts with HC3 train, validation, or test, or with either already-evaluated MAGE CSV. This check is exact after casefolding and collapsing whitespace; it does not rule out semantic near-duplicates.
- **Beemo** is a small human-edited-AI benchmark. The official parquet is 8,345,056 bytes and contains 2,187 prompts/examples. From its expert-human-edit field, this project selected 656 outputs whose original generation model terms are documented as MIT (`zephyr-7b-beta`) or Apache-2.0 (`Mistral-7B-Instruct-v0.1`, `Mixtral-8x7B-Instruct-v0.1`). The data card/repository warns that model output terms vary; the human prompt/source-output field can be CC BY-NC-4.0. That field and the LLM-edit columns were not used. The selected human-edited AI-origin texts are positive-only sensitivity probes: SVM AI recall 0.3003, DistilBERT 0.3476. No accuracy, precision, macro-F1, FPR, ROC-AUC, or Brier score is reported for this one-class slice. Exact overlap with HC3 or previous MAGE evaluation texts was zero.
- **Pangram/EditLens Grammarly** would provide a direct human-text editing/grammar-correction probe: the official card describes 1,768 Grammarly-edited samples from 200 human sources, nine edit instructions, and a ~3.95 MB parquet. The repository is gated; its file request returned HTTP 401 and requires authenticated access/acceptance. Open Pangram's project materials declare CC BY-NC-SA 4.0. No authentication was attempted, no Grammarly data was downloaded, and no result is claimed. This remains the specific gap for actual Grammarly-style edits of human writing.

The additional source cards: [GRADTEX](https://huggingface.co/datasets/elisabeth-pl-pl/GRADTEX), [Beemo authors' repository and license notes](https://github.com/Toloka/beemo), [Pangram Grammarly dataset card](https://huggingface.co/datasets/pangram/editlens_iclr_grammarly), and [Open Pangram terms and dataset description](https://www.pangram.com/blog/introducing-open-pangram).

Reproduce local acquisition, evaluation, and validation-only calibration analysis:

```powershell
python scripts\download_text_robustness_extra.py --dataset all
python training\evaluate_text_additional_robustness.py --dataset gradtex
python training\evaluate_text_additional_robustness.py --dataset gradtex_b
python training\evaluate_text_additional_robustness.py --dataset beemo
python training\analyze_text_calibration.py --device cuda --batch-size 8
```

These commands score frozen checkpoints. The GRADTEX and Beemo files are not used for training, model selection, or calibration.

MAGE links: [official repository](https://github.com/yafuly/MAGE), [dataset card and files](https://huggingface.co/datasets/yaful/MAGE). Other sources reviewed: [RAID paper](https://aclanthology.org/2024.acl-long.674/) and [repository](https://github.com/liamdugan/raid); [M4 paper](https://aclanthology.org/2024.eacl-long.83/) and [repository](https://github.com/mbzuai-nlp/M4).

Reproduce the explicit robustness download and evaluation:

```powershell
python scripts\download_text_robustness.py
python scripts\download_text_robustness.py --include-paraphrase
python training\evaluate_text_robustness.py --dataset data\text\robustness\mage\test_ood_set_gpt.csv --report training\reports\text_mage_ood_evaluation.json
python training\evaluate_text_robustness.py --dataset data\text\robustness\mage\test_ood_set_gpt_para.csv --paraphrased-ai-only --report training\reports\text_mage_paraphrase_ai_evaluation.json
```

The second download command is optional and only needed for the paraphrase probe. External test files are never used in fitting or threshold selection.

## Download

```powershell
python -m pip install -r requirements-ml.txt
python scripts\download_text_datasets.py --output-dir data\text\raw\hc3
python scripts\prepare_text_dataset.py --raw-dir data\text\raw\hc3 --output-dir data\text\processed\hc3 --max-groups-per-source 1000 --seed 42
python scripts\validate_text_dataset.py data\text\processed\hc3 --report training\reports\text_dataset_validation.json
```

Downloading is a separate explicit command. It contacts the public Hub and fetches only the two reviewed files; it does not require a token. Approximate compressed file sizes are exposed by the pinned repository file listing (55.4 MB and 2.2 MB).
