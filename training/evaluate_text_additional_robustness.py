"""Evaluate frozen Phase 2 checkpoints on GRADTEX Test C and Beemo expert edits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .evaluate_text_robustness import baseline_scores, score_report, transformer_scores
except ImportError:
    from evaluate_text_robustness import baseline_scores, score_report, transformer_scores


def _read_parquet(path: Path, columns: list[str]) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow (included with the datasets extra) to read the downloaded Parquet file") from exc
    return pq.read_table(path, columns=columns).to_pylist()


def exact_overlap_audit(rows: list[dict], hc3_dir: Path = Path("data/text/processed/hc3"),
                        mage_dir: Path = Path("data/text/robustness/mage")) -> dict:
    """Count exact normalized text overlaps without writing corpus text into reports."""
    normalize = lambda value: " ".join((value or "").split()).casefold()
    eval_texts = {normalize(row["text"]) for row in rows if normalize(row.get("text"))}
    corpora: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        path = hc3_dir / f"{split}.jsonl"
        if path.is_file():
            corpus = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    corpus.add(normalize(json.loads(line).get("text", "")))
                except json.JSONDecodeError:
                    continue
            corpora[f"hc3_{split}"] = corpus
    for filename in ("test_ood_set_gpt.csv", "test_ood_set_gpt_para.csv"):
        path = mage_dir / filename
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                corpora[f"mage_{filename}"] = {normalize(row.get("text", "")) for row in csv.DictReader(stream)}
    return {"normalization": "casefold + whitespace collapse", "evaluation_unique_texts": len(eval_texts),
            "overlaps_by_existing_split_or_external_report": {name: len(eval_texts & corpus) for name, corpus in corpora.items()},
            "known_corpora_compared": sorted(corpora)}


def load_gradtex_test_c(path: Path) -> list[dict]:
    """Load only the official held-out unseen-scenario split C."""
    rows = _read_parquet(path, ["text", "binary_label", "multiclass_label", "domain", "scenario",
                                "scenario_family", "generator_model", "model_family", "test_split", "evaluation_setup"])
    selected = []
    for row in rows:
        if row["test_split"] != "C":
            continue
        if row["evaluation_setup"] != "unseen_scenarios_evaluation":
            raise ValueError("GRADTEX split C does not match its documented unseen-scenario setup")
        text = row["text"]
        label = int(row["binary_label"])
        expected = {"HWT": 1, "MIX": 0}
        if row["multiclass_label"] not in expected or expected[row["multiclass_label"]] != label:
            raise ValueError("Unexpected GRADTEX Test C label mapping; refusing to score")
        if not isinstance(text, str) or not text.strip():
            continue
        selected.append({"text": text, "label": 0 if label == 0 else 1,
                         "domain": row["domain"], "source": row["model_family"] or "human",
                         "scenario": row["scenario"], "label_name": row["multiclass_label"]})
    if not selected or {row["label"] for row in selected} != {0, 1}:
        raise ValueError("GRADTEX Test C must contain both machine-involved MIX and human HWT samples")
    return selected


def load_gradtex_paired_scenario(path: Path, scenario: str) -> list[dict]:
    """Pair held-out source human text with a selected Test B generated/edited variant."""
    rows = _read_parquet(path, ["text", "human_source_text", "binary_label", "multiclass_label", "domain",
                                "scenario", "scenario_family", "generator_model", "model_family", "test_split",
                                "evaluation_setup"])
    generated = [row for row in rows if row["test_split"] == "B" and row["scenario"] == scenario]
    if not generated:
        raise ValueError(f"No GRADTEX Test B rows found for scenario {scenario!r}")
    selected = []
    expected_label = "MGT" if scenario == "generate_topic" else "MIX"
    for row in generated:
        if row["evaluation_setup"] != "in_domain_evaluation" or row["multiclass_label"] != expected_label:
            raise ValueError("Unexpected GRADTEX Test B scenario or label mapping; refusing to score")
        original, transformed = row["human_source_text"], row["text"]
        if not isinstance(original, str) or not original.strip() or not isinstance(transformed, str) or not transformed.strip():
            continue
        normalized = " ".join(original.split()).casefold()
        pair_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        shared = {"domain": row["domain"], "scenario": scenario,
                  "pair_id": pair_id}
        selected.append({"text": original, "label": 1, "source": "human_source_control", "label_name": "HWT", **shared})
        selected.append({"text": transformed, "label": 0, "source": row["model_family"] or "unknown_model",
                         "label_name": expected_label, **shared})
    if not selected:
        raise ValueError(f"No usable paired GRADTEX Test B rows for {scenario!r}")
    return selected


LICENSED_BEEMO_MODELS = {
    "HuggingFaceH4/zephyr-7b-beta",  # MIT
    "mistralai/Mistral-7B-Instruct-v0.1",  # Apache-2.0
    "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Apache-2.0
}


def load_beemo_expert_edits(path: Path) -> list[dict]:
    """Select only human expert-edited AI outputs from models with permissive terms.

    The No Robots human-output field and LLM-edited columns are deliberately not loaded.
    This is positive-only AI-origin sensitivity evidence; it has no human negatives.
    """
    rows = _read_parquet(path, ["id", "category", "model", "model_output", "human_edits"])
    selected = []
    for row in rows:
        if row["model"] not in LICENSED_BEEMO_MODELS:
            continue
        text = row["human_edits"]
        if not isinstance(text, str) or not text.strip():
            continue
        selected.append({"text": text, "label": 0, "source": row["model"], "domain": row["category"],
                         "scenario": "expert_human_edit", "original_ai_text": row["model_output"]})
    if not selected:
        raise ValueError("No eligible Beemo human-edited samples found")
    return selected


def _evaluate_slice(rows: list[dict], report: dict, device: str, batch_size: int) -> None:
    report["row_count"] = len(rows)
    if any("pair_id" in row for row in rows):
        report["unique_source_pairs"] = len({row["pair_id"] for row in rows if "pair_id" in row})
    report["class_counts"] = dict(Counter(row["label_name"] if "label_name" in row else "AI_ORIGIN_HUMAN_EDIT" for row in rows))
    group_fields = {"by_domain": "domain", "by_source": "source", "by_scenario": "scenario"}
    report.update({key: {} for key in (*group_fields, "by_length")})
    word_counts = [len(row["text"].split()) for row in rows]
    length_bins = ["under_100_words" if n < 100 else "100_to_299_words" if n < 300 else "300_or_more_words" for n in word_counts]
    report["length_counts"] = dict(Counter(length_bins))
    report["models"] = {}

    baseline_path = Path(report["baseline_checkpoint"])
    transformer_path = Path(report["transformer_checkpoint"])
    for model_name, scorer in (("tfidf_linear_svm", lambda: baseline_scores(baseline_path, rows)),
                               ("distilbert", lambda: transformer_scores(transformer_path, rows, device, batch_size))):
        try:
            probs, thresholds = scorer()
        except (FileNotFoundError, OSError):
            continue
        report["models"][model_name] = score_report([row["label"] for row in rows], probs, thresholds)
        for key, field in group_fields.items():
            for value in sorted({str(row[field]) for row in rows}):
                indices = [i for i, row in enumerate(rows) if str(row[field]) == value]
                report[key][value] = report[key].get(value, {})
                report[key][value][model_name] = score_report([rows[i]["label"] for i in indices], probs[indices])
        for value in sorted(set(length_bins)):
            indices = [i for i, group in enumerate(length_bins) if group == value]
            report["by_length"].setdefault(value, {})[model_name] = score_report(
                [rows[i]["label"] for i in indices], probs[indices])
    if not report["models"]:
        raise SystemExit("No frozen text checkpoints were found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gradtex", "gradtex_b", "beemo"), required=True)
    parser.add_argument("--gradtex-file", type=Path, default=Path("data/text/robustness/gradtex/test.parquet"))
    parser.add_argument("--beemo-file", type=Path, default=Path("data/text/robustness/beemo/data/train-00000-of-00001.parquet"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib"))
    parser.add_argument("--transformer-checkpoint", type=Path, default=Path("models/text_detector/transformer/distilbert_hc3_v1"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.dataset == "gradtex":
        rows = load_gradtex_test_c(args.gradtex_file)
        report = {"dataset": "GRADTEX", "revision": "553d859da0255d75a39c385c208f7522a2007f53",
                  "split": "test_split C / unseen_scenarios_evaluation",
                  "interpretation": "two-class machine-involved MIX vs purely human HWT; MIX is AI-assisted/partially rewritten, not fully AI-authored",
                  "not_used_for": ["training", "model selection", "threshold calibration"]}
        output = Path("training/reports/text_gradtex_test_c_evaluation.json")
    elif args.dataset == "gradtex_b":
        scenarios = ("generate_topic", "paraphrase", "polish_token")
        rows = [row for scenario in scenarios for row in load_gradtex_paired_scenario(args.gradtex_file, scenario)]
        report = {"dataset": "GRADTEX", "revision": "553d859da0255d75a39c385c208f7522a2007f53",
                  "split": "test_split B / paired human_source_text controls",
                  "interpretation": "binary AI involvement: HWT source vs MIX AI-polished/paraphrased or MGT fully generated; not a pure authorship claim",
                  "scenario_note": "Test B scenarios are in-domain within GRADTEX, but entirely external to HC3 model training. Scenario-specific paired sources are used where available.",
                  "not_used_for": ["training", "model selection", "threshold calibration"]}
        output = Path("training/reports/text_gradtex_test_b_selected_evaluation.json")
    else:
        rows = load_beemo_expert_edits(args.beemo_file)
        report = {"dataset": "Beemo", "revision": "9c014107fe9b85c4c784c1ce3a43b0b7b0a6d162",
                  "split": "full expert-edited column from eligible model sources; positive-only sensitivity probe",
                  "interpretation": "human expert edits of AI-origin outputs; no human negative examples; two-class accuracy/F1/FPR/AUC are not applicable",
                  "eligible_models": sorted(LICENSED_BEEMO_MODELS),
                  "not_used_for": ["training", "model selection", "threshold calibration"]}
        output = Path("training/reports/text_beemo_expert_edit_sensitivity.json")
    report["exact_overlap_audit"] = exact_overlap_audit(rows)
    report["baseline_checkpoint"] = str(args.baseline_checkpoint)
    report["transformer_checkpoint"] = str(args.transformer_checkpoint)
    _evaluate_slice(rows, report, args.device, args.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": report["dataset"], "row_count": report["row_count"], "class_counts": report["class_counts"], "models": report["models"]}, indent=2))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
