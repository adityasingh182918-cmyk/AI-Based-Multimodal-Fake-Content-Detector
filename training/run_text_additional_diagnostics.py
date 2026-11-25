"""Run both frozen text detectors over the same local evaluation examples.

No training, threshold selection, or raw text persistence is performed. The
per-example report stores predictions, source metadata and lengths only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from training.evaluate_text_additional_robustness import (  # noqa: E402
    load_beemo_expert_edits, load_gradtex_paired_scenario, load_gradtex_test_c,
)
from training.evaluate_text_robustness import read_rows as read_mage_rows  # noqa: E402
from training.text_baseline import load_model  # noqa: E402

AI = "AI_GENERATED"
HUMAN = "HUMAN_WRITTEN"


def _threshold_label(ai_score: float, thresholds: dict[str, Any]) -> str:
    if ai_score <= float(thresholds["real_max_ai_score"]):
        return HUMAN
    if ai_score >= float(thresholds["ai_min_ai_score"]):
        return AI
    return "UNCERTAIN"


def _length_bin(words: int) -> str:
    if words < 20:
        return "very_short_1_to_19_words"
    if words < 100:
        return "short_20_to_99_words"
    if words < 300:
        return "medium_100_to_299_words"
    return "long_300_plus_words"


def _load_datasets(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    if args.datasets in {"all", "mage"}:
        rows = read_mage_rows(args.mage_file)
        datasets["mage_gpt4_ood"] = [{"text": row["text"], "label": AI if row["label"] == 0 else HUMAN,
                                      "source": row["src"], "domain": row["src"].split("_", 1)[0],
                                      "scenario": "full_gpt4_ood"} for row in rows]
    if args.datasets in {"all", "gradtex"}:
        rows = load_gradtex_test_c(args.gradtex_file)
        datasets["gradtex_test_c"] = [{**row, "label": AI if row["label"] == 0 else HUMAN} for row in rows]
        paired = [row for scenario in ("generate_topic", "paraphrase", "polish_token")
                  for row in load_gradtex_paired_scenario(args.gradtex_file, scenario)]
        datasets["gradtex_test_b_selected"] = [{**row, "label": AI if row["label"] == 0 else HUMAN} for row in paired]
    if args.datasets in {"all", "beemo"}:
        rows = load_beemo_expert_edits(args.beemo_file)
        datasets["beemo_expert_edited_ai_positive_only"] = [{**row, "label": AI} for row in rows]
    return datasets


def _baseline_predictions(rows: list[dict[str, Any]], checkpoint: Path) -> list[dict[str, Any]]:
    from app.services.text_detector import split_text

    model = load_model(checkpoint)
    classes = list(model.classes_)
    ai_index = classes.index(AI)
    human_index = classes.index(HUMAN)
    sidecar = checkpoint.with_suffix(".json")
    thresholds = json.loads(sidecar.read_text(encoding="utf-8"))["decision_thresholds"] if sidecar.is_file() else {
        "real_max_ai_score": 0.5, "ai_min_ai_score": 0.5}
    result = []
    for i, row in enumerate(rows, 1):
        started = time.perf_counter()
        text = row["text"]
        chunks = split_text(text)
        weights = np.asarray([max(1, len(chunk.split())) for chunk in chunks], dtype=float)
        probs = model.predict_proba(chunks)
        ai_score = float(np.average(probs[:, ai_index], weights=weights))
        feature_matrix = model.named_steps["features"].transform(chunks)
        calibrated = model.named_steps["classifier"]
        margins = np.asarray([fold.estimator.decision_function(feature_matrix)
                              for fold in calibrated.calibrated_classifiers_])
        decision_values = margins.mean(axis=0)
        oriented_margin = (decision_values if classes[1] == AI else -decision_values)
        margin = float(np.average(oriented_margin, weights=weights))
        pred = AI if ai_score >= 0.5 else HUMAN
        words = len(text.split())
        result.append({"sample_id": f"sample_{i:05d}", "expected_class": row["label"],
                       "predicted_class_at_0_5": pred, "validation_threshold_label": _threshold_label(ai_score, thresholds),
                       "class_scores": {HUMAN: 1.0 - ai_score, AI: ai_score},
                       "score_type": "CalibratedClassifierCV probability (domain-dependent calibration)",
                       "decision_score": margin, "decision_score_type": "mean AI-oriented raw LinearSVC margin across calibration folds",
                       "word_count": words, "character_count": len(text), "length_bin": _length_bin(words),
                       "chunk_count": len(chunks), "api_accepted": len(text) >= 40 and words >= 5 and len(text) <= 50_000,
                       "source": row.get("source"), "domain": row.get("domain"), "scenario": row.get("scenario"),
                       "inference_time_ms": (time.perf_counter() - started) * 1000.0})
        if i % 1000 == 0:
            print(f"SVM scored {i}/{len(rows)}", flush=True)
    return result


def _transformer_predictions(rows: list[dict[str, Any]], checkpoint: Path, device: str) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    metadata = json.loads((checkpoint / "text_detector_metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    max_length = int(metadata["max_length"])
    temperature = float(metadata.get("inference_temperature", 1.0))
    thresholds = metadata["decision_thresholds"]
    results = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        text = row["text"]
        encoded = tokenizer(text, truncation=True, max_length=max_length, stride=32,
                             return_overflowing_tokens=True, return_tensors="pt", padding=True)
        count = int(encoded["input_ids"].shape[0])
        weights = encoded["attention_mask"].sum(dim=1).float()
        batches = []
        for start in range(0, count, 8):
            batch = {key: value[start:start + 8].to(device) for key, value in encoded.items()
                     if key in {"input_ids", "attention_mask"}}
            with torch.inference_mode():
                logits = model(**batch).logits.float() / temperature
            batches.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        window_scores = torch.cat(batches)
        ai_score = float((window_scores * weights).sum().item() / weights.sum().item())
        words = len(text.split())
        results.append({"sample_id": f"sample_{index:05d}", "expected_class": row["label"],
                        "predicted_class_at_0_5": AI if ai_score >= 0.5 else HUMAN,
                        "validation_threshold_label": _threshold_label(ai_score, thresholds),
                        "class_scores": {HUMAN: 1.0 - ai_score, AI: ai_score},
                        "score_type": ("validation-temperature-scaled softmax" if temperature != 1.0 else
                                       "uncalibrated softmax (temperature not deployed)"),
                        "decision_score": None, "decision_score_type": None,
                        "word_count": words, "character_count": len(text), "length_bin": _length_bin(words),
                        "chunk_count": count, "overlapping_window_token_weight_sum": int(weights.sum().item()),
                        "api_accepted": len(text) >= 40 and words >= 5 and len(text) <= 50_000,
                        "source": row.get("source"), "domain": row.get("domain"), "scenario": row.get("scenario"),
                        "inference_time_ms": (time.perf_counter() - started) * 1000.0})
        if index % 500 == 0:
            print(f"DistilBERT scored {index}/{len(rows)}", flush=True)
    del model, tokenizer
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.empty_cache()
    return results


def _summary(rows: list[dict[str, Any]], metrics_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    rows = rows if metrics_rows is None else metrics_rows
    y = [row["expected_class"] for row in rows]
    pred = [row["predicted_class_at_0_5"] for row in rows]
    labels = [HUMAN, AI]
    both = set(y) == {HUMAN, AI}
    fp_ids = [r["sample_id"] for r in rows if r["expected_class"] == HUMAN and r["predicted_class_at_0_5"] == AI]
    fn_ids = [r["sample_id"] for r in rows if r["expected_class"] == AI and r["predicted_class_at_0_5"] == HUMAN]
    matrix = confusion_matrix(y, pred, labels=labels).tolist()
    support = Counter(y)
    recalls = {label: (recall_score(y, pred, labels=[label], average=None, zero_division=0)[0]
                       if support[label] else None) for label in labels}
    threshold_counts = Counter(row["validation_threshold_label"] for row in rows)
    return {"sample_count": len(rows), "class_support": dict(support),
            "accuracy": float(accuracy_score(y, pred)) if both else None,
            "macro_f1": float(f1_score(y, pred, average="macro")) if both else None,
            "class_recall": recalls,
            "false_positive_count_human_as_ai": len(fp_ids), "false_negative_count_ai_as_human": len(fn_ids),
            "false_positive_ids": fp_ids, "false_negative_ids": fn_ids,
            "confusion_matrix_labels": labels, "confusion_matrix_rows_true_columns_predicted": matrix,
            "fpr_human_as_ai": len(fp_ids) / support[HUMAN] if support[HUMAN] else None,
            "fnr_ai_as_human": len(fn_ids) / support[AI] if support[AI] else None,
            "validation_threshold_decision_counts_including_uncertain": dict(threshold_counts),
            "api_rejected_count": sum(not r["api_accepted"] for r in rows)}


def _grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown")].append(row)
    return {name: _summary(values) for name, values in sorted(groups.items())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    datasets = _load_datasets(args)
    report: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Frozen checkpoint diagnostic only. No training, tuning, or threshold changes.",
        "length_bins": {"very_short_1_to_19_words": "1-19 words", "short_20_to_99_words": "20-99 words",
                        "medium_100_to_299_words": "100-299 words", "long_300_plus_words": "300+ words"},
        "text_backend_default_when_unset": "baseline HC3 TF-IDF Linear SVM",
        "models": {}, "datasets": {},
    }
    for dataset_name, samples in datasets.items():
        print(f"Scoring {dataset_name}: {len(samples)} samples", flush=True)
        svm = _baseline_predictions(samples, args.baseline_checkpoint)
        distilbert = _transformer_predictions(samples, args.transformer_checkpoint, args.device)
        single_class = len({r["label"] for r in samples}) == 1
        report["datasets"][dataset_name] = {
            "sample_count": len(samples),
            "class_support": dict(Counter(r["label"] for r in samples)),
            "evaluation_kind": "positive-only sensitivity probe" if single_class else "two-class evaluation",
            "not_used_for": ["training", "model selection", "calibration"],
            "models": {"tfidf_linear_svm": _summary(svm), "distilbert": _summary(distilbert)},
            "by_domain": {"tfidf_linear_svm": _grouped(svm, "domain"), "distilbert": _grouped(distilbert, "domain")},
            "by_source": {"tfidf_linear_svm": _grouped(svm, "source"), "distilbert": _grouped(distilbert, "source")},
            "by_scenario": {"tfidf_linear_svm": _grouped(svm, "scenario"), "distilbert": _grouped(distilbert, "scenario")},
            "by_length": {"tfidf_linear_svm": _grouped(svm, "length_bin"), "distilbert": _grouped(distilbert, "length_bin")},
            "predictions": {"tfidf_linear_svm": svm, "distilbert": distilbert},
        }
    svm_threshold_path = args.baseline_checkpoint.with_suffix(".json")
    svm_thresholds = json.loads(svm_threshold_path.read_text(encoding="utf-8"))["decision_thresholds"] if svm_threshold_path.is_file() else None
    transformer_metadata = json.loads((args.transformer_checkpoint / "text_detector_metadata.json").read_text(encoding="utf-8"))
    temperature = float(transformer_metadata.get("inference_temperature", 1.0))
    transformer_score_semantics = ("Validation-temperature-scaled softmax; calibration quality may shift by domain."
                                   if temperature != 1.0 else "Raw uncalibrated softmax.")
    report["models"] = {"tfidf_linear_svm": {"checkpoint": str(args.baseline_checkpoint),
                                               "score_semantics": "CalibratedClassifierCV outputs probabilities; calibration quality is domain-dependent.",
                                               "decision_thresholds_unchanged": svm_thresholds},
                         "distilbert": {"checkpoint": str(args.transformer_checkpoint),
                                        "score_semantics": transformer_score_semantics,
                                        "inference_temperature": temperature,
                                        "temperature_fit_split": transformer_metadata.get("temperature_calibration", {}).get("fit_split"),
                                        "max_length_tokens": transformer_metadata["max_length"],
                                        "stride_tokens": 32,
                                        "decision_thresholds_unchanged": transformer_metadata["decision_thresholds"]}}
    report["text_persisted"] = False
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=("all", "mage", "gradtex", "beemo"), default="all")
    parser.add_argument("--mage-file", type=Path, default=Path("data/text/robustness/mage/test_ood_set_gpt.csv"))
    parser.add_argument("--gradtex-file", type=Path, default=Path("data/text/robustness/gradtex/test.parquet"))
    parser.add_argument("--beemo-file", type=Path, default=Path("data/text/robustness/beemo/data/train-00000-of-00001.parquet"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib"))
    parser.add_argument("--transformer-checkpoint", type=Path, default=Path("models/text_detector/transformer/distilbert_hc3_v1"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("training/reports/text_additional_diagnostics.json"))
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {model: {"count": summary["sample_count"], "accuracy": summary["accuracy"],
                                    "macro_f1": summary["macro_f1"], "FPR": summary["fpr_human_as_ai"],
                                    "FNR": summary["fnr_ai_as_human"]}
                             for model, summary in data["models"].items()}
                      for name, data in result["datasets"].items()}, indent=2))
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
