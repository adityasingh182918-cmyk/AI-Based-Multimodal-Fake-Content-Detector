"""Evaluate frozen text checkpoints on the external MAGE GPT-4 OOD test file."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)

try:
    from .text_baseline import load_model
except ImportError:
    from text_baseline import load_model


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"text", "label", "src"}.issubset(rows[0]):
        raise ValueError("MAGE file must contain non-empty text, label, and src columns")
    valid = []
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1) or not row["text"].strip():
            continue
        valid.append({"text": row["text"], "label": label, "src": row["src"]})
    if not valid or {row["label"] for row in valid} != {0, 1}:
        raise ValueError("MAGE OOD file must contain both labels")
    return valid


def score_report(y_true, probs, thresholds=None) -> dict:
    y = np.asarray(y_true, dtype=int)
    y_ai = (y == 0).astype(int)  # MAGE's documented mapping is 0=machine, 1=human.
    ai_prob = np.asarray(probs, dtype=float)
    y_pred = np.where(ai_prob >= .5, 0, 1)
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, labels=[0, 1], zero_division=0)
    both_classes = set(np.unique(y)) == {0, 1}
    tp = int(((y == 0) & (y_pred == 0)).sum())
    fn = int(((y == 0) & (y_pred == 1)).sum())
    fp = int(((y == 1) & (y_pred == 0)).sum())
    tn = int(((y == 1) & (y_pred == 1)).sum())
    report = {"sample_count": int(len(y)), "class_support": {"AI_GENERATED": int((y == 0).sum()), "HUMAN_WRITTEN": int((y == 1).sum())},
              "accuracy": float(accuracy_score(y, y_pred)) if both_classes else None,
              "macro_f1": float(f1_score(y, y_pred, average="macro")) if both_classes else None,
              "weighted_f1": float(f1_score(y, y_pred, average="weighted")) if both_classes else None,
              "ai_generated_precision": float(precision[0]) if both_classes else None,
              "ai_generated_recall": float(recall[0]) if (y == 0).any() else None,
              "human_written_precision": float(precision[1]) if both_classes else None,
              "human_written_recall": float(recall[1]) if (y == 1).any() else None,
              "false_positive_rate_human_as_ai": float(fp / (fp + tn)) if fp + tn else None,
              "false_negative_rate_ai_as_human": float(fn / (fn + tp)) if fn + tp else None,
              "confusion_matrix_labels": ["AI_GENERATED", "HUMAN_WRITTEN"],
              "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
              "roc_auc_ai_generated": float(roc_auc_score(y_ai, ai_prob)) if both_classes else None,
              "average_precision_ai_generated": float(average_precision_score(y_ai, ai_prob)) if both_classes else None,
              "brier_score_ai_generated": float(brier_score_loss(y_ai, ai_prob)) if both_classes else None}
    if thresholds:
        low, high = float(thresholds["real_max_ai_score"]), float(thresholds["ai_min_ai_score"])
        decisions = np.where(ai_prob <= low, 1, np.where(ai_prob >= high, 0, -1))
        covered = decisions >= 0
        report["validation_thresholds"] = thresholds
        report["abstention"] = {"coverage": float(covered.mean()), "uncertain_count": int((~covered).sum())}
        if covered.any():
            if both_classes:
                report["abstention"]["accuracy_on_covered"] = float((decisions[covered] == y[covered]).mean())
            else:
                report["abstention"]["ai_recall_on_covered"] = float((decisions[covered] == 0).mean())
    return report


def baseline_scores(checkpoint: Path, rows):
    model = load_model(checkpoint)
    probs = model.predict_proba([r["text"] for r in rows])[:, list(model.classes_).index("AI_GENERATED")]
    sidecar = checkpoint.with_suffix(".json")
    thresholds = json.loads(sidecar.read_text(encoding="utf-8"))["decision_thresholds"] if sidecar.is_file() else None
    return probs, thresholds


def transformer_scores(checkpoint: Path, rows, device: str, batch_size: int):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    metadata = json.loads((checkpoint / "text_detector_metadata.json").read_text(encoding="utf-8"))
    temperature = float(metadata.get("inference_temperature", 1.0))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = tokenizer([r["text"] for r in rows[start:start + batch_size]], padding=True,
                              truncation=True, max_length=metadata["max_length"], return_tensors="pt").to(device)
            outputs.extend(torch.softmax(model(**batch).logits.float() / temperature, dim=1)[:, 1].cpu().numpy())
    return np.asarray(outputs), metadata["decision_thresholds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/text/robustness/mage/test_ood_set_gpt.csv"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib"))
    parser.add_argument("--transformer-checkpoint", type=Path, default=Path("models/text_detector/transformer/distilbert_hc3_v1"))
    parser.add_argument("--report", type=Path, default=Path("training/reports/text_mage_ood_evaluation.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-per-source", type=int, default=0,
                        help="0 evaluates the full file; otherwise deterministic cap for quick checks")
    parser.add_argument("--paraphrased-ai-only", action="store_true",
                        help="evaluate only MAGE sources ending in _gpt4_para; report single-class AI sensitivity")
    args = parser.parse_args()
    rows = read_rows(args.dataset)
    if args.paraphrased_ai_only:
        rows = [row for row in rows if row["src"].endswith("_gpt4_para") and row["label"] == 0]
        if not rows:
            raise SystemExit("No MAGE GPT-4 paraphrased AI samples found")
    if args.max_per_source:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["src"]].append(row)
        rows = [row for src in sorted(grouped) for row in grouped[src][:args.max_per_source]]
        if {row["label"] for row in rows} != {0, 1}:
            raise SystemExit("Capped subset must retain both MAGE labels; increase --max-per-source")
    domain_by_row = [row["src"].split("_", 1)[0] if "_" in row["src"] else row["src"] for row in rows]
    length_by_row = ["under_100_words" if len(row["text"].split()) < 100 else
                     "100_to_299_words" if len(row["text"].split()) < 300 else "300_or_more_words" for row in rows]
    report = {"dataset": str(args.dataset), "purpose": "external robustness evaluation only; not used for training, selection, or calibration",
              "evaluation_slice": "GPT-4 paraphrased AI only; sensitivity-only, no human negatives" if args.paraphrased_ai_only else "full MAGE OOD file",
              "row_count": len(rows), "source_counts": dict(Counter(row["src"] for row in rows)),
              "domain_counts": dict(Counter(domain_by_row)), "length_counts": dict(Counter(length_by_row)),
              "models": {}, "by_domain": {}, "by_source": {}, "by_length": {}}
    if args.baseline_checkpoint.is_file():
        probs, thresholds = baseline_scores(args.baseline_checkpoint, rows)
        report["models"]["tfidf_linear_svm"] = {"checkpoint": str(args.baseline_checkpoint), **score_report([r["label"] for r in rows], probs, thresholds)}
        for group_name, group_values in (("by_domain", domain_by_row), ("by_source", [r["src"] for r in rows]),
                                         ("by_length", length_by_row)):
            for group in sorted(set(group_values)):
                indices = [i for i, value in enumerate(group_values) if value == group]
                report[group_name].setdefault(group, {})["tfidf_linear_svm"] = score_report(
                    [rows[i]["label"] for i in indices], probs[indices])
    if args.transformer_checkpoint.is_dir():
        probs, thresholds = transformer_scores(args.transformer_checkpoint, rows, args.device, args.batch_size)
        report["models"]["distilbert"] = {"checkpoint": str(args.transformer_checkpoint), **score_report([r["label"] for r in rows], probs, thresholds)}
        for group_name, group_values in (("by_domain", domain_by_row), ("by_source", [r["src"] for r in rows]),
                                         ("by_length", length_by_row)):
            for group in sorted(set(group_values)):
                indices = [i for i, value in enumerate(group_values) if value == group]
                report[group_name].setdefault(group, {})["distilbert"] = score_report(
                    [rows[i]["label"] for i in indices], probs[indices])
    if not report["models"]:
        raise SystemExit("No local text checkpoints were found")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["models"], indent=2))
    print(f"Evaluation report written to {args.report}")


if __name__ == "__main__":
    main()
