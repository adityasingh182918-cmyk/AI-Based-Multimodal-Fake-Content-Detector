"""Evaluate a fine-tuned transformer on the untouched text test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

try:
    from .train_text_model import TextRows, infer_logits
    from .text_baseline import read_jsonl
except ImportError:
    from train_text_model import TextRows, infer_logits
    from text_baseline import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/text/processed/hc3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/text_detector/transformer/distilbert_hc3_v1"))
    parser.add_argument("--report", type=Path, default=Path("training/reports/text_transformer_evaluation.json"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        raise SystemExit(f"Transformer checkpoint directory does not exist: {args.checkpoint}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    metadata = json.loads((args.checkpoint / "text_detector_metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, local_files_only=True).to(device)
    rows = read_jsonl(args.dataset_dir / "test.jsonl")
    loader = DataLoader(TextRows(rows, tokenizer, metadata["max_length"]), batch_size=args.batch_size,
                        shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if device.type == "cuda" else None))
    logits, labels = infer_logits(model, loader, device)
    temperature = float(metadata.get("inference_temperature", 1.0))
    probs = torch.softmax(torch.tensor(logits) / temperature, dim=1).numpy()[:, 1]
    predicted = logits.argmax(axis=1)
    label_names = ["HUMAN_WRITTEN", "AI_GENERATED"]
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    m = {"sample_count": len(rows), "accuracy": float(accuracy_score(labels, predicted)),
         "macro_f1": float(f1_score(labels, predicted, average="macro")), "weighted_f1": float(f1_score(labels, predicted, average="weighted")),
         "false_positive_rate_human_as_ai": float(matrix[0, 1] / max(1, matrix[0].sum())),
         "false_negative_rate_ai_as_human": float(matrix[1, 0] / max(1, matrix[1].sum())),
         "confusion_matrix_labels": label_names, "confusion_matrix": matrix.tolist(),
         "classification_report": classification_report(labels, predicted, labels=[0, 1], target_names=label_names, output_dict=True, zero_division=0),
         "roc_auc_ai_generated": float(roc_auc_score(labels, probs)),
         "average_precision_ai_generated": float(average_precision_score(labels, probs)),
         "brier_score_ai_generated": float(brier_score_loss(labels, probs))}
    m["inference_temperature"] = temperature
    m["score_semantics"] = "validation-temperature-scaled softmax" if temperature != 1.0 else "uncalibrated softmax"
    thresholds = metadata["decision_thresholds"]
    decisions = np.where(probs <= thresholds["real_max_ai_score"], 0,
                         np.where(probs >= thresholds["ai_min_ai_score"], 1, -1))
    covered = decisions >= 0
    m["validation_calibrated_thresholds"] = thresholds
    m["abstention_test_metrics"] = {"coverage": float(covered.mean()), "uncertain_count": int((~covered).sum()),
                                    "accuracy_on_covered": float((decisions[covered] == labels[covered]).mean()) if covered.any() else None}
    source_slices = {}
    for source in sorted({row["source"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["source"] == source]
        source_labels, source_probs = labels[indices], probs[indices]
        source_predictions = logits[indices].argmax(axis=1)
        source_matrix = confusion_matrix(source_labels, source_predictions, labels=[0, 1])
        source_slices[source] = {"sample_count": len(indices), "accuracy": float(accuracy_score(source_labels, source_predictions)),
                                 "macro_f1": float(f1_score(source_labels, source_predictions, average="macro")),
                                 "false_positive_rate_human_as_ai": float(source_matrix[0, 1] / max(1, source_matrix[0].sum())),
                                 "false_negative_rate_ai_as_human": float(source_matrix[1, 0] / max(1, source_matrix[1].sum())),
                                 "confusion_matrix": source_matrix.tolist(),
                                 "roc_auc_ai_generated": float(roc_auc_score(source_labels, source_probs)) if len(set(source_labels.tolist())) == 2 else None}
    length_slices = {}
    for name, predicate in (("under_100_words", lambda n: n < 100),
                            ("100_to_299_words", lambda n: 100 <= n < 300),
                            ("300_or_more_words", lambda n: n >= 300)):
        indices = [i for i, row in enumerate(rows) if predicate(len(row["text"].split()))]
        if not indices:
            length_slices[name] = {"sample_count": 0, "metrics": "no samples in this bin"}
            continue
        subset_y, subset_pred = labels[indices], predicted[indices]
        subset_matrix = confusion_matrix(subset_y, subset_pred, labels=[0, 1])
        length_slices[name] = {"sample_count": len(indices),
                               "accuracy": float(accuracy_score(subset_y, subset_pred)),
                               "macro_f1": float(f1_score(subset_y, subset_pred, average="macro")),
                               "false_positive_rate_human_as_ai": float(subset_matrix[0, 1] / max(1, subset_matrix[0].sum())),
                               "false_negative_rate_ai_as_human": float(subset_matrix[1, 0] / max(1, subset_matrix[1].sum())),
                               "confusion_matrix": subset_matrix.tolist(),
                               "roc_auc_ai_generated": float(roc_auc_score(labels[indices], probs[indices])) if len(set(labels[indices].tolist())) == 2 else None}
    report = {"checkpoint": str(args.checkpoint), "test_split": str(args.dataset_dir / "test.jsonl"),
              "device": str(device), "metrics": m, "source_slices": source_slices, "length_slices": length_slices}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.report.with_suffix(".md").write_text(
        f"# DistilBERT HC3 test evaluation\n\nAccuracy: {m['accuracy']:.4f}; macro-F1: {m['macro_f1']:.4f}; "
        f"ROC-AUC: {m['roc_auc_ai_generated']:.4f}; coverage: {m['abstention_test_metrics']['coverage']:.4f}.\n\n"
        "In-domain HC3 results only; not evidence of robustness to other generators or genres.\n", encoding="utf-8")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
