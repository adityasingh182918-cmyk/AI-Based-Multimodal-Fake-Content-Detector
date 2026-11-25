"""Compare frozen text checkpoints on one prepared split with API-like chunk scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

try:
    from .text_baseline import load_model, read_jsonl
except ImportError:
    from text_baseline import load_model, read_jsonl

LABELS = ("HUMAN_WRITTEN", "AI_GENERATED")


def text_hash(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def metrics(rows, scores: np.ndarray, thresholds: dict | None = None) -> dict:
    y_names = [row["label"] for row in rows]
    y = np.asarray([int(value == "AI_GENERATED") for value in y_names])
    pred = (scores >= .5).astype(int)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    output = {"sample_count": len(rows), "class_counts": dict(Counter(y_names)),
              "accuracy_at_0_5": float(accuracy_score(y, pred)),
              "macro_f1_at_0_5": float(f1_score(y, pred, average="macro", zero_division=0)),
              "class_metrics_at_0_5": classification_report(y, pred, labels=[0, 1],
                  target_names=["HUMAN_WRITTEN", "AI_GENERATED"], output_dict=True, zero_division=0),
              "confusion_matrix_labels": ["HUMAN_WRITTEN", "AI_GENERATED"],
              "confusion_matrix": matrix.tolist(),
              "human_false_positive_rate": float(matrix[0, 1] / max(1, matrix[0].sum())),
              "ai_false_negative_rate": float(matrix[1, 0] / max(1, matrix[1].sum())),
              "ai_roc_auc": float(roc_auc_score(y, scores)) if len(set(y.tolist())) == 2 else None}
    if thresholds:
        low, high = thresholds["real_max_ai_score"], thresholds["ai_min_ai_score"]
        tri = np.where(scores <= low, 0, np.where(scores >= high, 1, -1))
        covered = tri >= 0
        output["validation_thresholds"] = thresholds
        output["abstention"] = {"coverage": float(covered.mean()), "uncertain_count": int((~covered).sum()),
                                "accuracy_on_covered": float((tri[covered] == y[covered]).mean()) if covered.any() else None}
    return output


def svm_scores(checkpoint: Path, rows) -> tuple[np.ndarray, list[int]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.services.text_detector import CHUNK_CHARS, split_text
    sys.path.pop(0)
    model = load_model(checkpoint)
    texts, weights, owners, counts = [], [], [], [0] * len(rows)
    for index, row in enumerate(rows):
        chunks = split_text(row["text"], CHUNK_CHARS)
        counts[index] = len(chunks)
        for chunk in chunks:
            texts.append(chunk); weights.append(max(1, len(chunk.split()))); owners.append(index)
    chunk_scores = model.predict_proba(texts)[:, list(model.classes_).index("AI_GENERATED")]
    sums, total_weights = np.zeros(len(rows)), np.zeros(len(rows))
    for owner, weight, score in zip(owners, weights, chunk_scores):
        sums[owner] += float(weight) * float(score); total_weights[owner] += weight
    return sums / np.maximum(total_weights, 1), counts


def transformer_scores(checkpoint: Path, rows, device: str, batch_size: int,
                       apply_temperature: bool = True) -> tuple[np.ndarray, list[int]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    metadata = json.loads((checkpoint / "text_detector_metadata.json").read_text(encoding="utf-8"))
    temperature = float(metadata.get("inference_temperature", 1.0)) if apply_temperature else 1.0
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    ai_index = int(metadata.get("label2id", {}).get("AI_GENERATED", 1))
    sums = np.zeros(len(rows), dtype=np.float64)
    weights = np.zeros(len(rows), dtype=np.float64)
    counts = [0] * len(rows)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            texts = [row["text"] for row in rows[start:start + batch_size]]
            encoded = tokenizer(texts, truncation=True, max_length=int(metadata["max_length"]), stride=32,
                                return_overflowing_tokens=True, return_tensors="pt", padding=True)
            owners = encoded.pop("overflow_to_sample_mapping").cpu().numpy().tolist()
            attention = encoded["attention_mask"]
            window_weights = attention.sum(dim=1).cpu().numpy().astype(np.float64)
            probabilities = []
            for window_start in range(0, attention.shape[0], batch_size):
                batch = {key: value[window_start:window_start + batch_size].to(device)
                         for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
                logits = model(**batch).logits.float()
                probabilities.extend(torch.softmax(logits / temperature, dim=1)[:, ai_index].cpu().numpy().tolist())
            for local_index, (owner, weight, score) in enumerate(zip(owners, window_weights, probabilities)):
                global_index = start + int(owner)
                sums[global_index] += float(weight) * float(score)
                weights[global_index] += float(weight)
                counts[global_index] += 1
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return sums / np.maximum(weights, 1), counts


def model_metadata(path: Path, transformer: bool) -> dict:
    metadata_path = path / "text_detector_metadata.json" if transformer else path.with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/text/processed/expanded_v1"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--old-svm", type=Path, default=Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib"))
    parser.add_argument("--new-svm", type=Path, default=Path("models/text_detector/svm_expanded_v1.joblib"))
    parser.add_argument("--old-transformer", type=Path, default=Path("models/text_detector/transformer/distilbert_hc3_v1"))
    parser.add_argument("--new-transformer", type=Path, default=Path("models/text_detector/distilbert_expanded_v1"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    rows = read_jsonl(args.dataset_dir / f"{args.split}.jsonl")
    report = {"dataset_dir": str(args.dataset_dir), "split": args.split,
              "selection_or_threshold_fitting": False, "row_count": len(rows), "models": {}, "per_example": []}
    candidates = [("old_svm", args.old_svm, False), ("expanded_svm", args.new_svm, False),
                  ("old_distilbert", args.old_transformer, True), ("expanded_distilbert", args.new_transformer, True)]
    all_scores = {}
    for name, checkpoint, is_transformer in candidates:
        exists = checkpoint.is_dir() if is_transformer else checkpoint.is_file()
        if not exists:
            report["models"][name] = {"status": "checkpoint_not_yet_available", "checkpoint": str(checkpoint)}
            continue
        scores, chunk_counts = (transformer_scores(checkpoint, rows, args.device, args.batch_size) if is_transformer
                                else svm_scores(checkpoint, rows))
        metadata = model_metadata(checkpoint, is_transformer)
        report["models"][name] = {"status": "evaluated", "checkpoint": str(checkpoint),
                                  "model_name": metadata.get("model_name", name),
                                  "model_version": metadata.get("model_version", checkpoint.name),
                                  "score_type": (("validation_temperature_scaled_softmax" if metadata.get("inference_temperature", 1.0) != 1.0
                                                  else "uncalibrated_softmax") if is_transformer
                                                 else "calibrated_classifier_probability; calibration may shift by source"),
                                  "metrics": metrics(rows, scores, metadata.get("decision_thresholds")),
                                  "by_source": {}, "by_domain": {}, "by_length": {}}
        all_scores[name] = (scores, chunk_counts)
        for group_name, group_values in (("by_source", lambda row: row.get("source", "unknown")),
                                         ("by_domain", lambda row: (row.get("source_group") or row.get("source", "unknown")).split("_", 1)[0]),
                                         ("by_length", lambda row: "very_short_under_20" if len(row["text"].split()) < 20 else
                                          "short_20_to_99" if len(row["text"].split()) < 100 else
                                          "medium_100_to_299" if len(row["text"].split()) < 300 else "long_300_plus")):
            buckets = defaultdict(list)
            for index, row in enumerate(rows):
                buckets[group_values(row)].append(index)
            for group, indices in buckets.items():
                subset_rows = [rows[i] for i in indices]
                report["models"][name][group_name][group] = metrics(subset_rows, scores[indices])

    for index, row in enumerate(rows):
        output = {"sample_id": text_hash(row["text"]), "expected": row["label"], "source": row.get("source"),
                  "domain": (row.get("source_group") or "").split("_", 1)[0], "word_count": len(row["text"].split())}
        for name, (scores, counts) in all_scores.items():
            score = float(scores[index])
            output[name] = {"predicted_at_0_5": "AI_GENERATED" if score >= .5 else "HUMAN_WRITTEN",
                            "ai_score": score, "chunk_or_window_count": counts[index]}
        report["per_example"].append(output)
    output_path = args.report or Path(f"training/reports/text_model_comparison_expanded_{args.split}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, item in report["models"].items():
        if item.get("status") == "evaluated":
            m = item["metrics"]
            print(f"{name}: accuracy={m['accuracy_at_0_5']:.4f}; macro-F1={m['macro_f1_at_0_5']:.4f}; AI FNR={m['ai_false_negative_rate']:.4f}; human FPR={m['human_false_positive_rate']:.4f}")
        else:
            print(f"{name}: checkpoint not available")
    print(f"Comparison report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
