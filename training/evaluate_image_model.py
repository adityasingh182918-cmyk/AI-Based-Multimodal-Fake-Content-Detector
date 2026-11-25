"""Evaluate a saved image detector on the held-out manifest test split only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from backend.app.services.image_detector import CLASS_NAMES, load_checkpoint  # noqa: E402
from training.data import ManifestImageDataset  # noqa: E402


def _auc(actual: list[int], positive_scores: list[float]) -> float | None:
    positives = sum(actual)
    negatives = len(actual) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(positive_scores, actual), key=lambda item: item[0])
    rank_sum = 0.0
    position = 1
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        average_rank = (position + position + (end - cursor) - 1) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[cursor:end])
        position += end - cursor
        cursor = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def calculate_metrics(actual: list[int], predicted: list[int], positive_scores: list[float] | None = None) -> dict[str, Any]:
    if len(actual) != len(predicted) or (positive_scores is not None and len(actual) != len(positive_scores)):
        raise ValueError("actual, predicted, and score arrays must have equal length")
    matrix = [[0, 0], [0, 0]]
    for truth, guess in zip(actual, predicted):
        matrix[truth][guess] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(2) if row != index)
        fn = sum(matrix[index][col] for col in range(2) if col != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": support,
        }
    total = len(actual)
    accuracy = sum(matrix[i][i] for i in range(2)) / total if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": per_class["AI_GENERATED"]["precision"],
        "recall": per_class["AI_GENERATED"]["recall"],
        "f1_score": per_class["AI_GENERATED"]["f1-score"],
        "real_false_positive_rate": matrix[0][1] / sum(matrix[0]) if sum(matrix[0]) else 0.0,
        "ai_generated_false_negative_rate": matrix[1][0] / sum(matrix[1]) if sum(matrix[1]) else 0.0,
        "precision_ai_generated": per_class["AI_GENERATED"]["precision"],
        "recall_ai_generated": per_class["AI_GENERATED"]["recall"],
        "f1_ai_generated": per_class["AI_GENERATED"]["f1-score"],
        "per_class": per_class,
        "confusion_matrix": {"labels": list(CLASS_NAMES), "rows_true_columns_predicted": matrix},
        "roc_auc_ai_generated": _auc(actual, positive_scores) if positive_scores is not None else None,
        "classification_report": {**per_class, "accuracy": accuracy, "macro_avg": {
            key: sum(float(per_class[label][key]) for label in CLASS_NAMES) / 2
            for key in ("precision", "recall", "f1-score")
        }, "weighted_avg": {
            key: sum(float(per_class[label][key]) * int(per_class[label]["support"]) for label in CLASS_NAMES) / total if total else 0.0
            for key in ("precision", "recall", "f1-score")
        }},
        "test_samples": total,
    }


def calculate_thresholded_metrics(actual: list[int], positive_scores: list[float], thresholds: dict[str, Any]) -> dict[str, Any]:
    if not actual or len(actual) != len(positive_scores):
        raise ValueError("actual labels and positive scores must be non-empty and have equal length")
    low = float(thresholds["real_max_ai_score"])
    high = float(thresholds["ai_min_ai_score"])
    matrix = [[0, 0, 0], [0, 0, 0]]
    predicted_names = ("REAL", "AI_GENERATED", "UNCERTAIN")
    for truth, score in zip(actual, positive_scores):
        prediction = 0 if score <= low else 1 if score >= high else 2
        matrix[truth][prediction] += 1
    total = len(actual)
    support_real = sum(matrix[0])
    support_ai = sum(matrix[1])
    covered = sum(matrix[row][0] + matrix[row][1] for row in range(2))
    correct = matrix[0][0] + matrix[1][1]
    return {
        "thresholds": {"real_max_ai_score": low, "ai_min_ai_score": high},
        "predicted_labels": list(predicted_names),
        "rows_true_columns_predicted": matrix,
        "samples": total,
        "coverage": covered / total,
        "uncertain_count": sum(matrix[row][2] for row in range(2)),
        "accuracy_on_covered": correct / covered if covered else None,
        "real_false_positive_rate": matrix[0][1] / support_real if support_real else 0.0,
        "ai_generated_false_negative_rate": matrix[1][0] / support_ai if support_ai else 0.0,
        "real_false_positive_count": matrix[0][1],
        "ai_generated_false_negative_count": matrix[1][0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/image/manifests/manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/image_detector/resnet18_best.pt"))
    parser.add_argument("--report", type=Path, default=Path("training/reports/image_model_evaluation.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers cannot be negative")
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        print(f"Evaluation dependencies are unavailable: {exc}. Install requirements-ml.txt and the selected PyTorch build first.", file=sys.stderr)
        return 2
    try:
        model, checkpoint, runtime = load_checkpoint(args.checkpoint, args.device)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 2
    device, _inference_preprocess, _torch = runtime
    try:
        dataset = ManifestImageDataset(args.dataset_root, args.manifest, "test", _inference_preprocess)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Held-out test data could not be loaded: {exc}", file=sys.stderr)
        return 2
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    actual: list[int] = []
    predicted: list[int] = []
    positive_scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            probabilities = torch.softmax(logits, dim=1)
            predicted.extend(probabilities.argmax(dim=1).cpu().tolist())
            positive_scores.extend(probabilities[:, 1].cpu().tolist())
            actual.extend(labels.tolist())
    metrics = calculate_metrics(actual, predicted, positive_scores)
    if isinstance(checkpoint.get("decision_thresholds"), dict):
        metrics["thresholded_decision"] = calculate_thresholded_metrics(actual, positive_scores, checkpoint["decision_thresholds"])
    if metrics["test_samples"] == 0:
        print("Test manifest has no held-out samples; no metrics were produced.", file=sys.stderr)
        return 2
    metadata_path = args.manifest.resolve().parent / "metadata.json"
    dataset_metadata = {}
    if metadata_path.is_file():
        try:
            dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "model_version": checkpoint.get("run_id", args.checkpoint.name),
        "architecture": checkpoint["architecture"],
        "training_device": checkpoint.get("training_device", "not recorded by this checkpoint"),
        "gpu_name": checkpoint.get("gpu_name", "not recorded by this checkpoint"),
        "preprocessing": {"image_size": checkpoint["image_size"], "resize_mode": checkpoint.get("resize_mode", "stretch"), "mean": checkpoint["mean"], "std": checkpoint["std"]},
        "dataset": dataset_metadata.get("dataset", checkpoint.get("dataset_identifier")),
        "dataset_revision": dataset_metadata.get("revision", checkpoint.get("dataset_revision")),
        "manifest": str(args.manifest.resolve()),
        "evaluation_split": "test",
        "positive_class": "AI_GENERATED",
        "score_note": "ROC-AUC uses uncalibrated softmax scores; no probability calibration has been applied.",
        "metrics": metrics,
        "limitations": "Results describe this held-out split only; performance does not guarantee reliability on unseen generators or real-world images.",
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Evaluation report saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
