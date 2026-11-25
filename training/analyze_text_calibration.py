"""Analyze text-score calibration; fit temperature only on HC3 validation data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss

try:
    from .evaluate_text_robustness import baseline_scores, transformer_scores
    from .text_baseline import read_jsonl
except ImportError:
    from evaluate_text_robustness import baseline_scores, transformer_scores
    from text_baseline import read_jsonl


def _to_ai_labels(rows: list[dict]) -> np.ndarray:
    labels = np.asarray([1 if row["label"] in ("AI_GENERATED", 1) else 0 for row in rows], dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration split must contain both classes")
    return labels


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    records = []
    value = 0.0
    for i in range(bins):
        selected = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        count = int(selected.sum())
        if not count:
            continue
        avg_p, rate = float(p[selected].mean()), float(y[selected].mean())
        value += count / len(y) * abs(avg_p - rate)
        records.append({"lower": float(edges[i]), "upper": float(edges[i + 1]), "count": count,
                        "mean_ai_probability": avg_p, "observed_ai_rate": rate})
    return float(value), records


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ece, bins = _ece(y, p)
    return {"sample_count": int(len(y)), "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "ece_10_equal_width_bins": ece,
            "reliability_bins": bins}


def _threshold_report(y: np.ndarray, p: np.ndarray, thresholds: dict) -> dict:
    low, high = thresholds["real_max_ai_score"], thresholds["ai_min_ai_score"]
    decision = np.where(p <= low, 0, np.where(p >= high, 1, -1))
    covered = decision >= 0
    return {"thresholds": thresholds, "coverage": float(covered.mean()),
            "uncertain_count": int((~covered).sum()),
            "covered_accuracy": float((decision[covered] == y[covered]).mean()) if covered.any() else None,
            "human_false_ai_rate_on_covered": float(((y[covered] == 0) & (decision[covered] == 1)).sum() /
                                                      max(1, (y[covered] == 0).sum())),
            "ai_false_human_rate_on_covered": float(((y[covered] == 1) & (decision[covered] == 0)).sum() /
                                                      max(1, (y[covered] == 1).sum()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/text/processed/hc3"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib"))
    parser.add_argument("--transformer-checkpoint", type=Path, default=Path("models/text_detector/transformer/distilbert_hc3_v1"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path("training/reports/text_calibration_analysis.json"))
    args = parser.parse_args()
    validation = read_jsonl(args.dataset_dir / "validation.jsonl")
    test = read_jsonl(args.dataset_dir / "test.jsonl")
    y_val, y_test = _to_ai_labels(validation), _to_ai_labels(test)
    report = {"scope": "HC3 calibration investigation; model scores frozen; temperature is fitted only on validation",
              "test_policy": "HC3 test and external test corpora are not used for temperature/threshold fitting",
              "models": {}}

    models = [("tfidf_linear_svm", lambda rows: baseline_scores(args.baseline_checkpoint, rows))]
    models.append(("distilbert", lambda rows: transformer_scores(args.transformer_checkpoint, rows, args.device, args.batch_size)))
    for name, score in models:
        val_probs, thresholds = score(validation)
        test_probs, _ = score(test)
        val_probs, test_probs = np.clip(val_probs, 1e-7, 1 - 1e-7), np.clip(test_probs, 1e-7, 1 - 1e-7)
        logits = np.log(val_probs / (1 - val_probs))
        result = minimize_scalar(lambda temp: log_loss(y_val, 1 / (1 + np.exp(-np.clip(logits / temp, -40, 40))), labels=[0, 1]),
                                 bounds=(0.25, 5.0), method="bounded")
        temperature = float(result.x)
        val_scaled = 1 / (1 + np.exp(-np.clip(logits / temperature, -40, 40)))
        test_logits = np.log(test_probs / (1 - test_probs))
        test_scaled = 1 / (1 + np.exp(-np.clip(test_logits / temperature, -40, 40)))
        result_row = {"temperature_scaling": {"temperature": temperature, "selected_on": "HC3 validation only",
                                                "deployed": False,
                                                "note": "No checkpoint or runtime threshold was changed."},
                      "validation_raw": _metrics(y_val, val_probs),
                      "validation_temperature_scaled": _metrics(y_val, val_scaled),
                      "test_raw": _metrics(y_test, test_probs),
                      "test_temperature_scaled_using_validation_temperature": _metrics(y_test, test_scaled),
                      "existing_abstention_thresholds": thresholds,
                      "threshold_behavior_validation_resubstitution": _threshold_report(y_val, val_probs, thresholds),
                      "threshold_behavior_heldout_test": _threshold_report(y_test, test_probs, thresholds)}
        report["models"][name] = result_row

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: {k: v for k, v in row[key].items() if k != "reliability_bins"}
                             for key in ("validation_raw", "validation_temperature_scaled", "test_raw",
                                         "test_temperature_scaled_using_validation_temperature", "threshold_behavior_heldout_test")}
                      for name, row in report["models"].items()}, indent=2))
    print(f"Calibration analysis written to {args.report}")


if __name__ == "__main__":
    main()
