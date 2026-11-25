"""Recompute expanded-model thresholds on validation using deployed inference paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

try:
    from .compare_text_models import svm_scores, transformer_scores
    from .text_baseline import read_jsonl
except ImportError:
    from compare_text_models import svm_scores, transformer_scores
    from text_baseline import read_jsonl


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (scores >= boundaries[index]) & (scores < boundaries[index + 1] if index < bins - 1 else scores <= boundaries[index + 1])
        if selected.any():
            ece += float(selected.mean()) * abs(float(labels[selected].mean()) - float(scores[selected].mean()))
    return ece


def validation_thresholds(scores: np.ndarray, labels: list[str], max_error: float = .05) -> dict:
    human = sorted(float(score) for score, label in zip(scores, labels) if label == "HUMAN_WRITTEN")
    ai = sorted(float(score) for score, label in zip(scores, labels) if label == "AI_GENERATED")
    low = human[max(0, int(len(human) * (1 - max_error)) - 1)]
    high = ai[min(len(ai) - 1, int(len(ai) * max_error))]
    if low >= high:
        low, high = 0.0, 1.0
    return {"method": "validation_quantile_abstention_api_aggregation", "max_class_error_rate": max_error,
            "real_max_ai_score": low, "ai_min_ai_score": high, "validation_rows": len(labels)}


def temperature_diagnostic(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Fit scalar temperature on validation for analysis only; never deployed here."""
    from scipy.optimize import minimize_scalar

    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped))
    def scaled(temp: float) -> np.ndarray:
        z = np.clip(logits / temp, -40, 40)
        return 1 / (1 + np.exp(-z))
    raw = {"brier": float(brier_score_loss(labels, scores)),
           "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
           "ece_10_bins": expected_calibration_error(labels, scores)}
    result = minimize_scalar(lambda temp: log_loss(labels, scaled(temp), labels=[0, 1]),
                             bounds=(0.05, 10.0), method="bounded")
    adjusted = scaled(float(result.x))
    return {"raw_validation": raw,
            "temperature_fit_on_validation_diagnostic_only": float(result.x),
            "temperature_scaled_validation_diagnostic_only": {
                "brier": float(brier_score_loss(labels, adjusted)),
                "log_loss": float(log_loss(labels, adjusted, labels=[0, 1])),
                "ece_10_bins": expected_calibration_error(labels, adjusted)},
            "temperature_deployed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/text/processed/expanded_v1"))
    parser.add_argument("--svm-checkpoint", type=Path, default=Path("models/text_detector/svm_expanded_v1.joblib"))
    parser.add_argument("--transformer-checkpoint", type=Path, default=Path("models/text_detector/distilbert_expanded_v1"))
    parser.add_argument("--report", type=Path, default=Path("training/reports/text_expanded_calibration.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    rows = read_jsonl(args.dataset_dir / "validation.jsonl")
    labels = [row["label"] for row in rows]
    y = np.asarray([int(label == "AI_GENERATED") for label in labels])
    report = {"dataset": str(args.dataset_dir / "validation.jsonl"), "sample_count": len(rows),
              "thresholds_use_validation_only": True, "models": {}}
    for name, checkpoint, is_transformer in (("svm_expanded_v1", args.svm_checkpoint, False),
                                              ("distilbert_expanded_v1", args.transformer_checkpoint, True)):
        if (checkpoint.is_dir() if is_transformer else checkpoint.is_file()) is False:
            raise SystemExit(f"Missing checkpoint: {checkpoint}")
        scores, _ = (transformer_scores(checkpoint, rows, args.device, args.batch_size, apply_temperature=False) if is_transformer
                     else svm_scores(checkpoint, rows))
        sidecar = checkpoint / "text_detector_metadata.json" if is_transformer else checkpoint.with_suffix(".json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        previous = metadata.get("decision_thresholds")
        calibration = temperature_diagnostic(scores, y)
        raw_metrics = calibration["raw_validation"]
        scaled_metrics = calibration["temperature_scaled_validation_diagnostic_only"]
        deploy_temperature = (is_transformer and all(scaled_metrics[key] < raw_metrics[key]
                            for key in ("brier", "log_loss", "ece_10_bins")))
        temperature = calibration["temperature_fit_on_validation_diagnostic_only"] if deploy_temperature else 1.0
        calibrated_scores = scores
        if deploy_temperature:
            clipped = np.clip(scores, 1e-7, 1 - 1e-7)
            logits = np.log(clipped / (1 - clipped)) / temperature
            calibrated_scores = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        calibration["temperature_deployed"] = deploy_temperature
        thresholds = validation_thresholds(calibrated_scores, labels)
        entry = {"checkpoint": str(checkpoint), "score_semantics": ("validation-temperature-scaled softmax" if deploy_temperature else
                 "uncalibrated softmax") if is_transformer else
                 "CalibratedClassifierCV probability; domain dependent",
                 "previous_thresholds": previous, "new_api_aggregation_validation_thresholds": thresholds,
                 "calibration_diagnostics": calibration,
                 "temperature_deployed": deploy_temperature,
                 "deployed_temperature": temperature,
                 "temperature_decision_rule": "deploy only when validation Brier, log-loss, and 10-bin ECE all improve; no independent test data used",
                 "prediction_rule_threshold": 0.5}
        report["models"][name] = entry
        metadata["decision_thresholds"] = thresholds
        metadata["decision_threshold_source"] = "expanded validation with deployed chunk/window aggregation; test/OOD not used"
        metadata["inference_temperature"] = temperature
        metadata["temperature_calibration"] = ({"method": "scalar temperature scaling",
                                                "temperature": temperature,
                                                "fit_split": str(args.dataset_dir / "validation.jsonl"),
                                                "validation_only": True,
                                                "deployed": deploy_temperature}
                                               if is_transformer else None)
        temp_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
        temp_sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temp_sidecar.replace(sidecar)
        print(f"{name}: thresholds {thresholds['real_max_ai_score']:.6f} / {thresholds['ai_min_ai_score']:.6f}; temperature diagnostic={entry['calibration_diagnostics']['temperature_fit_on_validation_diagnostic_only']:.4f} (not deployed)", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation-only calibration report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
