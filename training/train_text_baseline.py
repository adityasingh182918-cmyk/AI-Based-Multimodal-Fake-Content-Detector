"""Train word/character TF-IDF Logistic Regression and calibrated Linear SVM."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
try:
    from .text_baseline import build_model, read_jsonl, save_model
except ImportError:  # Direct script execution from the project root on Windows.
    from text_baseline import build_model, read_jsonl, save_model


def validation_thresholds(model, rows, max_class_error_rate: float = .05) -> dict:
    labels = [r["label"] for r in rows]
    scores = model.predict_proba([r["text"] for r in rows])[:, list(model.classes_).index("AI_GENERATED")]
    human_scores = sorted(float(s) for s, y in zip(scores, labels) if y == "HUMAN_WRITTEN")
    ai_scores = sorted(float(s) for s, y in zip(scores, labels) if y == "AI_GENERATED")
    # Select one-sided operating points on validation data, limiting each directional error.
    low_index = max(0, int(len(human_scores) * (1 - max_class_error_rate)) - 1)
    high_index = min(len(ai_scores) - 1, int(len(ai_scores) * max_class_error_rate))
    low, high = human_scores[low_index], ai_scores[high_index]
    if low >= high:
        low, high = 0.0, 1.0
    return {"method": "validation_quantile_abstention", "max_class_error_rate": max_class_error_rate,
            "real_max_ai_score": low, "ai_min_ai_score": high, "validation_rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/text/processed/hc3"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/text_detector/baseline"))
    parser.add_argument("--report", type=Path, default=Path("training/reports/text_baseline_validation.json"))
    args = parser.parse_args()
    train, validation = read_jsonl(args.dataset_dir / "train.jsonl"), read_jsonl(args.dataset_dir / "validation.jsonl")
    X_train, y_train = [r["text"] for r in train], [r["label"] for r in train]
    X_val, y_val = [r["text"] for r in validation], [r["label"] for r in validation]
    report = {"dataset": str(args.dataset_dir), "validation_only_model_selection": True, "models": {}}
    for kind, filename in (("logistic", "hc3_tfidf_logistic.joblib"), ("linear_svm", "hc3_tfidf_linear_svm.joblib")):
        started = time.perf_counter()
        model = build_model(kind)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        metrics = {"accuracy": accuracy_score(y_val, pred), "macro_f1": f1_score(y_val, pred, average="macro"),
                   "training_seconds": time.perf_counter() - started, "train_rows": len(train), "validation_rows": len(validation)}
        path = args.output_dir / filename
        save_model(model, path)
        thresholds = validation_thresholds(model, validation)
        if thresholds:
            path.with_suffix(".json").write_text(json.dumps({"decision_thresholds": thresholds}, indent=2) + "\n", encoding="utf-8")
        report["models"][kind] = {**metrics, "checkpoint": str(path), "decision_thresholds": thresholds}
        print(f"{kind}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, saved {path}")
    selected = max(report["models"], key=lambda name: report["models"][name]["macro_f1"])
    report["selected_model_by_validation_macro_f1"] = selected
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation comparison written to {args.report}")


if __name__ == "__main__":
    main()
