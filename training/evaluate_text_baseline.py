"""Evaluate a frozen TF-IDF model and optionally write length-sliced test metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, average_precision_score,
                             brier_score_loss)
try:
    from .text_baseline import LABELS, load_model, read_jsonl
except ImportError:  # Direct script execution from the project root on Windows.
    from text_baseline import LABELS, load_model, read_jsonl


def metrics(model, rows):
    y = [r["label"] for r in rows]
    pred = model.predict([r["text"] for r in rows])
    scores = model.predict_proba([r["text"] for r in rows])[:, list(model.classes_).index("AI_GENERATED")]
    matrix = confusion_matrix(y, pred, labels=LABELS)
    return {"sample_count": len(rows), "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=LABELS, average="macro"), "weighted_f1": f1_score(y, pred, average="weighted"),
            "precision_by_class": precision_score(y, pred, labels=LABELS, average=None, zero_division=0).tolist(),
            "recall_by_class": recall_score(y, pred, labels=LABELS, average=None, zero_division=0).tolist(),
            "false_positive_rate_human_as_ai": float(matrix[0, 1] / max(1, matrix[0].sum())),
            "false_negative_rate_ai_as_human": float(matrix[1, 0] / max(1, matrix[1].sum())),
            "confusion_matrix_labels": list(LABELS), "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0),
            "roc_auc_ai_generated": roc_auc_score([int(value == "AI_GENERATED") for value in y], scores),
            "average_precision_ai_generated": average_precision_score([int(value == "AI_GENERATED") for value in y], scores),
            "brier_score_ai_generated": brier_score_loss([int(value == "AI_GENERATED") for value in y], scores)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/text/processed/hc3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib"))
    parser.add_argument("--report", type=Path, default=Path("training/reports/text_evaluation.json"))
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    rows = read_jsonl(args.dataset_dir / "test.jsonl")
    result = metrics(model, rows)
    sidecar = args.checkpoint.with_suffix(".json")
    if sidecar.is_file():
        thresholds = json.loads(sidecar.read_text(encoding="utf-8"))["decision_thresholds"]
        scores = model.predict_proba([r["text"] for r in rows])[:, list(model.classes_).index("AI_GENERATED")]
        predictions = ["HUMAN_WRITTEN" if score <= thresholds["real_max_ai_score"] else
                       "AI_GENERATED" if score >= thresholds["ai_min_ai_score"] else "UNCERTAIN" for score in scores]
        result["validation_calibrated_thresholds"] = thresholds
        result["abstention_test_metrics"] = {"coverage": sum(p != "UNCERTAIN" for p in predictions) / len(predictions),
                                               "uncertain_count": predictions.count("UNCERTAIN"),
                                               "accuracy_on_covered": (sum(p == row["label"] for p, row in zip(predictions, rows) if p != "UNCERTAIN") /
                                                                        max(1, sum(p != "UNCERTAIN" for p in predictions)))}
    report = {"checkpoint": str(args.checkpoint), "test_split": str(args.dataset_dir / "test.jsonl"),
              "metrics": result, "source_slices": {}, "length_slices": {}}
    for source in sorted({row["source"] for row in rows}):
        subset = [row for row in rows if row["source"] == source]
        report["source_slices"][source] = metrics(model, subset) if {r["label"] for r in subset} == set(LABELS) else {
            "sample_count": len(subset), "metrics": "not enough class coverage"}
    for name, predicate in (("under_100_words", lambda n: n < 100), ("100_to_299_words", lambda n: 100 <= n < 300),
                            ("300_or_more_words", lambda n: n >= 300)):
        subset = [r for r in rows if predicate(len(r["text"].split()))]
        report["length_slices"][name] = metrics(model, subset) if subset and {r["label"] for r in subset} == set(LABELS) else {"sample_count": len(subset), "metrics": "not enough class coverage"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    m = report["metrics"]
    markdown = ("# HC3 text baseline test evaluation\n\n"
                f"Checkpoint: `{args.checkpoint}`  \nTest rows: {m['sample_count']}  \n"
                f"Accuracy: {m['accuracy']:.4f}  \nMacro F1: {m['macro_f1']:.4f}  \n"
                f"Weighted F1: {m['weighted_f1']:.4f}  \nROC-AUC (AI_GENERATED): {m['roc_auc_ai_generated']:.4f}  \n"
                f"Average precision (AI_GENERATED): {m['average_precision_ai_generated']:.4f}  \n"
                f"Brier score (AI_GENERATED): {m['brier_score_ai_generated']:.4f}\n\n"
                f"Confusion matrix labels `{LABELS}`: `{m['confusion_matrix']}`\n\n"
                "These are in-domain HC3 question-answer results only. They do not establish robustness to other generators, topics, or writing domains.\n")
    args.report.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Test report written to {args.report}")


if __name__ == "__main__":
    main()
