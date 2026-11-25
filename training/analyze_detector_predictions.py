"""Audit every labeled image/text row with the same detectors selected by FastAPI.

This is inference-only. It uses a labeled validation or held-out test split,
does not train or tune models, and writes per-example outputs without storing
the original text or copying image bytes into the report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IMAGE_LABELS = ("REAL", "AI_GENERATED")
TEXT_LABELS = ("HUMAN_WRITTEN", "AI_GENERATED")


def metrics(rows: list[dict[str, Any]], true_key: str, predicted_key: str,
            labels: tuple[str, str]) -> dict[str, Any]:
    prediction_labels = (*labels, "UNCERTAIN")
    matrix = [[0, 0, 0] for _ in labels]
    for row in rows:
        truth, guess = row[true_key], row[predicted_key]
        matrix[labels.index(truth)][prediction_labels.index(guess)] += 1
    total = len(rows)
    support = {label: sum(matrix[index]) for index, label in enumerate(labels)}
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, label in enumerate(labels):
        tp = matrix[index][prediction_labels.index(label)]
        predicted_count = sum(matrix[truth_index][prediction_labels.index(label)] for truth_index in range(len(labels)))
        fn = support[label] - tp
        fp = predicted_count - tp
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support[label] if support[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support[label]}
        f1_values.append(f1)
    covered = [row for row in rows if row[predicted_key] != "UNCERTAIN"]
    correct = sum(row[true_key] == row[predicted_key] for row in rows)
    positive = labels[1]
    negative = labels[0]
    return {
        "sample_count": total,
        "accuracy_uncertain_counts_as_incorrect": correct / total if total else None,
        "coverage": len(covered) / total if total else None,
        "accuracy_on_covered": (sum(row[true_key] == row[predicted_key] for row in covered) / len(covered)) if covered else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "false_positive_rate_real_as_ai": matrix[0][1] / support[negative] if support[negative] else None,
        "false_negative_rate_ai_as_human": matrix[1][0] / support[positive] if support[positive] else None,
        "uncertain_rate_by_true_class": {
            label: matrix[index][2] / support[label] if support[label] else None
            for index, label in enumerate(labels)
        },
        "per_class": per_class,
        "confusion_matrix_labels": {"true_rows": list(labels), "predicted_columns": list(prediction_labels)},
        "confusion_matrix_rows_true_columns_predicted": matrix,
    }


def group_metrics(rows: list[dict[str, Any]], key: str, labels: tuple[str, str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unspecified")].append(row)
    return {name: metrics(group, "expected_label", "predicted_label", labels)
            for name, group in sorted(groups.items())}


def markdown_report(result: dict[str, Any]) -> str:
    lines = ["# Detector error analysis", "", f"Evaluation split: `{result['evaluation_split']}`", "",
             "Inference only. No training, threshold tuning, or checkpoint changes were performed.",
             "Raw text and image bytes are not included in the report.", ""]
    for modality, title in (("image", "Image"), ("text", "Text")):
        section = result.get(modality)
        if section is None:
            continue
        m = section["metrics"]
        lines.extend([f"## {title}", "",
                      f"Checkpoint/backend: `{section.get('checkpoint', section.get('backend'))}`",
                      f"Successful predictions: {section['successful_predictions']} / {section['expected_input_samples']}",
                      f"Accuracy (UNCERTAIN counts incorrect): {m['accuracy_uncertain_counts_as_incorrect']:.4f}",
                      f"Coverage: {m['coverage']:.4f}",
                      f"Accuracy on covered predictions: {m['accuracy_on_covered']:.4f}" if m['accuracy_on_covered'] is not None else "Accuracy on covered predictions: n/a",
                      f"Macro-F1: {m['macro_f1']:.4f}",
                      f"False positives REAL/HUMAN as AI: {m['false_positive_rate_real_as_ai']:.4f}",
                      f"False negatives AI as REAL/HUMAN: {m['false_negative_rate_ai_as_human']:.4f}",
                      f"Confusion matrix (true rows, predicted columns `{m['confusion_matrix_labels']['predicted_columns']}`): `{m['confusion_matrix_rows_true_columns_predicted']}`",
                      f"Misclassified or uncertain outputs: {len(section['misclassified_samples'])}", ""])
        if modality == "text":
            lines.extend(["### Per-source accuracy", "", "| Source | Samples | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"])
            for source, stats in section["per_source_metrics"].items():
                lines.append(f"| {source} | {stats['sample_count']} | {stats['accuracy_uncertain_counts_as_incorrect']:.4f} | {stats['macro_f1']:.4f} |")
            lines.append("")
        if section["misclassified_samples"]:
            lines.extend(["### First 25 error/uncertain records", "", "| ID/path | Expected | Predicted | Source |", "|---|---|---|---|"])
            for row in section["misclassified_samples"][:25]:
                identifier = row.get("path", row.get("sample_id", ""))
                source = row.get("source", "")
                lines.append(f"| `{identifier}` | {row['expected_label']} | {row['predicted_label']} | {source} |")
            lines.append("")
    lines.extend(["## Limits", "",
                  "These metrics describe only the selected local split and the checkpoint named above. They do not establish real-world reliability. The image benchmark contains small 32×32 CIFAKE images; the HC3 text split is QA-domain data. User camera images or out-of-domain text can behave differently.", ""])
    return "\n".join(lines)


def read_image_rows(manifest: Path, dataset_root: Path, split: str) -> list[dict[str, str]]:
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("split") == split]
    if not rows:
        raise ValueError(f"No image rows for split={split!r} in {manifest}")
    root = dataset_root.resolve()
    for row in rows:
        if row.get("label") not in IMAGE_LABELS:
            raise ValueError(f"Unsupported image label in manifest: {row.get('label')!r}")
        path = (root / row.get("path", "")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/out-of-root image path: {row.get('path')}")
        row["_resolved_path"] = str(path)
    return rows


def read_text_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("split", split) == split]
    if not rows:
        raise ValueError(f"No text rows for split={split!r} in {path}")
    for row in rows:
        if row.get("label") not in TEXT_LABELS or not isinstance(row.get("text"), str):
            raise ValueError(f"Invalid text label or text field in {path}")
    return rows


def analyze_image(rows: list[dict[str, str]], detector: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        try:
            prediction = detector.predict(row["_resolved_path"])
            results.append({"sample_id": row.get("sha256") or Path(row["path"]).name,
                            "path": row["path"], "expected_label": row["label"],
                            "predicted_label": prediction["label"],
                            "class_scores": prediction.get("class_scores"),
                            "score_type": prediction.get("score_type"),
                            "checkpoint": prediction.get("checkpoint_identifier", prediction.get("model_version")),
                            "inference_device": prediction.get("inference_device"),
                            "processing_time_ms": prediction.get("processing_time_ms")})
        except Exception as exc:
            failures.append({"sample_id": row.get("sha256") or str(index), "path": row["path"],
                             "error_type": type(exc).__name__, "error": str(exc)})
        if index % 250 == 0:
            print(f"Image rows checked: {index}/{len(rows)}", flush=True)
    return results, failures


def analyze_text(rows: list[dict[str, Any]], detector: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        text = row["text"]
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            prediction = detector.predict(text)
            results.append({"sample_id": row.get("sample_id", text_hash[:16]),
                            "text_sha256": text_hash, "source": row.get("source", "unspecified"),
                            "expected_label": row["label"], "predicted_label": prediction["label"],
                            "class_scores": prediction.get("class_scores"),
                            "score_type": prediction.get("score_type"),
                            "model_name": prediction.get("model_name"),
                            "model_version": prediction.get("model_version"),
                            "chunk_count": prediction.get("chunk_count"),
                            "word_count": prediction.get("input_word_count", len(text.split())),
                            "character_count": prediction.get("input_character_count", len(text)),
                            "processing_time_ms": prediction.get("processing_time_ms")})
        except Exception as exc:
            failures.append({"sample_id": row.get("sample_id", text_hash[:16]),
                             "text_sha256": text_hash, "source": str(row.get("source", "unspecified")),
                             "error_type": type(exc).__name__, "error": str(exc)})
        if index % 250 == 0:
            print(f"Text rows checked: {index}/{len(rows)}", flush=True)
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test",
                        help="Use only a labeled held-out split; this script never trains or tunes thresholds.")
    parser.add_argument("--image-manifest", type=Path, default=Path("data/image/cifake_ready/manifests/manifest.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/image/cifake_ready"))
    parser.add_argument("--text-file", type=Path, default=Path("data/text/processed/hc3/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("training/reports/detector_error_analysis.json"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace only the requested report file.")
    args = parser.parse_args()
    if args.skip_image and args.skip_text:
        parser.error("At least one modality must be enabled")
    output_path = args.output.resolve()
    markdown_path = output_path.with_suffix(".md")
    if (output_path.exists() or markdown_path.exists()) and not args.overwrite:
        parser.error(f"Refusing to overwrite an existing report: {output_path} or {markdown_path}; choose another --output or pass --overwrite")

    os.environ["IMAGE_MODEL_DEVICE"] = args.device
    os.environ["TEXT_MODEL_DEVICE"] = args.device
    for candidate in (ROOT / "backend", ROOT):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from backend.app.main import app

    result: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": args.split,
        "evaluation_policy": "read-only inference; no training, threshold selection, or checkpoint modification",
        "raw_text_included": False,
        "image": None,
        "text": None,
    }
    image_detector = app.state.image_detector
    text_detector = app.state.text_detector
    if not args.skip_image:
        image_rows = read_image_rows(args.image_manifest, args.image_root, args.split)
        if not image_detector.initialize():
            raise RuntimeError(f"Active FastAPI image checkpoint failed to load: {image_detector.load_error}")
        predictions, errors = analyze_image(image_rows, image_detector)
        result["image"] = {
            "manifest": str(args.image_manifest.resolve()), "dataset_root": str(args.image_root.resolve()),
            "checkpoint": str(image_detector.checkpoint_path.resolve()),
            "model_version": image_detector.checkpoint_identifier,
            "expected_input_samples": len(image_rows), "successful_predictions": len(predictions),
            "failed_samples": errors, "metrics": metrics(predictions, "expected_label", "predicted_label", IMAGE_LABELS),
            "per_sample_predictions": predictions,
            "misclassified_samples": [row for row in predictions if row["expected_label"] != row["predicted_label"]],
        }
    if not args.skip_text:
        text_rows = read_text_rows(args.text_file, args.split)
        predictions, errors = analyze_text(text_rows, text_detector)
        result["text"] = {
            "dataset_file": str(args.text_file.resolve()), "backend": app.state.text_backend,
            "checkpoint": str(text_detector.checkpoint_path.resolve()),
            "model_version": getattr(text_detector, "_metadata", {}).get("model_version", text_detector.checkpoint_path.name),
            "expected_input_samples": len(text_rows), "successful_predictions": len(predictions),
            "failed_samples": errors, "metrics": metrics(predictions, "expected_label", "predicted_label", TEXT_LABELS),
            "per_source_metrics": group_metrics(predictions, "source", TEXT_LABELS),
            "per_sample_predictions": predictions,
            "misclassified_samples": [row for row in predictions if row["expected_label"] != row["predicted_label"]],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({"split": result["evaluation_split"],
                      "image": ({"samples": result["image"]["successful_predictions"],
                                 "errors": len(result["image"]["misclassified_samples"]),
                                 "accuracy": result["image"]["metrics"]["accuracy_uncertain_counts_as_incorrect"],
                                 "checkpoint": result["image"]["checkpoint"]} if result["image"] else None),
                      "text": ({"samples": result["text"]["successful_predictions"],
                                "errors": len(result["text"]["misclassified_samples"]),
                                "accuracy": result["text"]["metrics"]["accuracy_uncertain_counts_as_incorrect"],
                                "backend": result["text"]["backend"], "checkpoint": result["text"]["checkpoint"]} if result["text"] else None),
                      "json_report": str(output_path), "summary_report": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
