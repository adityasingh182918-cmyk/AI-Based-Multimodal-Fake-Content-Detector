"""Analyze private real-world image/text examples without saving their content.

Input manifest CSV columns: kind,path,expected_label[,sample_id]. Paths may be
absolute or relative to the manifest's directory. Reports contain measurements
and labels only; neither text, image bytes, nor input paths are copied to output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

IMAGE_LABELS = {"REAL", "AI_GENERATED"}
TEXT_LABELS = {"HUMAN_WRITTEN", "AI_GENERATED"}


def _input_path(raw_path: str, manifest: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    local = (manifest.parent / path).resolve()
    if local.is_file():
        return local
    return (ROOT / path).resolve()


def _image_metadata(path: Path, detector: Any) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        original_size = list(image.size)
        original_format = image.format
        original_mode = image.mode
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        oriented_size = list(oriented.size)
    detector._ensure_loaded()  # Use the same checkpoint-matched transform as inference.
    _, preprocess, _ = detector._runtime
    with Image.open(path) as image:
        image.load()
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        tensor = preprocess(oriented)
    return {
        "format": original_format,
        "original_mode": original_mode,
        "original_dimensions": {"width": original_size[0], "height": original_size[1]},
        "exif_oriented_dimensions": {"width": oriented_size[0], "height": oriented_size[1]},
        "preprocessing_output": {
            "shape_chw": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "value_min": float(tensor.min().item()),
            "value_max": float(tensor.max().item()),
            "channel_means": [float(value) for value in tensor.mean(dim=(1, 2)).tolist()],
            "resize_mode": detector._checkpoint.get("resize_mode", "stretch"),
            "image_size": int(detector._checkpoint.get("image_size", 224)),
            "mean": detector._checkpoint.get("mean"),
            "std": detector._checkpoint.get("std"),
            "exif_transpose": True,
            "rgb_conversion": True,
        },
    }


def _make_image_detector(checkpoint: str | None, device: str) -> Any:
    from app.services.image_detector import ImageDetector

    api_default = ROOT / "models" / "image_detector" / "resnet18_cifake_aug_v1_calibrated_strict.pt"
    configured = checkpoint or os.getenv("IMAGE_MODEL_CHECKPOINT") or api_default
    return ImageDetector(configured, device=device)


def _make_text_detector(backend: str, checkpoint: str | None, device: str) -> Any:
    if backend == "baseline":
        from app.services.text_detector import DEFAULT_CHECKPOINT, TextDetector

        configured = checkpoint or os.getenv("TEXT_MODEL_CHECKPOINT") or DEFAULT_CHECKPOINT
        return TextDetector(configured)
    from app.services.transformer_text_detector import TransformerTextDetector

    default = ROOT / "models" / "text_detector" / "transformer" / "distilbert_hc3_v1"
    return TransformerTextDetector(checkpoint or default, device=device)


def analyze(manifest: Path, kind: str, image_checkpoint: str | None = None,
            text_backend: str = "baseline", text_checkpoint: str | None = None,
            device: str = "auto") -> dict[str, Any]:
    image_detector = _make_image_detector(image_checkpoint, device) if kind == "image" else None
    text_detector = _make_text_detector(text_backend, text_checkpoint, device) if kind == "text" else None
    expected_labels = IMAGE_LABELS if kind == "image" else TEXT_LABELS
    rows: list[dict[str, str]] = []
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "expected_label"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest must include path and expected_label columns")
        for row in reader:
            row_kind = (row.get("kind") or kind).strip().lower()
            label = (row.get("expected_label") or "").strip().upper()
            if row_kind != kind:
                raise ValueError(f"Manifest row kind {row_kind!r} does not match requested kind {kind!r}")
            if label not in expected_labels:
                raise ValueError(f"Unsupported {kind} expected_label {label!r}")
            rows.append({"path": row["path"], "expected_label": label})
    if not rows:
        raise ValueError("Manifest contains no examples")

    checkpoint = (str(image_detector.checkpoint_path) if image_detector else
                  str(text_detector.checkpoint_path))
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = f"{kind}_{index:04d}"
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "expected_class": row["expected_label"],
            "predicted_class": None,
            "class_probabilities": None,
            "inference_time_ms": None,
            "checkpoint": checkpoint,
            "status": "ERROR",
        }
        path = _input_path(row["path"], manifest)
        try:
            if kind == "image":
                result.update(_image_metadata(path, image_detector))
                prediction = image_detector.predict(path)
                result.update({
                    "predicted_class": prediction["label"],
                    "class_probabilities": prediction["class_scores"],
                    "inference_time_ms": prediction["processing_time_ms"],
                    "model_version": prediction["model_version"],
                    "score_type": prediction["score_type"],
                })
            else:
                text = path.read_text(encoding="utf-8-sig")
                prediction = text_detector.predict(text)
                result.update({
                    "predicted_class": prediction["label"],
                    "class_probabilities": prediction["class_scores"],
                    "word_count": prediction["input_word_count"],
                    "character_count": prediction["input_character_count"],
                    "chunk_count": prediction["chunk_count"],
                    "inference_time_ms": prediction["processing_time_ms"],
                    "model_name": prediction["model_name"],
                    "score_type": prediction["score_type"],
                })
            result["misclassified"] = result["predicted_class"] != row["expected_label"]
            result["status"] = "OK"
        except Exception as exc:  # Keep private paths/content out of failure reports.
            result["error_type"] = type(exc).__name__
            result["misclassified"] = None
        examples.append(result)

    succeeded = [row for row in examples if row["status"] == "OK"]
    misclassified = sum(bool(row["misclassified"]) for row in succeeded)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": kind,
        "status": "COMPLETED" if len(succeeded) == len(examples) else "PARTIAL",
        "sample_count": len(examples),
        "successful_count": len(succeeded),
        "misclassified_count": misclassified,
        "checkpoint": checkpoint,
        "text_backend": text_backend if kind == "text" else None,
        "content_persisted": False,
        "input_paths_persisted": False,
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("image", "text"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-checkpoint")
    parser.add_argument("--text-backend", choices=("baseline", "transformer"))
    parser.add_argument("--text-checkpoint")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args(argv)
    try:
        text_backend = args.text_backend or os.getenv("TEXT_MODEL_BACKEND", "baseline").lower()
        device = args.device or os.getenv("TEXT_MODEL_DEVICE" if text_backend == "transformer" else "IMAGE_MODEL_DEVICE", "auto").lower()
        report = analyze(args.manifest, args.kind, args.image_checkpoint,
                         text_backend, args.text_checkpoint, device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Failure analysis could not run: {exc}", file=sys.stderr)
        return 2
    print(f"{report['task']} analysis: {report['successful_count']}/{report['sample_count']} succeeded; "
          f"{report['misclassified_count']} misclassified. Report: {args.output}")
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
