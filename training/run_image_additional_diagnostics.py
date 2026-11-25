"""Compare frozen image checkpoints and validation-only preprocessing sensitivities.

All transformations are diagnostic perturbations of the existing validation
split; they are not independent real-world accuracy estimates.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
from app.services.image_detector import AspectPadResize, DEFAULT_STD, DEFAULT_MEAN, load_checkpoint  # noqa: E402


def _metrics(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    names = ["REAL", "AI_GENERATED"]
    matrix = [[0, 0], [0, 0]]
    for truth, guess in zip(labels, predictions):
        matrix[names.index(truth)][names.index(guess)] += 1
    total = len(labels)
    return {
        "samples": total,
        "accuracy": sum(matrix[i][i] for i in range(2)) / total if total else None,
        "real_false_positive_rate": matrix[0][1] / sum(matrix[0]) if sum(matrix[0]) else None,
        "ai_generated_false_negative_rate": matrix[1][0] / sum(matrix[1]) if sum(matrix[1]) else None,
        "confusion_matrix_labels": names,
        "confusion_matrix_rows_true_columns_predicted": matrix,
    }


def _read_validation(manifest: Path, root: Path) -> list[dict[str, Any]]:
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("split") == "validation"]
    if not rows:
        raise ValueError(f"No validation rows found in {manifest}")
    for row in rows:
        if row.get("label") not in {"REAL", "AI_GENERATED"}:
            raise ValueError(f"Unsupported validation label: {row.get('label')!r}")
    result = []
    for index, row in enumerate(rows, start=1):
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Validation manifest path escapes dataset root")
        with Image.open(path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
        result.append({"sample_id": f"validation_{index:04d}", "label": row["label"], "image": image})
    return result


def _variant(image: Image.Image, name: str, size: int, mean: list[float], std: list[float],
             resize_mode: str, to_tensor: Any) -> Any:
    if name == "native_resolution":
        transformed = image.copy()
    elif name == "checkpoint_preprocess":
        if resize_mode == "pad":
            transformed = AspectPadResize(size)(image)
        else:
            from torchvision import transforms
            transformed = transforms.Resize((size, size))(image)
    elif name == "aspect_preserving_pad":
        transformed = AspectPadResize(size)(image)
    elif name == "square_stretch":
        transformed = image.resize((size, size), Image.Resampling.LANCZOS)
    elif name in {"jpeg_quality_95", "jpeg_quality_70"}:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95 if name.endswith("95") else 70, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            transformed = compressed.convert("RGB")
    elif name == "mild_blur":
        transformed = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    elif name == "brightness_90":
        transformed = ImageEnhance.Brightness(image).enhance(0.9)
    elif name == "brightness_110":
        transformed = ImageEnhance.Brightness(image).enhance(1.1)
    else:
        raise ValueError(f"Unknown diagnostic variant: {name}")
    if name in {"jpeg_quality_95", "jpeg_quality_70", "mild_blur", "brightness_90", "brightness_110"}:
        transformed = AspectPadResize(size)(transformed) if resize_mode == "pad" else transformed.resize(
            (size, size), Image.Resampling.LANCZOS)
    from torchvision import transforms

    return transforms.Normalize(mean, std)(to_tensor(transformed))


def _score_batch(model: Any, checkpoint: dict[str, Any], runtime: Any,
                 samples: list[dict[str, Any]], variant: str, batch_size: int = 1) -> list[dict[str, Any]]:
    import torch
    from torchvision.transforms import ToTensor

    device, _, _ = runtime
    transform = lambda image: _variant(image, variant, int(checkpoint.get("image_size", 224)),
                                       checkpoint.get("mean", DEFAULT_MEAN), checkpoint.get("std", DEFAULT_STD),
                                       checkpoint.get("resize_mode", "stretch"), ToTensor())
    output: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        rows = samples[start:start + batch_size]
        infer_started = time.perf_counter()
        tensor = torch.stack([transform(row["image"]) for row in rows]).to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(tensor), dim=1).cpu().tolist()
        elapsed_ms_per_item = (time.perf_counter() - infer_started) * 1000.0 / len(rows)
        for row, scores in zip(rows, probs):
            class_scores = {"REAL": float(scores[0]), "AI_GENERATED": float(scores[1])}
            raw = max(class_scores, key=class_scores.get)
            thresholds = checkpoint.get("decision_thresholds")
            if thresholds:
                ai_score = class_scores["AI_GENERATED"]
                if ai_score <= float(thresholds["real_max_ai_score"]):
                    decision = "REAL"
                elif ai_score >= float(thresholds["ai_min_ai_score"]):
                    decision = "AI_GENERATED"
                else:
                    decision = "UNCERTAIN"
            else:
                decision = raw
            output.append({"sample_id": row["sample_id"], "expected_class": row["label"],
                           "predicted_class": raw, "decision_label": decision,
                           "class_scores": class_scores, "inference_time_ms": elapsed_ms_per_item})
    return output


def _checkpoint_info(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {"file": path.name, "run_id": checkpoint.get("run_id"),
            "architecture": checkpoint.get("architecture"), "class_order": checkpoint.get("class_names"),
            "image_size": checkpoint.get("image_size", 224), "resize_mode": checkpoint.get("resize_mode", "stretch"),
            "mean": checkpoint.get("mean", DEFAULT_MEAN), "std": checkpoint.get("std", DEFAULT_STD),
            "decision_thresholds": checkpoint.get("decision_thresholds"),
            "pretrained_initialization": checkpoint.get("pretrained_initialization")}


def run(manifest: Path, dataset_root: Path, user_images: list[Path], device: str = "auto") -> dict[str, Any]:
    import torch

    resolved = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    validation = _read_validation(manifest, dataset_root)
    available_checkpoints = sorted((ROOT / "models" / "image_detector").glob("*.pt"))
    checkpoints: list[tuple[Path, Any, dict[str, Any], Any]] = []
    rejected: list[dict[str, str]] = []
    for path in available_checkpoints:
        try:
            model, metadata, runtime = load_checkpoint(path, resolved)
            checkpoints.append((path, model, metadata, runtime))
        except Exception as exc:
            rejected.append({"file": path.name, "reason": type(exc).__name__})
    if not checkpoints:
        raise RuntimeError("No valid local image checkpoints could be loaded")

    user_rows = []
    user_paths: list[Path] = []
    for idx, path in enumerate(user_images, start=1):
        user_paths.append(path)
        with Image.open(path) as opened:
            opened.load()
            original_size, image_format, image_mode = list(opened.size), opened.format, opened.mode
            image = ImageOps.exif_transpose(opened).convert("RGB")
            oriented_size = list(image.size)
        aspect = oriented_size[0] / oriented_size[1]
        user_rows.append({"sample_id": f"user_image_{idx:02d}", "label": "REAL", "image": image,
                          "image_metadata": {"format": image_format, "original_mode": image_mode,
                                             "original_dimensions": {"width": original_size[0], "height": original_size[1]},
                                             "exif_oriented_dimensions": {"width": oriented_size[0], "height": oriented_size[1]},
                                             "aspect_ratio_width_over_height": aspect,
                                             "aspect_pad_approximate_black_fraction": max(0.0, 1.0 - min(aspect, 1.0 / aspect))}})

    checkpoint_reports = []
    for path, model, metadata, runtime in checkpoints:
        t0 = time.perf_counter()
        predictions = _score_batch(model, metadata, runtime, validation, "checkpoint_preprocess")
        raw_metrics = _metrics([row["label"] for row in validation], [row["predicted_class"] for row in predictions])
        decisions = [row["decision_label"] for row in predictions]
        observed = [(row["label"], label) for row, label in zip(validation, decisions) if label != "UNCERTAIN"]
        decision_metrics = _metrics([x[0] for x in observed], [x[1] for x in observed]) if observed else None
        user_predictions = _score_batch(model, metadata, runtime, user_rows, "checkpoint_preprocess", batch_size=1)
        checkpoint_reports.append({"checkpoint": _checkpoint_info(path, metadata),
                                   "validation_raw_argmax_metrics": raw_metrics,
                                   "validation_saved_threshold_metrics_on_covered": decision_metrics,
                                   "validation_uncertain_count": sum(value == "UNCERTAIN" for value in decisions),
                                   "validation_predictions": predictions,
                                   "user_examples": [{**sample["image_metadata"], **prediction}
                                                     for sample, prediction in zip(user_rows, user_predictions)],
                                   "evaluation_seconds": round(time.perf_counter() - t0, 4)})

    api_checkpoint = Path(os.getenv("IMAGE_MODEL_CHECKPOINT", str(ROOT / "models/image_detector/resnet18_cifake_aug_v1_calibrated_strict.pt")))
    api_checkpoint = api_checkpoint if api_checkpoint.is_absolute() else ROOT / api_checkpoint
    api_model_tuple = next((item for item in checkpoints if item[0].resolve() == api_checkpoint.resolve()), None)
    sensitivity = None
    api_route_predictions = []
    if api_model_tuple:
        _, model, metadata, runtime = api_model_tuple
        from app.services.image_detector import ImageDetector
        service_detector = ImageDetector(api_checkpoint, device=resolved.type)
        for index, path in enumerate(user_paths, start=1):
            prediction = service_detector.predict(path)
            api_route_predictions.append({"sample_id": f"user_image_{index:02d}",
                                          "expected_class": "REAL",
                                          "predicted_class": prediction["label"],
                                          "class_scores": prediction["class_scores"],
                                          "model_version": prediction["model_version"],
                                          "inference_time_ms": prediction["processing_time_ms"],
                                          "score_type": prediction["score_type"]})
        names = ("native_resolution", "aspect_preserving_pad", "square_stretch", "jpeg_quality_95",
                 "jpeg_quality_70", "mild_blur", "brightness_90", "brightness_110")
        base = _score_batch(model, metadata, runtime, validation, "checkpoint_preprocess")
        variants = {}
        for name in names:
            preds = base if name == "aspect_preserving_pad" else _score_batch(model, metadata, runtime, validation, name)
            native_by_id = {row["sample_id"]: row for row in base}
            variants[name] = {
                "metrics_raw_argmax": _metrics([row["label"] for row in validation], [row["predicted_class"] for row in preds]),
                "changed_raw_predictions_vs_checkpoint_preprocessing": sum(
                    row["predicted_class"] != native_by_id[row["sample_id"]]["predicted_class"] for row in preds),
                "mean_absolute_ai_score_change_vs_checkpoint_preprocessing": sum(
                    abs(row["class_scores"]["AI_GENERATED"] - native_by_id[row["sample_id"]]["class_scores"]["AI_GENERATED"])
                    for row in preds) / len(preds),
                "predictions": preds,
            }
        sensitivity = {"anchor_checkpoint": _checkpoint_info(api_model_tuple[0], metadata),
                       "split": "existing validation only",
                       "interpretation": "Controlled perturbation sensitivity; variants are correlated views of the same images and are not separate real-world accuracy samples.",
                       "variants": variants}

    return {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validation_source": str(manifest), "validation_samples": len(validation),
            "loaded_checkpoints": len(checkpoints), "rejected_checkpoints": rejected,
            "fastapi_default_checkpoint": str(api_checkpoint),
            "fastapi_default_checkpoint_loaded": api_model_tuple is not None,
            "checkpoint_comparison": checkpoint_reports,
            "preprocessing_sensitivity": sensitivity,
            "fastapi_service_predictions": api_route_predictions,
            "user_example_note": "Four user-provided images were treated as REAL per the user's statement. Reports contain no source paths or image bytes.",
            "user_content_persisted": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/image/cifake_ready/manifests/manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/image/cifake_ready"))
    parser.add_argument("--user-images", nargs="*", type=Path, default=[])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("training/reports/image_additional_diagnostics.json"))
    args = parser.parse_args()
    report = run(args.manifest, args.dataset_root, args.user_images, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.user_images:
        by_checkpoint = {row["checkpoint"]["file"]: row["user_examples"] for row in report["checkpoint_comparison"]}
        sample_metadata = report["checkpoint_comparison"][0]["user_examples"]
        grouped = []
        for sample in sample_metadata:
            grouped.append({
                key: sample[key] for key in ("sample_id", "expected_class", "format", "original_mode",
                                               "original_dimensions", "exif_oriented_dimensions",
                                               "aspect_ratio_width_over_height", "aspect_pad_approximate_black_fraction")
            } | {"checkpoint_predictions": [dict(checkpoint=name, predicted_class=prediction["predicted_class"],
                                                 decision_label=prediction["decision_label"],
                                                 class_scores=prediction["class_scores"],
                                                 inference_time_ms=prediction["inference_time_ms"])
                                            for name, predictions in by_checkpoint.items()
                                            for prediction in predictions if prediction["sample_id"] == sample["sample_id"]]})
        user_report = {"status": "COMPLETED", "task": "user-reported REAL image false-positive reproduction",
                       "expected_label_source": "User stated these uploaded photographs are real; provenance is user-provided and not independently verified.",
                       "sample_count": len(grouped), "examples": grouped,
                       "fastapi_service_predictions": report["fastapi_service_predictions"],
                       "checkpoints_compared": list(by_checkpoint),
                       "fastapi_default_checkpoint": report["fastapi_default_checkpoint"],
                       "content_persisted": False, "source_paths_persisted": False,
                       "full_validation_and_sensitivity_report": str(args.output)}
        error_report = args.output.parent / "image_real_world_errors.json"
        error_report.write_text(json.dumps(user_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid_checkpoints": report["loaded_checkpoints"],
                      "validation_samples": report["validation_samples"],
                      "user_images": len(args.user_images),
                      "report": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
