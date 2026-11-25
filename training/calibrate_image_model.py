"""Choose validation-only REAL / AI_GENERATED / UNCERTAIN score thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from backend.app.services.image_detector import load_checkpoint  # noqa: E402
from training.data import ManifestImageDataset  # noqa: E402


def select_thresholds(actual: list[int], scores: list[float], max_class_error_rate: float) -> dict[str, Any]:
    if not actual or len(actual) != len(scores):
        raise ValueError("Validation labels and scores must be non-empty and have equal length")
    if set(actual) != {0, 1}:
        raise ValueError("Validation data must contain REAL (0) and AI_GENERATED (1)")
    if not 0 <= max_class_error_rate < 0.5:
        raise ValueError("max_class_error_rate must be in [0, 0.5)")
    candidates = sorted({0.0, 1.0, *(float(score) for score in scores)})
    total_real = sum(label == 0 for label in actual)
    total_ai = sum(label == 1 for label in actual)
    best: tuple[tuple[float, int, float], dict[str, Any]] | None = None
    for real_max in candidates:
        for ai_min in candidates:
            if real_max >= ai_min:
                continue
            real_correct = sum(y == 0 and p <= real_max for y, p in zip(actual, scores))
            real_fp = sum(y == 0 and p >= ai_min for y, p in zip(actual, scores))
            ai_fn = sum(y == 1 and p <= real_max for y, p in zip(actual, scores))
            ai_correct = sum(y == 1 and p >= ai_min for y, p in zip(actual, scores))
            real_error = real_fp / total_real
            ai_error = ai_fn / total_ai
            if real_error > max_class_error_rate or ai_error > max_class_error_rate:
                continue
            accepted = real_correct + real_fp + ai_fn + ai_correct
            coverage = accepted / len(actual)
            errors = real_fp + ai_fn
            candidate = {
                "real_max_ai_score": real_max,
                "ai_min_ai_score": ai_min,
                "max_class_error_rate_target": max_class_error_rate,
                "validation_real_false_positive_rate": real_error,
                "validation_ai_generated_false_negative_rate": ai_error,
                "validation_coverage": coverage,
                "validation_uncertain_count": len(actual) - accepted,
                "validation_samples": len(actual),
                "validation_decision_matrix": {
                    "true_labels": ["REAL", "AI_GENERATED"],
                    "predicted_labels": ["REAL", "AI_GENERATED", "UNCERTAIN"],
                    "rows_true_columns_predicted": [[real_correct, real_fp, total_real - real_correct - real_fp],
                                                     [ai_fn, ai_correct, total_ai - ai_fn - ai_correct]],
                },
            }
            rank = (coverage, -errors, ai_min - real_max)
            if best is None or rank > best[0]:
                best = (rank, candidate)
    if best is None:
        raise ValueError("No REAL/AI_GENERATED threshold pair satisfies the requested validation error target")
    return best[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--max-class-error-rate", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 0 <= args.min_coverage <= 1:
        parser.error("batch-size must be positive and min-coverage must be between 0 and 1")
    output = args.output_checkpoint.resolve()
    if output.exists():
        print(f"Refusing to overwrite calibrated checkpoint: {output}", file=sys.stderr)
        return 2
    try:
        import torch
        from torch.utils.data import DataLoader
        model, checkpoint, runtime = load_checkpoint(args.checkpoint.resolve(), "auto")
        device, preprocess, _ = runtime
        dataset = ManifestImageDataset(args.dataset_root, args.manifest, "validation", preprocess)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
        actual: list[int] = []
        scores: list[float] = []
        model.eval()
        with torch.inference_mode():
            for images, labels in loader:
                logits = model(images.to(device, non_blocking=device.type == "cuda"))
                scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
                actual.extend(labels.tolist())
        calibration = select_thresholds(actual, scores, args.max_class_error_rate)
        if calibration["validation_coverage"] < args.min_coverage:
            raise ValueError(
                f"Best validation coverage {calibration['validation_coverage']:.3f} is below required "
                f"minimum {args.min_coverage:.3f}; no calibrated checkpoint was saved"
            )
        calibration["calibrated_at_utc"] = datetime.now(timezone.utc).isoformat()
        calibration["method"] = "maximize validation coverage subject to per-class error-rate targets; softmax scores are uncalibrated"
        output.parent.mkdir(parents=True, exist_ok=True)
        calibrated = dict(checkpoint)
        calibrated["decision_thresholds"] = calibration
        tmp_path = output.with_suffix(output.suffix + ".tmp")
        torch.save(calibrated, tmp_path)
        tmp_path.replace(output)
        metadata_path = output.with_suffix(".metadata.json")
        metadata = {key: value for key, value in calibrated.items() if key != "state_dict"}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"source_checkpoint": str(args.checkpoint.resolve()),
                          "calibrated_checkpoint": str(output), "decision_thresholds": calibration}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        print(f"Validation calibration failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
