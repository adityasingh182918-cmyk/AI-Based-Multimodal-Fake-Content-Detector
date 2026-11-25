"""Evaluate a trained fusion checkpoint on held-out paired test features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from phase3.fusion.encoders import sparse_to_torch
from phase3.fusion.network import FUSION_LABELS, GatedFusionNetwork
from phase3.training.train_fusion import load_split


def metrics(expected, predicted, labels):
    return {"accuracy": float(accuracy_score(expected, predicted)),
            "macro_f1": float(f1_score(expected, predicted, average="macro", labels=list(range(len(labels))), zero_division=0)),
            "confusion_matrix_labels": labels,
            "confusion_matrix_rows_true_columns_predicted": confusion_matrix(expected, predicted, labels=list(range(len(labels)))).tolist(),
            "classification_report": classification_report(expected, predicted, labels=list(range(len(labels))),
                                                            target_names=labels, output_dict=True, zero_division=0)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("phase3/data/features"))
    parser.add_argument("--checkpoint", type=Path, default=Path("phase3/models/fusion/best_model.pt"))
    parser.add_argument("--report", type=Path, default=Path("phase3/reports/fusion_test_evaluation.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"No trained fusion checkpoint: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    image, text, labels = load_split(args.features_dir, "test")
    model = GatedFusionNetwork(checkpoint["image_dim"], checkpoint["text_dim"], checkpoint.get("hidden_dim", 128))
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.to(device).eval()
    fusion_pred, image_pred, text_pred = [], [], []
    with torch.inference_mode():
        for start in range(0, len(image), args.batch_size):
            ids = slice(start, start + args.batch_size)
            outputs = model(image[ids].to(device), sparse_to_torch(text[ids], device))
            fp = outputs["fusion_logits"].argmax(1).cpu().numpy()
            fusion_pred.extend(fp.tolist())
            image_pred.extend(np.where(np.isin(fp, [1, 3]), 1, 0).tolist())
            text_pred.extend(np.where(np.isin(fp, [1, 2]), 1, 0).tolist())
    true_fusion = labels["fusion_y"].tolist()
    true_image, true_text = labels["image_y"].tolist(), labels["text_y"].tolist()
    metadata = json.loads((args.features_dir / "metadata_test.json").read_text(encoding="utf-8"))
    frozen_img = [row["frozen_image_prediction"]["label"] for row in metadata["provenance_and_frozen_predictions"]]
    frozen_txt = [row["frozen_text_prediction"]["label"] for row in metadata["provenance_and_frozen_predictions"]]
    image_names, text_names = ["REAL", "AI_GENERATED"], ["HUMAN_WRITTEN", "AI_GENERATED"]
    # Compare both binary encoders on the same pairs. For their UNCERTAIN
    # decisions, use the larger raw class score only for this explicit
    # two-class diagnostic; keep the production threshold outcomes in metadata.
    image_base, text_base = [], []
    for row, label in zip(metadata["provenance_and_frozen_predictions"], frozen_img):
        image_base.append(1 if label == "AI_GENERATED" else 0 if label == "REAL" else
                          int(row["frozen_image_prediction"]["class_scores"]["AI_GENERATED"] >
                              row["frozen_image_prediction"]["class_scores"]["REAL"]))
    for row, label in zip(metadata["provenance_and_frozen_predictions"], frozen_txt):
        text_base.append(1 if label == "AI_GENERATED" else 0 if label == "HUMAN_WRITTEN" else
                         int(row["frozen_text_prediction"]["class_scores"]["AI_GENERATED"] >
                             row["frozen_text_prediction"]["class_scores"]["HUMAN_WRITTEN"]))
    report = {"checkpoint": str(args.checkpoint), "split": "paired test only",
              "selection_or_calibration_used_test": False,
              "support_by_joint_label": {name: true_fusion.count(index) for index, name in enumerate(FUSION_LABELS)},
              "fusion_four_class": metrics(true_fusion, fusion_pred, list(FUSION_LABELS)),
              "image_origin_from_fusion": metrics(true_image, image_pred, image_names),
              "text_origin_from_fusion": metrics(true_text, text_pred, text_names),
              "frozen_image_model": metrics(true_image, image_base, image_names),
              "frozen_text_model": metrics(true_text, text_base, text_names),
              "frozen_model_uncertain_handling": "For the two-class baseline comparison only, UNCERTAIN outcomes are converted by argmax of the stored raw class scores; these are not the deployed threshold predictions.",
              "sample_ids": labels["sample_ids"].tolist(),
              "interpretation": "Only report improvement against each frozen modality when evaluated on the same paired held-out examples; no claim is implied by code alone."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("support_by_joint_label", "fusion_four_class", "image_origin_from_fusion", "text_origin_from_fusion")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
