"""Train only the Phase 3 fusion layers from precomputed paired features."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score
from torch import nn

from phase3.fusion.encoders import sparse_to_torch
from phase3.fusion.network import FUSION_LABELS, GatedFusionNetwork

ROOT = Path(__file__).resolve().parents[2]


def load_split(feature_dir: Path, split: str):
    image = torch.from_numpy(np.load(feature_dir / f"image_features_{split}.npy")).float()
    text = sp.load_npz(feature_dir / f"text_features_{split}.npz").tocsr()
    labels = np.load(feature_dir / f"labels_{split}.npz")
    return image, text, labels


def score(model, image, text, labels, device, batch_size: int):
    model.eval()
    predictions, truth, losses = [], [], []
    with torch.inference_mode():
        for start in range(0, len(image), batch_size):
            take = slice(start, start + batch_size)
            sparse = sparse_to_torch(text[take], device)
            result = model(image[take].to(device), sparse)
            target = torch.as_tensor(labels["fusion_y"][take], dtype=torch.long, device=device)
            losses.append(float(nn.functional.cross_entropy(result["fusion_logits"], target).item()))
            predictions.extend(result["fusion_logits"].argmax(1).cpu().tolist())
            truth.extend(target.cpu().tolist())
    return {"macro_f1": float(f1_score(truth, predictions, average="macro")),
            "accuracy": float(np.mean(np.asarray(truth) == np.asarray(predictions))),
            "loss": float(np.mean(losses)), "predictions": predictions, "truth": truth}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("phase3/data/features"))
    parser.add_argument("--checkpoint", type=Path, default=Path("phase3/models/fusion/best_model.pt"))
    parser.add_argument("--last-checkpoint", type=Path, default=Path("phase3/models/fusion/last_checkpoint.pt"))
    parser.add_argument("--history", type=Path, default=Path("phase3/reports/fusion_training_history.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite fusion checkpoint: {args.checkpoint}")
    if args.last_checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite last checkpoint: {args.last_checkpoint}")
    for split in ("train", "validation"):
        if not (args.features_dir / f"image_features_{split}.npy").is_file():
            raise FileNotFoundError(f"Missing extracted {split} pair features; no fusion training was started")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    train_image, train_text, train_labels = load_split(args.features_dir, "train")
    val_image, val_text, val_labels = load_split(args.features_dir, "validation")
    train_counts = np.bincount(train_labels["fusion_y"], minlength=4)
    if np.any(train_counts == 0):
        raise ValueError(f"All four paired authenticity classes are required for training; support={train_counts.tolist()}")
    model = GatedFusionNetwork(train_image.shape[1], train_text.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_score, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_image))
        running_loss, steps = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            sparse = sparse_to_torch(train_text[ids], device)
            image_batch = train_image[ids].to(device)
            fusion_y = torch.as_tensor(train_labels["fusion_y"][ids], dtype=torch.long, device=device)
            image_y = torch.as_tensor(train_labels["image_y"][ids], dtype=torch.long, device=device)
            text_y = torch.as_tensor(train_labels["text_y"][ids], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image_batch, sparse)
            loss = (nn.functional.cross_entropy(outputs["fusion_logits"], fusion_y) +
                    0.25 * nn.functional.cross_entropy(outputs["image_logits"], image_y) +
                    0.25 * nn.functional.cross_entropy(outputs["text_logits"], text_y))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item()); steps += 1
        validation = score(model, val_image, val_text, val_labels, device, args.batch_size)
        scheduler.step(validation["loss"])
        row = {"epoch": epoch, "train_loss": running_loss / max(steps, 1),
               "validation_macro_f1": validation["macro_f1"], "validation_accuracy": validation["accuracy"],
               "validation_loss": validation["loss"], "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); print(json.dumps(row), flush=True)
        args.last_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "architecture": "gated_sparse_fusion_v1",
                    "image_dim": int(train_image.shape[1]), "text_dim": int(train_text.shape[1]),
                    "hidden_dim": 128, "fusion_labels": list(FUSION_LABELS), "seed": args.seed,
                    "epoch": epoch, "history": history}, args.last_checkpoint)
        if validation["macro_f1"] > best_score:
            best_score, wait = validation["macro_f1"], 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No validation-selected fusion state was produced")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "architecture": "gated_sparse_fusion_v1",
                "image_dim": int(train_image.shape[1]), "text_dim": int(train_text.shape[1]),
                "hidden_dim": 128, "fusion_labels": list(FUSION_LABELS), "seed": args.seed,
                "selected_epoch": max(history, key=lambda row: row["validation_macro_f1"])["epoch"]}, args.checkpoint)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text(json.dumps({"seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "patience": args.patience,
        "learning_rate": args.learning_rate, "device": str(device),
        "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=2)",
        "selected_epoch": max(history, key=lambda row: row["validation_macro_f1"])["epoch"],
        "history": history}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
