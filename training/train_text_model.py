"""Fine-tune DistilBERT on prepared text splits with dynamic padding and CUDA-safe batches."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

try:
    from .text_baseline import read_jsonl
except ImportError:
    from text_baseline import read_jsonl

MODEL_ID = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
LABEL2ID = {"HUMAN_WRITTEN": 0, "AI_GENERATED": 1}


class TextRows(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows, self.tokenizer, self.max_length = rows, tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)
        encoded["labels"] = LABEL2ID[row["label"]]
        return encoded


def infer_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**batch)
            logits.extend(output.logits.float().cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    return np.asarray(logits), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/text/processed/hc3"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/text_detector/transformer/distilbert_hc3_v1"))
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Optional local sequence-classification checkpoint to continue fine-tuning from.")
    parser.add_argument("--history", type=Path, default=Path("training/reports/text_transformer_history.json"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Select a compatible PyTorch build or explicitly pass --device cpu.")
    device = torch.device(args.device)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite nonempty checkpoint directory: {args.output_dir}. Choose a new path or pass --overwrite.")
    if args.init_checkpoint is not None and not (args.init_checkpoint / "config.json").is_file():
        raise SystemExit(f"Initialization checkpoint is missing config.json: {args.init_checkpoint}")
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("epochs, batch size, and gradient accumulation must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        print(f"GPU: {torch.cuda.get_device_name(0)}; memory={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
    started = time.perf_counter()
    source_model = str(args.init_checkpoint) if args.init_checkpoint is not None else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(
        source_model, local_files_only=True if args.init_checkpoint is not None else False,
        **({} if args.init_checkpoint is not None else {"revision": MODEL_REVISION}))
    model = AutoModelForSequenceClassification.from_pretrained(
        source_model, local_files_only=args.init_checkpoint is not None,
        **({} if args.init_checkpoint is not None else {"revision": MODEL_REVISION}), num_labels=2,
        id2label={0: "HUMAN_WRITTEN", 1: "AI_GENERATED"}, label2id=LABEL2ID)
    model.to(device)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if device.type == "cuda" else None)
    train_rows = read_jsonl(args.dataset_dir / "train.jsonl")
    val_rows = read_jsonl(args.dataset_dir / "validation.jsonl")
    train_loader = DataLoader(TextRows(train_rows, tokenizer, args.max_length), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(TextRows(val_rows, tokenizer, args.max_length), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    optimizer_steps = (len(train_loader) + args.gradient_accumulation - 1) // args.gradient_accumulation
    scheduler = get_linear_schedule_with_warmup(optimizer, int(optimizer_steps * args.epochs * .1), optimizer_steps * args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss / args.gradient_accumulation
            scaler.scale(loss).backward(); losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        logits, labels = infer_logits(model, val_loader, device)
        preds = logits.argmax(axis=1)
        val_f1 = float(f1_score(labels, preds, average="macro"))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": float(accuracy_score(labels, preds)),
                  "validation_macro_f1": val_f1, "elapsed_seconds": time.perf_counter() - started}
        history.append(record); print(json.dumps(record), flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            model.save_pretrained(args.output_dir, safe_serialization=True); tokenizer.save_pretrained(args.output_dir)
            # Validation-only quantile abstention operating points.
            probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
            human = sorted(float(p) for p, y in zip(probs, labels) if y == 0)
            generated = sorted(float(p) for p, y in zip(probs, labels) if y == 1)
            low = human[max(0, int(len(human) * .95) - 1)]
            high = generated[min(len(generated) - 1, int(len(generated) * .05))]
            if low >= high: low, high = 0.0, 1.0
            metadata = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                        "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
                        "model_name": "DistilBERT continued fine-tune",
                        "model_version": args.output_dir.name, "max_length": args.max_length,
                        "label2id": LABEL2ID, "selected_epoch": epoch,
                        "decision_thresholds": {"method": "validation_quantile_abstention", "max_class_error_rate": .05,
                                                "real_max_ai_score": low, "ai_min_ai_score": high}}
            (args.output_dir / "text_detector_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        else:
            stale += 1
            if stale >= 2:
                print("Early stopping: validation macro-F1 did not improve for two epochs.")
                break
    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text(json.dumps({"settings": vars(args) | {"dataset_dir": str(args.dataset_dir), "output_dir": str(args.output_dir), "history": str(args.history)},
                                       "best_epoch": best_epoch, "best_validation_macro_f1": best_f1, "epochs": history}, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Best validation macro-F1={best_f1:.4f} at epoch {best_epoch}; checkpoint: {args.output_dir}")


if __name__ == "__main__":
    main()
