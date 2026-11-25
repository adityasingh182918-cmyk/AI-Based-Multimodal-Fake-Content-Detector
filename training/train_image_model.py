"""Train the ResNet-18 image classifier with live CUDA-aware progress."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from backend.app.services.image_detector import (  # noqa: E402
    CLASS_NAMES, DEFAULT_MEAN, DEFAULT_STD, AspectPadResize, build_preprocess, build_resnet18,
)
from training.data import ManifestImageDataset  # noqa: E402


def validate_manifest(manifest: Path) -> None:
    with manifest.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"path", "label", "split"} <= set(rows[0]):
        raise ValueError("Manifest must contain path, label, and split columns")
    for split in ("train", "validation", "test"):
        labels = {row.get("label") for row in rows if row.get("split") == split}
        missing = set(CLASS_NAMES) - labels
        if missing:
            raise ValueError(f"{split} split is missing required class(es): {', '.join(sorted(missing))}")
    hash_splits: dict[str, str] = {}
    for row in rows:
        digest = row.get("sha256", "")
        split = row.get("split", "")
        if digest and digest in hash_splits and hash_splits[digest] != split:
            raise ValueError(f"Manifest hash leaks across {hash_splits[digest]} and {split} splits: {row.get('path')}")
        if digest:
            hash_splits[digest] = split


def calculate_metrics(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    matrix = [[0, 0], [0, 0]]
    for truth, guess in zip(actual, predicted):
        matrix[truth][guess] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(2) if row != index)
        fn = sum(matrix[index][col] for col in range(2) if col != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                           "support": support}
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(2))
    return {
        "accuracy": correct / total if total else 0.0,
        "precision_ai_generated": per_class["AI_GENERATED"]["precision"],
        "recall_ai_generated": per_class["AI_GENERATED"]["recall"],
        "f1_ai_generated": per_class["AI_GENERATED"]["f1"],
        "per_class": per_class,
        "confusion_matrix": {"labels": list(CLASS_NAMES), "rows_true_columns_predicted": matrix},
        "samples": total,
    }


class RunLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def run_epoch(model: Any, loader: Any, loss_fn: Any, optimizer: Any, device: Any, *, training: bool,
              amp_enabled: bool, scaler: Any, epoch: int, epochs: int, tqdm_class: Any) -> tuple[float, dict[str, Any], float]:
    import torch

    model.train(training)
    loss_total = 0.0
    total = 0
    actual: list[int] = []
    predicted: list[int] = []
    started = time.perf_counter()
    progress = tqdm_class(loader, desc=f"Epoch {epoch}/{epochs} {'train' if training else 'validation'}", unit="batch", leave=False)
    for batch_index, (images, labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_size = int(labels.size(0))
        loss_total += float(loss.detach().item()) * batch_size
        guesses = logits.detach().argmax(dim=1)
        actual.extend(labels.detach().cpu().tolist())
        predicted.extend(guesses.cpu().tolist())
        total += batch_size
        running = calculate_metrics(actual, predicted)
        lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix(loss=f"{loss_total / total:.4f}", accuracy=f"{running['accuracy']:.1%}",
                             progress=f"{batch_index / max(len(loader), 1):.0%}", lr=f"{lr:.2g}",
                             elapsed=f"{time.perf_counter() - started:.0f}s")
    if not total:
        raise ValueError("DataLoader is empty")
    return loss_total / total, calculate_metrics(actual, predicted), time.perf_counter() - started


def _gpu_diagnostics(torch: Any, requested_device: str) -> tuple[Any | None, str | None]:
    cuda_available = bool(torch.cuda.is_available())
    runtime = torch.version.cuda or "not available"
    device_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
    total_memory = torch.cuda.get_device_properties(0).total_memory if cuda_available else None
    selected = requested_device
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"PyTorch CUDA runtime: {runtime}")
    print(f"GPU device: {device_name}")
    print(f"GPU total memory: {total_memory / (1024**3):.2f} GiB" if total_memory else "GPU total memory: unavailable")
    print(f"Selected training device: {selected}")
    if requested_device == "cuda" and not cuda_available:
        return None, "CUDA was requested but is unavailable. Check the NVIDIA driver and install a CUDA-enabled torch/torchvision build; no CPU fallback was selected."
    return torch.device(requested_device), None


def _cpu_state(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item, torch) for item in value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/image/manifests/manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/image_detector/resnet18_best.pt"))
    parser.add_argument("--init-checkpoint", type=Path, help="Compatible ResNet-18 checkpoint to fine-tune; takes precedence over ImageNet initialization")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="CUDA is required by default; use cpu only when explicitly requested")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable CUDA mixed precision (default: on)")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.num_workers < 0 or args.learning_rate <= 0 or args.early_stopping_patience < 1:
        parser.error("epochs, batch-size, learning-rate and early-stopping-patience must be positive; image-size >=32 and num-workers >=0")
    if args.checkpoint.resolve().exists():
        print(f"Refusing to overwrite existing checkpoint: {args.checkpoint.resolve()}. Choose a new versioned --checkpoint path.", file=sys.stderr)
        return 2
    last_checkpoint = args.checkpoint.resolve().with_name(f"{args.checkpoint.resolve().stem}_last.pt")
    if last_checkpoint.exists():
        print(f"Refusing to overwrite existing last checkpoint: {last_checkpoint}. Choose a new versioned --checkpoint path.", file=sys.stderr)
        return 2
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        parser.error(f"Initialization checkpoint does not exist: {args.init_checkpoint}")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from tqdm.auto import tqdm
    except ImportError as exc:
        print(f"Training dependency unavailable: {exc}. Install backend requirements, requirements-ml.txt and the selected PyTorch build.", file=sys.stderr)
        return 2

    device, error = _gpu_diagnostics(torch, args.device)
    if error:
        print(error, file=sys.stderr)
        return 2
    try:
        validate_manifest(args.manifest)
        train_transform = transforms.Compose([
            AspectPadResize(args.image_size), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.1),
            transforms.ToTensor(), transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ])
        eval_transform = build_preprocess(args.image_size, DEFAULT_MEAN, DEFAULT_STD, "pad")
        train_data = ManifestImageDataset(args.dataset_root, args.manifest, "train", train_transform)
        validation_data = ManifestImageDataset(args.dataset_root, args.manifest, "validation", eval_transform)
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        print(f"Dataset/manifest setup failed: {exc}", file=sys.stderr)
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pin_memory = device.type == "cuda"
    current_batch_size = args.batch_size

    def make_loaders(batch_size: int, epoch: int):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                  generator=generator, pin_memory=pin_memory)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=pin_memory)
        return train_loader, validation_loader
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.amp and not amp_enabled:
        print("AMP requested but CPU device selected; mixed precision is disabled.")
    pretrained_initialization = bool(args.pretrained and args.init_checkpoint is None)
    try:
        model = build_resnet18(pretrained=pretrained_initialization).to(device)
        if args.init_checkpoint is not None:
            initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
            expected = {
                "architecture": "resnet18",
                "class_names": list(CLASS_NAMES),
                "image_size": args.image_size,
                "resize_mode": "pad",
                "mean": list(DEFAULT_MEAN),
                "std": list(DEFAULT_STD),
            }
            incompatible = {key: (initial.get(key), value) for key, value in expected.items()
                            if initial.get(key) != value}
            if incompatible:
                raise ValueError(f"Initialization checkpoint metadata is incompatible: {incompatible}")
            state = initial.get("state_dict")
            if not isinstance(state, dict):
                raise ValueError("Initialization checkpoint does not contain a state_dict")
            model.load_state_dict(state, strict=True)
    except Exception as exc:
        message = "Could not initialize ResNet-18 pretrained weights." if args.pretrained else "Could not initialize ResNet-18."
        print(f"{message} No random-weight fallback was used. Details: {exc}", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    started_at = datetime.now(timezone.utc).isoformat()
    reports_dir = REPO_ROOT / "training" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(reports_dir / "train_image_model.log")
    history_path = reports_dir / f"image_model_history_{run_id}.json"
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.{run_id}.candidate.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_fn = nn.CrossEntropyLoss()
    best_validation_loss = float("inf")
    best_metadata: dict[str, Any] | None = None
    completed_epochs = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    dataset_metadata: dict[str, Any] = {}
    metadata_source = args.manifest.resolve().parent / "metadata.json"
    if metadata_source.is_file():
        try:
            dataset_metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dataset_metadata = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (explicitly selected)"
    logger.log(f"Run {run_id}: device={device}; GPU={gpu_name}; train={len(train_data)}; validation={len(validation_data)}; batch={current_batch_size}; amp={amp_enabled}")
    logger.log(f"Model initialization: {args.init_checkpoint.resolve() if args.init_checkpoint else ('torchvision ResNet18_Weights.DEFAULT' if pretrained_initialization else 'random initialization explicitly requested')}")
    if pretrained_initialization:
        logger.log("Torchvision may retrieve official pretrained ResNet-18 weights if they are not cached locally.")
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
            epoch_started = time.perf_counter()
            epoch_model_state = _cpu_state(model.state_dict(), torch)
            epoch_optimizer_state = _cpu_state(optimizer.state_dict(), torch)
            epoch_scheduler_state = scheduler.state_dict()
            epoch_scaler_state = scaler.state_dict()
            epoch_torch_rng = torch.get_rng_state()
            epoch_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            epoch_python_rng = random.getstate()
            epoch_numpy_rng = np.random.get_state()
            while True:
                train_loader, validation_loader = make_loaders(current_batch_size, epoch)
                try:
                    train_loss, train_metrics, train_seconds = run_epoch(
                        model, train_loader, loss_fn, optimizer, device, training=True, amp_enabled=amp_enabled,
                        scaler=scaler, epoch=epoch, epochs=args.epochs, tqdm_class=tqdm,
                    )
                    val_loss, val_metrics, val_seconds = run_epoch(
                        model, validation_loader, loss_fn, optimizer, device, training=False, amp_enabled=amp_enabled,
                        scaler=scaler, epoch=epoch, epochs=args.epochs, tqdm_class=tqdm,
                    )
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or current_batch_size <= 1:
                        raise
                    current_batch_size = max(1, current_batch_size // 2)
                    memory_kind = "CUDA" if device.type == "cuda" else "host"
                    logger.log(f"{memory_kind} out of memory in epoch {epoch}; restoring epoch-start state and retrying with batch size {current_batch_size}.")
                    model.load_state_dict(epoch_model_state)
                    optimizer.load_state_dict(epoch_optimizer_state)
                    scheduler.load_state_dict(epoch_scheduler_state)
                    scaler.load_state_dict(epoch_scaler_state)
                    torch.set_rng_state(epoch_torch_rng)
                    if epoch_cuda_rng is not None:
                        torch.cuda.set_rng_state_all(epoch_cuda_rng)
                        torch.cuda.empty_cache()
                    random.setstate(epoch_python_rng)
                    np.random.set_state(epoch_numpy_rng)
            completed_epochs = epoch
            scheduler.step(val_loss)
            memory = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None
            epoch_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": val_loss,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_precision_ai_generated": val_metrics["precision_ai_generated"],
                "validation_recall_ai_generated": val_metrics["recall_ai_generated"],
                "validation_f1_ai_generated": val_metrics["f1_ai_generated"],
                "validation_per_class": val_metrics["per_class"],
                "validation_confusion_matrix": val_metrics["confusion_matrix"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": time.perf_counter() - epoch_started,
                "train_duration_seconds": train_seconds,
                "validation_duration_seconds": val_seconds,
                "gpu_peak_memory_gib": memory,
                "batch_size": current_batch_size,
            }
            history.append(epoch_row)
            logger.log(
                f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_accuracy={train_metrics['accuracy']:.4f} "
                f"val_loss={val_loss:.4f} val_accuracy={val_metrics['accuracy']:.4f} "
                f"val_precision={val_metrics['precision_ai_generated']:.4f} val_recall={val_metrics['recall_ai_generated']:.4f} "
                f"val_f1={val_metrics['f1_ai_generated']:.4f} duration={epoch_row['duration_seconds']:.1f}s "
                f"gpu_peak={f'{memory:.2f}GiB' if memory is not None else 'n/a'}"
            )
            last_checkpoint_path = last_checkpoint
            last_metadata = {
                "architecture": "resnet18", "class_names": list(CLASS_NAMES),
                "image_size": args.image_size, "resize_mode": "pad",
                "mean": DEFAULT_MEAN, "std": DEFAULT_STD,
                "epoch": epoch, "run_id": run_id,
                "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
            }
            last_temp = last_checkpoint_path.with_suffix(last_checkpoint_path.suffix + ".tmp")
            torch.save({**last_metadata, "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(), "history": history}, last_temp)
            last_temp.replace(last_checkpoint_path)
            history_path.write_text(json.dumps({"run_id": run_id, "epoch_metrics": history}, indent=2), encoding="utf-8")
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                epochs_without_improvement = 0
                best_metadata = {
                    "architecture": "resnet18",
                    "class_names": list(CLASS_NAMES),
                    "image_size": args.image_size,
                    "resize_mode": "pad",
                    "mean": DEFAULT_MEAN,
                    "std": DEFAULT_STD,
                    "pretrained_initialization": str(args.init_checkpoint.resolve()) if args.init_checkpoint else ("torchvision ResNet18_Weights.DEFAULT" if pretrained_initialization else "not pretrained"),
                    "dataset_identifier": dataset_metadata.get("dataset", "manifest source not recorded"),
                    "dataset_revision": dataset_metadata.get("revision", "not recorded in manifest metadata"),
                    "seed": args.seed,
                    "epochs_completed": completed_epochs,
                    "best_checkpoint_epoch": epoch,
                    "best_validation_metric": {"name": "validation_loss", "value": best_validation_loss},
                    "validation_accuracy": val_metrics["accuracy"],
                    "training_timestamp_utc": started_at,
                    "run_id": run_id,
                    "training_device": str(device),
                    "gpu_name": gpu_name,
                }
                checkpoint = {**best_metadata, "state_dict": model.state_dict()}
                temp_checkpoint = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
                torch.save(checkpoint, temp_checkpoint)
                temp_checkpoint.replace(candidate_path)
                logger.log(f"Updated this run's best validation-loss candidate: {candidate_path.name}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stopping_patience:
                    logger.log(f"Early stopping after {epoch} epochs; validation loss did not improve for {args.early_stopping_patience} consecutive epochs.")
                    break
    except KeyboardInterrupt:
        logger.log(f"Training interrupted after {completed_epochs} completed epoch(s); existing best checkpoint was preserved.")
        history_path.write_text(json.dumps({"run_id": run_id, "status": "interrupted", "epoch_metrics": history}, indent=2), encoding="utf-8")
        return 130
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            memory_kind = "CUDA" if device.type == "cuda" else "host"
            logger.log(f"{memory_kind} out of memory at batch size 1. Reduce model/input memory use and rerun; any previously saved best checkpoint is preserved.")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return 2
        raise
    if best_metadata is None or not candidate_path.is_file():
        logger.log("No finite validation-loss checkpoint was produced; the prior checkpoint was left untouched.")
        return 2
    best_metadata["epochs_completed"] = completed_epochs
    final_candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    final_candidate.update(best_metadata)
    final_temp = candidate_path.with_suffix(candidate_path.suffix + ".final.tmp")
    torch.save(final_candidate, final_temp)
    final_temp.replace(candidate_path)
    candidate_path.replace(checkpoint_path)
    metadata_path.write_text(json.dumps(best_metadata, indent=2), encoding="utf-8")
    logger.log(f"Promoted this run's best checkpoint to: {checkpoint_path}")
    history_path.write_text(json.dumps({"run_id": run_id, "status": "completed", "epoch_metrics": history}, indent=2), encoding="utf-8")
    logger.log(f"Training finished after {completed_epochs} epochs. The held-out test split was not used for model selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
