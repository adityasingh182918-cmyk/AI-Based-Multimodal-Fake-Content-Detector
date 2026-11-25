"""Stream a small, reproducible CIFAKE subset into the image pipeline format.

This reads the verified Hugging Face Parquet dataset lazily and saves only the
requested examples. The default produces 800 train, 100 validation, and 100
test images per class (2,000 total). Source test examples remain test-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

DATASET_ID = "dragonintelligence/CIFAKE-image-dataset"
DATASET_REVISION = "6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31"
CLASS_MAP = {"REAL": "REAL", "FAKE": "AI_GENERATED"}
FIELDS = ["path", "label", "split", "source_split", "sha256"]


class CIFAKEError(ValueError):
    """Raised when source rows or generated output do not meet expectations."""


def _label_name(raw: Any, label_names: list[str]) -> str:
    if isinstance(raw, int):
        if raw < 0 or raw >= len(label_names):
            raise CIFAKEError(f"Unknown CIFAKE label index: {raw}")
        name = label_names[raw].upper()
    else:
        name = str(raw).upper()
    if name not in CLASS_MAP:
        raise CIFAKEError(f"Expected CIFAKE labels FAKE/REAL, got {name!r}")
    return CLASS_MAP[name]


def take_balanced(records: Iterable[dict[str, Any]], per_class: int, label_names: list[str], *,
                  rejected: list[dict[str, Any]] | None = None, source_split: str = "unknown",
                  seen_image_hashes: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Read until each mapped class has the requested number of rows."""
    if per_class < 1:
        raise CIFAKEError("per_class must be positive")
    if seen_image_hashes is None:
        seen_image_hashes = set()
    chosen = {"REAL": [], "AI_GENERATED": []}
    for source_index, record in enumerate(records):
        label = _label_name(record.get("label"), label_names)
        try:
            image = record.get("image")
            if not isinstance(image, Image.Image):
                raise CIFAKEError("Image field did not decode to a PIL image")
            image.load()
            normalized = image.convert("RGB")
            image_hash = hashlib.sha256(
                f"{normalized.width}x{normalized.height}:".encode("ascii") + normalized.tobytes()
            ).hexdigest()
        except (CIFAKEError, UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            if rejected is None:
                raise CIFAKEError(f"Corrupt/unsupported image in source {source_split} at row {source_index}: {exc}") from exc
            rejected.append({"source_split": source_split, "source_index": str(source_index), "reason": str(exc)})
            continue
        if image_hash in seen_image_hashes:
            continue
        seen_image_hashes.add(image_hash)
        if len(chosen[label]) < per_class:
            chosen[label].append(record)
        if all(len(rows) == per_class for rows in chosen.values()):
            break
    missing = {label: len(rows) for label, rows in chosen.items() if len(rows) != per_class}
    if missing:
        raise CIFAKEError(f"Source split ended before obtaining {per_class} rows per class: {missing}")
    return chosen


def _save_image(record: dict[str, Any], path: Path) -> tuple[str, tuple[int, int]]:
    image = record.get("image")
    if not isinstance(image, Image.Image):
        raise CIFAKEError("Dataset row image field did not decode to a PIL image")
    image = image.convert("RGB")
    if image.width < 1 or image.height < 1:
        raise CIFAKEError("Dataset row contains an empty image")
    dimensions = (image.width, image.height)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)
    try:
        with Image.open(path) as check:
            check.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CIFAKEError(f"Saved image could not be decoded: {path}: {exc}") from exc
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, dimensions


def write_subset(
    train_records: Iterable[dict[str, Any]],
    test_records: Iterable[dict[str, Any]],
    output_root: Path,
    *,
    label_names: list[str],
    train_per_class: int = 800,
    validation_per_class: int = 100,
    test_per_class: int = 100,
    seed: int = 42,
) -> dict[str, int]:
    """Save balanced rows from source train/test streams and write manifests."""
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise CIFAKEError(f"Output directory is not empty; choose another path: {output_root}")
    rejected: list[dict[str, str]] = []
    # Validate and decode rows before writing any of the selected images.
    selected_hashes: set[str] = set()
    train_selected = take_balanced(train_records, train_per_class + validation_per_class, label_names,
                                   rejected=rejected, source_split="train", seen_image_hashes=selected_hashes)
    test_selected = take_balanced(test_records, test_per_class, label_names,
                                  rejected=rejected, source_split="test", seen_image_hashes=selected_hashes)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    dimensions: Counter[str] = Counter()

    for label in ("REAL", "AI_GENERATED"):
        for index, record in enumerate(train_selected[label]):
            split = "train" if index < train_per_class else "validation"
            source_split = "train"
            path = output_root / "images" / split / label / f"{index:05d}.jpg"
            digest, image_dimensions = _save_image(record, path)
            dimensions[f"{image_dimensions[0]}x{image_dimensions[1]}"] += 1
            previous = seen.get(digest)
            if previous is not None:
                raise CIFAKEError(f"Duplicate image bytes cross or repeat selected rows: {previous} and {path}")
            seen[digest] = str(path)
            rows.append({"path": path.relative_to(output_root).as_posix(), "label": label, "split": split,
                         "source_split": source_split, "sha256": digest})
        for index, record in enumerate(test_selected[label]):
            split = "test"
            source_split = "test"
            path = output_root / "images" / split / label / f"{index:05d}.jpg"
            digest, image_dimensions = _save_image(record, path)
            dimensions[f"{image_dimensions[0]}x{image_dimensions[1]}"] += 1
            previous = seen.get(digest)
            if previous is not None:
                raise CIFAKEError(f"Duplicate image bytes cross or repeat selected rows: {previous} and {path}")
            seen[digest] = str(path)
            rows.append({"path": path.relative_to(output_root).as_posix(), "label": label, "split": split,
                         "source_split": source_split, "sha256": digest})

    # Deterministic row ordering independent of the source stream's class order.
    rows.sort(key=lambda row: (row["split"], row["label"], row["path"]))
    for name in ("manifest", "train", "validation", "test"):
        target = manifest_dir / f"{name}.csv"
        subset = rows if name == "manifest" else [row for row in rows if row["split"] == name]
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(subset)
    with (manifest_dir / "rejected.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_split", "source_index", "reason"])
        writer.writeheader()
        writer.writerows(rejected)
    metadata = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "seed": seed,
        "requested_per_class": {"train": train_per_class, "validation": validation_per_class, "test": test_per_class},
        "rejected_corrupt_images": len(rejected),
        "image_dimensions": dict(sorted(dimensions.items())),
        "label_mapping": {"REAL": "REAL", "FAKE": "AI_GENERATED"},
        "counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
        "counts_by_split_and_class": {
            split: {label: sum(row["split"] == split and row["label"] == label for row in rows)
                    for label in ("REAL", "AI_GENERATED")}
            for split in ("train", "validation", "test")
        },
        "source_splits": {"train": "train", "validation": "train", "test": "test"},
    }
    (manifest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata["counts"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/image/cifake_ready"))
    parser.add_argument("--train-per-class", type=int, default=800)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.train_per_class, args.validation_per_class, args.test_per_class) < 1:
        parser.error("all per-class counts must be positive")
    try:
        from datasets import load_dataset
        from tqdm.auto import tqdm
    except ImportError:
        print("Missing optional dependency 'datasets'. Install it with: python -m pip install -r requirements-ml.txt", file=sys.stderr)
        return 2
    try:
        train = load_dataset(DATASET_ID, split="train", streaming=True, revision=DATASET_REVISION)
        test = load_dataset(DATASET_ID, split="test", streaming=True, revision=DATASET_REVISION)
        for split_name, stream in (("train", train), ("test", test)):
            if "label" not in stream.features or "image" not in stream.features:
                raise CIFAKEError(f"{split_name} split is missing image/label features")
            names = getattr(stream.features["label"], "names", None)
            if not names or {str(name).upper() for name in names} != set(CLASS_MAP):
                raise CIFAKEError(f"Unexpected label feature in {split_name}: {names}")
        label_names = list(train.features["label"].names)
        print(f"Streaming only enough rows for {args.train_per_class + args.validation_per_class}/class from source train and {args.test_per_class}/class from source test.")
        counts = write_subset(
            tqdm(train.shuffle(seed=args.seed, buffer_size=10_000), total=None, desc="Read source train", unit="image"),
            tqdm(test.shuffle(seed=args.seed + 1, buffer_size=10_000), total=None, desc="Read source test", unit="image"),
            args.output_root,
            label_names=label_names,
            train_per_class=args.train_per_class,
            validation_per_class=args.validation_per_class,
            test_per_class=args.test_per_class,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"CIFAKE subset preparation failed: {exc}", file=sys.stderr)
        return 2
    print(f"CIFAKE manifests and selected images written under {args.output_root.resolve()}: {counts}")
    print(f"Source: {DATASET_ID}@{DATASET_REVISION}; source test rows are test-only.")
    metadata = json.loads((args.output_root / "manifests" / "metadata.json").read_text(encoding="utf-8"))
    print(f"Counts by split/class: {metadata['counts_by_split_and_class']}")
    print(f"Observed image dimensions: {metadata['image_dimensions']}; model resize: 224x224; normalization: ImageNet mean/std")
    print(f"Corrupt/unsupported images skipped while sampling: {metadata['rejected_corrupt_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
