"""Validate extracted GenImage data and write reproducible, path-only manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

CLASS_FOLDERS = {"ai": "AI_GENERATED", "nature": "REAL"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MANIFEST_FIELDS = ["path", "label", "split", "source_split", "sha256"]
REJECTED_FIELDS = ["path", "reason"]


class DatasetValidationError(ValueError):
    """Raised when input data cannot safely produce all three splits."""


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise DatasetValidationError(f"Required class directory is missing: {directory}")
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def validate_dataset_structure(root: Path) -> dict[str, dict[str, list[Path]]]:
    """Require train and val folders with ai and nature class subfolders."""
    root = root.resolve()
    data: dict[str, dict[str, list[Path]]] = {}
    for source_split in ("train", "val"):
        data[source_split] = {}
        for folder, label in CLASS_FOLDERS.items():
            files = _image_files(root / source_split / folder)
            data[source_split][label] = files
            if not files:
                raise DatasetValidationError(f"No supported images found in {root / source_split / folder}")
    for label in CLASS_FOLDERS.values():
        if len(data["train"][label]) < 2:
            raise DatasetValidationError(f"Class {label} needs at least 2 train images to create train and validation splits")
        if len(data["val"][label]) < 1:
            raise DatasetValidationError(f"Class {label} needs at least 1 held-out image under val/")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise DatasetValidationError(f"Unreadable image {path}: {exc}") from exc


def create_manifests(
    root: Path,
    output_dir: Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    max_images_per_class: int | None = None,
    skip_corrupt: bool = False,
    force: bool = False,
) -> dict[str, int]:
    if not 0 < validation_fraction < 1:
        raise DatasetValidationError("validation_fraction must be between 0 and 1")
    if max_images_per_class is not None and max_images_per_class < 3:
        raise DatasetValidationError("max_images_per_class must be at least 3")
    root = root.resolve()
    data = validate_dataset_structure(root)
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    hash_labels: dict[str, str] = {}
    hash_splits: dict[str, str] = {}

    for label in ("REAL", "AI_GENERATED"):
        source_groups = {"train": data["train"][label], "val": data["val"][label]}
        if max_images_per_class:
            # Cap total class size consistently across both official source partitions.
            combined = [(part, path) for part, paths in source_groups.items() for path in paths]
            if len(combined) > max_images_per_class:
                chosen = set(rng.sample(range(len(combined)), max_images_per_class))
                combined = [item for index, item in enumerate(combined) if index in chosen]
                source_groups = {part: [p for src, p in combined if src == part] for part in ("train", "val")}
            if not source_groups["train"] or not source_groups["val"]:
                raise DatasetValidationError(f"Cap leaves class {label} without a train or held-out val example")

        for source_split, paths in source_groups.items():
            for path in paths:
                try:
                    _verify_image(path)
                except DatasetValidationError as exc:
                    if not skip_corrupt:
                        raise
                    rejected.append({"path": path.relative_to(root).as_posix(), "reason": str(exc)})
                    continue
                digest = _sha256(path)
                previous_label = hash_labels.get(digest)
                if previous_label and previous_label != label:
                    raise DatasetValidationError(f"Identical image bytes have conflicting labels: {path} ({previous_label} vs {label})")
                assigned = "test" if source_split == "val" else "train"
                previous_split = hash_splits.get(digest)
                if previous_split and previous_split != assigned:
                    raise DatasetValidationError(f"Duplicate image leaks across train/test partitions: {path}")
                hash_labels[digest] = label
                hash_splits[digest] = assigned
                rows.append({
                    "path": path.relative_to(root).as_posix(),
                    "label": label,
                    "split": assigned,
                    "source_split": source_split,
                    "sha256": digest,
                })

    # Deduplicate repeated identical files inside the same class and source partition.
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        unique.setdefault(row["sha256"], row)
    rows = list(unique.values())
    for label in ("REAL", "AI_GENERATED"):
        train_rows = [r for r in rows if r["label"] == label and r["split"] == "train"]
        test_rows = [r for r in rows if r["label"] == label and r["split"] == "test"]
        if len(train_rows) < 2 or not test_rows:
            raise DatasetValidationError(f"After deduplication, {label} needs >=2 train images and >=1 test image")
        rng.shuffle(train_rows)
        validation_count = max(1, round(len(train_rows) * validation_fraction))
        if validation_count >= len(train_rows):
            raise DatasetValidationError(f"Not enough unique {label} training images for the requested validation split")
        validation_hashes = {r["sha256"] for r in train_rows[:validation_count]}
        for row in rows:
            if row["label"] == label and row["split"] == "train" and row["sha256"] in validation_hashes:
                row["split"] = "validation"

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {"manifest": output_dir / "manifest.csv", "rejected": output_dir / "rejected.csv"}
    output_paths.update({name: output_dir / f"{name}.csv" for name in ("train", "validation", "test")})
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not force:
        raise DatasetValidationError("Manifest output exists; choose another --output-dir or pass --force: " + ", ".join(map(str, existing)))
    counts: Counter[str] = Counter()
    for split, path in output_paths.items():
        subset = rows if split == "manifest" else rejected if split == "rejected" else [row for row in rows if row["split"] == split]
        if split != "manifest":
            counts[split] = len(subset)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=output_dir, suffix=".tmp") as temp:
            writer = csv.DictWriter(temp, fieldnames=REJECTED_FIELDS if split == "rejected" else MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(subset)
            temp_path = Path(temp.name)
        temp_path.replace(path)
    return dict(counts)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Extracted root containing train/ai, train/nature, val/ai, val/nature")
    parser.add_argument("--output-dir", type=Path, default=Path("data/image/manifests"))
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of source train files used for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-class", type=int, help="Optional reproducible cap across source train+val files per class")
    parser.add_argument("--skip-corrupt", action="store_true", help="Exclude unreadable images and record their paths in rejected.csv")
    parser.add_argument("--force", action="store_true", help="Replace existing combined, per-split, and rejected CSVs")
    args = parser.parse_args(argv)
    try:
        counts = create_manifests(
            args.dataset_root, args.output_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            max_images_per_class=args.max_images_per_class,
            skip_corrupt=args.skip_corrupt,
            force=args.force,
        )
    except DatasetValidationError as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Manifest rows written to {args.output_dir.resolve()}: {counts}")
    print("Input images were not copied or modified. Source val/ entries are assigned to the held-out test split.")
    print("Unreadable files are fatal by default; --skip-corrupt explicitly excludes and records them in rejected.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
