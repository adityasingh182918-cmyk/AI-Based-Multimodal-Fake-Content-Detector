"""Combine the unchanged CIFAKE splits with user-labelled REAL training photos.

This intentionally does not make the four user photos part of validation or test.
The original CIFAKE validation and test rows remain byte-for-byte represented in
their original splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CIFAKE = ROOT / "data/image/cifake_ready"
DEFAULT_USER = ROOT / "data/image_detection/user_real/train"
DEFAULT_OUTPUT = ROOT / "data/image_detection/combined"


def phash(path: Path) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    # A compact, dependency-free perceptual hash. Compare the low-frequency 8x8
    # DCT coefficients; omit DC when choosing the median threshold.
    n = 32
    basis = np.cos(np.pi * (2 * np.arange(n)[:, None] + 1) * np.arange(n)[None, :] / (2 * n))
    basis[:, 0] *= 1 / np.sqrt(2)
    dct = (2 / n) * basis.T @ pixels @ basis
    low = dct[:8, :8].reshape(-1)
    threshold = np.median(low[1:])
    value = 0
    for bit in low:
        value = (value << 1) | int(bit > threshold)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def prepare(cifake_root: Path, user_root: Path, output: Path, *, project_root: Path = ROOT) -> dict[str, object]:
    project_root = project_root.resolve()
    cifake_root = cifake_root.resolve()
    user_root = user_root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output folder: {output}")
    cifake_manifest = cifake_root / "manifests/manifest.csv"
    user_manifest = user_root / "manifest.csv"
    with cifake_manifest.open("r", encoding="utf-8", newline="") as stream:
        base_rows = list(csv.DictReader(stream))
    with user_manifest.open("r", encoding="utf-8", newline="") as stream:
        user_rows = list(csv.DictReader(stream))
    if not base_rows or not user_rows:
        raise ValueError("CIFAKE and user manifests must both contain rows")

    rows: list[dict[str, str]] = []
    exact_hashes: dict[str, str] = {}
    try:
        cifake_prefix = cifake_root.relative_to(project_root)
        user_prefix = user_root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("CIFAKE and user image folders must be inside project_root") from exc
    for row in base_rows:
        rel = cifake_prefix / Path(row["path"])
        path = project_root / rel
        if not path.is_file():
            raise FileNotFoundError(f"CIFAKE manifest image is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get("sha256") and digest != row["sha256"]:
            raise ValueError(f"CIFAKE hash mismatch: {path}")
        exact_hashes[digest] = "CIFAKE"
        rows.append({"path": rel.as_posix(), "label": row["label"], "split": row["split"],
                     "source_split": row.get("source_split", ""), "source": "CIFAKE",
                     "sha256": digest, "group_id": f"cifake:{digest}"})

    user_phashes: list[tuple[str, int]] = []
    skipped_duplicates = 0
    for row in user_rows:
        path = user_root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"User image is missing: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"Invalid user image {path.name}: {exc}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in exact_hashes:
            skipped_duplicates += 1
            continue
        hash_value = phash(path)
        if any(hamming(hash_value, prior_hash) <= 4 for _, prior_hash in user_phashes):
            skipped_duplicates += 1
            continue
        user_phashes.append((row["path"], hash_value))
        rel = (user_prefix / row["path"]).as_posix()
        rows.append({"path": rel, "label": "REAL", "split": "train", "source_split": "user_provided",
                     "source": "user_provided", "sha256": digest, "group_id": f"user-real:{digest}"})
        exact_hashes[digest] = "user_provided"

    splits = ("train", "validation", "test")
    expected = {split: sum(row["split"] == split for row in base_rows) for split in splits}
    observed = {split: sum(row["split"] == split for row in rows if row["source"] == "CIFAKE") for split in splits}
    if observed != expected:
        raise ValueError(f"Original CIFAKE splits changed: expected {expected}, found {observed}")
    counts = {split: {label: sum(row["split"] == split and row["label"] == label for row in rows)
                      for label in ("REAL", "AI_GENERATED")} for split in splits}
    if any(not all(counts[split].values()) for split in splits):
        raise ValueError(f"Combined manifest must contain both labels in each split: {counts}")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "split", "source_split", "source", "sha256", "group_id"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": "CIFAKE plus four user-labelled REAL supplemental photos",
        "label_provenance": "user photos labelled REAL by the user; provenance not independently verified",
        "cifake_manifest": cifake_manifest.relative_to(project_root).as_posix(),
        "cifake_split_counts_preserved": observed,
        "user_labelled_train_images": len(user_phashes),
        "user_duplicates_skipped": skipped_duplicates,
        "counts_by_split_and_class": counts,
        "seed": 42,
        "test_policy": "user images are training-only; CIFAKE test remains held out and unchanged",
        "limitations": ["CIFAKE images are 32x32", "four user photos are too few to establish camera-domain generalization", "no external licensed multi-generator dataset was used"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifake-root", type=Path, default=DEFAULT_CIFAKE)
    parser.add_argument("--user-root", type=Path, default=DEFAULT_USER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = prepare(args.cifake_root, args.user_root, args.output)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
