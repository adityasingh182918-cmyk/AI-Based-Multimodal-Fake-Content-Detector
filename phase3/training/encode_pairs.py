"""Extract frozen copied-model features from a human-verified paired manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase3.fusion.encoders import FrozenEncoders  # noqa: E402

ORIGINS = {"REAL", "AI_GENERATED"}
TEXT_ORIGINS = {"HUMAN_WRITTEN", "AI_GENERATED"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_split(rows: list[dict[str, Any]], dataset_root: Path, split: str, encoders: FrozenEncoders,
                  output_dir: Path) -> dict[str, Any]:
    image_vectors, text_vectors = [], []
    image_y, text_y, fusion_y = [], [], []
    ids, groups, provenance = [], [], []
    for index, row in enumerate(rows, 1):
        if row.get("split") != split:
            raise ValueError(f"Sample {row.get('sample_id')} is stored in the wrong split manifest")
        sample_id = str(row.get("sample_id", "")).strip()
        group_id = str(row.get("group_id", "")).strip()
        image_origin, text_origin = row.get("image_origin"), row.get("text_origin")
        text = row.get("text")
        image_path = (dataset_root / row.get("image_path", "")).resolve()
        if not sample_id or not group_id or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid IDs/text in {split} row {index}")
        if image_origin not in ORIGINS or text_origin not in TEXT_ORIGINS:
            raise ValueError(f"Invalid or missing modality origin labels in sample {sample_id}")
        if not image_path.is_relative_to(dataset_root.resolve()) or not image_path.is_file():
            raise ValueError(f"Missing or out-of-root image for sample {sample_id}: {image_path}")
        image_bytes = image_path.read_bytes()
        image_feature, image_result = encoders.image(image_bytes)
        text_feature, text_result = encoders.text(text)
        image_vectors.append(image_feature.numpy().astype(np.float32, copy=False))
        text_vectors.append(text_feature)
        image_y.append(int(image_origin == "AI_GENERATED"))
        text_y.append(int(text_origin == "AI_GENERATED"))
        fusion_y.append({("REAL", "HUMAN_WRITTEN"): 0, ("AI_GENERATED", "AI_GENERATED"): 1,
                         ("REAL", "AI_GENERATED"): 2, ("AI_GENERATED", "HUMAN_WRITTEN"): 3}[(image_origin, text_origin)])
        ids.append(sample_id)
        groups.append(group_id)
        provenance.append({"image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                           "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                           "source": row.get("source"), "dataset_provenance": row.get("dataset_provenance"),
                           "relation": row.get("relation"),
                           "frozen_image_prediction": image_result,
                           "frozen_text_prediction": text_result})
        if index % 250 == 0:
            print(f"{split}: extracted {index}/{len(rows)}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"image_features_{split}.npy", np.stack(image_vectors).astype(np.float32))
    text_matrix = sp.vstack(text_vectors, format="csr", dtype=np.float32)
    sp.save_npz(output_dir / f"text_features_{split}.npz", text_matrix)
    np.savez_compressed(output_dir / f"labels_{split}.npz",
                        image_y=np.asarray(image_y, dtype=np.int64), text_y=np.asarray(text_y, dtype=np.int64),
                        fusion_y=np.asarray(fusion_y, dtype=np.int64), sample_ids=np.asarray(ids))
    metadata = {"split": split, "sample_count": len(rows), "image_feature_dim": int(image_vectors[0].shape[0]),
                "text_feature_dim": int(text_matrix.shape[1]), "text_feature_nnz": int(text_matrix.nnz),
                "group_ids": groups, "provenance_and_frozen_predictions": provenance}
    (output_dir / f"metadata_{split}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def validate_group_split(data_dir: Path, image_root: Path) -> None:
    group_splits: dict[str, str] = {}
    sample_ids: set[str] = set()
    content_splits: dict[str, str] = {}
    joint_labels: dict[str, set[int]] = {name: set() for name in ("train", "validation", "test")}
    image_root = image_root.resolve()
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing paired split manifest: {path}")
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty paired dataset split: {split}")
        for line_number, row in enumerate(rows, 1):
            sample_id = str(row.get("sample_id", "")).strip()
            group = str(row.get("group_id", "")).strip()
            image_origin, text_origin = row.get("image_origin"), row.get("text_origin")
            text = row.get("text")
            image_rel = row.get("image_path")
            for field in ("source", "dataset_provenance", "relation"):
                if not str(row.get(field, "")).strip():
                    raise ValueError(f"Missing {field} provenance in {split} row {line_number}")
            if not sample_id or not group or not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing sample/group ID or usable text in {split} row {line_number}")
            if image_origin not in ORIGINS or text_origin not in TEXT_ORIGINS:
                raise ValueError(f"Invalid modality-origin labels in sample {sample_id}")
            if not isinstance(image_rel, str) or not image_rel.strip():
                raise ValueError(f"Missing image_path in sample {sample_id}")
            image_path = (image_root / image_rel).resolve()
            if not image_path.is_relative_to(image_root) or not image_path.is_file():
                raise ValueError(f"Missing or out-of-root image in sample {sample_id}: {image_path}")
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Invalid image in sample {sample_id}: {image_path}") from exc
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id across splits: {sample_id}")
            sample_ids.add(sample_id)
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f"Leakage: group {group!r} appears in {group_splits[group]} and {split}")
            group_splits[group] = split
            image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            text_digest = hashlib.sha256(" ".join(text.split()).casefold().encode("utf-8")).hexdigest()
            for kind, digest in (("image", image_digest), ("normalized text", text_digest)):
                key = f"{kind}:{digest}"
                if key in content_splits and content_splits[key] != split:
                    raise ValueError(f"Exact {kind} duplicate leaks across {content_splits[key]} and {split}: {sample_id}")
                content_splits[key] = split
            joint_labels[split].add({("REAL", "HUMAN_WRITTEN"): 0, ("AI_GENERATED", "AI_GENERATED"): 1,
                                     ("REAL", "AI_GENERATED"): 2, ("AI_GENERATED", "HUMAN_WRITTEN"): 3}
                                    [(image_origin, text_origin)])
    for split, labels in joint_labels.items():
        if labels != set(range(4)):
            raise ValueError(f"All four image/text-origin combinations are required in {split}; present={sorted(labels)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("phase3/data/paired"))
    parser.add_argument("--image-root", type=Path, default=Path("phase3/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase3/data/features"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    validate_group_split(args.data_dir, args.image_root)
    encoders = FrozenEncoders(args.device)
    for split in ("train", "validation", "test"):
        rows = read_jsonl(args.data_dir / f"{split}.jsonl")
        extract_split(rows, args.image_root, split, encoders, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
