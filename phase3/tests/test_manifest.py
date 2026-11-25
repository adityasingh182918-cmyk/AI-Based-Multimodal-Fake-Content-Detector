import json

import pytest
from PIL import Image

from phase3.training.encode_pairs import validate_group_split


def _write_manifest(root, image_root, split, split_index):
    rows = []
    for class_index, (image_origin, text_origin) in enumerate([
        ("REAL", "HUMAN_WRITTEN"), ("AI_GENERATED", "AI_GENERATED"),
        ("REAL", "AI_GENERATED"), ("AI_GENERATED", "HUMAN_WRITTEN"),
    ]):
        rel = f"{split}/{class_index}.png"
        target = image_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (split_index * 40, class_index * 40, 20)).save(target)
        rows.append({"sample_id": f"{split}-{class_index}", "group_id": f"{split}-group-{class_index}",
                     "split": split, "image_path": rel, "text": f"Unique text for {split} example {class_index}.",
                     "image_origin": image_origin, "text_origin": text_origin, "source": "fixture",
                     "dataset_provenance": "synthetic test fixture", "relation": "paired fixture"})
    (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def _fixture(tmp_path):
    data_dir, image_root = tmp_path / "manifests", tmp_path / "images"
    data_dir.mkdir()
    for index, split in enumerate(("train", "validation", "test")):
        _write_manifest(data_dir, image_root, split, index)
    return data_dir, image_root


def test_valid_four_way_grouped_manifests_pass(tmp_path):
    data_dir, image_root = _fixture(tmp_path)
    validate_group_split(data_dir, image_root)


def test_group_leakage_across_splits_fails(tmp_path):
    data_dir, image_root = _fixture(tmp_path)
    rows = [json.loads(line) for line in (data_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["group_id"] = "train-group-0"
    (data_dir / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Leakage"):
        validate_group_split(data_dir, image_root)


def test_incomplete_class_combinations_fail(tmp_path):
    data_dir, image_root = _fixture(tmp_path)
    rows = [json.loads(line) for line in (data_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
    rows.pop()
    (data_dir / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="All four"):
        validate_group_split(data_dir, image_root)


def test_invalid_image_fails_before_feature_extraction(tmp_path):
    data_dir, image_root = _fixture(tmp_path)
    bad = image_root / "train/0.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Invalid image"):
        validate_group_split(data_dir, image_root)
