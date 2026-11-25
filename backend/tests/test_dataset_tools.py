from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_dataset import DatasetValidationError, create_manifests, validate_dataset_structure


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path, format="PNG")


def _make_dataset(root: Path) -> None:
    for class_name, base in (("ai", 20), ("nature", 100)):
        _write_image(root / "train" / class_name / "train-a.png", (base, 0, 0))
        _write_image(root / "train" / class_name / "train-b.png", (base, 0, 1))
        _write_image(root / "val" / class_name / "heldout.png", (base, 1, 0))


def test_validation_requires_both_documented_classes(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    (tmp_path / "train" / "ai" / "train-a.png").unlink()
    (tmp_path / "train" / "ai" / "train-b.png").unlink()

    with pytest.raises(DatasetValidationError, match="No supported images"):
        validate_dataset_structure(tmp_path)


def test_manifest_assigns_disjoint_reproducible_splits(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_dataset(root)
    output = tmp_path / "manifests"

    counts = create_manifests(root, output, seed=12)
    assert counts == {"train": 2, "validation": 2, "test": 2, "rejected": 0}
    assert (output / "manifest.csv").is_file()
    assert (output / "train.csv").is_file()
    assert (output / "validation.csv").is_file()
    assert (output / "test.csv").is_file()
    with pytest.raises(DatasetValidationError, match="output exists"):
        create_manifests(root, output, seed=12)


def test_corrupt_image_is_rejected(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    corrupt = tmp_path / "train" / "ai" / "train-a.png"
    corrupt.write_bytes(b"not an image")

    with pytest.raises(DatasetValidationError, match="Unreadable image"):
        create_manifests(tmp_path, tmp_path / "out")


def test_corrupt_image_can_be_explicitly_excluded_and_recorded(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    _write_image(tmp_path / "train" / "ai" / "extra.png", (20, 0, 2))
    corrupt = tmp_path / "train" / "ai" / "train-a.png"
    corrupt.write_bytes(b"not an image")

    counts = create_manifests(tmp_path, tmp_path / "out", skip_corrupt=True)

    assert counts["rejected"] == 1
    assert "train/ai/train-a.png" in (tmp_path / "out" / "rejected.csv").read_text(encoding="utf-8")


def test_duplicate_across_train_and_heldout_is_rejected(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    duplicate_bytes = (tmp_path / "train" / "ai" / "train-a.png").read_bytes()
    (tmp_path / "val" / "ai" / "heldout.png").write_bytes(duplicate_bytes)

    with pytest.raises(DatasetValidationError, match="leaks across train/test"):
        create_manifests(tmp_path, tmp_path / "out")
