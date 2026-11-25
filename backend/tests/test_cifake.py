from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_cifake import CIFAKEError, take_balanced, write_subset
from training.data import ManifestImageDataset
from training.train_image_model import validate_manifest


def _record(label: int, value: int) -> dict:
    return {
        "label": label,
        "image": Image.new("RGB", (8, 8), (value * 13 % 256, value * 47 % 256, value * 83 % 256)),
    }


def test_cifake_labels_use_verified_hugging_face_mapping() -> None:
    selected = take_balanced([_record(0, 10), _record(1, 20)], 1, ["FAKE", "REAL"])

    assert selected["AI_GENERATED"][0]["label"] == 0
    assert selected["REAL"][0]["label"] == 1


def test_small_subset_keeps_official_test_split_and_writes_pipeline_manifests(tmp_path: Path) -> None:
    train = [_record(label, 10 + label * 20 + i) for label in (0, 1) for i in range(4)]
    test = [_record(label, 100 + label * 20 + i) for label in (0, 1) for i in range(2)]
    root = tmp_path / "cifake"

    counts = write_subset(
        train, test, root, label_names=["FAKE", "REAL"],
        train_per_class=2, validation_per_class=2, test_per_class=2, seed=9,
    )

    assert counts == {"train": 4, "validation": 4, "test": 4}
    manifest = (root / "manifests" / "manifest.csv").read_text(encoding="utf-8")
    assert "AI_GENERATED" in manifest and "REAL" in manifest
    assert "test,train" not in manifest
    assert manifest.count(",test,test,") == 4
    assert len(list((root / "images" / "test").rglob("*.jpg"))) == 4
    validate_manifest(root / "manifests" / "manifest.csv")


def test_prepared_images_load_through_existing_pytorch_adapter(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    train = [_record(label, 10 + label * 20 + i) for label in (0, 1) for i in range(4)]
    test = [_record(label, 100 + label * 20 + i) for label in (0, 1) for i in range(2)]
    root = tmp_path / "cifake"
    write_subset(train, test, root, label_names=["FAKE", "REAL"], train_per_class=2,
                 validation_per_class=2, test_per_class=2, seed=9)

    dataset = ManifestImageDataset(root, root / "manifests" / "manifest.csv", "test")
    image, label = dataset[0]

    assert image.mode == "RGB"
    assert label.item() in (0, 1)
    assert isinstance(label, torch.Tensor)


def test_cifake_loader_rejects_wrong_label_schema_and_short_stream() -> None:
    with pytest.raises(CIFAKEError, match="Expected CIFAKE labels"):
        take_balanced([_record(0, 22)], 1, ["OTHER"])
    with pytest.raises(CIFAKEError, match="ended before obtaining"):
        take_balanced([_record(1, 23)], 1, ["FAKE", "REAL"])


def test_cifake_loader_records_corrupt_images_while_replacing_them() -> None:
    corrupt = {"label": 0, "image": b"not decoded"}
    rejected = []
    records = [corrupt, _record(0, 42), _record(1, 88)]

    selected = take_balanced(records, 1, ["FAKE", "REAL"], rejected=rejected, source_split="train")

    assert len(selected["AI_GENERATED"]) == 1
    assert len(selected["REAL"]) == 1
    assert len(rejected) == 1
    assert rejected[0]["source_split"] == "train"


def test_cifake_sampling_skips_duplicate_decoded_images() -> None:
    duplicate = _record(0, 42)
    selected = take_balanced(
        [duplicate, _record(0, 42), _record(0, 43), _record(0, 44), _record(1, 88), _record(1, 89)],
        2,
        ["FAKE", "REAL"],
        seen_image_hashes=set(),
    )

    assert len(selected["AI_GENERATED"]) == 2
    assert len(selected["REAL"]) == 2
    fake_pixels = {row["image"].getpixel((0, 0)) for row in selected["AI_GENERATED"]}
    assert len(fake_pixels) == 2


def test_cifake_preparation_refuses_nonempty_output(tmp_path: Path) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    (root / "keep.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(CIFAKEError, match="not empty"):
        write_subset([], [], root, label_names=["FAKE", "REAL"], train_per_class=1,
                     validation_per_class=1, test_per_class=1)

    assert (root / "keep.txt").read_text(encoding="utf-8") == "preserve"
