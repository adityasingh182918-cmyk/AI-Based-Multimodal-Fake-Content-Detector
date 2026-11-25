import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from scripts.import_user_real_images import import_images
from scripts.prepare_realworld_manifest import prepare


def _write_image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path, format="JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cifake(root: Path) -> None:
    images = root / "images"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for label_index, label in enumerate(("REAL", "AI_GENERATED")):
            relative = Path("images") / split / label / "one.jpg"
            path = images / split / label / "one.jpg"
            digest = _write_image(path, (40 + split_index * 40, 30 + label_index * 80, 100))
            rows.append({"path": relative.as_posix(), "label": label, "split": split,
                         "source_split": split, "sha256": digest})
    with (manifests / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_import_user_images_preserves_files_and_skips_exact_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    _write_image(source, (80, 120, 160))
    destination = tmp_path / "train"

    rows = import_images([source, source], destination)

    assert len(rows) == 1
    assert rows[0]["label"] == "REAL"
    assert rows[0]["split"] == "train"
    assert rows[0]["label_provenance"].startswith("user_labelled_real")
    assert (destination / rows[0]["path"]).read_bytes() == source.read_bytes()


def test_prepare_keeps_cifake_splits_and_adds_user_images_only_to_train(tmp_path: Path) -> None:
    cifake = tmp_path / "data/image/cifake_ready"
    user = tmp_path / "data/image_detection/user_real/train"
    output = tmp_path / "data/image_detection/combined"
    _write_cifake(cifake)
    user.mkdir(parents=True)
    user_image = user / "user_real_01.jpg"
    user_hash = _write_image(user_image, (220, 30, 90))
    with (user / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "split", "sha256"])
        writer.writeheader()
        writer.writerow({"path": user_image.name, "label": "REAL", "split": "train", "sha256": user_hash})

    result = prepare(cifake, user, output, project_root=tmp_path)

    with (output / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert result["user_labelled_train_images"] == 1
    assert result["cifake_split_counts_preserved"] == {"train": 2, "validation": 2, "test": 2}
    assert sum(row["source"] == "user_provided" and row["split"] == "train" for row in rows) == 1
    assert sum(row["source"] == "user_provided" and row["split"] == "test" for row in rows) == 0
    assert sum(row["source"] == "user_provided" and row["split"] == "validation" for row in rows) == 0


def test_prepare_refuses_to_overwrite_nonempty_output(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare(tmp_path / "missing-cifake", tmp_path / "missing-user", existing, project_root=tmp_path)
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "preserve"
