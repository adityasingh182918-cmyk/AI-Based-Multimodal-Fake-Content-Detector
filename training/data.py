"""Shared manifest-backed PyTorch image dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

CLASS_TO_INDEX = {"REAL": 0, "AI_GENERATED": 1}


class ManifestImageDataset:
    def __new__(cls, root: Path, manifest: Path, split: str, transform: Any = None):
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise RuntimeError("Install PyTorch and torchvision using the official Windows install selector") from exc

        class _Dataset(Dataset):
            def __init__(self) -> None:
                self.root = root.resolve()
                self.transform = transform
                with manifest.open("r", newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.rows = [row for row in rows if row.get("split") == split]
                if not self.rows:
                    raise ValueError(f"Manifest {manifest} contains no rows for split={split!r}")
                for row in self.rows:
                    if row.get("label") not in CLASS_TO_INDEX:
                        raise ValueError(f"Unsupported manifest label: {row.get('label')!r}")
                    image_path = (self.root / row["path"]).resolve()
                    if not image_path.is_relative_to(self.root):
                        raise ValueError(f"Manifest path escapes dataset root: {row['path']}")

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int):
                row = self.rows[index]
                image_path = self.root / row["path"]
                try:
                    with Image.open(image_path) as image:
                        image.load()
                        image = ImageOps.exif_transpose(image).convert("RGB")
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    raise ValueError(f"Could not read manifest image {image_path}: {exc}") from exc
                if self.transform:
                    image = self.transform(image)
                return image, torch.tensor(CLASS_TO_INDEX[row["label"]], dtype=torch.long)

        return _Dataset()
