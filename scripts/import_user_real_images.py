"""Copy user-labelled REAL images into an ignored supplemental training folder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


def inspect_image(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as original:
        original.load()
        raw_size = list(original.size)
        orientation = original.getexif().get(274, 1)
        oriented = ImageOps.exif_transpose(original).convert("RGB")
        oriented_size = list(oriented.size)
        pixel_hash = hashlib.sha256(
            f"{oriented.width}x{oriented.height}:".encode("ascii") + oriented.tobytes()
        ).hexdigest()
        return {
            "width": oriented_size[0],
            "height": oriented_size[1],
            "stored_width": raw_size[0],
            "stored_height": raw_size[1],
            "format": original.format,
            "exif_orientation": int(orientation),
            "sha256_file": digest,
            "sha256_oriented_pixels": pixel_hash,
        }


def import_images(sources: list[Path], destination: Path) -> list[dict[str, object]]:
    if not sources:
        raise ValueError("At least one source image is required")
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    hashes: set[str] = set()
    for index, source in enumerate(sources, start=1):
        if not source.is_file():
            raise FileNotFoundError(f"Image not found: {source}")
        try:
            meta = inspect_image(source)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f"Invalid image {source.name}: {exc}") from exc
        if meta["sha256_oriented_pixels"] in hashes:
            continue
        hashes.add(str(meta["sha256_oriented_pixels"]))
        suffix = source.suffix.lower() if source.suffix else ".img"
        target = destination / f"user_real_{index:02d}{suffix}"
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != meta["sha256_file"]:
                raise FileExistsError(f"Refusing to replace existing non-matching file: {target}")
        else:
            shutil.copy2(source, target)
        rows.append({
            "path": target.name,
            "label": "REAL",
            "split": "train",
            "source": "user_provided",
            "label_provenance": "user_labelled_real; not independently verified",
            **meta,
        })
    metadata_text = json.dumps({"source": "user_provided", "split": "train", "records": rows}, indent=2)
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    manifest_text = buffer.getvalue()
    for name, content in (("metadata.json", metadata_text), ("manifest.csv", manifest_text)):
        target = destination / name
        if target.exists():
            existing_text = target.read_text(encoding="utf-8")
            if name.endswith(".json"):
                try:
                    matches = json.loads(existing_text) == json.loads(content)
                except json.JSONDecodeError:
                    matches = False
            else:
                with target.open("r", encoding="utf-8", newline="") as stream:
                    existing_rows = list(csv.DictReader(stream))
                with io.StringIO(content, newline="") as stream:
                    generated_rows = list(csv.DictReader(stream))
                matches = existing_rows == generated_rows
            if not matches:
                raise FileExistsError(f"Refusing to replace existing different metadata file: {target}")
        else:
            target.write_text(content, encoding="utf-8", newline="")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/image_detection/user_real/train"))
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        rows = import_images(args.images, args.destination)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Imported {len(rows)} unique user-labelled REAL images to {args.destination}")
    for row in rows:
        print(f"{row['path']}: {row['width']}x{row['height']} {row['format']} orientation={row['exif_orientation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
