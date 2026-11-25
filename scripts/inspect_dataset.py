"""Inspect GenImage-style ZIP archives or extracted folders without extraction."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterable

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CLASS_LABELS = {"ai": "AI_GENERATED", "nature": "REAL"}


def _read_eocd(archive: Path) -> dict[str, int] | None:
    """Read the standard ZIP end record, including its multi-disk indicators."""
    size = archive.stat().st_size
    with archive.open("rb") as stream:
        stream.seek(max(0, size - 65557))
        tail = stream.read()
    signature = b"PK\x05\x06"
    offset = tail.rfind(signature)
    if offset < 0 or offset + 22 > len(tail):
        return None
    (_sig, disk_number, cd_disk, _entries_disk, entries_total,
     cd_size, cd_offset, comment_len) = struct.unpack_from("<4s4H2LH", tail, offset)
    if offset + 22 + comment_len > len(tail):
        return None
    return {
        "disk_number": disk_number,
        "central_directory_disk": cd_disk,
        "entries_total_eocd": entries_total,
        "central_directory_size": cd_size,
        "central_directory_offset": cd_offset,
    }


def _central_directory_summary(archive: Path, eocd: dict[str, int]) -> dict[str, object]:
    """Count central-directory records directly; useful when split parts are absent."""
    offset, size = eocd["central_directory_offset"], eocd["central_directory_size"]
    if offset + size > archive.stat().st_size:
        raise ValueError("central directory is not wholly present in this volume")
    ext_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    split_class_counts: Counter[str] = Counter()
    entry_count = file_count = image_count = uncompressed_bytes = 0
    zero_byte_images: list[str] = []
    with archive.open("rb") as stream:
        stream.seek(offset)
        end = offset + size
        while stream.tell() < end:
            header = stream.read(46)
            if len(header) != 46 or header[:4] != b"PK\x01\x02":
                raise ValueError(f"invalid central directory entry at byte {stream.tell() - len(header)}")
            values = struct.unpack("<4s6H3I5H2I", header)
            compressed_size, uncompressed_size = values[8], values[9]
            name_len, extra_len, comment_len = values[10], values[11], values[12]
            name = stream.read(name_len).decode("utf-8", errors="replace")
            extra = stream.read(extra_len)
            stream.seek(comment_len, 1)
            entry_count += 1
            if uncompressed_size == 0xFFFFFFFF or compressed_size == 0xFFFFFFFF:
                cursor = 0
                while cursor + 4 <= len(extra):
                    field_id, field_size = struct.unpack_from("<HH", extra, cursor)
                    payload = extra[cursor + 4:cursor + 4 + field_size]
                    cursor += 4 + field_size
                    if field_id != 0x0001:
                        continue
                    zip64_cursor = 0
                    if uncompressed_size == 0xFFFFFFFF and zip64_cursor + 8 <= len(payload):
                        uncompressed_size = struct.unpack_from("<Q", payload, zip64_cursor)[0]
                        zip64_cursor += 8
                    if compressed_size == 0xFFFFFFFF and zip64_cursor + 8 <= len(payload):
                        compressed_size = struct.unpack_from("<Q", payload, zip64_cursor)[0]
                    break
            uncompressed_bytes += uncompressed_size
            if name.endswith("/"):
                continue
            file_count += 1
            ext = Path(name).suffix.lower() or "[no extension]"
            if ext not in IMAGE_EXTENSIONS:
                continue
            image_count += 1
            if uncompressed_size == 0:
                zero_byte_images.append(name)
            ext_counts[ext] += 1
            split, cls = _class_and_split(name)
            if split:
                split_counts[split] += 1
            if cls:
                class_counts[CLASS_LABELS[cls]] += 1
            if split and cls:
                split_class_counts[f"{split}/{CLASS_LABELS[cls]}"] += 1
    return {
        "entry_count": entry_count,
        "file_count": file_count,
        "uncompressed_bytes_listed": uncompressed_bytes,
        "image_count": image_count,
        "zero_byte_image_count": len(zero_byte_images),
        "zero_byte_images_sample": zero_byte_images[:50],
        "image_count_by_extension": dict(sorted(ext_counts.items())),
        "image_count_by_split": dict(sorted(split_counts.items())),
        "image_count_by_class_folder": dict(sorted(class_counts.items())),
        "image_count_by_split_and_label": dict(sorted(split_class_counts.items())),
    }


def _parts_status(archive: Path, eocd: dict[str, int] | None) -> dict[str, object]:
    """Report numbered split volumes adjacent to an archive, without guessing completeness."""
    numbered = {}
    for part in archive.parent.glob(f"{archive.stem}.z*"):
        match = re.fullmatch(re.escape(archive.stem) + r"\.z(\d+)", part.name, re.IGNORECASE)
        if match:
            numbered[int(match.group(1))] = part
    disk_count = eocd["disk_number"] if eocd else 0
    if disk_count > 0:
        expected_numbers = range(1, disk_count + 1)
        missing = [f"{archive.stem}.z{i:02d}" for i in expected_numbers if i not in numbered]
        return {
            "archive_is_split": True,
            "final_volume_disk_number": disk_count,
            "numbered_parts_found": [p.name for _, p in sorted(numbered.items())],
            "missing_numbered_parts": missing,
            "parts_missing": bool(missing),
            "numbered_parts_present": bool(numbered),
            "completeness_note": (
                "ZIP end record identifies a multi-disk archive; required preceding volumes are missing."
                if missing
                else "ZIP end record identifies a multi-disk archive; all expected preceding volume names are present."
            ),
        }
    if not numbered:
        return {
            "archive_is_split": False,
            "numbered_parts_found": [],
            "missing_numbered_parts": [],
            "parts_missing": False,
            "numbered_parts_present": False,
            "completeness_note": "No adjacent .zNN parts were found. This does not prove the ZIP is complete.",
        }
    max_part = max(numbered)
    missing = [f"{archive.stem}.z{i:02d}" for i in range(1, max_part + 1) if i not in numbered]
    return {
        "numbered_parts_found": [p.name for _, p in sorted(numbered.items())],
        "missing_numbered_parts": missing,
        "parts_missing": bool(missing),
        "numbered_parts_present": True,
        "completeness_note": (
            "A sequence gap was found among adjacent volumes."
            if missing
            else "Adjacent numbered volumes are sequential; inspect the original source for the expected final volume count."
        ),
    }


def _class_and_split(member: str) -> tuple[str | None, str | None]:
    segments = [s.lower() for s in member.replace("\\", "/").split("/") if s]
    split = next((s for s in segments if s in {"train", "val", "validation", "test"}), None)
    cls = next((s for s in segments if s in CLASS_LABELS), None)
    return split, cls


def _validate_stream(stream: BinaryIO) -> str | None:
    try:
        with Image.open(stream) as image:
            image.load()
        return None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        return str(exc)


def inspect_archive(archive: Path, *, validate_images: bool = False) -> dict[str, object]:
    eocd = _read_eocd(archive)
    result: dict[str, object] = {
        "path": str(archive.resolve()),
        "archive_size_bytes": archive.stat().st_size,
        **_parts_status(archive, eocd),
    }
    if eocd:
        result["zip_disk_number"] = eocd["disk_number"]
        result["central_directory_disk"] = eocd["central_directory_disk"]
    if eocd and eocd["disk_number"] > 0:
        try:
            result.update(_central_directory_summary(archive, eocd))
            result["archive_opened"] = False
            result["image_validation_performed"] = False
            result["invalid_image_count"] = None
            result["integrity_note"] = "Split-volume metadata was readable, but image payload validation was skipped because required volumes are missing or multi-disk ZIP access is unsupported."
        except (OSError, ValueError, struct.error) as exc:
            result["archive_opened"] = False
            result["archive_error"] = f"Could not parse split ZIP central directory: {exc}"
        return result
    ext_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    split_class_counts: Counter[str] = Counter()
    invalid: list[str] = []
    entries_count = 0
    image_count = 0
    uncompressed_bytes = 0
    try:
        with zipfile.ZipFile(archive) as zf:
            entries = [info for info in zf.infolist() if not info.is_dir()]
            entries_count = len(entries)
            for info in entries:
                uncompressed_bytes += info.file_size
                ext = Path(info.filename).suffix.lower() or "[no extension]"
                ext_counts[ext] += 1
                if ext not in IMAGE_EXTENSIONS:
                    continue
                image_count += 1
                split, cls = _class_and_split(info.filename)
                if split:
                    split_counts[split] += 1
                if cls:
                    class_counts[CLASS_LABELS[cls]] += 1
                if split and cls:
                    split_class_counts[f"{split}/{CLASS_LABELS[cls]}"] += 1
                if validate_images:
                    try:
                        with zf.open(info) as stream:
                            error = _validate_stream(stream)
                    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                        error = str(exc)
                    if error:
                        invalid.append(f"{info.filename}: {error}")
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        result["archive_error"] = str(exc)
        result["archive_opened"] = False
        return result
    result.update({
        "archive_opened": True,
        "entry_count": entries_count,
        "uncompressed_bytes_listed": uncompressed_bytes,
        "image_count_by_extension": dict(sorted(ext_counts.items())),
        "image_count": image_count,
        "image_count_by_split": dict(sorted(split_counts.items())),
        "image_count_by_class_folder": dict(sorted(class_counts.items())),
        "image_count_by_split_and_label": dict(sorted(split_class_counts.items())),
        "image_validation_performed": validate_images,
        "invalid_image_count": len(invalid),
        "invalid_images_sample": invalid[:50],
        "integrity_note": (
            "Every listed image was decoded; this checks listed entries, not whether source files are missing."
            if validate_images
            else "Image payloads were not decoded. Use --validate-images for an exhaustive image-content scan."
        ),
    })
    return result


def inspect_directory(root: Path, *, validate_images: bool = False) -> dict[str, object]:
    result: dict[str, object] = {"path": str(root.resolve()), "input_type": "directory"}
    all_files = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS]
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    split_class_counts: Counter[str] = Counter()
    ext_counts = Counter(p.suffix.lower() or "[no extension]" for p in images)
    invalid: list[str] = []
    for path in images:
        rel = path.relative_to(root).as_posix()
        split, cls = _class_and_split(rel)
        if split:
            split_counts[split] += 1
        if cls:
            class_counts[CLASS_LABELS[cls]] += 1
        if split and cls:
            split_class_counts[f"{split}/{CLASS_LABELS[cls]}"] += 1
        if validate_images:
            with path.open("rb") as stream:
                error = _validate_stream(stream)
            if error:
                invalid.append(f"{rel}: {error}")
    result.update({
        "file_count": len(all_files),
        "image_count": len(images),
        "image_count_by_extension": dict(sorted(ext_counts.items())),
        "image_count_by_split": dict(sorted(split_counts.items())),
        "image_count_by_class_folder": dict(sorted(class_counts.items())),
        "image_count_by_split_and_label": dict(sorted(split_class_counts.items())),
        "image_validation_performed": validate_images,
        "invalid_image_count": len(invalid),
        "invalid_images_sample": invalid[:50],
    })
    return result


def render_report(report: dict[str, object]) -> str:
    lines = ["Dataset inspection report", "=" * 25]
    for key, value in report.items():
        lines.append(f"{key.replace('_', ' ').capitalize()}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="ZIP archive or extracted dataset directory")
    parser.add_argument("--validate-images", action="store_true", help="Decode every listed image (can take a long time)")
    parser.add_argument("--report-json", type=Path, help="Optional path to save the JSON report")
    args = parser.parse_args(argv)
    if not args.source.exists():
        parser.error(f"input does not exist: {args.source}")
    report = inspect_directory(args.source, validate_images=args.validate_images) if args.source.is_dir() else inspect_archive(args.source, validate_images=args.validate_images)
    print(render_report(report))
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 2 if report.get("parts_missing") or report.get("archive_error") else 0


if __name__ == "__main__":
    sys.exit(main())
