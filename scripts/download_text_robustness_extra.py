"""Download two small held-out robustness artifacts for evaluation only."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

SOURCES = {
    "gradtex": {
        "repo_id": "elisabeth-pl-pl/GRADTEX",
        "revision": "553d859da0255d75a39c385c208f7522a2007f53",
        "filename": "test.parquet",
        "relative_path": Path("gradtex/test.parquet"),
        "declared_license": "CC-BY-4.0",
        "expected_size_bytes": 64_081_445,
        "url": "https://huggingface.co/datasets/elisabeth-pl-pl/GRADTEX",
    },
    "beemo": {
        "repo_id": "Toloka/Beemo",
        "revision": "9c014107fe9b85c4c784c1ce3a43b0b7b0a6d162",
        "filename": "data/train-00000-of-00001.parquet",
        "relative_path": Path("beemo/data/train-00000-of-00001.parquet"),
        "declared_license": "MIT dataset code; source/model content terms vary",
        "expected_size_bytes": 8_345_056,
        "url": "https://github.com/Toloka/beemo",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gradtex", "beemo", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("data/text/robustness"))
    args = parser.parse_args()
    selected = list(SOURCES) if args.dataset == "all" else [args.dataset]
    required = sum(SOURCES[name]["expected_size_bytes"] for name in selected)
    free = shutil.disk_usage(args.output_dir if args.output_dir.exists() else args.output_dir.parent if args.output_dir.parent.exists() else Path.cwd()).free
    if free < required * 1.25:
        raise SystemExit(f"Insufficient free space: need about {required * 1.25 / 1024**2:.1f} MiB; have {free / 1024**2:.1f} MiB")

    records = {}
    for name in selected:
        source = SOURCES[name]
        destination = args.output_dir / source["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        path = Path(hf_hub_download(source["repo_id"], source["filename"], repo_type="dataset",
                                    revision=source["revision"], local_dir=args.output_dir / source["relative_path"].parts[0]))
        if path.stat().st_size != source["expected_size_bytes"]:
            raise RuntimeError(f"Downloaded size mismatch for {name}: {path.stat().st_size} bytes")
        records[name] = {**{k: str(v) if isinstance(v, Path) else v for k, v in source.items()},
                         "local_path": str(path), "actual_size_bytes": path.stat().st_size,
                         "use": "frozen-checkpoint evaluation only; never training/model selection/calibration"}
        print(f"Downloaded {name}: {path} ({path.stat().st_size:,} bytes)")

    metadata_path = args.output_dir / "additional_sources_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
