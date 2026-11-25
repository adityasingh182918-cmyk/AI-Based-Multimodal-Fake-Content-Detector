"""Download the small, labeled MAGE GPT-4 OOD test file for external evaluation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "yaful/MAGE"
REVISION = "342663f0a2b775455c023f5d36a1341ff0ec5402"
FILES = ["test_ood_set_gpt.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/text/robustness/mage"))
    parser.add_argument("--include-paraphrase", action="store_true",
                        help="also fetch the 3.6 MB paraphrase test file; MAGE treats all paraphrased outputs as machine-generated")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = FILES + (["test_ood_set_gpt_para.csv"] if args.include_paraphrase else [])
    downloaded = [Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION,
                                       local_dir=args.output_dir)) for name in files]
    metadata = {"repo_id": REPO_ID, "revision": REVISION, "files": files,
                "declared_license": "Apache-2.0", "purpose": "external robustness evaluation only; not used for training or tuning",
                "label_mapping": {"0": "AI_GENERATED", "1": "HUMAN_WRITTEN"},
                "schema": ["text", "label", "src"],
                "source_url": "https://github.com/yafuly/MAGE",
                "file_sizes_bytes": {path.name: path.stat().st_size for path in downloaded}}
    (args.output_dir / "download_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for path in downloaded:
        print(f"Downloaded {path} ({path.stat().st_size:,} bytes)")
    print("This file is reserved for evaluation; do not use it for model selection or threshold tuning.")


if __name__ == "__main__":
    main()
