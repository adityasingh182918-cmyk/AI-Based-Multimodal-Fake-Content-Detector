"""Download only the documented, license-reviewed English HC3 source files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Hello-SimpleAI/HC3"
REVISION = "4d0ff18143b5a7e1b1e79beb540c04549d1e59d3"
SOURCES = {
    "reddit_eli5": {"file": "reddit_eli5.jsonl", "license": "BSD License (source ELI5; see HC3 terms)"},
    "wiki_csai": {"file": "wiki_csai.jsonl", "license": "CC BY-SA (Wikipedia-derived; attribution/share-alike obligations apply)"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/text/raw/hc3"))
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        spec = SOURCES[source]
        path = Path(hf_hub_download(REPO_ID, spec["file"], repo_type="dataset", revision=REVISION,
                                    local_dir=args.output_dir))
        print(f"Downloaded {source}: {path} ({path.stat().st_size:,} bytes)")
    metadata = {"repo_id": REPO_ID, "revision": REVISION, "sources": {s: SOURCES[s] for s in args.sources}}
    (args.output_dir / "download_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Recorded source revision and license notes: {args.output_dir / 'download_metadata.json'}")


if __name__ == "__main__":
    main()
