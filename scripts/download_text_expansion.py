"""Stream and deterministically sample the documented MAGE official splits."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import sys
from pathlib import Path
from typing import Any

DATASET_ID = "yaful/MAGE"
DATASET_REVISION = "342663f0a2b775455c023f5d36a1341ff0ec5402"
LABELS = {0: "AI_GENERATED", 1: "HUMAN_WRITTEN"}
SPLITS = ("train", "validation", "test")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def rank_for(seed: int, split: str, label: str, source: str, text: str) -> str:
    payload = "\0".join((str(seed), split, label, source, normalized(text)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_split(dataset: Any, split: str, seed: int, ai_cap: int, human_cap: int) -> tuple[list[dict], dict]:
    # Keep the smallest stable hash ranks in each (label, documented source) stratum.
    # This gives deterministic source coverage without materializing the full corpus.
    heaps: dict[tuple[str, str], list[tuple[int, str, int, dict]]] = {}
    seen_rank: set[tuple[str, str, str]] = set()
    raw_count = valid_count = skipped_short = skipped_invalid = 0
    source_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {label: 0 for label in LABELS.values()}

    for raw_count, row in enumerate(dataset, start=1):
        try:
            text = row.get("text")
            numeric_label = int(row.get("label"))
            source = str(row.get("src") or "").strip()
            label = LABELS[numeric_label]
            if not isinstance(text, str) or not text.strip() or not source:
                raise ValueError("missing text or source")
        except (TypeError, ValueError, KeyError):
            skipped_invalid += 1
            continue
        text = text.strip()
        if len(text) < 40 or len(text.split()) < 5:
            skipped_short += 1
            continue
        valid_count += 1
        class_counts[label] += 1
        source_counts[source] = source_counts.get(source, 0) + 1
        key = (label, source)
        rank_key = (label, source, normalized(text))
        if rank_key in seen_rank:
            continue
        seen_rank.add(rank_key)
        cap = human_cap if label == "HUMAN_WRITTEN" else ai_cap
        rank = rank_for(seed, split, label, source, text)
        row_out = {"text": text, "label": label, "source": f"mage:{source}",
                   "source_group": source, "source_split": split,
                   "source_label": numeric_label}
        heap = heaps.setdefault(key, [])
        item = (-int(rank, 16), rank, raw_count, row_out)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif rank < heap[0][1]:
            heapq.heapreplace(heap, item)
        if raw_count % 50_000 == 0:
            print(f"{split}: scanned {raw_count:,} rows; retained candidates={sum(map(len, heaps.values())):,}", flush=True)

    selected = [item[3] for heap in heaps.values() for item in heap]
    selected.sort(key=lambda row: (row["label"], row["source_group"], rank_for(seed, split, row["label"], row["source_group"], row["text"])))
    counts = {label: sum(row["label"] == label for row in selected) for label in LABELS.values()}
    report = {"official_split": split, "rows_scanned": raw_count, "valid_rows": valid_count,
              "selected_rows": len(selected), "selected_class_counts": counts,
              "valid_class_counts": class_counts, "selected_source_count": len({r['source_group'] for r in selected}),
              "selected_source_counts": {source: sum(r["source_group"] == source for r in selected) for source in sorted({r['source_group'] for r in selected})},
              "source_row_counts": dict(sorted(source_counts.items())),
              "skipped_short": skipped_short, "skipped_invalid": skipped_invalid,
              "max_ai_per_source": ai_cap, "max_human_per_source": human_cap}
    return selected, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/text/raw/mage_expansion"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ai-per-source", type=int, default=40)
    parser.add_argument("--train-human-per-source", type=int, default=1200)
    parser.add_argument("--eval-ai-per-source", type=int, default=10)
    parser.add_argument("--eval-human-per-source", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("train-ai-per-source", "train-human-per-source", "eval-ai-per-source", "eval-human-per-source"):
        if getattr(args, name.replace("-", "_")) < 1:
            parser.error(f"--{name} must be positive")

    expected = [args.output_dir / f"{split}.jsonl" for split in SPLITS] + [args.output_dir / "metadata.json"]
    if any(path.exists() for path in expected) and not args.overwrite:
        parser.error(f"Output already exists under {args.output_dir}; choose a new --output-dir or pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        print("Install the project ML requirements first: python -m pip install -r requirements-ml.txt", file=sys.stderr)
        raise SystemExit(2) from exc

    metadata = {"dataset": DATASET_ID, "revision": DATASET_REVISION, "declared_license": "Apache-2.0",
                "label_mapping": {"0": "AI_GENERATED", "1": "HUMAN_WRITTEN"},
                "split_policy": "official MAGE train/validation/test kept separate; no local reshuffle",
                "selection": "streaming deterministic SHA-256 bottom-k per src and label; whitespace-only normalization is used for sampling rank",
                "seed": args.seed, "splits": {}}
    try:
        for split in SPLITS:
            stream = load_dataset(DATASET_ID, split=split, streaming=True, revision=DATASET_REVISION)
            ai_cap = args.train_ai_per_source if split == "train" else args.eval_ai_per_source
            human_cap = args.train_human_per_source if split == "train" else args.eval_human_per_source
            rows, report = sample_split(stream, split, args.seed, ai_cap, human_cap)
            tmp_path = args.output_dir / f"{split}.jsonl.tmp"
            with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp_path.replace(args.output_dir / f"{split}.jsonl")
            metadata["splits"][split] = report
            print(f"{split}: selected {len(rows):,} rows ({report['selected_class_counts']})", flush=True)
        metadata_path = args.output_dir / "metadata.json"
        tmp_meta = metadata_path.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp_meta.replace(metadata_path)
    except Exception:
        for path in args.output_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        raise
    print(f"MAGE sampled splits and metadata saved under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
