"""Clean reviewed HC3 JSONL sources and create deterministic question-group splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SOURCES = ("reddit_eli5", "wiki_csai")
SPLITS = ("train", "validation", "test")


def normalize_text(text: str) -> str:
    return re.sub(r"[\t\r\f\v ]+", " ", text.replace("\u0000", "")).strip()


def read_rows(raw_dir: Path, max_groups_per_source: int, seed: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    rejected = Counter()
    digest_seen: set[str] = set()
    for source in SOURCES:
        path = raw_dir / f"{source}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}. Run scripts/download_text_datasets.py first.")
        records = []
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    obj = json.loads(line)
                    group = str(obj.get("id") or f"{source}:{line_no}")
                    human, ai = obj.get("human_answers"), obj.get("chatgpt_answers")
                    if not isinstance(human, list) or not isinstance(ai, list):
                        rejected["missing_answer_lists"] += 1
                        continue
                    records.append((group, human, ai))
                except (json.JSONDecodeError, TypeError):
                    rejected["invalid_json"] += 1
        # Sample groups reproducibly by a seeded stable key so both classes and answers stay together.
        records.sort(key=lambda item: hashlib.sha256(f"{seed}:{source}:{item[0]}".encode()).hexdigest())
        if max_groups_per_source > 0:
            records = records[:max_groups_per_source]
        for group, humans, generated in records:
            # Pair a deterministic, equal number of human and generated answers per question.
            pair_count = min(len(humans), len(generated))
            if pair_count == 0:
                rejected["unpaired_question"] += 1
                continue
            for label, answers in (("HUMAN_WRITTEN", humans[:pair_count]), ("AI_GENERATED", generated[:pair_count])):
                if not isinstance(answers, list):
                    continue
                for answer_index, answer in enumerate(answers):
                    if not isinstance(answer, str):
                        rejected["non_string"] += 1
                        continue
                    text = normalize_text(answer)
                    if len(text) < 20:
                        rejected["empty_or_short"] += 1
                        continue
                    digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
                    if digest in digest_seen:
                        rejected["exact_duplicate"] += 1
                        continue
                    digest_seen.add(digest)
                    rows.append({"text": text, "label": label, "source": source,
                                 "group_id": f"{source}:{group}", "answer_index": answer_index})
    return rows, dict(rejected)


def assign_group_splits(rows: list[dict], seed: int) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    names = ["train", "validation", "test"]
    ratios = {"train": .70, "validation": .15, "test": .15}
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: -len(item[1]))
    targets = {name: len(rows) * ratios[name] for name in names}
    counts = Counter()
    for group, group_rows in items:
        dest = max(names, key=lambda name: (targets[name] - counts[name], -names.index(name)))
        for row in group_rows:
            row["split"] = dest
        counts[dest] += len(group_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/text/raw/hc3"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/text/processed/hc3"))
    parser.add_argument("--max-groups-per-source", type=int, default=1000,
                        help="0 uses all groups; default selects at most 1,000 questions per source")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, rejected = read_rows(args.raw_dir, args.max_groups_per_source, args.seed)
    if not rows or {row["label"] for row in rows} != {"HUMAN_WRITTEN", "AI_GENERATED"}:
        raise SystemExit("Prepared data must contain both verified HC3 answer labels")
    assign_group_splits(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter((r["split"], r["label"]) for r in rows)
    for split in SPLITS:
        target = args.output_dir / f"{split}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as out:
            for row in rows:
                if row["split"] == split:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {sum(counts[(split, label)] for label in ('HUMAN_WRITTEN','AI_GENERATED')):,} rows")
    metadata = {"dataset": "HC3", "repo_id": "Hello-SimpleAI/HC3", "revision": "4d0ff18143b5a7e1b1e79beb540c04549d1e59d3",
                "sources": list(SOURCES), "seed": args.seed, "max_groups_per_source": args.max_groups_per_source,
                "split_ratios_target": {"train": .70, "validation": .15, "test": .15},
                "split_label_counts": {split: {label: counts[(split, label)] for label in ("HUMAN_WRITTEN", "AI_GENERATED")} for split in SPLITS},
                "rejected": rejected, "group_split": "All answers to the same source question remain in a single split.",
                "limitations": "HC3 is primarily QA and ChatGPT-era English text; this is not representative of all current generators or genres."}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote manifests and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
