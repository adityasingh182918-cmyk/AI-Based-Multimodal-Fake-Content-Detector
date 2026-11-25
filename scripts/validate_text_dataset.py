"""Validate prepared text JSONL splits, labels, groups, duplicates, and counts."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

LABELS = {"HUMAN_WRITTEN", "AI_GENERATED"}


def validate(root: Path) -> dict:
    group_splits: dict[str, str] = {}
    digest_splits: dict[str, str] = {}
    report = {"splits": {}, "errors": []}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        counts = Counter()
        if not path.is_file():
            report["errors"].append(f"missing {path.name}")
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                text, label, group = row["text"], row["label"], row["group_id"]
                if not isinstance(text, str) or len(text.strip()) < 20 or label not in LABELS:
                    raise ValueError("invalid text or label")
                if group in group_splits and group_splits[group] != split:
                    report["errors"].append(f"group leakage: {group}")
                group_splits[group] = split
                digest = hashlib.sha256(text.casefold().encode()).hexdigest()
                if digest in digest_splits and digest_splits[digest] != split:
                    report["errors"].append(f"duplicate leakage across splits: {digest[:12]}")
                digest_splits[digest] = split
                counts[label] += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                report["errors"].append(f"{split}:{line_no}: {exc}")
        report["splits"][split] = dict(counts)
        if not all(counts[label] for label in LABELS):
            report["errors"].append(f"{split} must contain both labels")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=Path("data/text/processed/hc3"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset_dir)
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
