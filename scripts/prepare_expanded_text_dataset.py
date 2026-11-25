"""Join sampled official MAGE splits with the unchanged HC3 splits safely."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")
LABELS = {"HUMAN_WRITTEN", "AI_GENERATED"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row["text"].strip() or row.get("label") not in LABELS:
                raise ValueError(f"Invalid row at {path}:{line_no}")
            rows.append(row)
    return rows


def external_eval_texts(root: Path) -> tuple[set[str], dict[str, int]]:
    """Read only text columns from already-downloaded, evaluation-only artifacts."""
    seen: set[str] = set()
    report: dict[str, int] = {}
    mage_files = (root / "data/text/robustness/mage/test_ood_set_gpt.csv",
                  root / "data/text/robustness/mage/test_ood_set_gpt_para.csv")
    for path in mage_files:
        count = 0
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    text = row.get("text")
                    if isinstance(text, str) and text.strip():
                        seen.add(normalize(text)); count += 1
        report[str(path)] = count
    gradtex = root / "data/text/robustness/gradtex/test.parquet"
    count = 0
    if gradtex.is_file():
        try:
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(gradtex).iter_batches(columns=["text"], batch_size=1024):
                for value in batch.column(0).to_pylist():
                    if isinstance(value, str) and value.strip():
                        seen.add(normalize(value)); count += 1
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to audit exact duplicates against the existing GRADTEX evaluation set") from exc
    report[str(gradtex)] = count
    beemo = root / "data/text/robustness/beemo/data/train-00000-of-00001.parquet"
    count = 0
    if beemo.is_file():
        try:
            sys.path.insert(0, str(root / "training"))
            from evaluate_text_additional_robustness import load_beemo_expert_edits
            for row in load_beemo_expert_edits(beemo):
                text = row.get("text")
                if isinstance(text, str) and text.strip():
                    seen.add(normalize(text)); count += 1
        finally:
            sys.path.pop(0)
    report[str(beemo)] = count
    return seen, report


def summary(rows: list[dict]) -> dict:
    classes = Counter(row["label"] for row in rows)
    sources = Counter(row.get("source", "unknown") for row in rows)
    lengths = Counter()
    for row in rows:
        words = len(row["text"].split())
        bucket = "1-19" if words < 20 else "20-99" if words < 100 else "100-299" if words < 300 else "300+"
        lengths[bucket] += 1
    return {"rows": len(rows), "class_counts": dict(sorted(classes.items())),
            "source_counts": dict(sorted(sources.items())), "word_length_buckets": dict(lengths)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc3-dir", type=Path, default=Path("data/text/processed/hc3"))
    parser.add_argument("--mage-dir", type=Path, default=Path("data/text/raw/mage_expansion"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/text/processed/expanded_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = [args.output_dir / f"{s}.jsonl" for s in SPLITS] + [args.output_dir / "metadata.json"]
    if any(path.exists() for path in outputs) and not args.overwrite:
        parser.error(f"Output exists under {args.output_dir}; choose a new path or pass --overwrite")
    if not all((args.hc3_dir / f"{s}.jsonl").is_file() for s in SPLITS):
        parser.error(f"Missing original HC3 split under {args.hc3_dir}")
    if not all((args.mage_dir / f"{s}.jsonl").is_file() for s in SPLITS):
        parser.error(f"Missing sampled MAGE split under {args.mage_dir}; run scripts/download_text_expansion.py first")

    hc3 = {s: read_jsonl(args.hc3_dir / f"{s}.jsonl") for s in SPLITS}
    mage = {s: read_jsonl(args.mage_dir / f"{s}.jsonl") for s in SPLITS}
    heldout_texts, heldout_sources = external_eval_texts(Path.cwd())
    hc3_norms = {normalize(row["text"]) for rows in hc3.values() for row in rows}
    duplicate_counts = {"mage_exact_duplicates_with_hc3": 0, "mage_exact_duplicates_with_frozen_evaluation": 0,
                        "mage_cross_split_or_within_split_exact_duplicates": 0}
    filtered: dict[str, list[dict]] = {s: [] for s in SPLITS}
    kept_priorities: dict[str, str] = {}
    for split in ("test", "validation", "train"):
        local: set[str] = set()
        for row in mage[split]:
            key = normalize(row["text"])
            if key in hc3_norms:
                duplicate_counts["mage_exact_duplicates_with_hc3"] += 1
                continue
            if key in heldout_texts:
                duplicate_counts["mage_exact_duplicates_with_frozen_evaluation"] += 1
                continue
            if key in kept_priorities or key in local:
                duplicate_counts["mage_cross_split_or_within_split_exact_duplicates"] += 1
                continue
            local.add(key)
            kept_priorities[key] = split
            row = dict(row)
            row["split"] = split
            row["group_id"] = f"mage:{split}:{row.get('source_group','unknown')}:{hashlib.sha256(key.encode('utf-8')).hexdigest()}"
            row["provenance"] = "MAGE source label; official split; original document ID not published in sampled schema"
            filtered[split].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": "HC3 + sampled MAGE", "seed": args.seed,
                "hc3_original_splits_preserved": True,
                "mage_revision": "342663f0a2b775455c023f5d36a1341ff0ec5402",
                "mage_label_mapping": {"0": "AI_GENERATED", "1": "HUMAN_WRITTEN"},
                "split_policy": "retain official HC3 and MAGE train/validation/test assignments; no reshuffling across splits",
                "cleaning": "drop blank/invalid rows and exact duplicates by whitespace-collapsed casefolded text; preserve remaining text verbatim",
                "leakage_limitations": "MAGE does not expose original document IDs; exact duplicates against HC3 and frozen MAGE OOD/GRADTEX/Beemo sets are removed, but semantic or cross-generator document overlap cannot be fully verified.",
                "frozen_evaluation_artifact_rows_scanned": heldout_sources,
                "exact_duplicate_removals": duplicate_counts,
                "splits": {}}
    try:
        for split in SPLITS:
            combined = [dict(row, split=split, provenance=row.get("provenance", "HC3 prepared row")) for row in hc3[split]]
            combined.extend(filtered[split])
            combined.sort(key=lambda row: hashlib.sha256(f"{args.seed}\0{row['label']}\0{normalize(row['text'])}".encode("utf-8")).hexdigest())
            path = args.output_dir / f"{split}.jsonl"
            temp = path.with_suffix(".jsonl.tmp")
            with temp.open("w", encoding="utf-8", newline="\n") as handle:
                for row in combined:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            temp.replace(path)
            metadata["splits"][split] = {"hc3": summary(hc3[split]), "mage_after_deduplication": summary(filtered[split]),
                                         "combined": summary(combined)}
        temp_meta = args.output_dir / "metadata.json.tmp"
        temp_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_meta.replace(args.output_dir / "metadata.json")
    except Exception:
        for path in args.output_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        raise
    print(json.dumps({"output_dir": str(args.output_dir), "exact_duplicate_removals": duplicate_counts,
                      "splits": {s: metadata["splits"][s]["combined"] for s in SPLITS}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
