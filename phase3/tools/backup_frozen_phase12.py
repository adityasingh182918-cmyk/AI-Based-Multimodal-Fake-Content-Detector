"""Create a one-time local backup of frozen Phase 1/2 source and model files."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from snapshot_originals import ROOT, sha256


def main() -> int:
    manifest_path = ROOT / "phase3/reports/original_state_baseline.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Baseline snapshot missing; capture original state first")
    baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    backup = ROOT / "phase3/backups/frozen_phase12"
    if backup.exists() and any(backup.iterdir()):
        raise FileExistsError(f"Refusing to replace nonempty backup: {backup}")
    copied = 0
    for section in ("source_files", "original_model_files"):
        for relative, expected in baseline[section].items():
            source = ROOT / relative
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256(destination) != expected["sha256"]:
                raise RuntimeError(f"Backup hash mismatch for {relative}")
            copied += 1
            if copied % 50 == 0:
                print(f"Backed up and verified {copied} frozen files", flush=True)
    record = {"source_and_model_files_copied_and_verified": copied,
              "dataset_files_sha256_manifest": str(manifest_path),
              "note": "Dataset files were not duplicated; all 3,559 dataset-file hashes are in the baseline snapshot and they remain read-only."}
    (backup / "backup_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
