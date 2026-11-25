"""Hash frozen Phase 1/2 artifacts and capture fixed-input model predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

IMAGE_REL = Path("models/image_detector/resnet18_cifake_aug_v1_calibrated_strict.pt")
IMAGE_META_REL = Path("models/image_detector/resnet18_cifake_aug_v1_calibrated_strict.metadata.json")
TEXT_REL = Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib")
TEXT_META_REL = Path("models/text_detector/baseline/hc3_tfidf_linear_svm.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {p.relative_to(ROOT).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(paths) if p.is_file()}


def source_paths() -> list[Path]:
    paths: set[Path] = set()
    for relative in ("backend/app", "backend/tests", "training", "scripts", "frontend"):
        base = ROOT / relative
        if base.exists():
            paths.update(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".py", ".html", ".js", ".css"})
    paths.update(p for p in ROOT.glob("requirements*.txt") if p.is_file())
    return sorted(paths)


def select_examples() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    image_root = ROOT / "data/image/cifake_ready"
    with (image_root / "manifests/manifest.csv").open(encoding="utf-8", newline="") as stream:
        manifest = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    image_rows = []
    for label in ("REAL", "AI_GENERATED"):
        matches = sorted((row for row in manifest if row["label"] == label), key=lambda row: row["path"])
        image_rows.extend(matches[:2])

    text_path = ROOT / "data/text/processed/hc3/test.jsonl"
    text_rows = [json.loads(line) for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected_text: list[dict[str, str]] = []
    for expected in ("HUMAN_WRITTEN", "AI_GENERATED"):
        matches = [(index, row) for index, row in enumerate(text_rows)
                   if row.get("label") == expected and len(row.get("text", "")) >= 40 and len(row.get("text", "").split()) >= 5]
        for index, row in matches[:2]:
            selected_text.append({"record_index": str(index), "text": row["text"], "expected_label": expected})
    if len(image_rows) != 4 or len(selected_text) != 4:
        raise RuntimeError("Could not find the fixed 4-image / 4-text test examples")
    return image_rows, selected_text


def predictions() -> dict[str, Any]:
    import torch
    from app.services.image_detector import ImageDetector
    from app.services.text_detector import TextDetector

    image_path, text_path = ROOT / IMAGE_REL, ROOT / TEXT_REL
    image_rows, text_rows = select_examples()
    image_detector = ImageDetector(image_path, device="auto")
    image_outputs = []
    for row in image_rows:
        image_file = image_path.parent.parent.parent / "data/image/cifake_ready" / row["path"]
        result = image_detector.predict(image_file)
        image_outputs.append({"relative_path": row["path"], "manifest_sha256": row["sha256"],
                              "expected_label": row["label"], "predicted_label": result["label"],
                              "class_scores": {k: round(float(v), 9) for k, v in result["class_scores"].items()},
                              "model_version": result.get("model_version"),
                              "checkpoint_identifier": result.get("checkpoint_identifier")})

    text_detector = TextDetector(text_path)
    text_outputs = []
    for row in text_rows:
        result = text_detector.predict(row["text"])
        text_outputs.append({"record_index": int(row["record_index"]),
                             "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                             "expected_label": row["expected_label"], "predicted_label": result["label"],
                             "class_scores": {k: round(float(v), 9) for k, v in result["class_scores"].items()},
                             "model_version": result.get("model_version"),
                             "score_type": result.get("score_type"), "chunk_count": result.get("chunk_count")})
    return {"runtime": {"python": sys.version, "torch": torch.__version__,
                         "cuda_available": torch.cuda.is_available(),
                         "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                         "image_device_request": "auto", "text_backend": "baseline HC3 TF-IDF SVM"},
            "image_predictions": image_outputs, "text_predictions": text_outputs}


def checkpoint_configs() -> dict[str, Any]:
    import torch
    image = torch.load(ROOT / IMAGE_REL, map_location="cpu", weights_only=True)
    image_config = {key: image.get(key) for key in ("architecture", "class_names", "image_size", "resize_mode", "mean", "std", "run_id", "epoch", "decision_thresholds")}
    text_metadata = json.loads((ROOT / TEXT_META_REL).read_text(encoding="utf-8"))
    training_config = json.loads((ROOT / "training/configs/text_training_config.json").read_text(encoding="utf-8"))
    return {"phase1_image": {"checkpoint": IMAGE_REL.as_posix(), "configuration": image_config},
            "phase2_text": {"backend": "baseline", "checkpoint": TEXT_REL.as_posix(),
                            "metadata": text_metadata, "training_configuration": training_config}}


def copy_active_models() -> dict[str, str]:
    import shutil
    copies = {
        IMAGE_REL: Path("phase3/models/image_copy") / IMAGE_REL.name,
        IMAGE_META_REL: Path("phase3/models/image_copy") / IMAGE_META_REL.name,
        TEXT_REL: Path("phase3/models/text_copy") / TEXT_REL.name,
        TEXT_META_REL: Path("phase3/models/text_copy") / TEXT_META_REL.name,
    }
    hashes = {}
    for source_rel, destination_rel in copies.items():
        source, destination = ROOT / source_rel, ROOT / destination_rel
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing copied model: {destination}")
        shutil.copy2(source, destination)
        original_hash, copied_hash = sha256(source), sha256(destination)
        if original_hash != copied_hash:
            raise RuntimeError(f"Copy hash mismatch for {source_rel}")
        hashes[source_rel.as_posix()] = copied_hash
    return hashes


def current_state() -> dict[str, Any]:
    models = [p for p in (ROOT / "models").rglob("*") if p.is_file()]
    datasets = [p for p in (ROOT / "data").rglob("*") if p.is_file()]
    return {"source_files": file_manifest(source_paths()),
            "original_model_files": file_manifest(models),
            "dataset_files": file_manifest(datasets)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Compare current originals and predictions to saved baseline")
    args = parser.parse_args()
    out_dir = ROOT / "phase3/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / "original_state_baseline.json"
    if args.verify:
        if not baseline_path.is_file():
            raise FileNotFoundError("Run the initial snapshot before --verify")
        before = json.loads(baseline_path.read_text(encoding="utf-8"))
        after = {"captured_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                 **current_state(), "predictions": predictions()}
        sections = ("source_files", "original_model_files", "dataset_files")
        hash_differences: dict[str, dict[str, Any]] = {}
        added_files: dict[str, list[str]] = {}
        for key in sections:
            expected, current = before[key], after[key]
            changed = {path: {"before": expected[path], "after": current[path]}
                       for path in expected.keys() & current.keys() if expected[path] != current[path]}
            missing = {path: {"before": expected[path], "after": None}
                       for path in expected.keys() - current.keys()}
            if changed or missing:
                hash_differences[key] = {**changed, **missing}
            added_files[key] = sorted(current.keys() - expected.keys())
        prediction_match = before["predictions"] == after["predictions"]
        result = {"hashes_unchanged": not hash_differences,
                  "predictions_unchanged": prediction_match,
                  "hash_differences": hash_differences,
                  "added_files_after_baseline": added_files,
                  "prediction_before": before["predictions"], "prediction_after": after["predictions"]}
        (out_dir / "original_state_postcheck.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"hashes_unchanged": result["hashes_unchanged"],
                          "predictions_unchanged": result["predictions_unchanged"],
                          "hash_differences": result["hash_differences"],
                          "added_files_after_baseline": result["added_files_after_baseline"]}, indent=2))
        return 0 if all((result["hashes_unchanged"], prediction_match)) else 1

    if baseline_path.exists():
        raise FileExistsError(f"Refusing to replace baseline snapshot: {baseline_path}")
    copied_hashes = copy_active_models()
    report = {"captured_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
              "project": str(ROOT), "phase1_active_checkpoint": IMAGE_REL.as_posix(),
              "phase2_active_backend": "baseline", "phase2_active_checkpoint": TEXT_REL.as_posix(),
              "configurations": checkpoint_configs(), "copied_model_sha256": copied_hashes,
              **current_state(), "predictions": predictions()}
    baseline_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot": str(baseline_path), "copied_hashes": copied_hashes,
                      "source_file_count": len(report["source_files"]),
                      "checkpoint_file_count": len(report["original_model_files"]),
                      "dataset_file_count": len(report["dataset_files"]),
                      "predictions": report["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
