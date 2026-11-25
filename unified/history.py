"""Persistent, privacy-aware SQLite prediction history for the unified app."""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("UNIFIED_DB_PATH", str(ROOT / "unified/data/history.sqlite3"))).expanduser()
IMAGE_DIR = Path(os.getenv("UNIFIED_IMAGE_STORAGE", str(ROOT / "unified/storage/images"))).expanduser()


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=15000")
    return connection


@contextmanager
def db() -> Any:
    connection = connect()
    try:
        yield connection
    finally:
        connection.close()


def resolve_stored_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def initialize() -> None:
    with db() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL UNIQUE,
            phase INTEGER NOT NULL CHECK (phase IN (1,2,3)),
            detection_type TEXT NOT NULL CHECK (detection_type IN ('image','text','multimodal')),
            image_filename TEXT,
            image_storage_path TEXT,
            image_hash TEXT,
            text_hash TEXT,
            input_text TEXT,
            primary_prediction TEXT,
            image_prediction TEXT,
            text_prediction TEXT,
            fusion_prediction TEXT,
            confidence REAL,
            confidence_type TEXT,
            model_name TEXT,
            model_version TEXT,
            processing_time_ms REAL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            response_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_history_created ON prediction_history(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_history_phase ON prediction_history(phase, detection_type);
        CREATE INDEX IF NOT EXISTS idx_history_prediction ON prediction_history(primary_prediction);
        """)


def new_id() -> str:
    return uuid.uuid4().hex


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def save_record(record: dict[str, Any], *, image_bytes: bytes | None = None,
                retain_inputs: bool = False) -> str:
    """Store a completed or failed analysis. Inputs are retained only by opt-in."""
    analysis_id = record.get("analysis_id") or new_id()
    storage_rel: str | None = None
    stored_path: Path | None = None
    if retain_inputs and image_bytes:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        mime = str(record.get("image_mime_type") or "application/octet-stream").lower()
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
                  "image/gif": ".gif", "image/bmp": ".bmp", "image/tiff": ".tif"}.get(mime, ".bin")
        stored_path = (IMAGE_DIR / f"{analysis_id}{suffix}").resolve()
        if not stored_path.is_relative_to(IMAGE_DIR.resolve()):
            raise ValueError("Private image path escaped its storage directory")
        stored_path.write_bytes(image_bytes)
        storage_rel = str(stored_path.relative_to(ROOT)) if stored_path.is_relative_to(ROOT) else str(stored_path)
    try:
        with db() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""INSERT INTO prediction_history
                (analysis_id,phase,detection_type,image_filename,image_storage_path,image_hash,text_hash,input_text,
                 primary_prediction,image_prediction,text_prediction,fusion_prediction,confidence,confidence_type,
                 model_name,model_version,processing_time_ms,created_at,status,error_message,response_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (analysis_id, record["phase"], record["detection_type"], record.get("image_filename"), storage_rel,
                 record.get("image_hash"), record.get("text_hash"), record.get("input_text") if retain_inputs else None,
                 record.get("primary_prediction"), record.get("image_prediction"), record.get("text_prediction"),
                 record.get("fusion_prediction"), record.get("confidence"), record.get("confidence_type"),
                 record.get("model_name"), record.get("model_version"), record.get("processing_time_ms"),
                 record.get("created_at") or datetime.now(timezone.utc).isoformat(), record.get("status", "completed"),
                 record.get("error_message"), _as_json(record.get("response", {}))))
            connection.commit()
    except Exception:
        if stored_path and stored_path.exists():
            stored_path.unlink()
        raise
    return analysis_id


def _public(row: sqlite3.Row, *, details: bool = False) -> dict[str, Any]:
    result = dict(row)
    result["response"] = json.loads(result.pop("response_json"))
    result["image_retained"] = bool(result.pop("image_storage_path"))
    result["text_retained"] = result.get("input_text") is not None
    if not details:
        result.pop("input_text", None)
        result.pop("image_hash", None)
        result.pop("text_hash", None)
        result.pop("error_message", None)
        result.pop("response", None)
    return result


def get(analysis_id: str) -> dict[str, Any] | None:
    with db() as connection:
        row = connection.execute("SELECT * FROM prediction_history WHERE analysis_id=?", (analysis_id,)).fetchone()
    return _public(row, details=True) if row else None


def image_path(analysis_id: str) -> Path | None:
    with db() as connection:
        row = connection.execute("SELECT image_storage_path FROM prediction_history WHERE analysis_id=?", (analysis_id,)).fetchone()
    if not row or not row[0]:
        return None
    path = resolve_stored_path(row[0])
    return path if path.is_relative_to(IMAGE_DIR.resolve()) and path.is_file() else None


def list_records(*, offset: int = 0, limit: int = 25, detection_type: str | None = None,
                 prediction: str | None = None, start_date: str | None = None,
                 end_date: str | None = None) -> dict[str, Any]:
    conditions, parameters = [], []
    if detection_type and detection_type != "all":
        conditions.append("detection_type=?"); parameters.append(detection_type)
    if prediction and prediction != "all":
        conditions.append("(primary_prediction=? OR image_prediction=? OR text_prediction=? OR fusion_prediction=?)")
        parameters.extend([prediction] * 4)
    if start_date:
        conditions.append("created_at>=?"); parameters.append(start_date)
    if end_date:
        conditions.append("created_at<=?"); parameters.append(end_date)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    with db() as connection:
        total = connection.execute("SELECT COUNT(*) FROM prediction_history" + where, parameters).fetchone()[0]
        rows = connection.execute("SELECT * FROM prediction_history" + where +
                                  " ORDER BY created_at DESC LIMIT ? OFFSET ?", [*parameters, limit, offset]).fetchall()
    return {"total": total, "offset": offset, "limit": limit, "items": [_public(row) for row in rows]}


def stats() -> dict[str, Any]:
    with db() as connection:
        rows = connection.execute("SELECT detection_type, COUNT(*) AS n FROM prediction_history WHERE status!='failed' GROUP BY detection_type").fetchall()
        total = connection.execute("SELECT COUNT(*) FROM prediction_history WHERE status!='failed'").fetchone()[0]
        recent = connection.execute("SELECT * FROM prediction_history ORDER BY created_at DESC LIMIT 5").fetchall()
    counts = {"image": 0, "text": 0, "multimodal": 0}
    counts.update({row["detection_type"]: row["n"] for row in rows})
    return {"total_analyses": total, "counts": counts, "recent": [_public(row) for row in recent]}


def delete_record(analysis_id: str) -> bool:
    with db() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT image_storage_path FROM prediction_history WHERE analysis_id=?", (analysis_id,)).fetchone()
        if not row:
            connection.rollback(); return False
        connection.execute("DELETE FROM prediction_history WHERE analysis_id=?", (analysis_id,))
        connection.commit()
    if row[0]:
        path = resolve_stored_path(row[0])
        if path.is_relative_to(IMAGE_DIR.resolve()) and path.is_file():
            path.unlink()
    return True


def clear() -> int:
    with db() as connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute("SELECT image_storage_path FROM prediction_history WHERE image_storage_path IS NOT NULL").fetchall()
        count = connection.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
        connection.execute("DELETE FROM prediction_history")
        connection.commit()
    for row in rows:
        path = resolve_stored_path(row[0])
        if path.is_relative_to(IMAGE_DIR.resolve()) and path.is_file():
            path.unlink()
    return count


def export_csv() -> str:
    records = list_records(offset=0, limit=1_000_000)["items"]
    output = io.StringIO(newline="")
    fields = ["analysis_id", "phase", "detection_type", "primary_prediction", "image_prediction",
              "text_prediction", "fusion_prediction", "confidence", "confidence_type", "model_name",
              "model_version", "processing_time_ms", "created_at", "status"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(records)
    return output.getvalue()
