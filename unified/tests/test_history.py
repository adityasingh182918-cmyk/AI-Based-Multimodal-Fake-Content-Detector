import json

from unified import history


def _record(**changes):
    row = {"analysis_id": history.new_id(), "phase": 1, "detection_type": "image",
           "primary_prediction": "REAL", "image_prediction": "REAL", "text_prediction": None,
           "fusion_prediction": None, "confidence": .91, "confidence_type": "uncalibrated_softmax",
           "model_name": "resnet18", "model_version": "checkpoint-v1", "processing_time_ms": 8.2,
           "created_at": "2026-11-25T12:00:00+00:00", "status": "completed",
           "response": {"label": "REAL", "model_version": "checkpoint-v1"}}
    row.update(changes)
    return row


def test_sqlite_persists_details_filters_csv_delete_and_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.sqlite3")
    monkeypatch.setattr(history, "IMAGE_DIR", tmp_path / "private_uploads")
    history.initialize()
    row = _record(input_text="private original text", text_hash="abc", image_hash="def",
                  image_mime_type="image/png")
    saved = history.save_record(row, image_bytes=b"private image bytes", retain_inputs=True)

    details = history.get(saved)
    assert details["response"] == row["response"]
    assert details["input_text"] == "private original text"
    assert details["image_retained"] is True
    stored = history.image_path(saved)
    assert stored is not None and stored.read_bytes() == b"private image bytes"

    # Each query opens its own connection, so this also checks persistence past
    # the write transaction and subsequent connection closure.
    listing = history.list_records(detection_type="image", prediction="REAL")
    assert listing["total"] == 1 and listing["items"][0]["analysis_id"] == saved
    assert "private original text" not in history.export_csv()
    assert history.delete_record(saved) is True
    assert history.get(saved) is None
    assert not stored.exists()

    history.save_record(_record(detection_type="text", phase=2, primary_prediction="HUMAN_WRITTEN"))
    assert history.clear() == 1
    assert history.list_records()["total"] == 0
