import sqlite3

import pytest

from app.services.image_detector import InvalidImageError
from app.services.text_detector import InvalidTextError
from backend.tests.http_client import InProcessClient
from phase3.fusion.rule_based_analysis import build_analysis, combine_predictions
from unified import app as app_module
from unified import history


@pytest.mark.parametrize(
    ("image_label", "text_label", "expected"),
    [
        ("REAL", "HUMAN_WRITTEN", "AUTHENTIC CONTENT"),
        ("REAL", "AI_GENERATED", "MIXED CONTENT"),
        ("AI_GENERATED", "HUMAN_WRITTEN", "MIXED CONTENT"),
        ("AI_GENERATED", "AI_GENERATED", "AI-GENERATED CONTENT"),
        ("UNCERTAIN", "HUMAN_WRITTEN", "INCONCLUSIVE"),
        ("REAL", "UNCERTAIN", "INCONCLUSIVE"),
    ],
)
def test_rule_decision_table(image_label, text_label, expected):
    result = combine_predictions({"label": image_label}, {"label": text_label})
    assert result["label"] == expected
    assert result["explanation"]


def test_partial_analysis_has_no_combined_verdict():
    result = build_analysis(None, {"label": "HUMAN_WRITTEN"}, "bad image", None)
    assert result["analysis_status"] == "partial"
    assert result["combined_result"] is None
    assert result["trained_fusion_model"] is False
    assert result["failed_detectors"] == {"image": "bad image"}


class MockImageDetector:
    def __init__(self, label="REAL", error=None):
        self.label = label
        self.error = error
        self.result = {"label": label, "score": 0.87, "score_type": "uncalibrated_softmax",
                       "class_scores": {"REAL": 0.87, "AI_GENERATED": 0.13},
                       "model_version": "frozen-image-v1"}

    def predict(self, _data):
        if self.error:
            raise self.error
        return self.result


class MockTextDetector:
    def __init__(self, label="HUMAN_WRITTEN", error=None):
        self.label = label
        self.error = error
        self.result = {"label": label, "score": 0.73, "score_type": "calibrated_tfidf_probability",
                       "class_scores": {"HUMAN_WRITTEN": 0.73, "AI_GENERATED": 0.27},
                       "model_version": "frozen-text-v1"}

    def predict(self, _text):
        if self.error:
            raise self.error
        if len(_text) < 40 or len(_text.split()) < 5:
            raise InvalidTextError("Enter at least 40 characters and 5 words")
        return self.result


def configure(monkeypatch, tmp_path, image_detector=None, text_detector=None):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.sqlite3")
    monkeypatch.setattr(history, "IMAGE_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_module, "RETAIN_INPUTS", False)
    monkeypatch.setattr(app_module.phase12_app.state, "image_detector", image_detector or MockImageDetector())
    monkeypatch.setattr(app_module.phase12_app.state, "text_detector", text_detector or MockTextDetector())
    history.initialize()
    return InProcessClient(app_module.app)


def submit(client, *, image=b"valid fixture bytes", text="A sufficiently long related text sample with at least five words."):
    return client.post("/api/v1/predict/multimodal", data={"text": text},
                       files={"file": ("pair.png", image, "image/png")})


def test_multimodal_api_preserves_modality_outputs_and_history(tmp_path, monkeypatch):
    image_detector = MockImageDetector("REAL")
    text_detector = MockTextDetector("AI_GENERATED")
    client = configure(monkeypatch, tmp_path, image_detector, text_detector)

    response = submit(client)
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_type"] == "rule_based_multimodal"
    assert body["fusion_type"] == "rule_based"
    assert body["trained_fusion_model"] is False
    assert body["analysis_status"] == "completed"
    assert body["image_result"] == image_detector.result
    assert body["text_result"] == text_detector.result
    assert body["combined_result"]["label"] == "MIXED CONTENT"
    assert "confidence" not in body["combined_result"]
    assert body["timestamp"] and body["history_saved"] is True

    analysis_id = body["analysis_id"]
    history.initialize()  # simulates a new app/database connection after restart
    saved = history.get(analysis_id)
    assert saved["primary_prediction"] == "MIXED CONTENT"
    assert saved["text_retained"] is False and saved["image_retained"] is False
    assert saved["response"]["combined_result"]["label"] == "MIXED CONTENT"
    assert client.get("/api/v1/history?prediction=MIXED%20CONTENT").json()["total"] == 1
    assert "MIXED CONTENT" in client.get("/api/v1/history/export.csv").text


@pytest.mark.parametrize("failed_modality", ["image", "text"])
def test_detector_failure_returns_partial_without_verdict(failed_modality, tmp_path, monkeypatch):
    image_detector = MockImageDetector(error=InvalidImageError("invalid image")) if failed_modality == "image" else MockImageDetector()
    text_detector = MockTextDetector(error=InvalidTextError("empty text")) if failed_modality == "text" else MockTextDetector()
    client = configure(monkeypatch, tmp_path, image_detector, text_detector)

    response = submit(client)
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_status"] == "partial"
    assert body["combined_result"] is None
    assert failed_modality in body["failed_detectors"]
    assert body["history_saved"] is True
    assert (body["text_result"] is None) if failed_modality == "text" else (body["image_result"] is None)


def test_empty_text_is_a_partial_result_not_a_false_combined_verdict(tmp_path, monkeypatch):
    client = configure(monkeypatch, tmp_path, text_detector=MockTextDetector(error=InvalidTextError("Text must not be empty")))
    response = submit(client, text="")
    assert response.status_code == 200
    assert response.json()["analysis_status"] == "partial"
    assert response.json()["combined_result"] is None
    assert response.json()["image_result"]["label"] == "REAL"


def test_missing_image_or_text_returns_request_validation_error(tmp_path, monkeypatch):
    client = configure(monkeypatch, tmp_path)
    missing_image = client.post("/api/v1/predict/multimodal", data={"text": "A sufficiently long related text sample with at least five words."})
    missing_text = client.post("/api/v1/predict/multimodal", files={"file": ("pair.png", b"valid", "image/png")})
    assert missing_image.status_code == 422
    assert missing_text.status_code == 200
    assert missing_text.json()["analysis_status"] == "partial"
    assert missing_text.json()["combined_result"] is None


def test_history_database_failure_does_not_fail_prediction_endpoints(tmp_path, monkeypatch):
    client = configure(monkeypatch, tmp_path)

    def fail_save(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(history, "save_record", fail_save)
    combined = submit(client)
    image = client.post("/api/v1/predict/image", files={"file": ("a.png", b"valid", "image/png")})
    text = client.post("/api/v1/predict/text", json={"text": "A sufficiently long standalone text sample with at least five words."})
    assert combined.status_code == image.status_code == text.status_code == 200
    assert combined.json()["history_saved"] is False
    assert image.json()["history_saved"] is False
    assert text.json()["history_saved"] is False
