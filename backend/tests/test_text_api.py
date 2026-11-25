from app.main import app, create_text_detector
from app.services.text_detector import TextDetector
from app.services.transformer_text_detector import TransformerTextDetector
from backend.tests.http_client import InProcessClient

client = InProcessClient(app)


def test_text_status_reports_checkpoint_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(app.state, "text_detector", TextDetector(tmp_path / "missing.joblib"))
    response = client.get("/text-model-status")
    assert response.status_code == 200
    assert response.json()["ready_for_inference"] is False


def test_text_backend_selection_uses_versioned_checkpoints(tmp_path):
    backend, svm = create_text_detector("svm_expanded", str(tmp_path / "selected.joblib"))
    assert backend == "svm_expanded"
    assert isinstance(svm, TextDetector)
    assert svm.checkpoint_path == tmp_path / "selected.joblib"

    backend, transformer = create_text_detector("transformer_expanded", str(tmp_path / "expanded-transformer"))
    assert backend == "transformer_expanded"
    assert isinstance(transformer, TransformerTextDetector)
    assert transformer.checkpoint_path == tmp_path / "expanded-transformer"


def test_text_backend_selection_rejects_unknown_name():
    import pytest
    with pytest.raises(RuntimeError, match="TEXT_MODEL_BACKEND"):
        create_text_detector("silent_fallback")


def test_text_api_missing_checkpoint_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(app.state, "text_detector", TextDetector(tmp_path / "missing.joblib"))
    response = client.post("/api/v1/predict/text", json={"text": "This valid and sufficiently long sentence has many ordinary words."})
    assert response.status_code == 503


def test_text_api_rejects_empty_short_and_oversized_requests():
    for value in ("", "short", "x" * 50_001):
        response = client.post("/api/v1/predict/text", json={"text": value})
        assert response.status_code == 422


def test_text_api_returns_prediction_from_local_baseline(monkeypatch):
    from pathlib import Path

    checkpoint = Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib")
    if not checkpoint.is_file():
        return
    detector = TextDetector(checkpoint)
    monkeypatch.setattr(app.state, "text_detector", detector)
    response = client.post("/api/v1/predict/text", json={
        "text": "This answer contains a few more words so that it passes the required input length and is analyzed."
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] in {"HUMAN_WRITTEN", "AI_GENERATED", "UNCERTAIN"}
    assert set(body["class_scores"]) == {"HUMAN_WRITTEN", "AI_GENERATED"}
    assert body["input_word_count"] >= 5
    assert body["chunk_count"] == 1
    assert body["model_version"]
    assert body["confidence_type"] == body["score_type"]
    assert body["decision_status"] in {"thresholded", "uncertain"}


def test_expanded_svm_checkpoint_is_selectable_and_reports_version(monkeypatch):
    from pathlib import Path

    checkpoint = Path("models/text_detector/svm_expanded_v1.joblib")
    if not checkpoint.is_file():
        return
    monkeypatch.setattr(app.state, "text_backend", "svm_expanded")
    monkeypatch.setattr(app.state, "text_detector", TextDetector(checkpoint))
    status = client.get("/text-model-status").json()
    assert status["selected_backend"] == "svm_expanded"
    assert status["checkpoint_identifier"] == checkpoint.name
    assert status["model_version"] == "svm_expanded_v1"
    response = client.post("/api/v1/predict/text", json={
        "text": "This sample contains enough words to exercise the independently saved expanded SVM API checkpoint."
    })
    assert response.status_code == 200, response.text
    assert response.json()["model_version"] == "svm_expanded_v1"


def test_ui_exposes_three_independent_detection_modes():
    response = client.get("/ui")
    assert response.status_code == 200
    assert 'data-mode="image"' in response.text
    assert 'data-mode="text"' in response.text
    assert 'data-mode="combined"' in response.text
    assert "NOT A TRAINED FUSION MODEL" in response.text
    assert '<label for="textInput">Text to analyze</label>' in response.text
    assert 'id="file" type="file"' in response.text and 'input[type=file]:focus-visible+label' in response.text
    assert "Promise.all([fetch('/api/v1/predict/image'" in response.text
    assert "postText(value)" in response.text


def test_text_api_accepts_long_text_and_returns_multiple_chunks(monkeypatch):
    from pathlib import Path

    checkpoint = Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib")
    if not checkpoint.is_file():
        return
    monkeypatch.setattr(app.state, "text_detector", TextDetector(checkpoint))
    text = ("This neutral test paragraph contains enough words to exercise text windowing in the API. " * 80)
    response = client.post("/api/v1/predict/text", json={"text": text})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["input_character_count"] == len(text)
    assert body["input_word_count"] > 1000
    assert body["chunk_count"] > 1


def test_transformer_backend_can_predict_from_local_checkpoint(monkeypatch):
    from pathlib import Path

    checkpoint = Path("models/text_detector/transformer/distilbert_hc3_v1")
    if not (checkpoint / "text_detector_metadata.json").is_file():
        return
    detector = TransformerTextDetector(checkpoint, device="cpu")
    monkeypatch.setattr(app.state, "text_detector", detector)
    text = "The local model must aggregate a final partial overflow window without truncating its text. " * 50
    response = client.post("/api/v1/predict/text", json={
        "text": text
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["score_type"] == "uncalibrated_softmax"
    assert body["chunk_count"] > 1
    assert body["input_character_count"] == len(text)
    assert abs(sum(body["class_scores"].values()) - 1.0) < 1e-5


def test_expanded_transformer_checkpoint_preserves_overflow_inference(monkeypatch):
    from pathlib import Path

    checkpoint = Path("models/text_detector/distilbert_expanded_v1")
    if not (checkpoint / "text_detector_metadata.json").is_file():
        return
    monkeypatch.setattr(app.state, "text_backend", "transformer_expanded")
    detector = TransformerTextDetector(checkpoint, device="cpu")
    monkeypatch.setattr(app.state, "text_detector", detector)
    text = "An expanded checkpoint must include every overlapping token window and pad its final shorter window safely. " * 55
    response = client.post("/api/v1/predict/text", json={"text": text})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"] == "distilbert_expanded_v1"
    assert body["confidence_type"] == "validation_temperature_scaled_softmax"
    assert body["chunk_count"] > 1
