import json

from backend.tests.http_client import InProcessClient
from unified import app as app_module
from unified import history


class FakeImageDetector:
    checkpoint_path = type("PathInfo", (), {"name": "frozen-image.pt"})()
    architecture = "resnet18"
    checkpoint_identifier = "frozen-image-v1"
    checkpoint_available = True
    is_loaded = True

    def predict(self, _data):
        return {"label": "REAL", "predicted_label": "REAL", "score": .9,
                "score_type": "uncalibrated_softmax", "class_scores": {"REAL": .9, "AI_GENERATED": .1},
                "model_version": self.checkpoint_identifier, "processing_time_ms": 2.0}


class FakeTextDetector:
    checkpoint_path = type("PathInfo", (), {"name": "frozen-text.joblib", "stem": "frozen-text"})()
    _metadata = {"model_name": "frozen SVM", "model_version": "frozen-text-v1"}
    ready = True
    is_loaded = True

    def predict(self, text):
        if len(text) < 40 or len(text.split()) < 5:
            from app.services.text_detector import InvalidTextError
            raise InvalidTextError("short text")
        return {"label": "HUMAN_WRITTEN", "predicted_label": "HUMAN_WRITTEN", "score": .8,
                "score_type": "calibrated_tfidf_probability", "class_scores": {"HUMAN_WRITTEN": .8, "AI_GENERATED": .2},
                "model_name": "frozen SVM", "model_version": "frozen-text-v1", "processing_time_ms": 1.0}


class FakePhase3:
    fusion_available = False
    checkpoint_path = type("PathInfo", (), {"name": "fusion_v1.pt"})()

    def predict_image(self, _data):
        return {"label": "REAL", "class_scores": {"REAL": .9, "AI_GENERATED": .1},
                "score_type": "uncalibrated_softmax", "model_version": "phase3-image-copy"}

    def predict_text(self, _text):
        return {"label": "HUMAN_WRITTEN", "class_scores": {"HUMAN_WRITTEN": .8, "AI_GENERATED": .2},
                "score_type": "calibrated_tfidf_probability", "model_version": "phase3-text-copy"}


def test_unified_dashboard_serves_three_modes_and_inline_favicon():
    client = InProcessClient(app_module.app)
    page = client.get("/")
    assert page.status_code == 200
    assert '<link rel="icon" href="data:,">' in page.text
    assert 'data-view="image"' in page.text
    assert 'data-view="text"' in page.text
    assert 'data-view="multimodal"' in page.text


def test_unified_modes_history_and_rule_based_multimodal_are_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.sqlite3")
    monkeypatch.setattr(history, "IMAGE_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_module, "RETAIN_INPUTS", False)
    monkeypatch.setattr(app_module.phase12_app.state, "image_detector", FakeImageDetector())
    monkeypatch.setattr(app_module.phase12_app.state, "text_detector", FakeTextDetector())
    monkeypatch.setattr(app_module.phase12_app.state, "text_backend", "baseline")
    monkeypatch.setattr(app_module, "phase3_predictor", FakePhase3())
    history.initialize()
    client = InProcessClient(app_module.app)

    image = client.post("/api/v1/predict/image", files={"file": ("private.png", b"fixture", "image/png")})
    assert image.status_code == 200 and image.json()["label"] == "REAL"
    assert history.get(image.json()["analysis_id"])["image_hash"]

    text = "A sufficiently long human-written test passage that contains at least five words."
    text_result = client.post("/api/v1/predict/text", json={"text": text})
    assert text_result.status_code == 200 and text_result.json()["label"] == "HUMAN_WRITTEN"
    assert history.get(text_result.json()["analysis_id"])["text_retained"] is False

    paired = client.post("/api/v1/predict/multimodal", data={"text": text},
                         files={"file": ("pair.png", b"fixture", "image/png")})
    assert paired.status_code == 200
    assert paired.json()["image_prediction"]["label"] == "REAL"
    assert paired.json()["text_prediction"]["label"] == "HUMAN_WRITTEN"
    assert paired.json()["image_prediction"]["model_version"] == "frozen-image-v1"
    assert paired.json()["text_prediction"]["model_version"] == "frozen-text-v1"
    assert paired.json()["analysis_type"] == "rule_based_multimodal"
    assert paired.json()["fusion_type"] == "rule_based"
    assert paired.json()["trained_fusion_model"] is False
    assert paired.json()["combined_result"]["label"] == "AUTHENTIC CONTENT"
    assert paired.json()["image_result"] == paired.json()["image_prediction"]
    assert paired.json()["text_result"] == paired.json()["text_prediction"]
    assert paired.json()["fusion_prediction"] == "AUTHENTIC CONTENT"
    assert paired.json()["fusion_status"] == "RULE_BASED"
    assert "no trained fusion neural network" in paired.json()["fusion_message"]

    rows = client.get("/api/v1/history?detection_type=multimodal")
    assert rows.status_code == 200 and rows.json()["total"] == 1
    assert rows.json()["items"][0]["primary_prediction"] == "AUTHENTIC CONTENT"
    assert client.get("/api/v1/history?prediction=AUTHENTIC%20CONTENT").json()["total"] == 1
    detail = client.get("/api/v1/history/" + paired.json()["analysis_id"])
    assert detail.status_code == 200
    assert detail.json()["response"]["fusion_status"] == "RULE_BASED"
    assert detail.json()["model_name"] == "Rule-based multimodal analysis"
    assert detail.json()["model_version"] == "rule_based_multimodal_v1"
    assert detail.json()["image_retained"] is False and detail.json()["text_retained"] is False
    assert client.get("/api/v1/history/export.csv").text.count("analysis_id") >= 1
    assert client.get("/api/v1/dashboard").json()["total_analyses"] == 3
    status = client.get("/api/v1/status").json()
    assert status["phase3"]["ready"] is True
    assert status["phase3"]["fusion_type"] == "rule_based"
    assert status["phase3"]["trained_fusion_model"] is False
    assert status["phase3"]["training_status"] == "NOT_TRAINED"
    assert client.get("/api/v1/privacy").json()["retain_inputs"] is False
    deleted = client.request("DELETE", "/api/v1/history/" + image.json()["analysis_id"])
    assert deleted.status_code == 200
    assert client.request("DELETE", "/api/v1/history").status_code == 400
    cleared = client.request("DELETE", "/api/v1/history?confirm=true")
    assert cleared.status_code == 200 and cleared.json()["deleted_count"] == 2
