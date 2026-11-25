import json
from pathlib import Path

from backend.tests.http_client import InProcessClient
from phase3.app import app
from phase3.fusion.encoders import FrozenEncoders

ROOT = Path(__file__).resolve().parents[2]
client = InProcessClient(app)


def test_copied_encoder_outputs_match_pre_phase3_fixed_predictions():
    snapshot = json.loads((ROOT / "phase3/reports/original_state_baseline.json").read_text(encoding="utf-8"))
    encoders = FrozenEncoders("auto")
    for expected in snapshot["predictions"]["image_predictions"]:
        image_path = ROOT / "data/image/cifake_ready" / expected["relative_path"]
        _, output = encoders.image(image_path)
        assert output["label"] == expected["predicted_label"]
        for key, value in expected["class_scores"].items():
            assert abs(output["class_scores"][key] - value) < 1e-6
    text_rows = [json.loads(line) for line in (ROOT / "data/text/processed/hc3/test.jsonl").read_text(encoding="utf-8").splitlines()]
    for expected in snapshot["predictions"]["text_predictions"]:
        text = text_rows[expected["record_index"]]["text"]
        _, output = encoders.text(text)
        assert output["label"] == expected["predicted_label"]
        for key, value in expected["class_scores"].items():
            assert abs(output["class_scores"][key] - value) < 1e-6


def test_phase3_status_reports_copies_and_untrained_fusion():
    response = client.get("/phase3/status")
    assert response.status_code == 200
    body = response.json()
    assert body["image_copy_available"] and body["text_copy_available"]
    assert body["fusion_checkpoint_available"] is False
    assert body["fusion_training_status"] == "BLOCKED_MISSING_LICENSED_PAIRED_DATASET"


def test_phase3_multimodal_api_does_not_claim_untrained_fusion():
    response = client.post("/api/v1/predict/multimodal", data={"text": "This is a sufficiently long description with several words for a Phase 3 paired API validation request."},
                           files={"file": ("fixture.jpg", b"unread image bytes", "image/jpeg")})
    assert response.status_code == 503
    assert "not trained" in response.json()["detail"]


def test_phase3_copied_unimodal_api_routes_and_invalid_image_error():
    image_path = ROOT / "data/image/cifake_ready/images/test/REAL/00000.jpg"
    response = client.post("/api/v1/predict/image",
                           files={"file": ("sample.jpg", image_path.read_bytes(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["label"] in {"REAL", "AI_GENERATED", "UNCERTAIN"}

    invalid = client.post("/api/v1/predict/image",
                          files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert invalid.status_code == 400

    row = json.loads((ROOT / "data/text/processed/hc3/test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    text_response = client.post("/api/v1/predict/text", json={"text": row["text"]})
    assert text_response.status_code == 200
    assert text_response.json()["label"] in {"HUMAN_WRITTEN", "AI_GENERATED", "UNCERTAIN"}


def test_phase3_ui_is_separate_from_the_existing_frontend():
    response = client.get("/ui")
    assert response.status_code == 200
    assert "Image + Text Analysis" in response.text
    assert "Copied image model" in response.text and "Phase 3 fusion" in response.text
