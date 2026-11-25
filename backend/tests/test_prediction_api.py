import asyncio
import io
import importlib

from PIL import Image
import pytest

from app.main import app
from app.schemas.prediction import ImagePredictionResponse
from app.services.image_detector import CLASS_NAMES, DEFAULT_MEAN, DEFAULT_STD, ImageDetector, build_resnet18
from backend.tests.http_client import InProcessClient

client = InProcessClient(app)


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 100, 200)).save(stream, format="PNG")
    return stream.getvalue()


def test_model_status_reports_missing_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app.state, "image_detector", ImageDetector(tmp_path / "missing.pt"))

    response = client.get("/model-status")

    assert response.status_code == 200
    status = response.json()
    assert status["trained_checkpoint_available"] is False
    assert status["checkpoint_valid"] is False
    assert status["ready_for_inference"] is False
    assert status["architecture"] == "resnet18"


def test_prediction_returns_503_when_checkpoint_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app.state, "image_detector", ImageDetector(tmp_path / "missing.pt"))

    response = client.post(
        "/api/v1/predict/image",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]


def test_prediction_rejects_invalid_image(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app.state, "image_detector", ImageDetector(tmp_path / "missing.pt"))

    response = client.post(
        "/api/v1/predict/image",
        files={"file": ("sample.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400


def test_prediction_endpoint_loads_synthetic_checkpoint_and_returns_scores(monkeypatch, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    torch.set_num_threads(1)
    model = build_resnet18(pretrained=False)
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({"architecture": "resnet18", "class_names": list(CLASS_NAMES), "image_size": 64,
                "mean": DEFAULT_MEAN, "std": DEFAULT_STD, "run_id": "api-synthetic-test",
                "state_dict": model.state_dict()}, checkpoint)
    detector = ImageDetector(checkpoint, device="cpu")
    monkeypatch.setattr(app.state, "image_detector", detector)

    async def run_startup() -> None:
        async with app.router.lifespan_context(app):
            assert detector.is_loaded

    asyncio.run(run_startup())

    response = client.post(
        "/api/v1/predict/image",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASS_NAMES
    assert body["predicted_label"] == body["label"]
    assert body["score_type"] == "uncalibrated_softmax"
    assert set(body["class_scores"]) == set(CLASS_NAMES)
    assert body["architecture"] == "resnet18"
    assert body["model_version"] == "api-synthetic-test"
    assert "Experimental" in body["warning"]
    status = client.get("/model-status").json()
    assert status["trained_checkpoint_available"] is True
    assert status["checkpoint_valid"] is True
    assert status["ready_for_inference"] is True
    assert status["inference_device"] == "cpu"


def test_prediction_rejects_non_image_content_type(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app.state, "image_detector", ImageDetector(tmp_path / "missing.pt"))

    response = client.post("/api/v1/predict/image", files={"file": ("note.txt", b"text", "text/plain")})

    assert response.status_code == 415


def test_prediction_enforces_configured_upload_limit(monkeypatch, tmp_path) -> None:
    main_module = importlib.import_module("app.main")
    monkeypatch.setattr(main_module, "_max_upload_bytes", 10)
    monkeypatch.setattr(app.state, "image_detector", ImageDetector(tmp_path / "missing.pt"))

    response = client.post("/api/v1/predict/image", files={"file": ("sample.png", _png_bytes(), "image/png")})

    assert response.status_code == 413


def test_browser_ui_is_served_by_fastapi() -> None:
    response = client.get("/ui")

    assert response.status_code == 200
    assert "Image and Text Detection" in response.text
    assert "Image detection" in response.text
    assert "/api/v1/predict/image" in response.text


def test_prediction_response_schema_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError):
        ImagePredictionResponse(
            label="REAL", predicted_label="REAL", score=1.5, score_type="uncalibrated_softmax",
            class_scores={"REAL": 1.5, "AI_GENERATED": 0.0}, architecture="resnet18",
            model_version="test", warning="experimental",
        )
