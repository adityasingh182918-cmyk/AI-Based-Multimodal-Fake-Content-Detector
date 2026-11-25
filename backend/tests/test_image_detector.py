import io

import pytest
from PIL import Image

from app.services.image_detector import (AspectPadResize, CLASS_NAMES, DEFAULT_MEAN, DEFAULT_STD,
                                         apply_decision_thresholds,
                                         ImageDetector, InvalidImageError, ModelNotTrainedError, build_resnet18)


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 100, 200)).save(stream, format="PNG")
    return stream.getvalue()


def test_missing_checkpoint_has_explicit_error(tmp_path) -> None:
    detector = ImageDetector(tmp_path / "missing.pt")

    with pytest.raises(ModelNotTrainedError, match="No trained model checkpoint"):
        detector.predict(_png_bytes())


def test_invalid_image_is_rejected_before_checkpoint_loading(tmp_path) -> None:
    detector = ImageDetector(tmp_path / "missing.pt")

    with pytest.raises(InvalidImageError, match="not a supported"):
        detector.predict(b"not an image")


def test_aspect_resize_pads_instead_of_stretching() -> None:
    source = Image.new("RGB", (40, 20), (255, 0, 0))
    result = AspectPadResize(32)(source)

    assert result.size == (32, 32)
    assert result.getpixel((16, 0)) == (0, 0, 0)
    assert result.getpixel((16, 16))[0] > 240


def test_exif_orientation_is_applied_before_inference() -> None:
    source = Image.new("RGB", (24, 12), (20, 50, 80))
    exif = source.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    source.save(stream, format="JPEG", exif=exif)

    oriented = ImageDetector._open_image(stream.getvalue())

    assert oriented.size == (12, 24)


def test_validation_thresholds_support_abstention_band() -> None:
    thresholds = {"real_max_ai_score": 0.25, "ai_min_ai_score": 0.75}

    assert apply_decision_thresholds(0.20, thresholds) == "REAL"
    assert apply_decision_thresholds(0.25, thresholds) == "REAL"
    assert apply_decision_thresholds(0.50, thresholds) == "UNCERTAIN"
    assert apply_decision_thresholds(0.75, thresholds) == "AI_GENERATED"


def test_synthetic_checkpoint_loads_and_prediction_uses_checkpoint_preprocessing(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    torch.set_num_threads(1)
    model = build_resnet18(pretrained=False)
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({
        "architecture": "resnet18",
        "class_names": list(CLASS_NAMES),
        "image_size": 64,
        "mean": DEFAULT_MEAN,
        "std": DEFAULT_STD,
        "run_id": "synthetic-test-checkpoint",
        "state_dict": model.state_dict(),
    }, checkpoint)

    detector = ImageDetector(checkpoint, device="cpu")
    assert detector.initialize() is True
    assert detector.is_loaded is True
    tensor = detector._runtime[1](detector._open_image(_png_bytes()))
    assert tuple(tensor.shape) == (3, 64, 64)
    result = detector.predict(_png_bytes())

    assert result["predicted_label"] in CLASS_NAMES
    assert result["model_version"] == "synthetic-test-checkpoint"
    assert set(result["class_scores"]) == set(CLASS_NAMES)
    assert sum(result["class_scores"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["score_type"] == "uncalibrated_softmax"
