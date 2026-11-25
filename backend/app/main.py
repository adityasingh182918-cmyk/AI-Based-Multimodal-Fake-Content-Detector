"""FastAPI application entry point."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas.prediction import (ImagePredictionResponse, ModelStatusResponse, TextModelStatusResponse,
                                    TextPredictionRequest, TextPredictionResponse)
from app.schemas.status import HealthResponse, WelcomeResponse
from app.services.image_detector import ImageDetector, InvalidImageError, ModelNotTrainedError
from app.services.text_detector import (MAX_TEXT_CHARS, InvalidTextError, TextDetector,
                                        TextModelNotReadyError)
from app.services.transformer_text_detector import TransformerTextDetector

PROJECT_NAME = "AI-Based Multimodal Fake Content Detector"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    detector: ImageDetector = application.state.image_detector
    if detector.checkpoint_path.is_file() and not detector.initialize():
        print(f"Image model checkpoint is not ready: {detector.load_error or 'initialization failed'}")
    elif detector.is_loaded:
        print(f"Image detector ready: {detector.architecture} on {detector.inference_device}")
    yield

app = FastAPI(
    title=PROJECT_NAME,
    description="Experimental image and text detection APIs. Predictions require locally trained checkpoints.",
    version="0.2.0",
    lifespan=lifespan,
)
_default_checkpoint = Path(__file__).resolve().parents[2] / "models" / "image_detector" / "resnet18_realworld_v1.pt"
_max_upload_bytes = int(os.getenv("IMAGE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
if _max_upload_bytes < 1:
    raise RuntimeError("IMAGE_MAX_UPLOAD_BYTES must be a positive integer")
app.state.image_detector = ImageDetector(
    os.getenv("IMAGE_MODEL_CHECKPOINT", str(_default_checkpoint)),
    device=os.getenv("IMAGE_MODEL_DEVICE", "auto").lower(),
)
TEXT_BACKEND_CHECKPOINTS = {
    "baseline": ("svm", Path("models/text_detector/baseline/hc3_tfidf_linear_svm.joblib")),
    "svm_expanded": ("svm", Path("models/text_detector/svm_expanded_v1.joblib")),
    "transformer": ("transformer", Path("models/text_detector/transformer/distilbert_hc3_v1")),
    "transformer_expanded": ("transformer", Path("models/text_detector/distilbert_expanded_v1")),
}


def create_text_detector(backend: str | None = None, checkpoint_override: str | None = None):
    selected = (backend or os.getenv("TEXT_MODEL_BACKEND", "baseline")).lower()
    if selected not in TEXT_BACKEND_CHECKPOINTS:
        raise RuntimeError("TEXT_MODEL_BACKEND must be baseline, svm_expanded, transformer, or transformer_expanded")
    kind, relative_path = TEXT_BACKEND_CHECKPOINTS[selected]
    default_path = Path(__file__).resolve().parents[2] / relative_path
    checkpoint = Path(checkpoint_override or os.getenv("TEXT_MODEL_CHECKPOINT") or default_path)
    if kind == "transformer":
        detector = TransformerTextDetector(checkpoint, device=os.getenv("TEXT_MODEL_DEVICE", "auto").lower())
    else:
        detector = TextDetector(checkpoint)
    return selected, detector


_text_backend, app_text_detector = create_text_detector()
app.state.text_backend = _text_backend
app.state.text_detector = app_text_detector


@app.get("/", response_model=WelcomeResponse, tags=["status"])
def read_root() -> WelcomeResponse:
    """Return the project welcome message."""
    return WelcomeResponse(
        message="Welcome to the project API.",
        project=PROJECT_NAME,
    )


@app.get("/health", response_model=HealthResponse, tags=["status"])
def read_health() -> HealthResponse:
    """Report that the API process is responding."""
    return HealthResponse(status="ok")


@app.get("/ui", include_in_schema=False)
def image_detection_ui() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html", media_type="text/html")


@app.get("/model-status", response_model=ModelStatusResponse, tags=["image"])
def model_status() -> ModelStatusResponse:
    detector: ImageDetector = app.state.image_detector
    file_exists = detector.checkpoint_path.is_file()
    valid = detector.checkpoint_available
    return ModelStatusResponse(
        trained_checkpoint_available=file_exists,
        checkpoint_valid=valid,
        checkpoint_identifier=detector.checkpoint_path.name,
        architecture=detector.architecture,
        inference_device=detector.inference_device,
        loaded=detector.is_loaded,
        ready_for_inference=valid and detector.is_loaded,
        max_upload_bytes=_max_upload_bytes,
        message=("Image model is ready." if valid and detector.is_loaded else
                 "No valid trained checkpoint is available; train the image model before prediction."),
    )


@app.get("/text-model-status", response_model=TextModelStatusResponse, tags=["text"])
def text_model_status() -> TextModelStatusResponse:
    detector: TextDetector = app.state.text_detector
    available = detector.checkpoint_available if hasattr(detector, "checkpoint_available") else detector.checkpoint_path.is_file()
    metadata_path = (detector.checkpoint_path / "text_detector_metadata.json" if detector.checkpoint_path.is_dir()
                     else detector.checkpoint_path.with_suffix(".json"))
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        metadata = {}
    fallback_name = ("DistilBERT HC3 fine-tune" if hasattr(detector, "checkpoint_available")
                     else "HC3 word+character TF-IDF Linear SVM")
    return TextModelStatusResponse(checkpoint_available=available, loaded=detector.is_loaded,
                                   ready_for_inference=available, checkpoint_identifier=detector.checkpoint_path.name,
                                   selected_backend=app.state.text_backend,
                                   model_name=metadata.get("model_name", fallback_name),
                                   model_version=metadata.get("model_version", detector.checkpoint_path.stem),
                                   max_characters=MAX_TEXT_CHARS,
                                   message="Text model checkpoint is available." if available else
                                   "Text model is not trained; run the documented baseline training command.")


@app.post("/api/v1/predict/image", response_model=ImagePredictionResponse, tags=["image"])
async def predict_image(file: UploadFile = File(...)) -> ImagePredictionResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        await file.close()
        raise HTTPException(status_code=415, detail="Upload must have an image content type")
    try:
        image_bytes = await file.read(_max_upload_bytes + 1)
    finally:
        await file.close()
    if len(image_bytes) > _max_upload_bytes:
        limit_mib = _max_upload_bytes / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Image exceeds the configured {limit_mib:g} MiB upload limit")
    detector: ImageDetector = app.state.image_detector
    try:
        result = detector.predict(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotTrainedError as exc:
        print(f"Image prediction unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Image model is not ready; check model status and server configuration.") from exc
    except RuntimeError as exc:
        print(f"Image inference runtime failure: {exc}")
        raise HTTPException(status_code=503, detail="Image inference is unavailable; check the server environment and logs.") from exc
    return ImagePredictionResponse(**result)


@app.post("/api/v1/predict/text", response_model=TextPredictionResponse, tags=["text"])
def predict_text(request: TextPredictionRequest) -> TextPredictionResponse:
    detector: TextDetector = app.state.text_detector
    try:
        result = detector.predict(request.text)
    except InvalidTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TextModelNotReadyError as exc:
        print(f"Text prediction unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Text model is not ready; train it and check text model status.") from exc
    return TextPredictionResponse(**result)
