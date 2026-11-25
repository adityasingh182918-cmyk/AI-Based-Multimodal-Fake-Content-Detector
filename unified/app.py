"""Single local gateway/UI over frozen Phase 1/2 and isolated Phase 3 services."""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app as phase12_app  # noqa: E402
from app.services.image_detector import InvalidImageError, ModelNotTrainedError  # noqa: E402
from app.services.text_detector import InvalidTextError, TextModelNotReadyError  # noqa: E402
from phase3.inference import Phase3Predictor  # noqa: E402
from phase3.fusion.rule_based_analysis import build_analysis  # noqa: E402
from unified import history  # noqa: E402

MAX_IMAGE_BYTES = int(os.getenv("IMAGE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
RETAIN_INPUTS = os.getenv("UNIFIED_RETAIN_INPUTS", "false").strip().lower() in {"1", "true", "yes"}
PHASE3_DEVICE = os.getenv("PHASE3_DEVICE", "auto")
FUSION_CHECKPOINT = os.getenv("PHASE3_FUSION_CHECKPOINT", str(ROOT / "phase3/models/fusion/best_model.pt"))
phase3_predictor = Phase3Predictor(PHASE3_DEVICE, FUSION_CHECKPOINT)
logger = logging.getLogger(__name__)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _model_name(phase: int, result: dict[str, Any] | None = None) -> tuple[str, str]:
    if phase == 1:
        detector = phase12_app.state.image_detector
        return detector.architecture, detector.checkpoint_identifier
    if phase == 2:
        detector = phase12_app.state.text_detector
        return str(getattr(detector, "_metadata", {}).get("model_name", phase12_app.state.text_backend)), \
            str(getattr(detector, "_metadata", {}).get("model_version", detector.checkpoint_path.stem))
    return "Rule-based multimodal analysis", str((result or {}).get("model_version", "rule_based_phase3_v1"))


def _history_record(phase: int, detection_type: str, response: dict[str, Any], *,
                    image_bytes: bytes | None = None, filename: str | None = None,
                    image_mime_type: str | None = None, text: str | None = None,
                    status: str = "completed", error: str | None = None,
                    started_at: float | None = None) -> str:
    image_result = (response.get("image_result", response.get("image_prediction"))
                    if phase == 3 else (response if phase == 1 else None))
    text_result = (response.get("text_result", response.get("text_prediction"))
                   if phase == 3 else (response if phase == 2 else None))
    combined = response.get("combined_result") or {}
    fusion_label = combined.get("label") or response.get("fusion_prediction")
    primary = (response.get("label") if phase in (1, 2) else
               fusion_label or ("PARTIAL ANALYSIS" if response.get("analysis_status") == "partial" else "INCONCLUSIVE"))
    model_name, model_version = _model_name(phase, response)
    if phase == 3:
        model_name = "Rule-based multimodal analysis"
        model_version = str(response.get("model_version", "rule_based_phase3_v1"))
    record = {"analysis_id": history.new_id(), "phase": phase, "detection_type": detection_type,
              "image_filename": Path(filename).name[:255] if filename else None,
              "image_hash": sha256(image_bytes) if image_bytes is not None else None,
              "image_mime_type": image_mime_type or "application/octet-stream",
              "text_hash": sha256(text) if text is not None else None,
              "input_text": text, "primary_prediction": primary,
              "image_prediction": (image_result or {}).get("label"),
              "text_prediction": (text_result or {}).get("label"), "fusion_prediction": fusion_label,
              "confidence": response.get("score", (response.get("fusion_confidence") or {}).get("value")),
              "confidence_type": response.get("score_type", (response.get("fusion_confidence") or {}).get("type")),
              "model_name": model_name, "model_version": model_version,
              "processing_time_ms": response.get("processing_time_ms", (time.perf_counter() - started_at) * 1000 if started_at else None),
              "created_at": now_iso(), "status": status, "error_message": error, "response": response}
    return history.save_record(record, image_bytes=image_bytes, retain_inputs=RETAIN_INPUTS)


def _with_history(response: dict[str, Any], analysis_id: str) -> dict[str, Any]:
    return {**response, "analysis_id": analysis_id, "history_saved": analysis_id is not None,
            "input_retention": "enabled" if RETAIN_INPUTS else "disabled; hashes and results only",
            **({} if analysis_id is not None else {"history_warning": "Analysis completed, but history could not be saved."})}


def _save_history_safely(*args: Any, **kwargs: Any) -> str | None:
    try:
        return _history_record(*args, **kwargs)
    except Exception:
        logger.exception("Could not save prediction history; returning detector result without history.")
        return None


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type and not file.content_type.startswith("image/"):
        await file.close()
        raise HTTPException(status_code=415, detail="Upload must have an image content type")
    try:
        data = await file.read(MAX_IMAGE_BYTES + 1)
    finally:
        await file.close()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit")
    return data


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        history.initialize()
        app.state.history_ready = True
    except Exception:
        app.state.history_ready = False
        logger.exception("History database is unavailable; predictions can continue without persistence.")
    detector = phase12_app.state.image_detector
    if detector.checkpoint_path.is_file() and not detector.is_loaded:
        detector.initialize()
    yield


app = FastAPI(title="AI-Based Multimodal Fake Content Detector", version="1.0.0",
              description="One local UI and API gateway for Phase 1 image, Phase 2 text, and isolated Phase 3 multimodal analysis.",
              lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).resolve().parent / "static/index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "application": "unified", "history_available": getattr(app.state, "history_ready", None),
            "history_database": str(history.DB_PATH)}


@app.get("/api/v1/status")
def model_status():
    image = phase12_app.state.image_detector
    text = phase12_app.state.text_detector
    image_ready = bool(image.checkpoint_available)
    text_ready = bool(getattr(text, "checkpoint_available", getattr(text, "ready", False)))
    trained_fusion_ready = phase3_predictor.fusion_available
    phase3_image_copy = (ROOT / "phase3/models/image_copy/resnet18_cifake_aug_v1_calibrated_strict.pt").is_file()
    phase3_text_copy = (ROOT / "phase3/models/text_copy/hc3_tfidf_linear_svm.joblib").is_file()
    return {"phase1": {"ready": image_ready, "loaded": image.is_loaded,
                       "checkpoint": image.checkpoint_path.name, "architecture": image.architecture,
                       "model_version": image.checkpoint_identifier,
                       "message": "Image checkpoint available." if image_ready else "Image checkpoint missing."},
            "phase2": {"ready": text_ready, "loaded": text.is_loaded,
                       "backend": phase12_app.state.text_backend, "checkpoint": text.checkpoint_path.name,
                       "model_name": getattr(text, "_metadata", {}).get("model_name", phase12_app.state.text_backend),
                       "model_version": getattr(text, "_metadata", {}).get("model_version", text.checkpoint_path.stem),
                       "message": "Text checkpoint available." if text_ready else "Text checkpoint missing."},
            "phase3": {"architecture_available": True, "ready": True,
                       "image_copy_available": phase3_image_copy, "text_copy_available": phase3_text_copy,
                       "fusion_type": "rule_based", "trained_fusion_model": False,
                       "trained_fusion_available": trained_fusion_ready,
                       "training_status": "CHECKPOINT_PRESENT" if trained_fusion_ready else "NOT_TRAINED",
                       "model_version": "rule_based_multimodal_v1",
                       "message": "Rule-based multimodal analysis is ready. No trained fusion neural network is available."}}


@app.post("/api/v1/predict/image")
async def predict_image(file: UploadFile = File(...)):
    filename = file.filename
    mime_type = file.content_type
    data = await _read_image(file)
    started = time.perf_counter()
    detector = phase12_app.state.image_detector
    try:
        result = detector.predict(data)
    except InvalidImageError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except ModelNotTrainedError as exc:
        raise HTTPException(503, detail="Image model is not ready; check model status and configuration.") from exc
    except RuntimeError as exc:
        raise HTTPException(503, detail="Image inference is unavailable; check server configuration.") from exc
    analysis_id = _save_history_safely(1, "image", result, image_bytes=data, filename=filename,
                                  image_mime_type=mime_type,
                                  started_at=started)
    return _with_history(result, analysis_id)


@app.post("/api/v1/predict/text")
def predict_text(request: TextRequest):
    started = time.perf_counter()
    detector = phase12_app.state.text_detector
    try:
        result = detector.predict(request.text)
    except InvalidTextError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except TextModelNotReadyError as exc:
        raise HTTPException(503, detail="Text model is not ready; check model status and configuration.") from exc
    analysis_id = _save_history_safely(2, "text", result, text=request.text, started_at=started)
    return _with_history(result, analysis_id)


@app.post("/api/v1/predict/multimodal")
async def predict_multimodal(file: UploadFile = File(...), text: str = Form("")):
    filename = file.filename
    mime_type = file.content_type
    if len(text) > 50_000:
        raise HTTPException(422, detail="Text exceeds the 50,000-character limit")
    data = await _read_image(file)
    started = time.perf_counter()
    image_result: dict[str, Any] | None = None
    text_result: dict[str, Any] | None = None
    image_error: str | None = None
    text_error: str | None = None
    # Reuse the same live instances as the Phase 1/2 endpoints. Each modality
    # is isolated so one detector failure does not discard the other result.
    try:
        image_result = phase12_app.state.image_detector.predict(data)
    except InvalidImageError:
        image_error = "The uploaded image could not be read by the image detector."
    except (ModelNotTrainedError, RuntimeError, ValueError, OSError):
        image_error = "Image analysis is unavailable because the image detector failed."
    except Exception:
        logger.exception("Unexpected Phase 1 error during multimodal analysis.")
        image_error = "Image analysis failed unexpectedly."
    try:
        text_result = phase12_app.state.text_detector.predict(text)
    except InvalidTextError as exc:
        text_error = str(exc)
    except TextModelNotReadyError:
        text_error = "Text analysis is unavailable because the text detector is not ready."
    except (RuntimeError, ValueError, OSError):
        text_error = "Text analysis is unavailable because the text detector failed."
    except Exception:
        logger.exception("Unexpected Phase 2 error during multimodal analysis.")
        text_error = "Text analysis failed unexpectedly."

    result = {"phase": 3, **build_analysis(image_result, text_result, image_error, text_error),
              "fusion_prediction": None, "fusion_class_scores": None,
              "fusion_status": "RULE_BASED",
              "fusion_message": "The combined verdict follows the Phase 1/2 labels; no trained fusion neural network is available.",
              "model_version": "rule_based_multimodal_v1", "timestamp": now_iso(),
              "processing_time_ms": (time.perf_counter() - started) * 1000,
              "image_prediction": image_result, "text_prediction": text_result}
    if result["combined_result"] is not None:
        result["fusion_prediction"] = result["combined_result"]["label"]
    status = "partial" if result["analysis_status"] == "partial" else "completed"
    analysis_id = _save_history_safely(3, "multimodal", result, image_bytes=data, filename=filename,
                                       image_mime_type=mime_type,
                                       text=text, status=status, started_at=started)
    return _with_history(result, analysis_id)


@app.get("/api/v1/dashboard")
def dashboard():
    return history.stats()


@app.get("/api/v1/privacy")
def privacy_status():
    return {"retain_inputs": RETAIN_INPUTS,
            "database_path": str(history.DB_PATH),
            "image_storage_path": str(history.IMAGE_DIR) if RETAIN_INPUTS else None,
            "raw_inputs_saved_by_default": False,
            "note": "History has no authentication; keep this local app on a trusted machine."}


@app.get("/api/v1/history")
def history_list(offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100),
                 detection_type: str | None = None, prediction: str | None = None,
                 start_date: str | None = None, end_date: str | None = None):
    if detection_type and detection_type not in {"all", "image", "text", "multimodal"}:
        raise HTTPException(422, detail="Invalid detection_type filter")
    return history.list_records(offset=offset, limit=limit, detection_type=detection_type,
                                prediction=prediction, start_date=start_date, end_date=end_date)


@app.get("/api/v1/history/export.csv")
def history_export():
    return Response(history.export_csv(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=prediction-history.csv"})


@app.get("/api/v1/history/{analysis_id}")
def history_detail(analysis_id: str):
    record = history.get(analysis_id)
    if not record:
        raise HTTPException(404, detail="History record not found")
    return record


@app.get("/api/v1/history/{analysis_id}/image")
def history_image(analysis_id: str):
    path = history.image_path(analysis_id)
    if not path:
        raise HTTPException(404, detail="No retained image is available for this record")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                        headers={"Content-Disposition": "inline"})


@app.delete("/api/v1/history/{analysis_id}")
def delete_history_record(analysis_id: str):
    if not history.delete_record(analysis_id):
        raise HTTPException(404, detail="History record not found")
    return {"deleted": True, "analysis_id": analysis_id}


@app.delete("/api/v1/history")
def clear_history(confirm: bool = Query(False)):
    if not confirm:
        raise HTTPException(400, detail="Set confirm=true to delete all history")
    return {"deleted_count": history.clear()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unified.app:app", host="127.0.0.1", port=int(os.getenv("UNIFIED_PORT", "8000")), reload=False)
