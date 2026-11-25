"""Standalone Phase 3 API; runs separately so Phase 1/2 app files stay frozen."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from phase3.fusion.encoders import split_text
from phase3.inference import FusionNotReadyError, Phase3Predictor

app = FastAPI(title="AI Content Detector — Phase 3 Fusion", version="0.1.0",
              description="Isolated multimodal fusion API; fusion is unavailable until a paired dataset is trained.")
app.state.predictor = Phase3Predictor(device=os.getenv("PHASE3_DEVICE", "auto"),
                                      checkpoint=os.getenv("PHASE3_FUSION_CHECKPOINT", "phase3/models/fusion/best_model.pt"))


class Phase3TextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "phase": 3}


@app.get("/phase3/status")
def status():
    predictor: Phase3Predictor = app.state.predictor
    return {"phase": 3, "image_copy_available": Path("phase3/models/image_copy/resnet18_cifake_aug_v1_calibrated_strict.pt").is_file(),
            "text_copy_available": Path("phase3/models/text_copy/hc3_tfidf_linear_svm.joblib").is_file(),
            "fusion_checkpoint_available": predictor.fusion_available,
            "fusion_training_status": "BLOCKED_MISSING_LICENSED_PAIRED_DATASET" if not predictor.fusion_available else "CHECKPOINT_PRESENT",
            "fusion_checkpoint": str(predictor.checkpoint_path),
            "message": ("Phase 3 fusion is not trained: no paired dataset with verified labels and suitable terms is available."
                        if not predictor.fusion_available else "Fusion checkpoint is available; predictions remain experimental.")}


@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).resolve().parent / "static/index.html", media_type="text/html")


@app.post("/api/v1/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, "Upload must have an image content type")
    data = await file.read(10 * 1024 * 1024 + 1)
    await file.close()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 10 MiB")
    try:
        return app.state.predictor.predict_image(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/predict/text")
def predict_text(request: Phase3TextRequest):
    if len(request.text) < 40 or len(request.text.split()) < 5:
        raise HTTPException(422, "Enter at least 40 characters and 5 words")
    if len(request.text) > 50_000:
        raise HTTPException(422, "Text exceeds the 50,000-character limit")
    try:
        split_text(request.text)
        return app.state.predictor.predict_text(request.text)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/v1/predict/multimodal")
async def predict_multimodal(file: UploadFile = File(...), text: str = Form(...)):
    if len(text) < 40 or len(text.split()) < 5 or len(text) > 50_000:
        raise HTTPException(422, "Text must be between 40 and 50,000 characters and contain at least 5 words")
    data = await file.read(10 * 1024 * 1024 + 1)
    await file.close()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 10 MiB")
    predictor: Phase3Predictor = app.state.predictor
    if not predictor.fusion_available:
        raise HTTPException(503, "Phase 3 fusion is not trained; a valid, licensed paired dataset is required")
    try:
        return predictor.predict_multimodal(data, text)
    except (FusionNotReadyError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phase3.app:app", host="127.0.0.1", port=int(os.getenv("PHASE3_PORT", "8001")), reload=False)
