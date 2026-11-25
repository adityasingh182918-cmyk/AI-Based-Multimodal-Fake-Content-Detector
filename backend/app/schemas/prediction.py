"""Image prediction and model readiness API schemas."""

from typing import Literal

from pydantic import BaseModel, Field

ImageLabel = Literal["REAL", "AI_GENERATED", "UNCERTAIN"]


class ModelStatusResponse(BaseModel):
    trained_checkpoint_available: bool
    checkpoint_valid: bool
    checkpoint_identifier: str
    architecture: str
    inference_device: str
    loaded: bool
    ready_for_inference: bool
    max_upload_bytes: int
    message: str


class ImagePredictionResponse(BaseModel):
    label: ImageLabel
    predicted_label: ImageLabel
    score: float = Field(ge=0.0, le=1.0, description="Winning uncalibrated softmax score, or maximum class score when label is UNCERTAIN")
    score_type: Literal["uncalibrated_softmax"]
    class_scores: dict[ImageLabel, float]
    architecture: str
    model_version: str
    inference_device: str | None = None
    processing_time_ms: float | None = Field(default=None, ge=0.0)
    warning: str


TextLabel = Literal["HUMAN_WRITTEN", "AI_GENERATED", "UNCERTAIN"]


class TextPredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)


class TextPredictionResponse(BaseModel):
    label: TextLabel
    predicted_label: TextLabel
    score: float = Field(ge=0.0, le=1.0)
    score_type: Literal["calibrated_tfidf_probability", "uncalibrated_softmax", "validation_temperature_scaled_softmax"]
    class_scores: dict[Literal["HUMAN_WRITTEN", "AI_GENERATED"], float]
    model_name: str
    model_version: str | None = None
    confidence_type: Literal["calibrated_tfidf_probability", "uncalibrated_softmax", "validation_temperature_scaled_softmax"] | None = None
    decision_status: Literal["thresholded", "uncertain"] | None = None
    processing_time_ms: float = Field(ge=0.0)
    input_word_count: int = Field(ge=0)
    input_character_count: int = Field(ge=0)
    chunk_count: int = Field(ge=1)
    thresholds: dict[str, float]
    warning: str


class TextModelStatusResponse(BaseModel):
    checkpoint_available: bool
    loaded: bool
    ready_for_inference: bool
    checkpoint_identifier: str
    selected_backend: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    max_characters: int
    message: str
