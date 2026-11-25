"""Optional DistilBERT inference service; loaded only when selected in configuration."""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from app.services.text_detector import MAX_TEXT_CHARS, MIN_TEXT_CHARS, InvalidTextError, TextModelNotReadyError


class TransformerTextDetector:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto", max_length: int = 256,
                 inference_batch_size: int = 8) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.requested_device = device
        self.max_length = max_length
        self.inference_batch_size = inference_batch_size
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._thresholds = {"real_max_ai_score": .5, "ai_min_ai_score": .5}
        self._temperature = 1.0
        self._metadata: dict[str, Any] = {"model_name": "DistilBERT HC3 fine-tune",
                                          "model_version": self.checkpoint_path.name}
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def checkpoint_available(self) -> bool:
        return (self.checkpoint_path / "config.json").is_file() and (self.checkpoint_path / "text_detector_metadata.json").is_file()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def ready(self) -> bool:
        return self.checkpoint_available

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.checkpoint_available:
                raise TextModelNotReadyError(f"Transformer text checkpoint is incomplete: {self.checkpoint_path}")
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                metadata = json.loads((self.checkpoint_path / "text_detector_metadata.json").read_text(encoding="utf-8"))
                self._metadata = metadata
                if self.requested_device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested for text inference but CUDA is unavailable")
                device_name = ("cuda" if torch.cuda.is_available() else "cpu") if self.requested_device == "auto" else self.requested_device
                if device_name not in {"cuda", "cpu"}:
                    raise ValueError(f"Unsupported text inference device: {device_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_path, local_files_only=True)
                model.to(torch.device(device_name)).eval()
                thresholds = metadata["decision_thresholds"]
                low, high = float(thresholds["real_max_ai_score"]), float(thresholds["ai_min_ai_score"])
                if not 0 <= low < high <= 1:
                    raise ValueError("Invalid validation-derived text decision thresholds")
                temperature = float(metadata.get("inference_temperature", 1.0))
                if not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError("Invalid validation-fitted inference temperature")
                self._model, self._tokenizer, self._torch = model, tokenizer, torch
                self._device, self._thresholds = torch.device(device_name), {"real_max_ai_score": low, "ai_min_ai_score": high}
                self._temperature = temperature
                self.max_length = int(metadata["max_length"])
            except Exception as exc:
                self._load_error = str(exc)
                self._model = None
                raise TextModelNotReadyError(f"Unable to load transformer text model: {exc}") from exc

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise InvalidTextError("Text must not be empty")
        if len(text) < MIN_TEXT_CHARS or len(text.split()) < 5:
            raise InvalidTextError(f"Enter at least {MIN_TEXT_CHARS} characters and 5 words")
        if len(text) > MAX_TEXT_CHARS:
            raise InvalidTextError(f"Text exceeds the {MAX_TEXT_CHARS:,}-character limit")
        started = time.perf_counter()
        self._ensure_loaded()
        encoded = self._tokenizer(text, truncation=True, max_length=self.max_length, stride=32,
                                  return_overflowing_tokens=True, return_tensors="pt", padding=True)
        chunk_count = int(encoded["input_ids"].shape[0])
        overflow_map = encoded.pop("overflow_to_sample_mapping", None)
        weights = encoded["attention_mask"].sum(dim=1).float()
        logits = []
        for start in range(0, chunk_count, self.inference_batch_size):
            batch = {key: value[start:start + self.inference_batch_size].to(self._device)
                     for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
            with self._torch.inference_mode():
                output = self._model(**batch)
                logits.append(output.logits.float().cpu())
        scores = self._torch.softmax(self._torch.cat(logits) / self._temperature, dim=1)[:, 1]
        ai_score = float((scores * weights).sum().item() / weights.sum().item())
        if ai_score <= self._thresholds["real_max_ai_score"]:
            label = "HUMAN_WRITTEN"
        elif ai_score >= self._thresholds["ai_min_ai_score"]:
            label = "AI_GENERATED"
        else:
            label = "UNCERTAIN"
        score_type = ("validation_temperature_scaled_softmax" if self._temperature != 1.0 else "uncalibrated_softmax")
        score_warning = ("Scores use a scalar temperature fitted only on the HC3+MAGE validation split; calibration may shift out of domain. "
                         if self._temperature != 1.0 else "Raw softmax scores are uncalibrated. ")
        return {"label": label, "predicted_label": label,
                "class_scores": {"HUMAN_WRITTEN": 1.0 - ai_score, "AI_GENERATED": ai_score},
                "score": ai_score if label == "AI_GENERATED" else 1.0 - ai_score,
                "score_type": score_type,
                "confidence_type": score_type,
                "decision_status": "uncertain" if label == "UNCERTAIN" else "thresholded",
                "model_name": self._metadata.get("model_name", "DistilBERT HC3 fine-tune"),
                "model_version": self._metadata.get("model_version", self.checkpoint_path.name),
                "processing_time_ms": (time.perf_counter() - started) * 1000.0,
                "input_word_count": len(text.split()), "input_character_count": len(text), "chunk_count": chunk_count,
                "thresholds": self._thresholds,
                "warning": f"Experimental text detector. {score_warning}Validation-selected thresholds and overlapping-window aggregation are not validated for out-of-domain or document-level detection. Scores are not proof of authorship."}
