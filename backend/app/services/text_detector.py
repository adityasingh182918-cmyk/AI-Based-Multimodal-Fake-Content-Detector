"""Lazy, reusable inference service for the local HC3 TF-IDF baseline."""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "models" / "text_detector" / "baseline" / "hc3_tfidf_linear_svm.joblib"
MAX_TEXT_CHARS = 50_000
MIN_TEXT_CHARS = 40
CHUNK_CHARS = 2_000


class TextModelNotReadyError(RuntimeError):
    pass


class InvalidTextError(ValueError):
    pass


def split_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split all content in order, preferring paragraph/sentence boundaries."""
    if not text.strip():
        raise InvalidTextError("Text must not be empty")
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = max(remaining.rfind("\n", 0, max_chars + 1), remaining.rfind(". ", 0, max_chars + 1))
        if cut < max_chars // 2:
            cut = max_chars
        else:
            cut += 1
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


class TextDetector:
    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._model: Any | None = None
        self._thresholds = {"real_max_ai_score": .5, "ai_min_ai_score": .5}
        self._metadata: dict[str, Any] = {"model_name": "HC3 word+character TF-IDF Linear SVM",
                                          "model_version": self.checkpoint_path.stem}
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def ready(self) -> bool:
        return self.checkpoint_path.is_file()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                if not self.checkpoint_path.is_file():
                    raise TextModelNotReadyError(f"Text model checkpoint is missing: {self.checkpoint_path}")
                try:
                    import joblib
                    self._model = joblib.load(self.checkpoint_path)
                    metadata_path = self.checkpoint_path.with_suffix(".json")
                    if metadata_path.is_file():
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        self._metadata.update(metadata)
                        self._thresholds = metadata["decision_thresholds"]
                    low = float(self._thresholds["real_max_ai_score"])
                    high = float(self._thresholds["ai_min_ai_score"])
                    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 1):
                        raise ValueError("invalid calibrated decision thresholds")
                except Exception as exc:
                    self._model = None
                    self._load_error = str(exc)
                    raise TextModelNotReadyError(f"Unable to load text model: {exc}") from exc

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise InvalidTextError("Text must not be empty")
        if len(text) < MIN_TEXT_CHARS or len(text.split()) < 5:
            raise InvalidTextError(f"Enter at least {MIN_TEXT_CHARS} characters and 5 words")
        if len(text) > MAX_TEXT_CHARS:
            raise InvalidTextError(f"Text exceeds the {MAX_TEXT_CHARS:,}-character limit")
        started = time.perf_counter()
        chunks = split_text(text)
        self._ensure_loaded()
        class_names = list(self._model.classes_)
        ai_index = class_names.index("AI_GENERATED")
        weights = [max(1, len(chunk.split())) for chunk in chunks]
        chunk_scores = self._model.predict_proba(chunks)
        ai_score = sum(float(row[ai_index]) * weight for row, weight in zip(chunk_scores, weights)) / sum(weights)
        if ai_score <= self._thresholds["real_max_ai_score"]:
            label = "HUMAN_WRITTEN"
        elif ai_score >= self._thresholds["ai_min_ai_score"]:
            label = "AI_GENERATED"
        else:
            label = "UNCERTAIN"
        version = self._metadata.get("model_version", self.checkpoint_path.stem)
        data_scope = "HC3+MAGE expanded training mixture" if version == "svm_expanded_v1" else "HC3 QA domain"
        return {"label": label, "predicted_label": label,
                "class_scores": {"HUMAN_WRITTEN": 1.0 - ai_score, "AI_GENERATED": ai_score},
                "score": ai_score if label == "AI_GENERATED" else 1.0 - ai_score,
                "score_type": "calibrated_tfidf_probability", "confidence_type": "calibrated_tfidf_probability",
                "decision_status": "uncertain" if label == "UNCERTAIN" else "thresholded",
                "model_name": self._metadata.get("model_name", "HC3 word+character TF-IDF Linear SVM"),
                "model_version": version,
                "processing_time_ms": (time.perf_counter() - started) * 1000.0,
                "input_word_count": len(text.split()), "input_character_count": len(text), "chunk_count": len(chunks),
                "thresholds": {"real_max_ai_score": float(self._thresholds["real_max_ai_score"]),
                               "ai_min_ai_score": float(self._thresholds["ai_min_ai_score"])},
                "warning": f"Experimental {data_scope} estimate; calibrated scores may shift by source and domain. Chunk scores are word-weighted and aggregation is not a validated document-level method. Scores are not proof of authorship."}
