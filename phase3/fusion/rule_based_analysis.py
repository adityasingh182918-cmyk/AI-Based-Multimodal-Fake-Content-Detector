"""Transparent label-level multimodal decisions; this is not a trained model."""
from __future__ import annotations

from typing import Any

LIMITATIONS = [
    "This is rule-based multimodal analysis, not a trained fusion neural network.",
    "It combines the image and text detector labels; it does not verify semantic consistency or factual accuracy.",
    "The combined verdict has no separate confidence score.",
]


def _label(result: dict[str, Any] | None) -> str | None:
    value = result.get("label") if isinstance(result, dict) else None
    return str(value).strip().upper() if value is not None else None


def combine_predictions(image_result: dict[str, Any] | None,
                        text_result: dict[str, Any] | None) -> dict[str, str]:
    """Apply the documented decision table to existing detector labels."""
    image_label = _label(image_result)
    text_label = _label(text_result)

    if image_label == "UNCERTAIN" or text_label == "UNCERTAIN":
        return {
            "label": "INCONCLUSIVE",
            "explanation": "At least one detector returned an uncertain result, so no combined category can be assigned.",
        }

    if image_label == "REAL" and text_label == "HUMAN_WRITTEN":
        return {
            "label": "AUTHENTIC CONTENT",
            "explanation": "The image detector classified the image as real, and the text detector classified the text as human-written.",
        }
    if image_label == "REAL" and text_label == "AI_GENERATED":
        return {
            "label": "MIXED CONTENT",
            "explanation": "The image detector classified the image as real, while the text detector classified the text as AI-generated.",
        }
    if image_label == "AI_GENERATED" and text_label == "HUMAN_WRITTEN":
        return {
            "label": "MIXED CONTENT",
            "explanation": "The image detector classified the image as AI-generated, while the text detector classified the text as human-written.",
        }
    if image_label == "AI_GENERATED" and text_label == "AI_GENERATED":
        return {
            "label": "AI-GENERATED CONTENT",
            "explanation": "Both detectors classified their input as AI-generated.",
        }

    return {
        "label": "INCONCLUSIVE",
        "explanation": "A detector returned an unrecognized label, so no combined category can be assigned.",
    }


def build_analysis(image_result: dict[str, Any] | None,
                   text_result: dict[str, Any] | None,
                   image_error: str | None = None,
                   text_error: str | None = None) -> dict[str, Any]:
    """Format modality outputs and a combined verdict without changing scores."""
    failures: dict[str, str] = {}
    if image_result is None:
        failures["image"] = image_error or "Image analysis could not be completed."
    if text_result is None:
        failures["text"] = text_error or "Text analysis could not be completed."

    partial = bool(failures)
    combined = None if partial else combine_predictions(image_result, text_result)
    return {
        "analysis_type": "rule_based_multimodal",
        "fusion_type": "rule_based",
        "trained_fusion_model": False,
        "image_result": image_result,
        "text_result": text_result,
        "combined_result": combined,
        "analysis_status": "partial" if partial else "completed",
        "failed_detectors": failures,
        "limitations": LIMITATIONS.copy(),
    }
