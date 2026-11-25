"""Phase 3 inference orchestration over isolated Phase 1/2 model copies."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from phase3.fusion.encoders import FrozenEncoders, sparse_to_torch
from phase3.fusion.network import FUSION_LABELS, GatedFusionNetwork

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUSION_CHECKPOINT = ROOT / "phase3/models/fusion/best_model.pt"


class FusionNotReadyError(RuntimeError):
    pass


class Phase3Predictor:
    def __init__(self, device: str = "auto", checkpoint: str | Path = DEFAULT_FUSION_CHECKPOINT) -> None:
        self.device_request = device
        self.checkpoint_path = Path(checkpoint)
        self.encoders: FrozenEncoders | None = None
        self.fusion_model: GatedFusionNetwork | None = None
        self.load_error: str | None = None

    @property
    def fusion_available(self) -> bool:
        return self.checkpoint_path.is_file()

    def _ensure_encoders(self) -> FrozenEncoders:
        if self.encoders is None:
            self.encoders = FrozenEncoders(self.device_request)
        return self.encoders

    def _ensure_fusion(self, encoders: FrozenEncoders) -> GatedFusionNetwork:
        if not self.checkpoint_path.is_file():
            raise FusionNotReadyError("Fusion checkpoint is absent; prepare licensed paired data and train Phase 3 explicitly")
        if self.fusion_model is None:
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=encoders.device, weights_only=True)
                model = GatedFusionNetwork(int(checkpoint["image_dim"]), int(checkpoint["text_dim"]),
                                           int(checkpoint.get("hidden_dim", 128)))
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                self.fusion_model = model.to(encoders.device).eval()
            except Exception as exc:
                self.load_error = str(exc)
                raise FusionNotReadyError(f"Unable to load Phase 3 fusion checkpoint: {exc}") from exc
        return self.fusion_model

    def predict_image(self, image: bytes) -> dict[str, Any]:
        encoders = self._ensure_encoders()
        _, prediction = encoders.image(image)
        return prediction

    def predict_text(self, text: str) -> dict[str, Any]:
        _, prediction = self._ensure_encoders().text(text)
        return prediction

    @torch.inference_mode()
    def predict_multimodal(self, image: bytes, text: str) -> dict[str, Any]:
        encoders = self._ensure_encoders()
        image_feature, image_prediction = encoders.image(image)
        text_feature, text_prediction = encoders.text(text)
        model = self._ensure_fusion(encoders)
        outputs = model(image_feature.unsqueeze(0).to(encoders.device), sparse_to_torch(text_feature, encoders.device))
        probabilities = torch.softmax(outputs["fusion_logits"].float(), dim=1)[0].cpu()
        predicted_index = int(probabilities.argmax().item())
        label = FUSION_LABELS[predicted_index]
        ai_image = label in {"AI_IMAGE_AI_TEXT", "AI_IMAGE_HUMAN_TEXT"}
        ai_text = label in {"AI_IMAGE_AI_TEXT", "REAL_IMAGE_AI_TEXT"}
        fusion_scores = {name: round(float(probabilities[index]), 8) for index, name in enumerate(FUSION_LABELS)}
        return {"image_prediction": image_prediction, "text_prediction": text_prediction,
                "image_feature_representation": [round(float(v), 7) for v in outputs["image_embedding"][0].cpu().tolist()],
                "text_feature_representation": [round(float(v), 7) for v in outputs["text_embedding"][0].cpu().tolist()],
                "fusion_prediction": label, "fusion_class_scores": fusion_scores,
                "fusion_confidence": {"value": fusion_scores[label], "type": "uncalibrated_softmax"},
                "uncertainty_status": "not_calibrated",
                "model_version": "gated_sparse_fusion_v1",
                "signals_agree": (None if image_prediction["label"] == "UNCERTAIN" or text_prediction["label"] == "UNCERTAIN"
                                  else ((image_prediction["label"] == "AI_GENERATED") == ai_image and
                                        (text_prediction["label"] == "AI_GENERATED") == ai_text)),
                "feature_dimensions": {"image_base": 512, "text_base_sparse": encoders.text_dim,
                                       "image_projected": len(outputs["image_embedding"][0]),
                                       "text_projected": len(outputs["text_embedding"][0])},
                "score_warning": "Fusion probabilities are raw softmax, not calibrated confidence; the fusion task has not been validated for other datasets."}


def image_sha256(image: bytes) -> str:
    return hashlib.sha256(image).hexdigest()
