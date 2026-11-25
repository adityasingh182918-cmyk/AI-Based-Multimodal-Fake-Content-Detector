"""Reusable inference service for the trained ResNet-18 image detector."""

from __future__ import annotations

import io
import math
import pickle
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

CLASS_NAMES = ("REAL", "AI_GENERATED")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "models" / "image_detector" / "resnet18_best.pt"
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


class ModelNotTrainedError(RuntimeError):
    """Raised when no compatible trained checkpoint is available."""


class InvalidImageError(ValueError):
    """Raised when input bytes cannot be decoded as an image."""


class AspectPadResize:
    """Resize without distorting aspect ratio, padding to a square canvas."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.LANCZOS, color=(0, 0, 0))


def build_preprocess(image_size: int, mean: list[float], std: list[float], resize_mode: str = "stretch") -> Any:
    """Return deterministic preprocessing compatible with a checkpoint."""
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Install torch and torchvision from the official PyTorch selector before prediction") from exc
    if resize_mode not in {"stretch", "pad"}:
        raise ValueError(f"Unsupported checkpoint resize_mode: {resize_mode}")
    resize = AspectPadResize(image_size) if resize_mode == "pad" else transforms.Resize((image_size, image_size))
    return transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def apply_decision_thresholds(ai_score: float, thresholds: dict[str, Any]) -> str:
    if ai_score <= float(thresholds["real_max_ai_score"]):
        return "REAL"
    if ai_score >= float(thresholds["ai_min_ai_score"]):
        return "AI_GENERATED"
    return "UNCERTAIN"


def build_resnet18(*, pretrained: bool = False) -> Any:
    """Build the architecture shared by training and inference."""
    try:
        import torch.nn as nn
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError("Install torch and torchvision from the official PyTorch selector to use image inference") from exc
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def _resolve_device(requested: str | Any = "auto") -> Any:
    import torch

    if isinstance(requested, str):
        if requested == "cuda" and not torch.cuda.is_available():
            raise ModelNotTrainedError("CUDA was explicitly requested but is unavailable; check the NVIDIA driver and CUDA-enabled PyTorch build.")
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested not in {"cuda", "cpu"}:
            raise ModelNotTrainedError(f"Unsupported inference device setting: {requested}")
        return torch.device(requested)
    return requested


def load_checkpoint(checkpoint_path: Path, device: Any = "auto") -> tuple[Any, dict[str, Any], Any]:
    """Load checkpoint metadata, weights, and the checkpoint-matched preprocessing."""
    if not checkpoint_path.is_file():
        raise ModelNotTrainedError(f"No trained model checkpoint found at {checkpoint_path}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch and torchvision from the official PyTorch selector before prediction") from exc
    device = _resolve_device(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as exc:
        raise ModelNotTrainedError(f"Unable to read model checkpoint at {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "resnet18":
        raise ModelNotTrainedError("Checkpoint is missing compatible ResNet-18 metadata")
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ModelNotTrainedError(f"Checkpoint class_names must be {list(CLASS_NAMES)}")
    try:
        image_size = int(checkpoint.get("image_size", 224))
        mean = [float(value) for value in checkpoint.get("mean", DEFAULT_MEAN)]
        std = [float(value) for value in checkpoint.get("std", DEFAULT_STD)]
        if image_size < 32 or len(mean) != 3 or len(std) != 3 or not all(math.isfinite(v) for v in mean + std) or any(v <= 0 for v in std):
            raise ValueError("invalid preprocessing configuration")
        thresholds = checkpoint.get("decision_thresholds")
        if thresholds is not None:
            if not isinstance(thresholds, dict):
                raise ValueError("decision_thresholds must be an object")
            low = float(thresholds["real_max_ai_score"])
            high = float(thresholds["ai_min_ai_score"])
            if not math.isfinite(low) or not math.isfinite(high) or not 0 <= low < high <= 1:
                raise ValueError("decision thresholds must satisfy 0 <= real_max_ai_score < ai_min_ai_score <= 1")
        state_dict = checkpoint["state_dict"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelNotTrainedError(f"Checkpoint metadata is invalid: {exc}") from exc
    model = build_resnet18(pretrained=False)
    try:
        model.load_state_dict(state_dict)
    except (RuntimeError, TypeError) as exc:
        raise ModelNotTrainedError(f"Checkpoint weights do not match the expected architecture: {exc}") from exc
    model.to(device)
    model.eval()
    resize_mode = checkpoint.get("resize_mode", "stretch")
    try:
        preprocess = build_preprocess(image_size, mean, std, resize_mode)
    except ValueError as exc:
        raise ModelNotTrainedError(f"Checkpoint preprocessing is invalid: {exc}") from exc
    return model, checkpoint, (device, preprocess, torch)


class ImageDetector:
    """Load a checkpoint once and reuse the model for subsequent requests."""

    def __init__(self, checkpoint_path: Path | str = DEFAULT_CHECKPOINT, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.requested_device = device
        self._model: Any = None
        self._checkpoint: dict[str, Any] | None = None
        self._runtime: Any = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def checkpoint_available(self) -> bool:
        """True only after a checkpoint has passed compatibility checks."""
        return self._checkpoint is not None

    @property
    def architecture(self) -> str:
        return "resnet18"

    @property
    def inference_device(self) -> str:
        if self._runtime is not None:
            return str(self._runtime[0])
        if self.requested_device in {"cuda", "cpu"}:
            return self.requested_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "unavailable"

    @property
    def checkpoint_identifier(self) -> str:
        version = str((self._checkpoint or {}).get("run_id") or self.checkpoint_path.name)
        if isinstance((self._checkpoint or {}).get("decision_thresholds"), dict):
            return f"{version}-validation-abstention"
        return version

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def initialize(self) -> bool:
        """Attempt startup loading; the API stays available if no model is trained."""
        if not self.checkpoint_path.is_file():
            return False
        try:
            self._ensure_loaded()
            return True
        except (ModelNotTrainedError, RuntimeError, OSError) as exc:
            self._load_error = str(exc)
            return False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                if self._load_error:
                    raise ModelNotTrainedError(self._load_error)
                model, checkpoint, runtime = load_checkpoint(self.checkpoint_path, self.requested_device)
                self._model, self._checkpoint, self._runtime = model, checkpoint, runtime

    @staticmethod
    def _open_image(image: bytes | bytearray | str | Path | Image.Image) -> Image.Image:
        try:
            if isinstance(image, Image.Image):
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
            if isinstance(image, (str, Path)):
                with Image.open(image) as opened:
                    opened.load()
                    return ImageOps.exif_transpose(opened).convert("RGB")
            with Image.open(io.BytesIO(bytes(image))) as opened:
                opened.load()
                return ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, TypeError, SyntaxError, Image.DecompressionBombError) as exc:
            raise InvalidImageError(f"Input is not a supported, readable image: {exc}") from exc

    def predict(self, image: bytes | bytearray | str | Path | Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        pil_image = self._open_image(image)
        self._ensure_loaded()
        device, preprocess, torch = self._runtime
        tensor = preprocess(pil_image).unsqueeze(0).to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            scores = torch.softmax(self._model(tensor), dim=1)[0].detach().cpu().tolist()
        class_scores = {name: float(scores[index]) for index, name in enumerate(CLASS_NAMES)}
        thresholds = (self._checkpoint or {}).get("decision_thresholds")
        if isinstance(thresholds, dict):
            label = apply_decision_thresholds(class_scores["AI_GENERATED"], thresholds)
        else:
            index = max(range(len(scores)), key=scores.__getitem__)
            label = CLASS_NAMES[index]
        label_score = class_scores.get(label, max(class_scores.values()))
        warning = "Experimental estimate; uncalibrated softmax scores are not proof of authenticity."
        if isinstance(thresholds, dict):
            warning = "UNCERTAIN uses a validation-selected abstention band; uncalibrated softmax scores are not proof of authenticity."
        return {
            "label": label,
            "predicted_label": label,
            "score": label_score,
            "score_type": "uncalibrated_softmax",
            "class_scores": class_scores,
            "architecture": self.architecture,
            "model_version": self.checkpoint_identifier,
            "inference_device": str(device),
            "processing_time_ms": (time.perf_counter() - started) * 1000.0,
            "warning": warning,
        }
