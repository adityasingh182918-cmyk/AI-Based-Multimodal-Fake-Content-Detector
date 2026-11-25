"""Frozen feature extractors loaded exclusively from Phase 3 model copies."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy.sparse as sp
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
IMAGE_COPY = ROOT / "phase3/models/image_copy/resnet18_cifake_aug_v1_calibrated_strict.pt"
TEXT_COPY = ROOT / "phase3/models/text_copy/hc3_tfidf_linear_svm.joblib"
TEXT_META = ROOT / "phase3/models/text_copy/hc3_tfidf_linear_svm.json"
IMAGE_LABELS = ("REAL", "AI_GENERATED")
TEXT_LABELS = ("HUMAN_WRITTEN", "AI_GENERATED")


def split_text(text: str, max_chars: int = 2_000) -> list[str]:
    """Phase 3-local copy of the frozen service's ordered text chunking policy."""
    if not text.strip():
        raise ValueError("Text must not be empty")
    chunks, remaining = [], text.strip()
    while len(remaining) > max_chars:
        cut = max(remaining.rfind("\n", 0, max_chars + 1), remaining.rfind(". ", 0, max_chars + 1))
        cut = max_chars if cut < max_chars // 2 else cut + 1
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


class FrozenEncoders:
    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is unavailable")
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        self.device = torch.device(device)
        self.image_model, image_checkpoint = self._load_image()
        self.image_preprocess = self._image_preprocess(image_checkpoint)
        self.text_model = joblib.load(TEXT_COPY)
        self.text_meta = json.loads(TEXT_META.read_text(encoding="utf-8"))
        self.text_feature_transform = self.text_model.named_steps["features"]
        self.text_class_names = list(self.text_model.classes_)
        self.text_thresholds = self.text_meta["decision_thresholds"]
        self.text_dim = int(self.text_feature_transform.transform(["feature dimension probe"]).shape[1])

    def _load_image(self) -> tuple[Any, dict[str, Any]]:
        if not IMAGE_COPY.is_file():
            raise FileNotFoundError(f"Phase 3 image checkpoint copy missing: {IMAGE_COPY}")
        checkpoint = torch.load(IMAGE_COPY, map_location="cpu", weights_only=True)
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(IMAGE_LABELS))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(self.device).eval()
        return model, checkpoint

    @staticmethod
    def _image_preprocess(checkpoint: dict[str, Any]):
        size = int(checkpoint.get("image_size", 224))
        if checkpoint.get("resize_mode", "stretch") == "pad":
            resize = lambda image: ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS, color=(0, 0, 0))
        else:
            resize = transforms.Resize((size, size))
        return transforms.Compose([resize, transforms.ToTensor(),
                                   transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"])])

    @staticmethod
    def _open_image(image: bytes | str | Path | Image.Image) -> Image.Image:
        try:
            if isinstance(image, Image.Image):
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
            if isinstance(image, (str, Path)):
                with Image.open(image) as opened:
                    opened.load()
                    return ImageOps.exif_transpose(opened).convert("RGB")
            with Image.open(io.BytesIO(image)) as opened:
                opened.load()
                return ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, TypeError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Invalid image: {exc}") from exc

    @torch.inference_mode()
    def image(self, image: bytes | str | Path | Image.Image) -> tuple[torch.Tensor, dict[str, Any]]:
        pil = self._open_image(image)
        tensor = self.image_preprocess(pil).unsqueeze(0).to(self.device)
        model = self.image_model
        x = model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
        feature = torch.flatten(model.avgpool(x), 1)
        logits = model.fc(feature)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu()
        scores = {label: float(probabilities[index]) for index, label in enumerate(IMAGE_LABELS)}
        thresholds = model_checkpoint_thresholds(IMAGE_COPY)
        ai_score = scores["AI_GENERATED"]
        label = ("REAL" if ai_score <= thresholds["real_max_ai_score"] else
                 "AI_GENERATED" if ai_score >= thresholds["ai_min_ai_score"] else "UNCERTAIN")
        return feature[0].detach().cpu(), {"label": label, "class_scores": scores,
                                           "score_type": "uncalibrated_softmax",
                                           "model_version": "resnet18_cifake_aug_v1_calibrated_strict"}

    def text(self, text: str) -> tuple[sp.csr_matrix, dict[str, Any]]:
        chunks = split_text(text)
        words = np.asarray([max(1, len(chunk.split())) for chunk in chunks], dtype=np.float64)
        rows = self.text_feature_transform.transform(chunks).tocsr()
        feature = sp.csr_matrix((words / words.sum()).reshape(1, -1)) @ rows
        chunk_scores = self.text_model.predict_proba(chunks)
        ai_index = self.text_class_names.index("AI_GENERATED")
        ai_score = float(np.average(chunk_scores[:, ai_index], weights=words))
        thresholds = self.text_thresholds
        label = ("HUMAN_WRITTEN" if ai_score <= thresholds["real_max_ai_score"] else
                 "AI_GENERATED" if ai_score >= thresholds["ai_min_ai_score"] else "UNCERTAIN")
        return feature, {"label": label,
                         "class_scores": {"HUMAN_WRITTEN": 1.0 - ai_score, "AI_GENERATED": ai_score},
                         "score_type": "calibrated_tfidf_probability",
                         "model_version": "hc3_tfidf_linear_svm", "chunk_count": len(chunks)}


def model_checkpoint_thresholds(path: Path) -> dict[str, float]:
    import json
    metadata = Path(str(path).replace(".pt", ".metadata.json"))
    if metadata.is_file():
        data = json.loads(metadata.read_text(encoding="utf-8"))
        return data["decision_thresholds"]
    return {"real_max_ai_score": 0.5, "ai_min_ai_score": 0.5}


def sparse_to_torch(matrix: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()
