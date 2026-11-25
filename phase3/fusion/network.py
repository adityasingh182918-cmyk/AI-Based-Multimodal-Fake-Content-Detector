"""Trainable gated fusion head over frozen Phase 1/2 representations."""
from __future__ import annotations

import torch
from torch import nn

FUSION_LABELS = (
    "REAL_IMAGE_HUMAN_TEXT",
    "AI_IMAGE_AI_TEXT",
    "REAL_IMAGE_AI_TEXT",
    "AI_IMAGE_HUMAN_TEXT",
)


class SparseProjection(nn.Module):
    """Linear projection that accepts a sparse TF-IDF batch without densifying it."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = torch.sparse.mm(features, self.weight) if features.layout != torch.strided else features @ self.weight
        return projected + self.bias


class GatedFusionNetwork(nn.Module):
    """Learn modality projections, a vector gate, and 4-way joint authenticity."""

    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.image_projection = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.text_projection = nn.Sequential(SparseProjection(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fusion_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, len(FUSION_LABELS)))
        self.image_head = nn.Linear(hidden_dim, 2)
        self.text_head = nn.Linear(hidden_dim, 2)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> dict[str, torch.Tensor]:
        image = self.image_projection(image_features)
        text = self.text_projection(text_features)
        gate = self.gate(torch.cat((image, text), dim=-1))
        fused = gate * image + (1.0 - gate) * text
        return {"fusion_logits": self.fusion_head(fused), "image_logits": self.image_head(image),
                "text_logits": self.text_head(text), "image_embedding": image,
                "text_embedding": text, "gate": gate}
