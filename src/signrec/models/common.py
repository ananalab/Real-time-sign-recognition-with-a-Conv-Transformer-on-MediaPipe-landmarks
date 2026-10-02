"""Building blocks shared by the sequence models (all ONNX-exportable)."""

from __future__ import annotations

import torch
from torch import nn


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over time of ``h [B, T, D]`` restricted to ``mask [B, T]`` frames."""
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


def masked_max(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    return h.masked_fill(~m, -1e4).amax(1)


class InputProjection(nn.Module):
    """Per-frame feature projection from ``F`` to ``dim``."""

    def __init__(self, n_features: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, dim: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(z))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
