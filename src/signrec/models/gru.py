"""Bidirectional GRU with masked mean and max pooling."""

from __future__ import annotations

import torch
from torch import nn

from signrec.models.common import ClassifierHead, InputProjection, masked_max, masked_mean


class GRUClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dim: int = 192,
        hidden: int = 192,
        layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj = InputProjection(n_features, dim, dropout)
        self.gru = nn.GRU(
            dim, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(4 * hidden)
        self.head = ClassifierHead(4 * hidden, n_classes, dropout)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.proj(x) * mask.unsqueeze(-1).to(x.dtype))
        return self.norm(torch.cat([masked_mean(h, mask), masked_max(h, mask)], dim=-1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, mask))
