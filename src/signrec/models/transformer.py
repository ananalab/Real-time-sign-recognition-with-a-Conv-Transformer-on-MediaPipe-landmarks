"""1D depthwise convolutions interleaved with a small Transformer encoder.

Same idea as the top Google ISLR solutions: depthwise-separable convolutions for local temporal
patterns, self-attention for global context. Attention is written with plain matmuls so the model
exports cleanly to ONNX and runs in onnxruntime-web.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from signrec.models.common import ClassifierHead, InputProjection, masked_mean


class ConvBlock(nn.Module):
    """Pre-norm residual block: pointwise expand, GLU, depthwise conv, BN, SiLU, projection."""

    def __init__(self, dim: int, kernel: int, expand: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 2 * expand * dim)
        self.glu = nn.GLU(dim=-1)
        self.dw = nn.Conv1d(
            expand * dim, expand * dim, kernel, padding=kernel // 2, groups=expand * dim
        )
        self.bn = nn.BatchNorm1d(expand * dim)
        self.act = nn.SiLU()
        self.out = nn.Linear(expand * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(x.dtype)
        h = self.glu(self.expand(self.norm(x))) * m
        h = self.act(self.bn(self.dw(h.transpose(1, 2)))).transpose(1, 2)
        return x + self.drop(self.out(h)) * m


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        att = att.masked_fill(~mask[:, None, None, :], -1e4).softmax(-1)
        h = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(h)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.n1(x), mask))
        return x + self.drop(self.mlp(self.n2(x)))


class ConvTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        max_len: int = 64,
        dim: int = 192,
        stages: int = 2,
        convs_per_stage: int = 3,
        kernel: int = 17,
        expand: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj = InputProjection(n_features, dim, dropout)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        blocks: list[nn.Module] = []
        for _ in range(stages):
            blocks += [ConvBlock(dim, kernel, expand, dropout) for _ in range(convs_per_stage)]
            blocks.append(TransformerBlock(dim, heads, mlp_ratio, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        self.head = ClassifierHead(dim, n_classes, dropout)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        for blk in self.blocks:
            h = blk(h, mask)
        return self.norm(masked_mean(h, mask))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, mask))
