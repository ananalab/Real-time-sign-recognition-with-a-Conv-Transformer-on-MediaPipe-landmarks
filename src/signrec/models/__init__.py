"""Model registry."""

from __future__ import annotations

from torch import nn

from signrec.models.gru import GRUClassifier
from signrec.models.transformer import ConvTransformerClassifier

REGISTRY: dict[str, type[nn.Module]] = {
    "gru": GRUClassifier,
    "transformer": ConvTransformerClassifier,
}


def build_model(name: str, n_features: int, n_classes: int, **kwargs: object) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](n_features=n_features, n_classes=n_classes, **kwargs)
