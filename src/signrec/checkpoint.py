"""Loading of training checkpoints (torch only, no ONNX dependency)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from signrec.models import build_model


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict]:
    """Rebuild the model stored in ``best.pt``; returns ``(model in eval mode, checkpoint)``."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    kw = {k: v for k, v in ck["model"].items() if k != "name"}
    model = build_model(ck["model"]["name"], ck["n_features"], len(ck["labels"]), **kw)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck
