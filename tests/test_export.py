import numpy as np
import pytest
import torch

from signrec.export import benchmark, export_onnx, parity, quantize, session
from signrec.models import build_model
from signrec.preprocess import PreprocessConfig

PCFG = PreprocessConfig(T=32)
SMALL = {
    "gru": {"dim": 32, "hidden": 32, "layers": 1, "dropout": 0.0},
    "transformer": {"max_len": 32, "dim": 32, "stages": 1, "convs_per_stage": 1, "heads": 4},
}


@pytest.mark.parametrize("name", ["gru", "transformer"])
def test_onnx_parity_and_int8(name, tmp_path) -> None:
    torch.manual_seed(0)
    model = build_model(name, PCFG.n_features, 10, **SMALL[name]).eval()
    path = export_onnx(model, PCFG.T, PCFG.n_features, tmp_path / "m.onnx")
    x = np.random.default_rng(0).standard_normal((4, PCFG.T, PCFG.n_features)).astype(np.float32)
    m = np.ones((4, PCFG.T), np.float32)
    m[1, 20:] = 0
    x[1, 20:] = 0
    assert parity(model, session(path), x, m) < 1e-4
    q = quantize(path, tmp_path / "q.onnx")
    assert q.stat().st_size < path.stat().st_size
    assert parity(model, session(q), x, m) < 0.05  # int8 stays close
    assert benchmark(session(q), PCFG.T, PCFG.n_features, n=5)["p50_ms"] > 0
