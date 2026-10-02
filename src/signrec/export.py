"""ONNX export, dynamic INT8 quantisation and CPU benchmark of a trained checkpoint.

    uv run python -m signrec.export --ckpt models/<run>/best.pt [--out app/web/models]

The exported graph takes ``x [B, T, F] float32`` and ``mask [B, T] float32`` (1 = real frame)
and returns ``probs [B, C]``; a float mask keeps the model easy to feed from JavaScript.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from signrec.checkpoint import load_checkpoint
from signrec.preprocess import PreprocessConfig, preprocess
from signrec.splits import load_split, split_indices
from signrec.store import PackedDataset


class _ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(x, mask > 0.5).softmax(-1)


def export_onnx(model: nn.Module, T: int, n_features: int, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, T, n_features)
    m = torch.ones(1, T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            _ExportWrapper(model).eval(),  # wrapper must be in eval mode: export restores it
            (x, m),
            str(out),
            input_names=["x", "mask"],
            output_names=["probs"],
            dynamic_axes={"x": {0: "batch"}, "mask": {0: "batch"}, "probs": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    model.eval()
    return out


def quantize(fp32: str | Path, int8: str | Path) -> Path:
    """Dynamic INT8 quantisation of the weight matrices, keeping the input projection in FP32.

    The input features have rare but large outliers (|x| up to ~200 while 99.9 % are below ~2), so
    a per-tensor dynamic activation scale on the first layer destroys their resolution (validation
    top-1 drops from 68 % to 41 % on ASL). Only matmuls with constant weights are quantised.
    """
    graph = onnx.load(str(fp32)).graph
    first = [n.name for n in graph.node if n.op_type in ("MatMul", "Gemm") and "/proj/" in n.name]
    quantize_dynamic(
        str(fp32),
        str(int8),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=first[:1],
        extra_options={"MatMulConstBOnly": True},
    )
    return Path(int8)


def session(path: str | Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_predict(sess: ort.InferenceSession, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return sess.run(["probs"], {"x": x.astype(np.float32), "mask": mask.astype(np.float32)})[0]


def parity(model: nn.Module, sess: ort.InferenceSession, x: np.ndarray, mask: np.ndarray) -> float:
    """Max absolute difference between PyTorch and ONNX probabilities."""
    model.eval()
    with torch.no_grad():
        ref = model(torch.from_numpy(x), torch.from_numpy(mask.astype(bool))).softmax(-1).numpy()
    return float(np.abs(ref - onnx_predict(sess, x, mask)).max())


def benchmark(sess: ort.InferenceSession, T: int, F: int, n: int = 500) -> dict[str, float]:
    x = np.random.default_rng(0).standard_normal((1, T, F)).astype(np.float32)
    m = np.ones((1, T), np.float32)
    for _ in range(20):
        onnx_predict(sess, x, m)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        onnx_predict(sess, x, m)
        times.append((time.perf_counter() - t0) * 1e3)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None, help="packed dataset for parity/accuracy checks")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", default=None, help="default: next to the checkpoint")
    ap.add_argument("--n-eval", type=int, default=2000)
    args = ap.parse_args()

    model, ck = load_checkpoint(args.ckpt)
    pcfg = PreprocessConfig.from_dict(ck["preprocess"])
    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent
    fp32 = export_onnx(model, pcfg.T, pcfg.n_features, out_dir / "model_fp32.onnx")
    int8 = quantize(fp32, out_dir / "model_int8.onnx")
    report: dict = {"run": ck["run"], "T": pcfg.T, "n_features": pcfg.n_features}

    xs = ms = ys = None
    if args.data and args.split:
        ds = PackedDataset(args.data)
        rows = split_indices(ds.index, load_split(args.split))["val"]
        lab = {g: i for i, g in enumerate(ck["labels"])}
        rows = [r for r in rows if ds.index.label.iloc[r] in lab][: args.n_eval]
        feats = [preprocess(ds.sequence(r), pcfg, y_scale=ds.y_scale) for r in rows]
        xs = np.stack([f[0] for f in feats])
        ms = np.stack([f[1] for f in feats])
        ys = np.array([lab[ds.index.label.iloc[r]] for r in rows])

    for name, path in (("fp32", fp32), ("int8", int8)):
        sess = session(path)
        entry = {"size_mb": path.stat().st_size / 2**20, **benchmark(sess, pcfg.T, pcfg.n_features)}
        if xs is not None:
            probs = np.concatenate(
                [
                    onnx_predict(sess, xs[i : i + 256], ms[i : i + 256])
                    for i in range(0, len(xs), 256)
                ]
            )
            entry["val_top1"] = float((probs.argmax(1) == ys).mean())
            entry["max_abs_diff_vs_torch"] = parity(model, sess, xs[:100], ms[:100])
        report[name] = entry
        print(name, json.dumps(entry))

    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(ck["labels"], ensure_ascii=False))
    (out_dir / "preprocess.json").write_text(json.dumps(pcfg.to_dict(), indent=1))
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
