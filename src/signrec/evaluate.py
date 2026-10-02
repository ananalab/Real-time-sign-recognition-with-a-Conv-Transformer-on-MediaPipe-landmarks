"""Evaluate a checkpoint on a split and write a Markdown + JSON report.

    uv run python -m signrec.evaluate --ckpt models/<run>/best.pt --data data/processed/<ds> \
        --split splits/<ds>_signer.json --subset val [--gifs 20]

Outputs ``reports/eval/<run>_<subset>.{md,json,npz}`` (the npz keeps probabilities for figures).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from signrec.checkpoint import load_checkpoint
from signrec.dataset import SignDataset
from signrec.metrics import (
    confused_pairs,
    confusion,
    per_class_accuracy,
    per_group_accuracy,
    summary,
)
from signrec.preprocess import PreprocessConfig, hand_presence, preprocess
from signrec.splits import load_split, split_indices
from signrec.store import PackedDataset

ROOT = Path(__file__).resolve().parents[2]


def missing_rate(seq: np.ndarray) -> float:
    """Share of frames where neither hand is detected."""
    return float((~hand_presence(seq).any(axis=1)).mean())


def cpu_latency(model: torch.nn.Module, x: np.ndarray, m: np.ndarray, n: int = 200) -> dict:
    model = model.cpu().eval()
    torch.set_num_threads(1)
    xt, mt = torch.from_numpy(x[None]), torch.from_numpy(m[None])
    times = []
    with torch.no_grad():
        for _ in range(10):
            model(xt, mt)
        for _ in range(n):
            t0 = time.perf_counter()
            model(xt, mt)
            times.append((time.perf_counter() - t0) * 1e3)
    return {
        "model_p50_ms": float(np.median(times)),
        "model_p95_ms": float(np.percentile(times, 95)),
    }


def evaluate(ckpt: str, data: str, split: str, subset: str, n_gifs: int = 0) -> dict:
    model, ck = load_checkpoint(ckpt)
    pcfg = PreprocessConfig.from_dict(ck["preprocess"])
    labels: list[str] = ck["labels"]
    ds = PackedDataset(data)
    rows = split_indices(ds.index, load_split(split))[subset]
    lab = {g: i for i, g in enumerate(labels)}
    rows = np.array([r for r in rows if ds.index.label.iloc[r] in lab])
    label_map = {ds.label_to_id[g]: i for g, i in lab.items()}
    sd = SignDataset(ds, rows, pcfg, None, label_map)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    probs, y = [], []
    with torch.no_grad():
        for x, m, t in DataLoader(sd, batch_size=256, num_workers=4):
            probs.append(model(x.to(device), m.to(device)).softmax(-1).cpu().numpy())
            y.append(t.numpy())
    probs, y = np.concatenate(probs), np.concatenate(y)
    pred = probs.argmax(1)
    n_classes = len(labels)
    met = summary(probs, y, n_classes)

    signers = ds.index.signer_id.iloc[rows].astype(str).to_numpy()
    by_signer = per_group_accuracy(y, pred, signers)
    cls_acc = per_class_accuracy(y, pred, n_classes)
    worst = np.argsort(np.nan_to_num(cls_acc, nan=2.0))[:10]
    cm = confusion(y, pred, n_classes)
    pairs = confused_pairs(cm, labels, k=20)

    miss = np.array([missing_rate(ds.sequence(int(r))) for r in rows])
    bins = pd.cut(
        miss, [-0.01, 0.0, 0.1, 0.25, 0.5, 1.0], labels=["0", "0-10%", "10-25%", "25-50%", ">50%"]
    )
    by_missing = pd.DataFrame({"bin": bins, "ok": pred == y}).groupby("bin", observed=False)["ok"]
    by_missing = by_missing.agg(["mean", "size"]).rename(columns={"mean": "acc", "size": "n"})

    x0, m0 = preprocess(ds.sequence(int(rows[0])), pcfg, y_scale=ds.y_scale)
    t0 = time.perf_counter()
    for r in rows[:200]:
        preprocess(ds.sequence(int(r)), pcfg, y_scale=ds.y_scale)
    pre_ms = (time.perf_counter() - t0) / min(200, len(rows)) * 1e3
    lat = {**cpu_latency(model, x0, m0), "preprocess_ms": pre_ms}

    run = ck["run"]
    out = ROOT / "reports" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{run}_{subset}"
    np.savez_compressed(stem.with_suffix(".npz"), probs=probs.astype(np.float16), y=y, rows=rows,
                        signers=signers, missing=miss)  # fmt: skip
    result = {
        "run": run,
        "subset": subset,
        "n": int(len(y)),
        "n_classes": n_classes,
        **met,
        "signer_acc_mean": float(by_signer["acc"].mean()),
        "signer_acc_std": float(by_signer["acc"].std()),
        "latency": lat,
        "by_signer": by_signer.reset_index().to_dict("records"),
        "by_missing": by_missing.reset_index().astype({"bin": str}).to_dict("records"),
        "worst_classes": [{"label": labels[c], "acc": float(cls_acc[c])} for c in worst],
        "confused_pairs": pairs.to_dict("records"),
    }
    stem.with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False))

    md = [
        f"# Evaluation of `{run}` on `{subset}`",
        "",
        f"- samples: {len(y)} · classes: {n_classes} · signers: {len(by_signer)}",
        f"- **top-1 {met['top1']:.4f}** · top-5 {met['top5']:.4f} · macro-F1 {met['macro_f1']:.4f}",
        f"- per-signer top-1: {result['signer_acc_mean']:.4f} ± {result['signer_acc_std']:.4f}",
        f"- CPU latency (1 thread): preprocess {pre_ms:.2f} ms, model p50 "
        f"{lat['model_p50_ms']:.2f} ms / p95 {lat['model_p95_ms']:.2f} ms",
        "",
        "## Per signer",
        by_signer.to_markdown(floatfmt=".4f"),
        "",
        "## Accuracy vs share of frames without any hand",
        by_missing.to_markdown(floatfmt=".4f"),
        "",
        "## 10 worst classes",
        pd.DataFrame(result["worst_classes"]).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 20 most confused pairs (rate = share of the true class)",
        pairs.to_markdown(index=False, floatfmt=".3f"),
    ]
    stem.with_suffix(".md").write_text("\n".join(md) + "\n")

    if n_gifs:
        from signrec.viz import save_gif

        err = np.flatnonzero(pred != y)
        rng = np.random.default_rng(0)
        for i in rng.permutation(err)[:n_gifs]:
            seq = ds.sequence(int(rows[i]))
            name = f"{labels[y[i]]}__as__{labels[pred[i]]}_{rows[i]}.gif".replace("/", "-")
            save_gif(seq, out / f"{run}_errors" / name, f"{labels[y[i]]} as {labels[pred[i]]}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", default="val", choices=["train", "val", "test"])
    ap.add_argument("--gifs", type=int, default=0)
    a = ap.parse_args()
    r = evaluate(a.ckpt, a.data, a.split, a.subset, a.gifs)
    print({k: r[k] for k in ("run", "subset", "n", "top1", "top5", "macro_f1")})


if __name__ == "__main__":
    main()
