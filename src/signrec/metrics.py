"""Classification metrics for isolated sign recognition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    topk = np.argpartition(-probs, kth=min(k, probs.shape[1]) - 1, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y, pred, average="macro", labels=np.arange(n_classes), zero_division=0))


def per_group_accuracy(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"g": groups, "ok": y == pred})
    out = df.groupby("g")["ok"].agg(["mean", "size"]).rename(columns={"mean": "acc", "size": "n"})
    return out.sort_values("acc")


def per_class_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    acc = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = y == c
        if m.any():
            acc[c] = (pred[m] == c).mean()
    return acc


def confused_pairs(cm: np.ndarray, labels: list[str], k: int = 20) -> pd.DataFrame:
    """Most frequent (true, predicted) off-diagonal pairs, as a share of the true class."""
    off = cm.astype(float).copy()
    np.fill_diagonal(off, 0)
    rate = off / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    flat = np.argsort(-off, axis=None)[:k]
    rows = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        rows.append(
            {"true": labels[i], "pred": labels[j], "count": int(off[i, j]), "rate": rate[i, j]}
        )
    return pd.DataFrame(rows)


def summary(probs: np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    return {
        "top1": topk_accuracy(probs, y, 1),
        "top5": topk_accuracy(probs, y, 5),
        "macro_f1": macro_f1(y, pred, n_classes),
    }


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y, pred, labels=np.arange(n_classes))
