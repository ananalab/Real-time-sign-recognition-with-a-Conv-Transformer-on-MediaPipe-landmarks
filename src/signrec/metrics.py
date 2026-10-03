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


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray,
    level: str = "signer",
    n: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap of the accuracy difference ``mean(a) - mean(b)`` on the same clips.

    ``a`` and ``b`` are per-clip scores (1/0, or a mean over seeds). ``level="signer"`` resamples
    signers with replacement, then clips within each drawn signer (hierarchical bootstrap: the
    uncertainty includes the choice of signers). ``level="clip"`` resamples clips within each
    signer only (the signers are taken as fixed). Returns the observed difference, a 95 %
    percentile interval and a two-sided bootstrap p-value.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    keys, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(keys))]
    diffs = np.empty(n)
    for i in range(n):
        drawn = rng.integers(0, len(keys), len(keys)) if level == "signer" else range(len(keys))
        tot, cnt = 0.0, 0
        for g in drawn:
            idx = members[g]
            pick = idx[rng.integers(0, len(idx), len(idx))]
            tot += d[pick].sum()
            cnt += len(pick)
        diffs[i] = tot / cnt
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # add-one estimate: a p-value is never reported as exactly zero
    p = 2 * (min((diffs <= 0).sum(), (diffs >= 0).sum()) + 1) / (n + 1)
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p": min(p, 1.0)}


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on paired binary outcomes (binomial test on discordant pairs)."""
    from scipy.stats import binomtest

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": float(p)}
