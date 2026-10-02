import numpy as np
import pandas as pd

from signrec.metrics import confused_pairs, confusion, per_group_accuracy, summary, topk_accuracy
from signrec.splits import random_split, signer_split, split_indices


def _index(n_signers: int = 10, per: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n_signers * per)],
            "signer_id": np.repeat([f"P{k}" for k in range(n_signers)], per),
        }
    )


def test_signer_split_disjoint_and_complete() -> None:
    idx = _index()
    sp = signer_split(idx, n_val=2, n_test=3, seed=1)
    s = sp["signers"]
    assert not (set(s["train"]) & set(s["val"]) or set(s["train"]) & set(s["test"]))
    assert not set(s["val"]) & set(s["test"])
    rows = split_indices(idx, sp)
    assert sum(len(v) for v in rows.values()) == len(idx)
    for a in rows:
        for b in rows:
            if a != b:
                assert not set(idx.signer_id.iloc[rows[a]]) & set(idx.signer_id.iloc[rows[b]])


def test_forced_test_signers() -> None:
    sp = signer_split(_index(), n_val=2, n_test=0, test_signers=["P0", "P1"])
    assert sp["signers"]["test"] == ["P0", "P1"] and len(sp["signers"]["val"]) == 2


def test_random_split_leaks_signers() -> None:
    idx = _index()
    rows = split_indices(idx, random_split(idx, 0.1, 0.2))
    assert sum(len(v) for v in rows.values()) == len(idx)
    assert set(idx.signer_id.iloc[rows["train"]]) & set(idx.signer_id.iloc[rows["test"]])


def test_metrics() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
    y = np.array([0, 1, 1])
    assert topk_accuracy(probs, y, 1) == 1 / 3
    assert topk_accuracy(probs, y, 2) == 1.0
    s = summary(probs, y, 3)
    assert set(s) == {"top1", "top5", "macro_f1"}
    cm = confusion(y, probs.argmax(1), 3)
    pairs = confused_pairs(cm, ["a", "b", "c"])
    assert set(zip(pairs["true"], pairs["pred"], strict=True)) == {("b", "c"), ("b", "a")}
    acc = per_group_accuracy(y, probs.argmax(1), np.array(["x", "x", "y"]))
    assert acc.loc["y", "acc"] == 0.0
