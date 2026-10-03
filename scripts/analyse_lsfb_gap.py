"""Why is LSFB validation accuracy higher than test accuracy? -> reports/lsfb_gap.json.

Validation signers are drawn from the official training signers, test signers are the official
test signers. The script compares the two sets on factors that can explain an accuracy gap:

- recording sessions: LSFB clips come from dialogues between two signers; a held-out signer whose
  dialogue partner is a training signer shares the conversation (topics, signs) with the
  training data;
- data: clips per signer, durations, missing hands, class distribution;
- predictions (three seeds of the from-scratch model, all data): accuracy of each split, per
  clip and per signer, test accuracy re-weighted to the validation class distribution, and
  accuracy of signers whose partner is or is not in the training set.

    uv run python scripts/analyse_lsfb_gap.py [--run lsfb-scratch-nall]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signrec.preprocess import hand_presence
from signrec.splits import load_split, split_indices
from signrec.store import PackedDataset

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (bits) between two discrete distributions."""
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def describe(ds: PackedDataset, rows: np.ndarray, train_counts: np.ndarray) -> dict:
    idx = ds.index.iloc[rows]
    no_hand = [float((~hand_presence(ds.sequence(int(r))).any(1)).mean()) for r in rows]
    counts = np.bincount(ds.targets()[rows], minlength=ds.n_classes).astype(float)
    return {
        "n_signers": int(idx.signer_id.nunique()),
        "n_clips": len(rows),
        "clips_per_signer_median": float(idx.groupby("signer_id").size().median()),
        "duration_s_median": float((idx.n_frames / idx.fps).median()),
        "no_hand_share_mean": float(np.mean(no_hand)),
        "class_js_vs_train_bits": js_divergence(counts + 1e-12, train_counts + 1e-12),
        "mean_train_examples_of_class": float(train_counts[ds.targets()[rows]].mean()),
    }


def predictions(run: str, subset: str) -> dict | None:
    paths = [ROOT / "reports" / "eval" / f"{run}-s{s}_{subset}.npz" for s in SEEDS]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    zs = [np.load(p, allow_pickle=True) for p in paths]
    correct = np.mean([z["probs"].argmax(1) == z["y"] for z in zs], axis=0)
    return {"rows": zs[0]["rows"], "y": zs[0]["y"], "correct": correct, "n_seeds": len(zs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lsfb-scratch-nall")
    a = ap.parse_args()
    ds = PackedDataset(ROOT / "data/processed/lsfb_isol")
    split = load_split(ROOT / "splits/lsfb_isol_signer.json")
    rows = split_indices(ds.index, split)
    labels = json.loads((ROOT / "app/web/models/lsfb/labels.json").read_text())
    keep = ds.index.label.isin(labels).to_numpy()
    rows = {k: v[keep[v]] for k, v in rows.items()}
    y = ds.targets()
    train_counts = np.bincount(y[rows["train"]], minlength=ds.n_classes).astype(float)

    session = ds.index.sample_id.str.split("_").str[0]
    signer = ds.index.signer_id.astype(str)
    pairs = pd.DataFrame({"session": session, "signer": signer}).drop_duplicates()
    partners = (
        pairs.merge(pairs.rename(columns={"signer": "partner"}), on="session")
        .query("signer != partner")
        .groupby("signer")["partner"]
        .apply(set)
    )
    train_signers = set(split["signers"]["train"])

    def partner_in_train(s: str) -> bool:
        return bool(partners.get(s, set()) & train_signers)

    report: dict = {"splits": {}}
    for k in ("val", "test"):
        held = split["signers"][k]
        report["splits"][k] = {
            **describe(ds, rows[k], train_counts),
            "signers_with_partner_in_train": int(sum(partner_in_train(s) for s in held)),
        }

    val, test = predictions(a.run, "val"), predictions(a.run, "test")
    if val is not None and test is not None:
        acc = {"val": float(val["correct"].mean()), "test": float(test["correct"].mean())}
        # test accuracy if its classes were distributed as in the validation set
        pv = np.bincount(val["y"], minlength=len(labels)) / len(val["y"])
        pt = np.bincount(test["y"], minlength=len(labels)) / len(test["y"])
        w = np.where(pt[test["y"]] > 0, pv[test["y"]] / pt[test["y"]], 0.0)
        acc["test_reweighted_to_val_classes"] = float((w * test["correct"]).sum() / w.sum())
        for name, pred in (("val", val), ("test", test)):
            s = signer.iloc[pred["rows"]].to_numpy()
            inside = np.array([partner_in_train(x) for x in s])
            for flag, label in ((True, "partner_in_train"), (False, "partner_not_in_train")):
                if (inside == flag).any():
                    acc[f"{name}_{label}"] = float(pred["correct"][inside == flag].mean())
                    acc[f"{name}_{label}_n_signers"] = int(len(np.unique(s[inside == flag])))
            per_signer = pd.Series(pred["correct"]).groupby(s).mean()
            acc[f"{name}_per_signer_mean"] = float(per_signer.mean())
            acc[f"{name}_per_signer_std"] = float(per_signer.std())
            acc[f"{name}_per_signer_sem"] = float(per_signer.std() / np.sqrt(len(per_signer)))
            # share of the clips contributed by the three most prolific signers
            sizes = pd.Series(s).value_counts()
            acc[f"{name}_top3_signer_clip_share"] = float(sizes.iloc[:3].sum() / sizes.sum())
        report["accuracy"] = {
            "run": a.run,
            "n_seeds_val": val["n_seeds"],
            "n_seeds_test": test["n_seeds"],
            **acc,
        }

    out = ROOT / "reports" / "lsfb_gap.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
