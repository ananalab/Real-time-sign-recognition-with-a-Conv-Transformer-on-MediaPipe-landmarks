"""Freeze the evaluation splits in splits/ (versioned).

uv run python scripts/make_splits.py --dataset kaggle_asl
uv run python scripts/make_splits.py --dataset lsfb_isol
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signrec.splits import random_split, save_split, signer_split, split_indices

ROOT = Path(__file__).resolve().parents[1]


def describe(index: pd.DataFrame, split: dict, name: str) -> None:
    rows = split_indices(index, split)
    parts = []
    for k, v in rows.items():
        parts.append(f"{k}: {len(v)} samples / {index.signer_id.iloc[v].nunique()} signers")
    print(f"{name}: " + " | ".join(parts))


def lsfb_selection_report(index: pd.DataFrame, split: dict) -> None:
    """Vocabulary-selection statistics quoted in the paper."""
    inst = pd.read_csv(ROOT / "data/raw/lsfb_isol/instances.csv")
    selected = inst[inst["sign"].isin(set(index["label"]))]
    tr = split_indices(index, split)["train"]
    per_class = index.iloc[tr]["label"].value_counts()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "lsfb_selection.json").write_text(
        json.dumps(
            {
                "all_instances": int(len(inst)),
                "all_glosses": int(inst["sign"].nunique()),
                "all_signers": int(inst["signer"].nunique()),
                "selected_instances": int(len(selected)),
                "kept_instances": int(len(index)),
                "dropped_share": float(1 - len(index) / len(selected)),
            },
            indent=1,
        )
    )
    (reports / "lsfb_train_per_class.json").write_text(
        json.dumps(
            {
                "min": int(per_class.min()),
                "median": float(per_class.median()),
                "max": int(per_class.max()),
            },
            indent=1,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["kaggle_asl", "lsfb_isol"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = ROOT / "data/processed" / a.dataset
    index = pd.read_parquet(root / "index.parquet")
    out = ROOT / "splits"
    if a.dataset == "kaggle_asl":
        # 21 participants: 14 train / 3 val / 4 test, all signer-disjoint
        sp = signer_split(index, n_val=3, n_test=4, seed=a.seed)
        sp["dataset"] = a.dataset
        save_split(sp, out / "kaggle_asl_signer.json")
        describe(index, sp, "signer")
        # sample-level split with the same proportions, for the leakage study
        n = len(index)
        r = split_indices(index, sp)
        rnd = random_split(index, len(r["val"]) / n, len(r["test"]) / n, seed=a.seed)
        rnd["dataset"] = a.dataset
        save_split(rnd, out / "kaggle_asl_random.json")
        describe(index, rnd, "random")
    else:
        # official signer-independent test set; validation signers drawn from the official train
        off = pd.read_parquet(root / "official_split.parquet").merge(
            index[["sample_id", "signer_id"]]
        )
        test_signers = sorted(off.loc[off.official_split == "test", "signer_id"].unique())
        train_signers = off.loc[off.official_split == "train", "signer_id"].nunique()
        sp = signer_split(index, n_val=max(1, round(0.15 * train_signers)), n_test=0,
                          seed=a.seed, test_signers=test_signers)  # fmt: skip
        sp["dataset"] = a.dataset
        save_split(sp, out / "lsfb_isol_signer.json")
        describe(index, sp, "signer")
        lsfb_selection_report(index, sp)


if __name__ == "__main__":
    main()
