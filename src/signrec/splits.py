"""Signer-independent train/val/test splits, frozen as JSON in ``splits/``.

A split file stores the *signer* lists (and, for the leakage study, explicit sample ids), so it is
small, human-readable and versioned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def signer_split(
    index: pd.DataFrame,
    n_val: int,
    n_test: int,
    seed: int = 0,
    test_signers: list[str] | None = None,
) -> dict:
    """Disjoint signer sets. ``test_signers`` forces a given test set (e.g. an official one)."""
    rng = np.random.default_rng(seed)
    signers = sorted(index["signer_id"].astype(str).unique())
    if test_signers is None:
        perm = list(rng.permutation(signers))
        test = sorted(perm[:n_test])
        rest = perm[n_test:]
    else:
        test = sorted(set(test_signers))
        rest = [s for s in rng.permutation(signers) if s not in set(test)]
    val = sorted(rest[:n_val])
    train = sorted(rest[n_val:])
    return {
        "strategy": "signer",
        "seed": seed,
        "signers": {"train": train, "val": val, "test": test},
    }


def random_split(index: pd.DataFrame, frac_val: float, frac_test: float, seed: int = 0) -> dict:
    """Sample-level random split (signers leak across splits), only used for the leakage study."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(index["sample_id"].astype(str).to_numpy())
    n_te, n_va = int(len(ids) * frac_test), int(len(ids) * frac_val)
    return {
        "strategy": "random",
        "seed": seed,
        "samples": {
            "test": sorted(ids[:n_te]),
            "val": sorted(ids[n_te : n_te + n_va]),
            "train": sorted(ids[n_te + n_va :]),
        },
    }


def split_indices(index: pd.DataFrame, split: dict) -> dict[str, np.ndarray]:
    """Row positions of each split in ``index``."""
    if split["strategy"] == "signer":
        s = index["signer_id"].astype(str)
        return {k: np.flatnonzero(s.isin(split["signers"][k]).to_numpy()) for k in SPLITS}
    ids = index["sample_id"].astype(str)
    return {k: np.flatnonzero(ids.isin(split["samples"][k]).to_numpy()) for k in SPLITS}


def save_split(split: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=1))


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
