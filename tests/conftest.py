"""Synthetic fixtures (no real data in the repository)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signrec.io.kaggle import KAGGLE_OFFSETS

SIZES = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}


def make_kaggle_long(n_frames: int = 5, missing_left: bool = True, seed: int = 0) -> pd.DataFrame:
    """Kaggle-like long table whose x encodes the 543-point index (for order checks)."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        for t in ("face", "left_hand", "pose", "right_hand"):
            for i in range(SIZES[t]):
                g = KAGGLE_OFFSETS[t] + i
                x = np.nan if (missing_left and t == "left_hand") else float(g)
                rows.append((100 + f, f"{100 + f}-{t}-{i}", t, i, x, rng.random(), rng.random()))
    return pd.DataFrame(rows, columns=["frame", "row_id", "type", "landmark_index", "x", "y", "z"])


@pytest.fixture
def kaggle_parquet(tmp_path):
    p = tmp_path / "seq.parquet"
    make_kaggle_long().to_parquet(p)
    return p


def random_sequence(T: int = 30, seed: int = 0, nan_left: bool = False) -> np.ndarray:
    """Plausible random sequence in the unified format."""
    from signrec.landmarks import N_LANDMARKS, POSE_POS, SLICES

    rng = np.random.default_rng(seed)
    seq = (0.5 + 0.1 * rng.standard_normal((T, N_LANDMARKS, 3))).astype(np.float32)
    seq[:, POSE_POS["l_shoulder"], :2] = [0.65, 0.6]  # image-left = signer's left shoulder
    seq[:, POSE_POS["r_shoulder"], :2] = [0.35, 0.6]
    if nan_left:
        seq[:, SLICES["left_hand"]] = np.nan
    return seq
