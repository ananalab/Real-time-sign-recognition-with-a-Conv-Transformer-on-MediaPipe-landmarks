"""Read Google ISLR (Kaggle ``asl-signs``) parquet files as unified ``[T, L, 3]`` arrays.

Each parquet has one row per (frame, landmark) with columns
``frame, row_id, type, landmark_index, x, y, z``; ``type`` in {face, left_hand, pose, right_hand}.
Frames hold 543 landmarks in the order face (468) | left_hand (21) | pose (33) | right_hand (21).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signrec.landmarks import GROUPS

KAGGLE_OFFSETS: dict[str, int] = {"face": 0, "left_hand": 468, "pose": 489, "right_hand": 522}
KAGGLE_N: int = 543


def unified_from_kaggle_index() -> np.ndarray:
    """Positions in the 543-point Kaggle frame of each unified landmark, in unified order."""
    idx: list[int] = []
    for _, source, sub in GROUPS:
        idx += [KAGGLE_OFFSETS[source] + i for i in sub]
    return np.asarray(idx, dtype=np.int64)


_UNIFIED = unified_from_kaggle_index()


def long_to_frames(df: pd.DataFrame) -> np.ndarray:
    """Long-format landmark table to a dense ``[T, 543, 3]`` float32 array (NaN where absent)."""
    frame_ids, t = np.unique(df["frame"].to_numpy(), return_inverse=True)
    offsets = df["type"].map(KAGGLE_OFFSETS).to_numpy()
    if np.isnan(offsets.astype(float)).any():
        raise ValueError(f"unknown landmark type in {set(df['type'])}")
    g = offsets.astype(np.int64) + df["landmark_index"].to_numpy(dtype=np.int64)
    out = np.full((len(frame_ids), KAGGLE_N, 3), np.nan, dtype=np.float32)
    out[t, g] = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return out


def load_kaggle_sequence(path: str | Path) -> np.ndarray:
    """Read one Kaggle parquet and return the unified ``[T, L, 3]`` float32 sequence."""
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    return long_to_frames(df)[:, _UNIFIED]
