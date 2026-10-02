"""Classical baselines: per-coordinate temporal statistics + linear model / gradient boosting."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from signrec.preprocess import PreprocessConfig, preprocess
from signrec.store import PackedDataset


def stat_features(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean, std, min and max of every position feature over the valid frames."""
    v = x[mask] if mask.any() else x
    return np.concatenate([v.mean(0), v.std(0), v.min(0), v.max(0)]).astype(np.float32)


def _one(seq: np.ndarray, cfg: PreprocessConfig, y_scale: float) -> np.ndarray:
    x, m = preprocess(seq, cfg, y_scale=y_scale)
    return stat_features(x, m)


def build_stat_matrix(ds: PackedDataset, rows: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Statistical feature matrix for the given dataset rows (positions only, real frames)."""
    cfg = PreprocessConfig(**{**cfg.to_dict(), "length_mode": "pad", "velocity": False})
    feats = Parallel(n_jobs=-1, batch_size=256)(
        delayed(_one)(ds.sequence(i), cfg, ds.y_scale) for i in rows
    )
    return np.stack(feats)
