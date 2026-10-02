"""Preprocessing: raw unified sequence ``[T, L, 3]`` to model features ``[T_out, F]`` + mask.

:func:`preprocess` is used for training, evaluation and, through a line-by-line JavaScript port
checked by fixtures, the browser demo. All steps are pure numpy.

Order of operations: trim, canonical hand (mirror), spatial normalisation, fixed length,
features, NaN to 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from signrec.landmarks import N_LANDMARKS, POSE_POS, SLICES, mirror_permutation

_MIRROR = np.asarray(mirror_permutation())
_LH, _RH = SLICES["left_hand"], SLICES["right_hand"]
_EPS = 1e-6


@dataclass(frozen=True)
class PreprocessConfig:
    T: int = 64
    length_mode: str = "resample"  # resample | pad
    use_z: bool = False
    parts: tuple[str, ...] = field(default=("left_hand", "right_hand", "pose", "lips"))
    trim: bool = True
    canonical_hand: bool = True
    hand_local: bool = True
    velocity: bool = True
    min_shoulder_width: float = 0.02

    @classmethod
    def from_dict(cls, d: dict) -> PreprocessConfig:
        return cls(**{**d, "parts": tuple(d.get("parts", cls.parts))})

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: object) -> PreprocessConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) | overrides)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["parts"] = list(self.parts)
        return d

    @property
    def n_dims(self) -> int:
        return 3 if self.use_z else 2

    @property
    def landmark_ids(self) -> np.ndarray:
        return np.concatenate([np.arange(N_LANDMARKS)[SLICES[p]] for p in self.parts])

    @property
    def n_features(self) -> int:
        n = len(self.landmark_ids) * self.n_dims
        if self.hand_local:
            n += 21 * self.n_dims
        return n * (2 if self.velocity else 1)


def hand_presence(seq: np.ndarray) -> np.ndarray:
    """``[T, 2]`` booleans: is the left / right hand detected in each frame."""
    lh = ~np.isnan(seq[:, _LH, 0]).all(axis=1)
    rh = ~np.isnan(seq[:, _RH, 0]).all(axis=1)
    return np.stack([lh, rh], axis=1)


def trim_handless(seq: np.ndarray) -> np.ndarray:
    """Remove leading and trailing frames without any detected hand (keep all if none)."""
    any_hand = hand_presence(seq).any(axis=1)
    if not any_hand.any():
        return seq
    idx = np.flatnonzero(any_hand)
    return seq[idx[0] : idx[-1] + 1]


def mirror(seq: np.ndarray) -> np.ndarray:
    """Horizontal flip in image coordinates (x -> 1 - x) with left/right landmark swap."""
    out = seq[:, _MIRROR].copy()
    out[..., 0] = 1.0 - out[..., 0]
    return out


def canonicalize_hand(seq: np.ndarray) -> np.ndarray:
    """Mirror the sequence when the left-hand slot is seen in more frames than the right one."""
    n_left, n_right = hand_presence(seq).sum(axis=0)
    return mirror(seq) if n_left > n_right else seq


def normalize(seq: np.ndarray, min_shoulder_width: float = 0.02) -> np.ndarray:
    """Centre on the mean shoulder midpoint and divide by the mean shoulder width (x/y only).

    Constants are computed over the whole sequence so motion is preserved. If shoulders are
    missing or degenerate, fall back to the centroid and spread of all visible points.
    """
    out = seq.copy()
    ls, rs = seq[:, POSE_POS["l_shoulder"], :2], seq[:, POSE_POS["r_shoulder"], :2]
    ok = ~(np.isnan(ls).any(axis=1) | np.isnan(rs).any(axis=1))
    width = np.linalg.norm(ls[ok] - rs[ok], axis=1).mean() if ok.any() else 0.0
    if width >= min_shoulder_width:
        center = ((ls[ok] + rs[ok]) / 2).mean(axis=0)
        scale = width
    else:
        pts = seq[..., :2].reshape(-1, 2)
        pts = pts[~np.isnan(pts).any(axis=1)]
        if len(pts) == 0:
            return out
        center = pts.mean(axis=0)
        scale = max(float(pts.std(axis=0).mean()) * 4.0, _EPS)
    out[..., :2] = (seq[..., :2] - center) / scale
    out[..., 2] = seq[..., 2] / scale
    return out


def resample_at(seq: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``seq`` at fractional frame positions ``pos`` (in ``[0, T-1]``).

    Where an endpoint is missing (NaN), the value of the nearest frame is used instead.
    """
    n = seq.shape[0]
    lo = np.clip(np.floor(pos).astype(np.int64), 0, n - 1)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo).astype(seq.dtype)[:, None, None]
    a, b = seq[lo], seq[hi]
    out = a * (1 - w) + b * w
    bad = np.isnan(out)
    if bad.any():
        nearest = np.where(w < 0.5, a, b)
        out[bad] = nearest[bad]
    return out


def resample(seq: np.ndarray, T: int) -> np.ndarray:
    """Uniform linear temporal resampling to ``T`` frames."""
    n = seq.shape[0]
    if n == T:
        return seq.copy()
    if n == 1:
        return np.repeat(seq, T, axis=0)
    return resample_at(seq, np.linspace(0.0, n - 1, T))


def fix_length(seq: np.ndarray, T: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``([T, L, C], mask[T])``; ``pad`` keeps the real frames and zero-pads the rest."""
    if mode == "resample":
        return resample(seq, T), np.ones(T, dtype=bool)
    if mode == "pad":
        if seq.shape[0] > T:
            return resample(seq, T), np.ones(T, dtype=bool)
        out = np.full((T, *seq.shape[1:]), np.nan, dtype=seq.dtype)
        out[: seq.shape[0]] = seq
        mask = np.zeros(T, dtype=bool)
        mask[: seq.shape[0]] = True
        return out, mask
    raise ValueError(f"unknown length_mode {mode!r}")


def hand_local(seq: np.ndarray) -> np.ndarray:
    """Dominant (right slot) hand relative to its wrist, scaled by the wrist-middle MCP length."""
    hand = seq[:, _RH]
    wrist = hand[:, :1]
    size = np.linalg.norm(hand[:, 9, :2] - hand[:, 0, :2], axis=-1)[:, None, None]
    return (hand - wrist) / np.maximum(size, _EPS)


def preprocess(
    seq: np.ndarray,
    cfg: PreprocessConfig,
    spatial_aug: Callable[[np.ndarray], np.ndarray] | None = None,
    y_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a unified sequence ``[T, L, 3]`` (NaN = missing) into ``(features, mask)``.

    Shapes: features ``[cfg.T, F]``, mask ``[cfg.T]``.

    ``y_scale`` makes image coordinates isotropic: MediaPipe divides x by the frame width and y by
    its height, so y is multiplied by height / width (display aspect) of the source video.
    ``spatial_aug`` (training only) is applied to the normalised coordinates.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[1] != N_LANDMARKS or seq.shape[0] == 0:
        raise ValueError(f"expected [T>0, {N_LANDMARKS}, 3], got {seq.shape}")
    if y_scale != 1.0:
        seq = seq.copy()
        seq[..., 1] *= np.float32(y_scale)
    if cfg.trim:
        seq = trim_handless(seq)
    if cfg.canonical_hand:
        seq = canonicalize_hand(seq)
    seq = normalize(seq, cfg.min_shoulder_width)
    if spatial_aug is not None:
        seq = spatial_aug(seq)
    seq, mask = fix_length(seq, cfg.T, cfg.length_mode)

    d = cfg.n_dims
    blocks = [seq[:, cfg.landmark_ids, :d]]
    if cfg.hand_local:
        blocks.append(hand_local(seq)[..., :d])
    pos = np.concatenate(blocks, axis=1).reshape(cfg.T, -1)
    feats = [pos]
    if cfg.velocity:
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        vel[1:][~(mask[1:] & mask[:-1])] = 0.0
        feats.append(vel)
    x = np.concatenate(feats, axis=1)
    x[~np.isfinite(x)] = 0.0
    x[~mask] = 0.0
    return x.astype(np.float32), mask
