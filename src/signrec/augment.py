"""Training-time augmentations, in two stages.

1. ``augment_raw`` acts on the raw sequence ``[T, L, 3]`` (image coordinates, before
   preprocessing): non-uniform time warping, frame dropping, landmark/hand dropping, noise.
2. ``spatial`` acts on the *normalised* sequence inside ``preprocess`` (body frame: origin at the
   shoulder centre, unit = shoulder width): rotation, anisotropic scaling and translation.

Spatial transforms are applied after normalisation on purpose: an image-space translation or
isotropic scaling would be cancelled exactly by the shoulder normalisation.
All functions take an explicit ``numpy.random.Generator`` and never modify their input.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from signrec.preprocess import resample_at


@dataclass(frozen=True)
class AugmentConfig:
    p_spatial: float = 0.8  # probability of applying the spatial transform
    rotate_deg: float = 15.0
    scale: tuple[float, float] = (0.8, 1.2)  # per-axis scale factors
    shift: float = 0.1  # in shoulder widths
    speed: tuple[float, float] = (0.8, 1.2)  # global speed factor
    warp: float = 0.3  # local speed variation (0 = uniform)
    warp_segments: int = 4
    frame_drop: float = 0.1
    landmark_drop: float = 0.05
    hand_drop: float = 0.05  # probability of hiding a whole hand in a frame
    noise_std: float = 0.002  # image units

    @classmethod
    def from_yaml(cls, path: str | Path) -> AugmentConfig:
        d = yaml.safe_load(Path(path).read_text())
        for k in ("scale", "speed"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def time_warp(seq: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Global speed change plus a smooth, monotonic, non-uniform warp of the time axis."""
    n = seq.shape[0]
    n_out = max(2, int(round(n / rng.uniform(*cfg.speed))))
    k = cfg.warp_segments
    speeds = rng.uniform(1 - cfg.warp, 1 + cfg.warp, size=k)
    knots = np.concatenate([[0.0], np.cumsum(speeds)]) / speeds.sum()
    u = np.linspace(0.0, 1.0, n_out)
    pos = np.interp(u, np.linspace(0.0, 1.0, k + 1), knots) * (n - 1)
    return resample_at(seq, pos)


def drop_frames(seq: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    keep = rng.random(seq.shape[0]) >= p
    return seq[keep] if keep.sum() >= 2 else seq


def drop_landmarks(seq: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    out = seq.copy()
    out[rng.random(seq.shape[:2]) < cfg.landmark_drop] = np.nan
    for sl in (slice(0, 21), slice(21, 42)):
        out[rng.random(seq.shape[0]) < cfg.hand_drop, sl] = np.nan
    return out


def augment_raw(seq: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Temporal and dropout augmentations on a raw sequence."""
    if seq.shape[0] < 2:
        return seq
    out = time_warp(seq, rng, cfg)
    out = drop_frames(out, rng, cfg.frame_drop)
    out = drop_landmarks(out, rng, cfg)
    return out + rng.normal(0.0, cfg.noise_std, size=out.shape).astype(out.dtype)


def spatial(seq: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Rotation, per-axis scaling and translation of a normalised sequence (x/y only)."""
    if rng.random() >= cfg.p_spatial:
        return seq
    th = np.deg2rad(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    sx, sy = rng.uniform(*cfg.scale, size=2)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([sx, sy])
    t = rng.uniform(-cfg.shift, cfg.shift, size=2)
    out = seq.copy()
    out[..., :2] = seq[..., :2] @ R.T.astype(np.float32) + t.astype(np.float32)
    return out


def make_augmenter(
    cfg: AugmentConfig, rng: np.random.Generator
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """The two stages bound to one random generator: ``(raw_fn, spatial_fn)``."""
    return (lambda s: augment_raw(s, rng, cfg)), (lambda s: spatial(s, rng, cfg))
