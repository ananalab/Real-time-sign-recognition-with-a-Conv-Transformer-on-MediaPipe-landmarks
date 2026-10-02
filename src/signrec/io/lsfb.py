"""Read LSFB-ISOL v2 poses as unified ``[T, L, 3]`` arrays.

LSFB v2 ships one ``.npy`` per body part and instance, ``poses[_raw]/<part>/<id>.npy``:
pose ``[T, 33, 3]``, left_hand/right_hand ``[T, 21, 3]``, face ``[T, 478, 3]`` (float16).
These follow the MediaPipe Tasks numbering, which shares the indices used in ``landmarks.py``
(the 478-point face mesh extends the 468-point one with 10 iris points at the end).
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Mapping

import numpy as np

from signrec.landmarks import GROUPS

LSFB_SOURCE = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol"
LSFB_PARTS = ("pose", "left_hand", "right_hand", "face")


def parts_to_unified(parts: Mapping[str, np.ndarray]) -> np.ndarray:
    """Combine per-part arrays (same T) into the unified ``[T, L, 3]`` float32 sequence."""
    lengths = {k: v.shape[0] for k, v in parts.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"parts have different lengths: {lengths}")
    blocks = [
        np.asarray(parts[source], dtype=np.float32)[:, list(sub)] for _, source, sub in GROUPS
    ]
    return np.concatenate(blocks, axis=1)


def fetch_lsfb_remote(
    instance_id: str, raw: bool = True, source: str = LSFB_SOURCE, timeout: float = 60.0
) -> np.ndarray:
    """Download the four part files of one instance and return the unified sequence.

    Nothing is written to disk: only the unified subset is kept by the caller.
    """
    folder = "poses_raw" if raw else "poses"
    parts = {}
    for p in LSFB_PARTS:
        url = f"{source}/{folder}/{p}/{instance_id}.npy"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            parts[p] = np.load(io.BytesIO(r.read()))
    return parts_to_unified(parts)
