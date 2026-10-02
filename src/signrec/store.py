"""Packed on-disk storage of the unified format.

A processed dataset lives in ``data/processed/<dataset>/``:

- ``frames.npy``  float16 ``[sum(T_i), L, 3]``, all sequences concatenated (memory-mapped on read);
- ``index.parquet`` one row per sample:
  ``sample_id, dataset, label, signer_id, n_frames, fps, offset``;
- ``labels.json``  label to class index;
- ``meta.json`` (optional) dataset-level constants, e.g. ``y_scale`` (see ``preprocess``).

``PackedDataset.sequence(i)`` returns a ``[T, L, 3]`` float32 array (NaN = missing landmark).
One packed file instead of ~10^5 small ``.npy`` files keeps loading fast on a laptop.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_COLUMNS = ["sample_id", "dataset", "label", "signer_id", "n_frames", "fps", "offset"]


@dataclass
class PackedWriter:
    """Accumulates sequences in memory and writes them as one packed dataset."""

    dataset: str
    chunks: list[np.ndarray] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    _offset: int = 0

    def add(self, seq: np.ndarray, sample_id: str, label: str, signer_id: str, fps: float) -> None:
        if seq.ndim != 3 or seq.shape[-1] != 3 or seq.shape[0] == 0:
            raise ValueError(f"expected [T>0, L, 3], got {seq.shape}")
        self.chunks.append(seq.astype(np.float16))
        self.rows.append(
            {
                "sample_id": sample_id,
                "dataset": self.dataset,
                "label": label,
                "signer_id": str(signer_id),
                "n_frames": int(seq.shape[0]),
                "fps": float(fps),
                "offset": self._offset,
            }
        )
        self._offset += seq.shape[0]

    def write(
        self, out_dir: str | Path, labels: Iterable[str] | None = None, meta: dict | None = None
    ) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        frames = np.concatenate(self.chunks, axis=0) if self.chunks else np.zeros((0, 0, 3))
        np.save(out / "frames.npy", frames.astype(np.float16))
        index = pd.DataFrame(self.rows, columns=INDEX_COLUMNS)
        index.to_parquet(out / "index.parquet", index=False)
        names = sorted(set(index["label"])) if labels is None else list(labels)
        (out / "labels.json").write_text(json.dumps({n: i for i, n in enumerate(names)}, indent=1))
        if meta:
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
        return out


class PackedDataset:
    """Read access to a packed dataset (frames are memory-mapped)."""

    def __init__(self, root: str | Path, mmap: bool = True) -> None:
        self.root = Path(root)
        self.index = pd.read_parquet(self.root / "index.parquet")
        self.frames = np.load(self.root / "frames.npy", mmap_mode="r" if mmap else None)
        self.label_to_id: dict[str, int] = json.loads((self.root / "labels.json").read_text())
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        meta = self.root / "meta.json"
        self.meta: dict = json.loads(meta.read_text()) if meta.exists() else {}
        self.y_scale = float(self.meta.get("y_scale", 1.0))
        self._offsets = self.index["offset"].to_numpy()
        self._lengths = self.index["n_frames"].to_numpy()

    def __getstate__(self) -> dict:
        # DataLoader workers re-open the memory map instead of pickling the frames.
        state = self.__dict__.copy()
        state["frames"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.frames = np.load(self.root / "frames.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.index)

    @property
    def n_classes(self) -> int:
        return len(self.label_to_id)

    def sequence(self, i: int) -> np.ndarray:
        o, n = int(self._offsets[i]), int(self._lengths[i])
        return np.asarray(self.frames[o : o + n], dtype=np.float32)

    def targets(self) -> np.ndarray:
        return self.index["label"].map(self.label_to_id).to_numpy(dtype=np.int64)
