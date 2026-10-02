"""PyTorch dataset over a packed dataset: augmentation (training only), then preprocess()."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from signrec.augment import AugmentConfig, make_augmenter
from signrec.preprocess import PreprocessConfig, preprocess
from signrec.store import PackedDataset


class SignDataset(Dataset):
    """Items are ``(features [T, F] float32, mask [T] bool, label int64)``.

    Augmentation is deterministic given ``(seed, epoch, index)``; call :meth:`set_epoch`
    before each epoch (workers are re-created every epoch so they see the new value).
    """

    def __init__(
        self,
        ds: PackedDataset,
        rows: np.ndarray,
        pcfg: PreprocessConfig,
        acfg: AugmentConfig | None = None,
        label_map: dict[int, int] | None = None,
        seed: int = 0,
    ) -> None:
        self.ds, self.rows, self.pcfg, self.acfg = ds, np.asarray(rows), pcfg, acfg
        targets = ds.targets()[self.rows]
        self.targets = np.array([label_map[t] for t in targets]) if label_map else targets
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.ds.sequence(int(self.rows[i]))
        spatial = None
        if self.acfg is not None:
            rng = np.random.default_rng((self.seed, self.epoch, int(self.rows[i])))
            raw, spatial = make_augmenter(self.acfg, rng)
            seq = raw(seq)
        x, m = preprocess(seq, self.pcfg, spatial, self.ds.y_scale)
        return torch.from_numpy(x), torch.from_numpy(m), torch.tensor(self.targets[i])
