"""Stream LSFB-ISOL poses from the UNamur server into the packed unified format.

Only the unified landmark subset of each instance is kept (a few KB per clip), so the
~10 GB of raw poses never touch the disk. Interrupted runs resume from ``_cache/``.

    uv run python scripts/convert_lsfb.py --config configs/data/lsfb_isol.yaml [--limit 500]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from signrec.io.lsfb import fetch_lsfb_remote
from signrec.store import PackedWriter


def select_vocabulary(instances: pd.DataFrame, top_n: int, min_signers: int) -> list[str]:
    """Most frequent signs among those performed by at least ``min_signers`` signers."""
    g = instances.groupby("sign").agg(n=("id", "size"), signers=("signer", "nunique"))
    g = g[g["signers"] >= min_signers].sort_values(["n", "sign"], ascending=[False, True])
    return g.head(top_n).index.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/lsfb_isol.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    raw = Path(cfg.raw_dir)
    inst = pd.read_csv(raw / "instances.csv")
    vocab = select_vocabulary(inst, cfg.top_n, cfg.min_signers)
    inst = inst[inst["sign"].isin(vocab)].reset_index(drop=True)
    if args.limit:
        inst = inst.sample(n=min(args.limit, len(inst)), random_state=0).reset_index(drop=True)

    cache = Path(cfg.out_dir) / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in inst["id"] if not (cache / f"{i}.npy").exists()]
    print(f"vocabulary={len(vocab)} instances={len(inst)} to_fetch={len(todo)}")

    def job(iid: str) -> str:
        seq = fetch_lsfb_remote(iid, raw=cfg.raw_poses)
        np.save(cache / f"{iid}.npy", seq.astype(np.float16))
        return iid

    failed: list[str] = []
    with ThreadPoolExecutor(cfg.workers) as ex:
        futures = {ex.submit(job, i): i for i in todo}
        for f in tqdm(as_completed(futures), total=len(futures), unit="clip"):
            if f.exception() is not None:
                failed.append(futures[f])
    if failed:
        print(f"{len(failed)} failed downloads (re-run to retry): {failed[:5]}")

    splits = {}
    for s in ("train", "test"):
        splits.update({i: s for i in json.loads((raw / f"metadata/splits/{s}.json").read_text())})

    w = PackedWriter(cfg.name)
    kept = []
    for r in inst.itertuples():
        p = cache / f"{r.id}.npy"
        if not p.exists():
            continue
        seq = np.load(p).astype(np.float32)
        if seq.shape[0] < cfg.min_frames:
            continue
        w.add(seq, r.id, r.sign, r.signer, cfg.fps)
        kept.append({"sample_id": r.id, "official_split": splits.get(r.id, "unknown")})
    meta = {"y_scale": cfg.y_scale, "y_scale_note": cfg.y_scale_note}
    out = w.write(cfg.out_dir, labels=sorted(vocab), meta=meta)
    pd.DataFrame(kept).to_parquet(out / "official_split.parquet", index=False)
    print(f"wrote {len(kept)} sequences to {out}")


if __name__ == "__main__":
    main()
