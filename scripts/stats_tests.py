"""Paired significance tests between models evaluated on the same clips -> reports/stats.json.

Every comparison uses the per-clip predictions saved by signrec.evaluate (reports/eval/*.npz)
for the three seeds of each model:

- per-clip correctness averaged over seeds, compared with a paired bootstrap, either
  hierarchical over signers (``signer``) or over clips within signers (``clip``);
- an exact McNemar test on the predictions of the seed ensembles;
- the mean and standard deviation over seeds of the accuracy difference.

Comparisons: Conv-Transformer vs. BiGRU on the ASL test signers, and the ASL-initialised LSFB
models vs. the model trained from scratch on the LSFB test signers.

    uv run python scripts/stats_tests.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from signrec.metrics import mcnemar, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports" / "eval"
SEEDS = (0, 1, 2)


def load(prefix: str, subset: str, seeds: tuple[int, ...] = SEEDS) -> dict | None:
    """Predictions of the given seeds of a model, or None if any is missing."""
    paths = [EVAL / f"{prefix}-s{s}_{subset}.npz" for s in seeds]
    if not all(p.exists() for p in paths):
        return None
    zs = [np.load(p, allow_pickle=True) for p in paths]
    for z in zs[1:]:
        assert (z["rows"] == zs[0]["rows"]).all(), f"{prefix}: clips differ across seeds"
    probs = [z["probs"].astype(np.float32) for z in zs]
    y = zs[0]["y"]
    correct = np.stack([p.argmax(1) == y for p in probs])
    return {
        "rows": zs[0]["rows"],
        "signers": zs[0]["signers"],
        "y": y,
        "correct": correct,  # [seed, clip]
        "ensemble_correct": np.mean(probs, axis=0).argmax(1) == y,
    }


def compare(a: dict, b: dict, levels: tuple[str, ...]) -> dict:
    assert (a["rows"] == b["rows"]).all(), "models were evaluated on different clips"
    per_seed = a["correct"].mean(1) - b["correct"].mean(1)
    out = {
        "n_clips": int(len(a["y"])),
        "n_signers": int(len(np.unique(a["signers"]))),
        "acc_a": float(a["correct"].mean()),
        "acc_b": float(b["correct"].mean()),
        "seed_diff_mean": float(per_seed.mean()),
        "seed_diff_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
        "mcnemar_ensemble": mcnemar(a["ensemble_correct"], b["ensemble_correct"]),
    }
    for level in levels:
        out[f"bootstrap_{level}"] = paired_bootstrap(
            a["correct"].mean(0), b["correct"].mean(0), a["signers"], level=level, n=5000
        )
    return out


def main() -> None:
    res: dict = {}
    tr, gru = load("asl-transformer", "test"), load("asl-gru", "test")
    if tr and gru:
        res["asl_transformer_vs_gru_test"] = compare(tr, gru, ("signer", "clip"))
    scratch = load("lsfb-scratch-nall", "test")
    for other in ("ft", "ftfreeze"):
        b = load(f"lsfb-{other}-nall", "test")
        if scratch and b:
            res[f"lsfb_{other}_vs_scratch_test"] = compare(b, scratch, ("signer", "clip"))
    out = ROOT / "reports" / "stats.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({', '.join(res)})")


if __name__ == "__main__":
    main()
