"""Build (and optionally push) the Kaggle script that converts asl-signs to the unified format.

The kernel inlines the exact source of ``landmarks.py``, ``io/kaggle.py`` and ``store.py``,
so the conversion run on Kaggle is the code of this repository (checked by tests).

    uv run python scripts/build_kaggle_kernel.py [--push]
    kaggle kernels status <user>/asl-signs-unified-subset
    kaggle kernels output <user>/asl-signs-unified-subset -p data/processed/
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["src/signrec/landmarks.py", "src/signrec/io/kaggle.py", "src/signrec/store.py"]
OUT_DIR = ROOT / "kaggle" / "convert_asl"

MAIN = """
# ---------------------------------------------------------------- kernel entry point
import glob
import os
import time
from multiprocessing import Pool


def _find_root() -> str:
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if not hits:
        raise FileNotFoundError("asl-signs not mounted: add the competition as a data source")
    return os.path.dirname(hits[0])


DATA = _find_root()


def _work(rel_path: str) -> np.ndarray:
    return load_kaggle_sequence(os.path.join(DATA, rel_path)).astype(np.float16)


if __name__ == "__main__":
    t0 = time.time()
    meta = pd.read_csv(os.path.join(DATA, "train.csv"))
    sign_map = json.load(open(os.path.join(DATA, "sign_to_prediction_index_map.json")))
    labels = [s for s, _ in sorted(sign_map.items(), key=lambda kv: kv[1])]
    writer = PackedWriter("kaggle_asl")
    with Pool(os.cpu_count()) as pool:
        seqs = pool.imap(_work, meta["path"], chunksize=64)
        for i, (row, seq) in enumerate(zip(meta.itertuples(), seqs)):
            writer.add(seq, str(row.sequence_id), row.sign, str(row.participant_id), 30.0)
            if i % 5000 == 0:
                print(f"{i}/{len(meta)} sequences, {time.time() - t0:.0f}s", flush=True)
    out = writer.write("/kaggle/working/kaggle_asl", labels=labels)
    stats = {
        "n_sequences": len(writer.rows),
        "n_frames": int(sum(r["n_frames"] for r in writer.rows)),
        "n_landmarks": N_LANDMARKS,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)
"""


def build() -> str:
    parts = ["from __future__ import annotations", "import json"]
    for m in MODULES:
        src = (ROOT / m).read_text()
        lines = [
            ln
            for ln in src.splitlines()
            if not ln.startswith("from signrec")
            and ln.strip() != "from __future__ import annotations"
        ]
        parts.append(f"# ---------------------------------------------------------------- {m}")
        parts.append("\n".join(lines))
    parts.append(MAIN)
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="anaisazouaoui")
    ap.add_argument("--slug", default="asl-signs-unified-subset")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "kernel.py").write_text(build())
    meta = {
        "id": f"{args.user}/{args.slug}",
        "title": args.slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["asl-signs"],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel written to {OUT_DIR}")
    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(OUT_DIR)], check=True)


if __name__ == "__main__":
    main()
