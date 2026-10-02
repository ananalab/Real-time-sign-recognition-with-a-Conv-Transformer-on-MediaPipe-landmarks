"""Publish the processed LSFB-ISOL data as a *private* Kaggle dataset (for GPU training).

LSFB-ISOL is distributed under CC BY 4.0; the dataset description carries the citations.

    uv run python scripts/upload_lsfb_dataset.py [--update]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/processed/lsfb_isol"
STAGE = ROOT / "kaggle/datasets/lsfb-isol-unified"
FILES = ["frames.npy", "index.parquet", "labels.json", "official_split.parquet", "meta.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="anaisazouaoui")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    for f in FILES:
        os.link(SRC / f, STAGE / f)  # hard links: no extra disk space
    meta = {
        "title": "LSFB-ISOL unified landmarks",
        "id": f"{a.user}/lsfb-isol-unified",
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": (
            "100 most frequent glosses of LSFB-ISOL v2 (raw MediaPipe poses) reduced to an "
            "89-landmark subset. Source: Fink et al., LSFB-CONT and LSFB-ISOL, IJCNN 2021; "
            "Meurant, Corpus LSFB, University of Namur, 2015."
        ),
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    if a.update:
        cmd = ["kaggle", "datasets", "version", "-p", str(STAGE), "-m", "update"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(STAGE)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
