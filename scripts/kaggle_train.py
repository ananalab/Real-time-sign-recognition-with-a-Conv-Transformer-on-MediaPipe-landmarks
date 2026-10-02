"""Run a list of training experiments on a Kaggle GPU and fetch the results.

The current source tree (src/, configs/, splits/) is embedded in the kernel script as a tarball,
so what runs on Kaggle is exactly the local code. Data come from Kaggle sources:
the conversion kernel output (asl-signs) and/or a private dataset with the processed LSFB data.

    uv run python scripts/kaggle_train.py --experiments experiments/asl_main.yaml --push
    uv run python scripts/kaggle_train.py --experiments experiments/asl_main.yaml --fetch

Experiment file (YAML)::

    name: asl-main               # kernel slug suffix
    sources: [kaggle_asl]        # kaggle_asl and/or lsfb_isol
    pretrained_from: []          # optional kernels whose models/*/best.pt are made available
    runs:
      - run_name: asl-transformer-s0
        overrides: [seed=0]
        evaluate: [val, test]    # optional
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
USER = "anaisazouaoui"
SOURCES = {
    "kaggle_asl": {"kernel": f"{USER}/asl-signs-unified-subset"},
    "lsfb_isol": {"dataset": f"{USER}/lsfb-isol-unified"},
}
PACKAGED = ["src", "configs", "splits", "pyproject.toml"]

RUNNER = r'''
import base64, glob, io, json, os, shutil, subprocess, sys, tarfile

CODE = "__CODE__"
SPEC = json.loads(r"""__SPEC__""")
WORK = "/kaggle/working/repo"
OUT = "/kaggle/working/out"

tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(WORK)
os.makedirs(f"{WORK}/data/processed", exist_ok=True)
for name in SPEC["sources"]:
    slug = name.replace("_", "-")
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}/index.parquet", recursive=True)
                  + glob.glob(f"/kaggle/input/**/{slug}-unified/index.parquet", recursive=True))
    if not hits:
        sys.exit(f"source {name} not mounted")
    os.symlink(os.path.dirname(hits[0]), f"{WORK}/data/processed/{name}")
for kernel in SPEC.get("pretrained_from", []):
    pattern = f"/kaggle/input/**/{kernel.split('/')[-1]}/**/models/*/best.pt"
    for ck in glob.glob(pattern, recursive=True):
        dst = f"{WORK}/models/{os.path.basename(os.path.dirname(ck))}"
        os.makedirs(dst, exist_ok=True)
        shutil.copy(ck, f"{dst}/best.pt")
for mod in ("omegaconf", "tabulate"):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod], check=True)

env = {**os.environ, "PYTHONPATH": f"{WORK}/src"}
py = [sys.executable, "-m"]
os.chdir(WORK)
subprocess.run(["nvidia-smi", "-L"])
status = []
for run in SPEC["runs"]:
    name = run["run_name"]
    cmd = py + ["signrec.train", "--config", "configs/train.yaml", "mlflow=false",
                f"run_name={name}", *run.get("overrides", [])]
    print("\n>>>", " ".join(cmd), flush=True)
    ok = subprocess.run(cmd, env=env).returncode == 0
    if ok:
        cfg = __import__("yaml").safe_load(open(f"reports/runs/{name}/config.yaml"))
        for subset in run.get("evaluate", []):
            subprocess.run(py + ["signrec.evaluate", "--ckpt", f"models/{name}/best.pt",
                                 "--data", cfg["data"]["root"], "--split", cfg["data"]["split"],
                                 "--subset", subset], env=env)
    status.append({"run": name, "ok": ok})
    shutil.copytree("reports", f"{OUT}/reports", dirs_exist_ok=True)
    if os.path.isdir(f"models/{name}"):
        shutil.copytree(f"models/{name}", f"{OUT}/models/{name}", dirs_exist_ok=True)
    json.dump(status, open(f"{OUT}/status.json", "w"), indent=1)
shutil.rmtree(WORK, ignore_errors=True)
print(json.dumps(status, indent=1))
'''


def pack_code() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in PACKAGED:
            tar.add(ROOT / item, arcname=item, filter=_skip_cache)
    return base64.b64encode(buf.getvalue()).decode()


def _skip_cache(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if "__pycache__" in info.name else info


def build(spec: dict, out_dir: Path, gpu: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = RUNNER.replace("__CODE__", pack_code()).replace("__SPEC__", json.dumps(spec))
    (out_dir / "kernel.py").write_text(script)
    meta = {
        "id": f"{USER}/signrec-{spec['name']}",
        "title": f"signrec-{spec['name']}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "competition_sources": [],
        "dataset_sources": [
            SOURCES[s]["dataset"] for s in spec["sources"] if "dataset" in SOURCES[s]
        ],
        "kernel_sources": [SOURCES[s]["kernel"] for s in spec["sources"] if "kernel" in SOURCES[s]]
        + list(spec.get("pretrained_from", [])),
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return out_dir


def fetch(spec: dict) -> None:
    """Download the kernel output and merge reports/ and models/ into the local tree."""
    tmp = ROOT / "kaggle" / "output" / spec["name"]
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    kernel = f"{USER}/signrec-{spec['name']}"
    subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(tmp)], check=True)
    for d in ("reports", "models"):
        if (tmp / "out" / d).exists():
            shutil.copytree(tmp / "out" / d, ROOT / d, dirs_exist_ok=True)
    print((tmp / "out" / "status.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.experiments).read_text())
    if a.fetch:
        fetch(spec)
        return
    out = build(spec, ROOT / "kaggle" / f"train_{spec['name']}", gpu=not a.cpu)
    print(f"kernel written to {out} ({len(spec['runs'])} runs)")
    if a.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], check=True)


if __name__ == "__main__":
    main()
