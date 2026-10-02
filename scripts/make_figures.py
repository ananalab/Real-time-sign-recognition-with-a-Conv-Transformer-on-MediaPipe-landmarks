"""Generate every figure and LaTeX table of the paper from data/ and reports/.

    uv run python scripts/make_figures.py [--only data|results]

Figures go to paper/figures/ (PDF + PNG), tables to paper/tables/. Nothing is typed by hand:
each number in the paper comes from a file written by the training/evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signrec import viz
from signrec.landmarks import POSE_POS, SLICES
from signrec.metrics import summary
from signrec.preprocess import PreprocessConfig, canonicalize_hand, normalize, trim_handless
from signrec.store import PackedDataset

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
DATASETS = {"kaggle_asl": "ASL (Google ISLR)", "lsfb_isol": "LSFB-ISOL"}
GIF_SIGNS = {"kaggle_asl": ["hello", "thankyou"], "lsfb_isol": ["AUSSI"]}  # shown in the README


def available() -> dict[str, PackedDataset]:
    out = {}
    for name in DATASETS:
        p = ROOT / "data" / "processed" / name
        if (p / "index.parquet").exists():
            out[name] = PackedDataset(p)
    return out


def part_missing_rates(ds: PackedDataset, max_n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Share of frames where each body part is entirely missing (random subsample)."""
    rng = np.random.default_rng(seed)
    rows = rng.permutation(len(ds))[:max_n]
    acc = {k: [] for k in ("left_hand", "right_hand", "any_hand", "pose", "lips")}
    for r in rows:
        s = ds.sequence(int(r))[..., 0]
        miss = {
            p: np.isnan(s[:, SLICES[p]]).all(1) for p in ("left_hand", "right_hand", "pose", "lips")
        }
        miss["any_hand"] = miss["left_hand"] & miss["right_hand"]
        for k, v in miss.items():
            acc[k].append(v.mean())
    return pd.DataFrame({k: [np.mean(v)] for k, v in acc.items()})


def fig_dataset_overview(dss: dict[str, PackedDataset]) -> dict:
    """(a) examples per class, (b) clip duration, (c) examples per signer."""
    fig, axes = plt.subplots(1, 3, figsize=(viz.FULL_W, 1.9))
    stats = {}
    for k, (name, ds) in enumerate(dss.items()):
        idx = ds.index
        color = viz.SERIES[k]
        per_class = idx.label.value_counts().to_numpy()
        axes[0].plot(np.arange(1, len(per_class) + 1), per_class, color=color, label=DATASETS[name])
        dur = idx.n_frames / idx.fps
        bins = np.linspace(0, 6, 61)
        axes[1].hist(
            dur.clip(upper=6), bins=bins, histtype="step", density=True, color=color, lw=1.2
        )
        per_signer = np.sort(idx.signer_id.value_counts().to_numpy())[::-1]
        axes[2].plot(np.arange(1, len(per_signer) + 1), per_signer, color=color)
        stats[name] = {
            "n_samples": int(len(idx)),
            "n_classes": int(idx.label.nunique()),
            "n_signers": int(idx.signer_id.nunique()),
            "per_class_min": int(per_class.min()),
            "per_class_median": float(np.median(per_class)),
            "per_class_max": int(per_class.max()),
            "duration_median_s": float(dur.median()),
            "frames_median": float(idx.n_frames.median()),
            "frames_p95": float(idx.n_frames.quantile(0.95)),
            "fps": float(idx.fps.iloc[0]),
        }
    axes[0].set(
        xlabel="class rank", ylabel="examples", title="(a) examples per class", yscale="log"
    )
    axes[1].set(
        xlabel="clip duration (s; last bin ≥ 6 s)", ylabel="density", title="(b) clip duration"
    )
    axes[2].set(
        xlabel="signer rank", ylabel="examples", title="(c) examples per signer", yscale="log"
    )
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=1.2)
    viz.save(fig, FIG / "dataset_overview")
    return stats


def fig_missing(dss: dict[str, PackedDataset]) -> pd.DataFrame:
    rates = pd.concat({DATASETS[n]: part_missing_rates(ds) for n, ds in dss.items()}).droplevel(1)
    parts = ["left_hand", "right_hand", "any_hand", "pose", "lips"]
    nice = ["left\nhand", "right\nhand", "both\nhands", "upper\nbody", "lips"]
    fig, ax = plt.subplots(figsize=(viz.COL_W, 1.9))
    w = 0.38
    x = np.arange(len(parts))
    for k, (name, row) in enumerate(rates.iterrows()):
        vals = row[parts].to_numpy() * 100
        bars = ax.bar(x + (k - 0.5) * w, vals, w * 0.92, color=viz.SERIES[k], label=name)
        ax.bar_label(bars, fmt="%.0f", fontsize=6, padding=1.5, color=viz.INK_2)
    ax.set_xticks(x, nice)
    ax.set_ylabel("frames with part missing (%)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    viz.save(fig, FIG / "missing_parts")
    return rates


def fig_normalisation(dss: dict[str, PackedDataset], n: int = 400, seed: int = 0) -> None:
    """Per-clip mean positions of the nose, shoulders and dominant wrist in three coordinate frames.

    (a) image coordinates; (b) body-centred, without aspect correction; (c) body-centred with the
    per-dataset aspect correction. Each point is one clip.
    """
    rng = np.random.default_rng(seed)
    cfg = PreprocessConfig()
    joints = [
        POSE_POS["nose"],
        POSE_POS["l_shoulder"],
        POSE_POS["r_shoulder"],
        SLICES["right_hand"].start,
    ]
    fig, axes = plt.subplots(1, 3, figsize=(viz.FULL_W, 2.05))
    for k, (name, ds) in enumerate(dss.items()):
        pts = {0: [], 1: [], 2: []}
        for r in rng.permutation(len(ds))[:n]:
            seq = canonicalize_hand(trim_handless(ds.sequence(int(r))))
            iso = seq.copy()
            iso[..., 1] *= ds.y_scale
            frames = (
                seq,
                normalize(seq, cfg.min_shoulder_width),
                normalize(iso, cfg.min_shoulder_width),
            )
            for j, arr in enumerate(frames):
                pts[j] += [np.nanmean(arr[:, idx, :2], axis=0) for idx in joints]
        for j, ax in enumerate(axes):
            p = np.array(pts[j])
            ax.scatter(
                p[:, 0], p[:, 1], s=1.2, color=viz.SERIES[k], alpha=0.35, lw=0, label=DATASETS[name]
            )
    axes[0].set(
        title="(a) image coordinates",
        xlabel="x / width",
        ylabel="y / height",
        xlim=(-0.1, 1.1),
        ylim=(1.6, -0.1),
    )
    axes[1].set(
        title="(b) body frame, no aspect correction",
        xlabel="shoulder widths",
        xlim=(-2.2, 2.2),
        ylim=(3.0, -1.6),
    )
    axes[2].set(
        title="(c) body frame, aspect-corrected",
        xlabel="shoulder widths",
        xlim=(-2.2, 2.2),
        ylim=(3.0, -1.6),
    )
    for ax in axes:
        ax.set_aspect("equal")
    for ax in axes[1:]:
        ax.annotate("nose", xy=(0, -0.5), xytext=(1.0, -1.25), fontsize=6.5, color=viz.INK_2,
                    arrowprops={"arrowstyle": "-", "color": viz.INK_3, "lw": 0.6})  # fmt: skip
    leg = axes[0].legend(loc="lower left", markerscale=5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout(w_pad=0.8)
    viz.save(fig, FIG / "normalisation")


def typical_clip(ds: PackedDataset, label: str | None) -> int:
    """Row of a clip with a visible hand in most frames and a typical length (~30 frames)."""
    idx = ds.index
    cand = idx.index[idx.label == label] if label in set(idx.label) else idx.index
    best, best_score = int(cand[0]), -1.0
    for r in cand[:300]:
        s = ds.sequence(int(r))
        score = (~np.isnan(s[:, 21:42, 0]).all(1) | ~np.isnan(s[:, :21, 0]).all(1)).mean()
        score -= abs(len(s) - 30) / 300
        if score > best_score:
            best, best_score = int(r), score
    return best


def fig_strips(dss: dict[str, PackedDataset]) -> None:
    rows = []
    for name, ds in dss.items():
        idx = ds.index
        best = typical_clip(ds, {"kaggle_asl": "hello", "lsfb_isol": "AUSSI"}.get(name))
        seq = ds.sequence(best)
        rows.append((f"{DATASETS[name].split(' ')[0]} “{idx.label[best]}”", seq))
    viz.save(viz.plot_strips(rows), FIG / "examples")


def sign_gifs(dss: dict[str, PackedDataset]) -> None:
    """Animated skeleton of a typical clip for a few signs, in reports/figures/."""
    for name, ds in dss.items():
        for sign in GIF_SIGNS.get(name, []):
            out = ROOT / "reports" / "figures" / f"{name}_{sign}.gif"
            viz.save_gif(ds.sequence(typical_clip(ds, sign)), out, title=sign)


def data_figures() -> None:
    dss = available()
    if not dss:
        print("no processed dataset found")
        return
    stats = fig_dataset_overview(dss)
    rates = fig_missing(dss)
    for name, r in rates.iterrows():
        key = next(k for k, v in DATASETS.items() if v == name)
        stats[key]["missing"] = {k: float(v) for k, v in r.items()}
    fig_normalisation(dss)
    fig_strips(dss)
    sign_gifs(dss)
    (ROOT / "reports" / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def run_metrics(run: str) -> dict | None:
    return _load(ROOT / "reports" / "runs" / run / "metrics.json")


def eval_report(run: str, subset: str) -> dict | None:
    return _load(ROOT / "reports" / "eval" / f"{run}_{subset}.json")


def seeds(prefix: str, n: int = 3) -> list[str]:
    return [f"{prefix}-s{i}" for i in range(n) if run_metrics(f"{prefix}-s{i}")]


def mean_std(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def pct(v: float, d: int = 1) -> str:
    return f"{100 * v:.{d}f}"


def pm(values: list[float], d: int = 1) -> str:
    m, s = mean_std(values)
    return f"{pct(m, d)} $\\pm$ {pct(s, d)}" if len(values) > 1 else pct(m, d)


class Numbers:
    """Collects \\newcommand macros written to paper/tables/numbers.tex."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def __setitem__(self, k: str, v: object) -> None:
        self.values[k] = str(v)

    def write(self) -> None:
        old = {}
        path = TAB / "numbers.tex"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith("\\newcommand{\\"):
                    name = line.split("{\\", 1)[1].split("}", 1)[0]
                    old[name] = line.split("}{", 1)[1].rsplit("}", 1)[0]
        merged = {k: v for k, v in {**old, **self.values}.items() if k.isalpha()}
        lines = ["% generated by scripts/make_figures.py, do not edit"]
        lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(merged.items())]
        path.write_text("\n".join(lines) + "\n")


def js_parity_report() -> None:
    """Re-run the Python/JavaScript preprocessing parity check and store its report."""
    if shutil.which("node") is None:
        return
    subprocess.run([sys.executable, "scripts/export_web_assets.py"], cwd=ROOT, check=True,
                   capture_output=True)  # fmt: skip
    out = subprocess.run(["node", "app/web/test/parity.mjs", "tests/fixtures/js_parity.json"],
                         cwd=ROOT, capture_output=True, text=True)  # fmt: skip
    (ROOT / "reports" / "js_parity.json").write_text(out.stdout)


def numbers_from_data(nums: Numbers) -> None:
    stats = _load(ROOT / "reports" / "dataset_stats.json") or {}
    if "kaggle_asl" in stats:
        a = stats["kaggle_asl"]
        nums["numAslSamples"] = f"{a['n_samples']:,}"
        nums["numAslClasses"] = a["n_classes"]
        nums["numAslSigners"] = a["n_signers"]
    if "lsfb_isol" in stats:
        b = stats["lsfb_isol"]
        nums["numLsfbClasses"] = b["n_classes"]
        nums["numLsfbFps"] = int(b["fps"])
    lsfb_meta = _load(ROOT / "reports" / "lsfb_selection.json")
    if lsfb_meta:
        nums["numLsfbAllInstances"] = f"{lsfb_meta['all_instances']:,}"
        nums["numLsfbAllGlosses"] = f"{lsfb_meta['all_glosses']:,}"
        nums["numLsfbAllSigners"] = lsfb_meta["all_signers"]
        nums["numLsfbDroppedPct"] = f"{100 * lsfb_meta['dropped_share']:.1f}"
    bp = _load(ROOT / "reports" / "browser_parity.json")
    if bp:
        nums["numBrowserSame"] = f"{bp['same_top1']}/{bp['n_clips']}"
        nums["numBrowserDiff"] = (
            f"{bp['max_abs_prob_diff']:.0e}".replace("e-0", "\\times10^{-").replace(
                "e-", "\\times10^{-"
            )
            + "}"
        )
        nums["numExampleAcc"] = pct(bp["top1_on_examples"])
    js_parity_report()
    par = _load(ROOT / "reports" / "js_parity.json")
    if par:
        e = max(r["maxDiff"] for r in par)
        mant, exp = f"{e:.0e}".split("e")
        nums["numJsParity"] = f"${mant}\\times10^{{{int(exp)}}}$"


def table_datasets() -> None:
    stats = _load(ROOT / "reports" / "dataset_stats.json") or {}
    splits = {"kaggle_asl": "kaggle_asl_signer.json", "lsfb_isol": "lsfb_isol_signer.json"}
    rows = []
    for name, st in stats.items():
        sp = _load(ROOT / "splits" / splits[name])
        s = sp["signers"] if sp else {"train": [], "val": [], "test": []}
        rows.append(
            f"{DATASETS[name]} & {st['n_classes']} & {st['n_samples']:,} & {st['n_signers']} & "
            f"{len(s['train'])}/{len(s['val'])}/{len(s['test'])} & {st['per_class_min']}--"
            f"{st['per_class_max']} & {st['duration_median_s']:.2f}\\,s & {int(st['fps'])} & "
            f"{100 * st['missing']['any_hand']:.0f}\\,\\% \\\\"
        )
    body = "\n".join(rows)
    (TAB / "datasets.tex").write_text(
        "\\begin{tabular}{lrrrcrrrr}\n\\toprule\n"
        " & & & & Signers & Clips & & & No \\\\\n"
        "Dataset & Classes & Clips & Signers & tr/val/test & per class & Duration & fps & hand \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_main(nums: Numbers) -> None:
    """Main ASL results: baselines and the two sequence models (mean ± std over seeds)."""
    base = _load(ROOT / "reports" / "baselines" / "kaggle_asl_kaggle_asl_signer.json")
    lines = []
    if base:
        c = base["chance"]
        lines.append(f"Chance & -- & {pct(c['top1'], 2)} & {pct(c['top1'], 2)} & "
                     f"{pct(c['top5'], 1)} & -- & -- \\\\")  # fmt: skip
        for key, name in (("logreg", "Statistics + log.\\ regression"),
                          ("lgbm", "Statistics + LightGBM")):  # fmt: skip
            if key in base:
                b = base[key]
                te = b.get("test", {})
                lines.append(
                    f"{name} & -- & {pct(b['val']['top1'])} & {pct(te.get('top1', float('nan')))} & "
                    f"{pct(te.get('top5', float('nan')))} & {pct(te.get('macro_f1', float('nan')))} & -- \\\\"
                )
        lines.append("\\midrule")
    for prefix, name, key in (
        ("asl-gru", "BiGRU", "Gru"),
        ("asl-transformer", "Conv-Transformer", "Tr"),
    ):
        runs = seeds(prefix)
        if not runs:
            continue
        val = [run_metrics(r)["best_val"]["top1"] for r in runs]
        tests = [eval_report(r, "test") for r in runs]
        tests = [t for t in tests if t]
        params = run_metrics(runs[0])["n_params"] / 1e6
        nums[f"num{key}Params"] = f"{params:.2f}"
        if not tests:
            continue
        t1 = [t["top1"] for t in tests]
        nums[f"num{key}TestTop"] = pm(t1)
        nums[f"num{key}TestTopMean"] = pct(mean_std(t1)[0])
        nums[f"num{key}ValTopMean"] = pct(mean_std(val)[0])
        nums[f"num{key}SignerStd"] = pct(np.mean([t["signer_acc_std"] for t in tests]))
        lines.append(
            f"{name} & {params:.2f}\\,M & {pm(val)} & {pm(t1)} & {pm([t['top5'] for t in tests])} & "
            f"{pm([t['macro_f1'] for t in tests])} & {pm([t['signer_acc_std'] for t in tests])} \\\\"
        )
    (TAB / "main_results.tex").write_text(
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        " & & \\multicolumn{1}{c}{Validation} & \\multicolumn{4}{c}{Test (4 unseen signers)} \\\\\n"
        "\\cmidrule(lr){3-3}\\cmidrule(lr){4-7}\n"
        "Model & Params & Top-1 & Top-1 & Top-5 & Macro-F1 & Signer s.d. \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


ABLATIONS = [
    ("abl-no-augment", "no augmentation"),
    ("abl-no-lips", "no lips"),
    ("abl-hands-only", "hands only"),
    ("abl-no-handlocal", "no hand-local features"),
    ("abl-no-velocity", "no velocities"),
    ("abl-no-canonical", "no canonical hand"),
    ("abl-xyz", "with depth $z$"),
    ("abl-pad", "pad + mask (no resampling)"),
    ("abl-T32", "$T'=32$"),
    ("abl-T96", "$T'=96$"),
]
DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def latex_name(run: str) -> str:
    """LaTeX macro name for an ablation run (macro names may only contain letters)."""
    name = "".join(w.capitalize() for w in run.removeprefix("abl-").split("-"))
    return "numAbl" + "".join(DIGITS[int(c)] if c.isdigit() else c for c in name)


def fig_ablations(nums: Numbers) -> None:
    ref = run_metrics("asl-transformer-s0")
    # runs that stopped early on a plateau were re-run for the full schedule ("-full")
    found = [(r, lab, run_metrics(f"{r}-full") or run_metrics(r)) for r, lab in ABLATIONS]
    rows = [(lab, m) for _, lab, m in found if m]
    if not ref or not rows:
        return
    r1 = ref["best_val"]["top1"]

    def stopped_early(m: dict) -> bool:
        h = pd.read_csv(ROOT / "reports" / "runs" / m["run"] / "history.csv")
        return len(h) < 48  # the schedule has 50 epochs

    flag = {lab: stopped_early(m) for lab, m in rows}
    nums["numAblEarlyStopped"] = sum(flag.values())
    deltas = [
        (
            lab.replace("$", "").replace("T'", "T′") + (" †" if flag[lab] else ""),
            100 * (m["best_val"]["top1"] - r1),
        )
        for lab, m in rows
    ]
    deltas.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(viz.COL_W * 1.3, 0.22 * len(deltas) + 0.6))
    y = np.arange(len(deltas))
    vals = np.array([d for _, d in deltas])
    colors = [viz.SERIES[7] if v < 0 else viz.SERIES[0] for v in vals]  # diverging blue / red
    ax.barh(y, vals, height=0.62, color=colors)
    for yi, v in zip(y, vals, strict=True):
        ax.text(v + (0.15 if v >= 0 else -0.15), yi, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.5, color=viz.INK_2)  # fmt: skip
    ax.set_yticks(y, [lab for lab, _ in deltas])
    ax.axvline(0, color=viz.INK_3, lw=0.8)
    ax.set_xlabel(f"Δ validation top-1 (points) vs. full model ({pct(r1)} %)")
    ax.grid(axis="y", visible=False)
    lo, hi = vals.min(), vals.max()
    pad = 0.18 * (hi - lo + 1)
    ax.set_xlim(min(lo - pad, -pad), max(hi + pad, pad))
    fig.tight_layout()
    viz.save(fig, FIG / "ablations")
    nums["numRefValTop"] = pct(r1)
    for run, _, m in found:
        if m:
            nums[latex_name(run)] = f"{100 * (m['best_val']['top1'] - r1):+.1f}"
    dagger = "$^\\dagger$"
    lines = [f"{lab}{dagger if flag[lab] else ''} & {pct(m['best_val']['top1'])} & "
             f"{100 * (m['best_val']['top1'] - r1):+.1f} \\\\" for lab, m in rows]  # fmt: skip
    (TAB / "ablations.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\nVariant & Val.\\ top-1 & $\\Delta$ \\\\\n\\midrule\n"
        f"Full model & {pct(r1)} & -- \\\\\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def leakage(nums: Numbers) -> None:
    rnd = eval_report("abl-random-split", "test")
    sig = [eval_report(r, "test") for r in seeds("asl-transformer")]
    sig = [s for s in sig if s]
    if rnd and sig:
        nums["numLeakRandom"] = pct(rnd["top1"])
        nums["numLeakSigner"] = pct(mean_std([s["top1"] for s in sig])[0])
        nums["numLeakGap"] = f"{100 * (rnd['top1'] - mean_std([s['top1'] for s in sig])[0]):.1f}"


def fig_training_curves() -> None:
    fig, ax = plt.subplots(figsize=(viz.COL_W, 1.9))
    drawn = False
    for k, (prefix, name) in enumerate(
        (("asl-transformer", "Conv-Transformer"), ("asl-gru", "BiGRU"))
    ):
        hist = [pd.read_csv(ROOT / "reports" / "runs" / r / "history.csv") for r in seeds(prefix)]
        if not hist:
            continue
        n = min(len(h) for h in hist)
        acc = np.stack([h["val_top1"].to_numpy()[:n] for h in hist]) * 100
        ep = np.arange(1, n + 1)
        ax.plot(ep, acc.mean(0), color=viz.SERIES[k], label=name)
        if len(hist) > 1:
            ax.fill_between(ep, acc.min(0), acc.max(0), color=viz.SERIES[k], alpha=0.15, lw=0)
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set(xlabel="epoch", ylabel="validation top-1 (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    viz.save(fig, FIG / "training_curves")


def fig_per_signer_and_missing() -> None:
    reps = {name: [eval_report(r, "test") for r in seeds(prefix)]
            for prefix, name in (("asl-transformer", "Conv-Transformer"), ("asl-gru", "BiGRU"))}  # fmt: skip
    reps = {k: [r for r in v if r] for k, v in reps.items()}
    reps = {k: v for k, v in reps.items() if v}
    if not reps:
        return
    fig, axes = plt.subplots(
        1, 2, figsize=(viz.FULL_W, 1.95), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    ax = axes[0]
    signers = sorted({d["g"] for r in next(iter(reps.values())) for d in r["by_signer"]})
    for k, (name, rs) in enumerate(reps.items()):
        acc = (
            np.array([[{d["g"]: d["acc"] for d in r["by_signer"]}[s] for s in signers] for r in rs])
            * 100
        )
        x = np.arange(len(signers)) + (k - 0.5) * 0.18
        ax.errorbar(x, acc.mean(0), yerr=acc.std(0) if len(rs) > 1 else None, fmt="o", ms=4,
                    color=viz.SERIES[k], label=name, capsize=2, lw=1)  # fmt: skip
    ax.set_xticks(np.arange(len(signers)), [f"signer {s}" for s in signers])
    ax.set_ylabel("test top-1 (%)")
    ax.set_title("(a) accuracy per unseen signer")
    ax.set_xlim(-0.5, len(signers) - 0.5)
    ax = axes[1]
    first = next(iter(reps.values()))
    bins = [d["bin"] for d in first[0]["by_missing"]]
    w = 0.38
    for k, (name, rs) in enumerate(reps.items()):
        acc = np.array([[d["acc"] for d in r["by_missing"]] for r in rs]) * 100
        n = [d["n"] for d in rs[0]["by_missing"]]
        x = np.arange(len(bins)) + (k - 0.5) * w
        ax.bar(x, np.nan_to_num(acc.mean(0)), w * 0.92, color=viz.SERIES[k], label=name)
    ax.set_xticks(np.arange(len(bins)), [f"{b}\n(n={c:,})" for b, c in zip(bins, n, strict=True)])
    ax.set_xlabel("share of frames with no hand detected")
    ax.set_ylabel("test top-1 (%)")
    handles, labels = ax.get_legend_handles_labels()
    ax.set_title("(b) accuracy vs. missing hands")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.0))
    viz.save(fig, FIG / "signers_missing")


def table_confusions(k: int = 12) -> None:
    runs = seeds("asl-transformer")
    rep = eval_report(runs[0], "test") if runs else None
    if not rep:
        return
    rows = [f"{p['true']} & {p['pred']} & {p['count']} & {100 * p['rate']:.0f}\\,\\% \\\\"
            for p in rep["confused_pairs"][:k]]  # fmt: skip
    (TAB / "confusions.tex").write_text(
        "\\begin{tabular}{llrr}\n\\toprule\nTrue sign & Predicted & Count & Rate \\\\\n\\midrule\n"
        + "\n".join(rows).replace("_", "\\_")
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def error_numbers(nums: Numbers) -> None:
    """Per-signer range, effect of missing hands and hardest classes (Conv-Transformer, test)."""
    reps = [r for r in (eval_report(x, "test") for x in seeds("asl-transformer")) if r]
    if not reps:
        return
    by: dict[str, list[float]] = {}
    for r in reps:
        for d in r["by_signer"]:
            by.setdefault(str(d["g"]), []).append(d["acc"])
    means = {k: float(np.mean(v)) for k, v in by.items()}
    nums["numSignerMin"] = pct(min(means.values()))
    nums["numSignerMax"] = pct(max(means.values()))
    bins = {d["bin"]: d for d in reps[0]["by_missing"]}
    total = sum(d["n"] for d in bins.values())
    nums["numMissHighAcc"] = pct(
        np.mean([{d["bin"]: d for d in r["by_missing"]}[">50%"]["acc"] for r in reps])
    )
    nums["numMissHighShare"] = f"{100 * bins['>50%']['n'] / total:.0f}"
    nums["numMissLowAcc"] = pct(
        np.mean([{d["bin"]: d for d in r["by_missing"]}["0-10%"]["acc"] for r in reps])
    )
    worst = reps[0]["worst_classes"][:3]
    nums["numWorstClasses"] = ", ".join(
        f"\\textsc{{{w['label']}}} ({pct(w['acc'], 0)}\\,\\%)" for w in worst
    )


def long_and_ensembles(nums: Numbers) -> None:
    """150-epoch run and seed ensembles (probabilities averaged over the three seeds)."""
    long = run_metrics("asl-transformer-long")
    if long:
        nums["numLongVal"] = pct(long["best_val"]["top1"])

    def ens(runs: list[str], nc: int) -> dict | None:
        files = [ROOT / "reports" / "eval" / f"{r}_test.npz" for r in runs]
        if not all(f.exists() for f in files):
            return None
        ds = [np.load(f) for f in files]
        return summary(np.mean([d["probs"].astype(np.float32) for d in ds], 0), ds[0]["y"], nc)

    for key, runs, nc in (("AslEns", [f"asl-transformer-s{i}" for i in range(3)], 250),
                          ("LsfbEns", [f"lsfb-scratch-nall-s{i}" for i in range(3)], 100)):  # fmt: skip
        e = ens(runs, nc)
        if e:
            nums[f"num{key}Top"] = pct(e["top1"])
            nums[f"num{key}FOne"] = pct(e["macro_f1"])


def transfer_checks(nums: Numbers) -> None:
    """Linear probe and low-learning-rate fine-tuning (validation top-1, seed 0)."""
    for key, run in (("ProbeAll", "lsfb-probe-nall-s0"), ("ProbeFive", "lsfb-probe-n5-s0"),
                     ("LowlrTwentyFive", "lsfb-ftlowlr-n25-s0"),
                     ("ScratchTwentyFive", "lsfb-scratch-n25-s0"), ("FtTwentyFive", "lsfb-ft-n25-s0"),
                     ("ScratchFive", "lsfb-scratch-n5-s0"), ("FtFive", "lsfb-ft-n5-s0")):  # fmt: skip
        m = run_metrics(run)
        if m:
            nums[f"num{key}"] = pct(m["best_val"]["top1"])
    base = _load(ROOT / "reports" / "baselines" / "lsfb_isol_lsfb_isol_signer.json")
    if base:
        nums["numLsfbLogregVal"] = pct(base["logreg"]["val"]["top1"])


def results_figures() -> None:
    nums = Numbers()
    numbers_from_data(nums)
    table_datasets()
    table_main(nums)
    fig_ablations(nums)
    leakage(nums)
    fig_training_curves()
    fig_per_signer_and_missing()
    table_confusions()
    error_numbers(nums)
    long_and_ensembles(nums)
    transfer_checks(nums)
    transfer_figures(nums)
    table_transfer(nums)
    export_tables(nums)
    nums.write()


LC_SIZES = [5, 10, 25, 50, 100]


def transfer_figures(nums: Numbers) -> None:
    """Learning curves on LSFB: from scratch vs. ASL-pretrained (mean ± std over seeds)."""
    settings = (("lsfb-scratch", "from scratch"), ("lsfb-ft", "ASL-pretrained, fine-tuned"))
    curves = {}
    for k, (prefix, name) in enumerate(settings):
        xs, ms, ss = [], [], []
        for n in [*LC_SIZES, "all"]:
            runs = [r for r in seeds(f"{prefix}-n{n}") if run_metrics(r)]
            if not runs:
                continue
            vals = [run_metrics(r)["best_val"]["top1"] for r in runs]
            xs.append(n)
            m, s = mean_std(vals)
            ms.append(100 * m)
            ss.append(100 * s)
        if xs:
            curves[name] = (xs, np.array(ms), np.array(ss), viz.SERIES[k])
    if not curves:
        return
    full = _load(ROOT / "reports" / "lsfb_train_per_class.json") or {}
    xmax = full.get("median", 300)
    fig, ax = plt.subplots(figsize=(viz.COL_W * 1.2, 2.0))
    for name, (xs, m, s, c) in curves.items():
        xv = np.array([xmax if x == "all" else x for x in xs], dtype=float)
        ax.plot(xv, m, marker="o", ms=3.5, color=c, label=name)
        ax.fill_between(xv, m - s, m + s, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ticks = [*LC_SIZES, xmax]
    ax.set_xticks(ticks, [*map(str, LC_SIZES), "all"])
    ax.minorticks_off()
    ax.set(xlabel="training examples per sign", ylabel="LSFB validation top-1 (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    viz.save(fig, FIG / "transfer_curve")
    nums["numLcMin"] = LC_SIZES[0]
    if all(name in curves for _, name in settings):
        (xa, ma, _, _), (xb, mb, _, _) = curves[settings[0][1]], curves[settings[1][1]]
        common = [x for x in xa if x in xb]
        gains = {x: mb[xb.index(x)] - ma[xa.index(x)] for x in common}
        if LC_SIZES[0] in gains:
            nums["numGainFew"] = f"{gains[LC_SIZES[0]]:.1f}"
        if "all" in gains:
            nums["numGainAll"] = f"{gains['all']:.1f}"
            nums["numScratchLeadAll"] = f"{-gains['all']:.1f}"


def table_transfer(nums: Numbers) -> None:
    """LSFB with all training data: baselines, from scratch and ASL-pretrained (mean ± std)."""
    base = _load(ROOT / "reports" / "baselines" / "lsfb_isol_lsfb_isol_signer.json")
    lines = []
    if base:
        c = base["chance"]
        lines.append(f"Chance & {pct(c['top1'])} & {pct(c['top1'])} & {pct(c['top5'])} & -- \\\\")
        for key, name in (
            ("logreg", "Statistics + log.\\ regression"),
            ("lgbm", "Statistics + LightGBM"),
        ):
            if key in base:
                b = base[key]
                lines.append(f"{name} & {pct(b['val']['top1'])} & {pct(b['test']['top1'])} & "
                             f"{pct(b['test']['top5'])} & {pct(b['test']['macro_f1'])} \\\\")  # fmt: skip
        lines.append("\\midrule")
    settings = (("lsfb-scratch-nall", "Conv-Transformer, from scratch", "Scratch"),
                ("lsfb-ft-nall", "\\quad ASL-pretrained, fine-tuned", "Ft"),
                ("lsfb-ftfreeze-nall", "\\quad ASL-pretrained, encoder frozen 5 epochs", "Freeze"))  # fmt: skip
    for prefix, name, key in settings:
        runs = seeds(prefix)
        if not runs:
            continue
        val = [run_metrics(r)["best_val"]["top1"] for r in runs]
        tests = [t for t in (eval_report(r, "test") for r in runs) if t]
        cells = [pm(val)]
        if tests:
            cells += [pm([t["top1"] for t in tests]), pm([t["top5"] for t in tests]),
                      pm([t["macro_f1"] for t in tests])]  # fmt: skip
            nums[f"numLsfb{key}Test"] = pct(mean_std([t["top1"] for t in tests])[0])
            nums[f"numLsfb{key}TestFOne"] = pct(mean_std([t["macro_f1"] for t in tests])[0])
        else:
            cells += ["--"] * 3
        nums[f"numLsfb{key}Val"] = pct(mean_std(val)[0])
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    (TAB / "transfer.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        " & Validation & \\multicolumn{3}{c}{Test (40 unseen signers)} \\\\\n\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}\n"
        "Model & Top-1 & Top-1 & Top-5 & Macro-F1 \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def export_tables(nums: Numbers) -> None:
    """Size and latency (native and browser) of FP32 / INT8 models; INT8 accuracy if exported."""
    lat = _load(ROOT / "reports" / "latency.json")
    if not lat:
        return
    acc = {}
    for mid in ("asl", "lsfb"):
        rep = _load(ROOT / "app" / "web" / "models" / mid / "export_report.json")
        if rep:
            acc[rep["run"].split("-s")[0]] = rep
    rows = []
    for arch, name in (("asl-transformer", "Conv-Transformer"), ("asl-gru", "BiGRU")):
        for tag, prec in (("fp32", "FP32"), ("int8", "INT8")):
            e = lat.get(f"{arch}_{tag}")
            if not e:
                continue
            web = e.get("browser", {})
            top1 = acc.get(arch, {}).get(tag, {}).get("val_top1")
            rows.append(
                f"{name} & {prec} & {e['size_mb']:.1f}\\,MB & {e['native']['p50_ms']:.1f} & "
                f"{web.get('p50_ms', float('nan')):.1f} & {web.get('p95_ms', float('nan')):.1f} & "
                f"{pct(top1) if top1 is not None else '--'} \\\\"
            )
    (TAB / "export.tex").write_text(
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        " & & & Native & \\multicolumn{2}{c}{Browser (WebAssembly)} & \\\\\n\\cmidrule(lr){5-6}\n"
        "Model & Precision & Size & p50 (ms) & p50 (ms) & p95 (ms) & Val.\\ top-1 \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    t8 = lat.get("asl-transformer_int8")
    if t8:
        nums["numIntSizeMB"] = f"{t8['size_mb']:.1f}"
        nums["numIntLatency"] = f"{t8.get('browser', t8['native'])['p50_ms']:.0f}"
        nums["numFpSizeMB"] = f"{lat['asl-transformer_fp32']['size_mb']:.1f}"
        nums["numIntNative"] = f"{t8['native']['p50_ms']:.1f}"
        nums["numFpNative"] = f"{lat['asl-transformer_fp32']['native']['p50_ms']:.1f}"
        fp = lat["asl-transformer_fp32"]
        nums["numFpBrowser"] = f"{fp.get('browser', fp['native'])['p50_ms']:.0f}"
        nums["numIntSlowdown"] = f"{t8['native']['p50_ms'] / fp['native']['p50_ms']:.1f}"
        nums["numIntShrink"] = f"{fp['size_mb'] / t8['size_mb']:.1f}"
        g = lat.get("asl-gru_fp32")
        if g:
            nums["numGruBrowser"] = f"{g.get('browser', g['native'])['p50_ms']:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["data", "results"], default=None)
    a = ap.parse_args()
    viz.paper_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "data"):
        data_figures()
    if a.only in (None, "results"):
        results_figures()


if __name__ == "__main__":
    main()
