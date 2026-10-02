"""Training entry point.

    uv run python -m signrec.train --config configs/train.yaml [key=value ...]

Writes ``models/<run>/best.pt`` (weights + everything needed for inference) and
``reports/runs/<run>/{config.yaml,history.csv,metrics.json}``; also logs to MLflow (``mlruns/``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from signrec.augment import AugmentConfig
from signrec.dataset import SignDataset
from signrec.metrics import summary
from signrec.models import build_model
from signrec.models.common import count_parameters
from signrec.preprocess import PreprocessConfig
from signrec.splits import load_split, split_indices
from signrec.store import PackedDataset

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def resolve_preprocess(cfg: DictConfig) -> PreprocessConfig:
    over = cfg.get("preprocess_overrides")
    over = OmegaConf.to_container(over, resolve=True) if OmegaConf.is_config(over) else {}
    pcfg = PreprocessConfig.from_yaml(ROOT / cfg.preprocess, **over)
    max_len = cfg_model(cfg).get("max_len")
    if max_len is not None and max_len < pcfg.T:
        raise ValueError("preprocess.T exceeds the model positional embedding length")
    return pcfg


def cfg_model(cfg: DictConfig) -> dict:
    m = cfg.model
    if isinstance(m, str):
        m = OmegaConf.load(ROOT / m)
    d = dict(OmegaConf.to_container(m, resolve=True) if OmegaConf.is_config(m) else m)
    over = cfg.get("model_overrides")
    if OmegaConf.is_config(over):
        d.update(OmegaConf.to_container(over, resolve=True))
    return d


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Data:
    ds: PackedDataset
    rows: dict[str, np.ndarray]
    label_map: dict[int, int]  # dataset class id -> model class id
    labels: list[str]  # model class id -> gloss


def prepare_data(cfg: DictConfig) -> Data:
    ds = PackedDataset(ROOT / cfg.data.root)
    rows = split_indices(ds.index, load_split(ROOT / cfg.data.split))
    y = ds.targets()
    classes = np.arange(ds.n_classes)
    if cfg.data.get("n_classes"):
        classes = classes[: cfg.data.n_classes]
    rows = {k: v[np.isin(y[v], classes)] for k, v in rows.items()}
    if cfg.data.get("max_per_class"):
        rng = np.random.default_rng(cfg.seed)
        tr = rows["train"]
        picked = [rng.permutation(tr[y[tr] == c])[: cfg.data.max_per_class] for c in classes]
        rows["train"] = np.sort(np.concatenate(picked))
    label_map = {int(c): i for i, c in enumerate(classes)}
    labels = [ds.id_to_label[int(c)] for c in classes]
    return Data(ds, rows, label_map, labels)


def make_loader(dataset: SignDataset, batch: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=shuffle and len(dataset) > batch,
    )


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for x, m, y in loader:
        logits = model(x.to(device), m.to(device))
        probs.append(logits.float().softmax(-1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def load_encoder(model: nn.Module, path: str | Path) -> list[str]:
    """Initialise every non-head parameter from a checkpoint; returns the missing keys."""
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {path}: {unexpected[:5]}")
    return list(missing)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if not name.startswith("head."):
            p.requires_grad_(trainable)


def train(cfg: DictConfig) -> dict:
    """Train one model; return the best validation metrics and artefact paths."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.train.device)
    pcfg = resolve_preprocess(cfg)
    mcfg = cfg_model(cfg)
    acfg = AugmentConfig.from_yaml(ROOT / cfg.augment) if cfg.get("augment") else None
    data = prepare_data(cfg)
    n_classes = len(data.labels)
    dataset_name = Path(cfg.data.root).name
    run = cfg.run_name or f"{dataset_name}-{mcfg['name']}-{time.strftime('%Y%m%d-%H%M%S')}"

    train_ds = SignDataset(data.ds, data.rows["train"], pcfg, acfg, data.label_map, cfg.seed)
    val_ds = SignDataset(data.ds, data.rows["val"], pcfg, None, data.label_map)
    tl = make_loader(train_ds, cfg.train.batch_size, True, cfg.train.num_workers)
    if cfg.train.get("cache_val"):
        # validation features are deterministic: compute them once instead of every epoch
        items = [val_ds[i] for i in range(len(val_ds))]
        val_ds = torch.utils.data.TensorDataset(*(torch.stack(t) for t in zip(*items, strict=True)))
        vl = make_loader(val_ds, 256, False, 0)
    else:
        vl = make_loader(val_ds, 256, False, cfg.train.num_workers)

    model_kwargs = {k: v for k, v in mcfg.items() if k != "name"}
    model = build_model(mcfg["name"], pcfg.n_features, n_classes, **model_kwargs).to(device)
    if cfg.transfer.get("init_from"):
        load_encoder(model, ROOT / cfg.transfer.init_from)
    freeze_epochs = int(cfg.transfer.get("freeze_epochs") or 0)
    n_params = count_parameters(model)
    print(
        f"[{run}] {mcfg['name']} {n_params / 1e6:.2f}M params | {n_classes} classes | "
        f"train={len(train_ds)} val={len(val_ds)} | device={device}"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    steps_per_epoch = max(len(tl), 1)
    total = cfg.train.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_with_warmup(s, total, cfg.train.warmup_epochs * steps_per_epoch)
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    use_amp = device.type == "cuda"  # fp16 autocast brings no speed-up on MPS (measured)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = ROOT / "reports" / "runs" / run
    ckpt_dir = ROOT / "models" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    tracker = _MlflowTracker(cfg, run, n_params) if cfg.get("mlflow") else None
    history, best, bad_epochs = [], {"macro_f1": -1.0}, 0
    t_start = time.time()
    for epoch in range(cfg.train.epochs):
        set_encoder_trainable(model, epoch >= freeze_epochs)
        train_ds.set_epoch(epoch)
        model.train()
        t0, losses = time.time(), []
        for x, m, y in tl:
            x, m, y = x.to(device), m.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x, m), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        probs, yv = predict(model, vl, device)
        met = summary(probs, yv, n_classes)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "lr": sched.get_last_lr()[0],
            "seconds": time.time() - t0,
            **{f"val_{k}": v for k, v in met.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if tracker:
            tracker.log(row, epoch)
        improved = met["macro_f1"] > best["macro_f1"]
        print(
            f"  ep {epoch:3d} loss {row['train_loss']:.3f} | val top1 {met['top1']:.4f} "
            f"top5 {met['top5']:.4f} f1 {met['macro_f1']:.4f} | {row['seconds']:.0f}s"
            + (" *" if improved else ""),
            flush=True,
        )
        if improved:
            best = {**met, "epoch": epoch}
            bad_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": mcfg,
                    "preprocess": pcfg.to_dict(),
                    "labels": data.labels,
                    "n_features": pcfg.n_features,
                    "run": run,
                    "git": git_hash(),
                },
                ckpt_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                print(f"  early stopping at epoch {epoch}")
                break

    result = {
        "run": run,
        "model": mcfg["name"],
        "dataset": dataset_name,
        "n_params": n_params,
        "n_classes": n_classes,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "train_minutes": (time.time() - t_start) / 60,
        "git": git_hash(),
        "best_val": best,
        "checkpoint": str((ckpt_dir / "best.pt").relative_to(ROOT)),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    if tracker:
        tracker.finish(result)
    return result


class _MlflowTracker:
    def __init__(self, cfg: DictConfig, run: str, n_params: int) -> None:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
        mlflow.set_experiment(Path(cfg.data.root).name)
        mlflow.start_run(run_name=run)
        flat = pd.json_normalize(OmegaConf.to_container(cfg, resolve=True)).iloc[0].to_dict()
        mlflow.log_params({k: str(v)[:250] for k, v in flat.items()})
        mlflow.log_params({"n_params": n_params, "git": git_hash()})

    def log(self, row: dict, step: int) -> None:
        self.mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "epoch"}, step=step)

    def finish(self, result: dict) -> None:
        self.mlflow.log_metrics({f"best_{k}": float(v) for k, v in result["best_val"].items()})
        self.mlflow.end_run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides (key=value)")
    args = ap.parse_args()
    print(json.dumps(train(load_config(args.config, args.overrides)), indent=1))


if __name__ == "__main__":
    main()
