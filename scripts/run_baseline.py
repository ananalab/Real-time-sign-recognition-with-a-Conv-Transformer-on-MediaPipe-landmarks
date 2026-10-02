"""Classical baselines on temporal statistics of the preprocessed landmarks.

    uv run python scripts/run_baseline.py --data data/processed/kaggle_asl \
        --split splits/kaggle_asl_signer.json [--models logreg lgbm] [--eval-test]

Writes ``reports/baselines/<dataset>_<split>.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signrec.baseline import build_stat_matrix  # torch-free: avoids an OpenMP clash with LightGBM
from signrec.metrics import summary
from signrec.preprocess import PreprocessConfig
from signrec.splits import load_split, split_indices
from signrec.store import PackedDataset

ROOT = Path(__file__).resolve().parents[1]


def fit_logreg(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
    return clf.fit(X, y)


def fit_lgbm(
    X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray, n_classes: int, seed: int
) -> object:
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.3,
        reg_lambda=1.0,
        max_bin=63,
        random_state=seed,
        verbose=-1,
    )
    # early stopping on the classification error: the multiclass log-loss starts rising long
    # before the accuracy stops improving
    clf.fit(
        X,
        y,
        eval_X=(Xv,),
        eval_y=(yv,),
        eval_metric="multi_error",
        callbacks=[lgb.early_stopping(50, first_metric_only=False, verbose=False)],
    )
    return clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--preprocess", default="configs/preprocess.yaml")
    ap.add_argument("--models", nargs="+", default=["logreg", "lgbm"])
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ds = PackedDataset(a.data)
    split = load_split(a.split)
    rows = split_indices(ds.index, split)
    y_all = ds.targets()
    pcfg = PreprocessConfig.from_yaml(a.preprocess)
    subsets = ["train", "val"] + (["test"] if a.eval_test else [])
    t0 = time.time()
    X = {k: build_stat_matrix(ds, rows[k], pcfg) for k in subsets}
    y = {k: y_all[rows[k]] for k in subsets}
    print(f"features {X['train'].shape} in {time.time() - t0:.0f}s")
    scaler = StandardScaler().fit(X["train"])
    X = {k: scaler.transform(v).astype(np.float32) for k, v in X.items()}

    n_classes = ds.n_classes
    out = ROOT / "reports" / "baselines" / f"{Path(a.data).name}_{Path(a.split).stem}.json"
    previous = json.loads(out.read_text()) if out.exists() else {}
    results: dict = {
        **previous,
        "dataset": Path(a.data).name,
        "split": Path(a.split).stem,
        "n_features": int(X["train"].shape[1]),
        "chance": {"top1": 1 / n_classes, "top5": 5 / n_classes},
    }
    for name in a.models:
        t = time.time()
        if name == "logreg":
            clf = fit_logreg(X["train"], y["train"], a.seed)
        else:
            clf = fit_lgbm(X["train"], y["train"], X["val"], y["val"], n_classes, a.seed)
        entry = {"fit_minutes": (time.time() - t) / 60}
        if name == "lgbm":
            entry["best_iteration"] = int(clf.best_iteration_ or clf.n_estimators)
        for k in subsets[1:]:
            probs = np.zeros((len(y[k]), n_classes), np.float32)
            probs[:, clf.classes_] = clf.predict_proba(X[k])
            entry[k] = summary(probs, y[k], n_classes)
        results[name] = entry
        print(name, json.dumps(entry), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
