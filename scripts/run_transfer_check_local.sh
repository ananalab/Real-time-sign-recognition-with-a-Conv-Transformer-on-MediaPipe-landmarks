#!/bin/bash
# Reduced version of experiments/lsfb_transfer_check.yaml, run locally on the M1 (MPS) once the
# Kaggle GPU quota was used up: linear probe with all data, then low-learning-rate fine-tuning with
# 25 examples per sign. The other runs of the plan were dropped (several hours each on the M1).
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

COMMON=(
  mlflow=false seed=0
  data.root=data/processed/lsfb_isol
  data.split=splits/lsfb_isol_signer.json
  train.cache_val=true
  transfer.init_from=models/asl-transformer-s0/best.pt
)

uv run python -m signrec.train --config configs/train.yaml "${COMMON[@]}" \
  run_name=lsfb-probe-nall-s0 train.epochs=60 train.patience=15 transfer.freeze_epochs=60 \
  > logs/lsfb-probe-nall-s0.log 2>&1

uv run python -m signrec.train --config configs/train.yaml "${COMMON[@]}" \
  run_name=lsfb-ftlowlr-n25-s0 train.epochs=120 train.patience=30 train.lr=2.0e-4 \
  train.warmup_epochs=1 data.max_per_class=25 train.num_workers=0 \
  > logs/lsfb-ftlowlr-n25-s0.log 2>&1
