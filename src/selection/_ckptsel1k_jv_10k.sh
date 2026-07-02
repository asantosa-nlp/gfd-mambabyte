#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PYBIN:-$ROOT/.venv/bin/python}"
RESULT_DIR="$ROOT/results/ckptsel1k_jv_10k"
CONFIG="$ROOT/configs/mb_jv_zs_large_v5cleanS10k_ckptsel1k.yaml"

mkdir -p "$RESULT_DIR"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH="$ROOT/src/decoding" "$PYBIN" -u           "$ROOT/src/scoring/run_exp_mb.py" "$CONFIG" --split search --resume           2>&1 | tee -a "$RESULT_DIR/run.log"
