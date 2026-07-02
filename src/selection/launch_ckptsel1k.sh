#!/usr/bin/env bash
# Launch a 1000-utt checkpoint selection run on a specified local or remote GPU.
#
# Usage:
#   bash src/selection/launch_ckptsel1k.sh <jv|su> <20k|40k> <local|remote> <gpu>
#
# Examples:
#   bash src/selection/launch_ckptsel1k.sh jv 20k local 5
#   bash src/selection/launch_ckptsel1k.sh jv 40k local 6
#   bash src/selection/launch_ckptsel1k.sh su 20k remote 2
#   bash src/selection/launch_ckptsel1k.sh su 40k remote 3
#
# Local GPUs available (check nvidia-smi first): 5, 6  (NEVER use local 7)
# Remote GPUs available: 2, 3, 5, 6, 7  (NEVER use remote 0; 1 and 4 permitted if needed)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-<REMOTE_USER@REMOTE_HOST>}"
IMAGE="${IMAGE:-nvidia-cuda-ffmpeg:12.8.1-python3.12}"
PYBIN="${PYBIN:-$ROOT/.venv/bin/python}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ $# -eq 4 ]] || die "usage: $0 <jv|su> <20k|40k> <local|remote> <gpu>"

LANG="$1"
CKPT="$2"
MODE="$3"
GPU="$4"

[[ "$LANG" == jv || "$LANG" == su ]]   || die "lang must be jv or su"
[[ "$CKPT" == 10k || "$CKPT" == 20k || "$CKPT" == 40k ]] || die "ckpt must be 10k, 20k or 40k"
[[ "$MODE" == local || "$MODE" == remote ]] || die "mode must be local or remote"
[[ "$GPU" =~ ^[0-9]+$ ]]               || die "gpu must be a number"

# Guard against known-bad GPUs
if [[ "$MODE" == local && "$GPU" == 7 ]]; then
    die "Local GPU 7 is broken — never use it"
fi
if [[ "$MODE" == remote && "$GPU" == 0 ]]; then
    die "Remote GPU 0 is permanently reserved"
fi

# Resolve config and result dir
# NOTE: the public wrapper keeps the original 10k/20k/40k entry points for
# compatibility. A 30k checkpoint-selection config exists in configs/mb_*S30k_ckptsel1k*.yaml
# and can be launched directly if needed.
SFX=""
[[ "$LANG" == su ]] && SFX="_minb15"
if [[ "$CKPT" == 10k ]]; then
    CONFIG="$ROOT/configs/mb_${LANG}_zs_large_v5cleanS10k_ckptsel1k${SFX}.yaml"
elif [[ "$CKPT" == 20k ]]; then
    CONFIG="$ROOT/configs/mb_${LANG}_zs_large_v5cleanS20k_ckptsel1k${SFX}.yaml"
else
    CONFIG="$ROOT/configs/mb_${LANG}_zs_large_v5clean_ckptsel1k${SFX}.yaml"
fi
RESULT_DIR="$ROOT/results/ckptsel1k_${LANG}_${CKPT}"
LOG="$ROOT/results/ckptsel1k_${LANG}_${CKPT}.log"
CONTAINER="ckptsel1k_${LANG}_${CKPT}"
RUNNER="$ROOT/.generated/selection_runners/_ckptsel1k_${LANG}_${CKPT}.sh"

[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"

echo "======================================================"
echo " Checkpoint selection 1000-utt"
echo "  lang=$LANG  ckpt=$CKPT  mode=$MODE  gpu=$GPU"
echo "  config=$(basename $CONFIG)"
echo "  result=$RESULT_DIR"
echo "  log=$LOG"
echo "======================================================"

# Write per-run runner script
mkdir -p "$RESULT_DIR"
mkdir -p "$(dirname "$RUNNER")"
if [[ "$MODE" == local ]]; then
    cat > "$RUNNER" << RUNNER_EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p $RESULT_DIR
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT/src/decoding $PYBIN -u \\
  $ROOT/src/scoring/run_exp_mb.py $CONFIG --split search --resume \\
  2>&1 | tee -a $RESULT_DIR/run.log
RUNNER_EOF
    chmod +x "$RUNNER"
    nohup bash "$RUNNER" > "$LOG" 2>&1 &
    echo "Launched locally on GPU $GPU  PID=$!  log=$LOG"
else
    die "remote mode is redacted in the public release; run the local mode or adapt this script for your own host/container layout"
fi

echo ""
echo "Monitor: tail -f $LOG"
