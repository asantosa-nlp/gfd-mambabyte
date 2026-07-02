#!/usr/bin/env bash
# Rescore Whisper N-best lists with adapted MambaByte.
#
# Usage:
#   bash experiments/rescoring/run_rescore_whisper_nbest.sh --cuda 0 --lang jv
#   bash experiments/rescoring/run_rescore_whisper_nbest.sh --cuda 1 --lang su --nbest-file path/to/su_nbest.jsonl

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

CUDA_VISIBLE_DEVICES_VALUE="0"
LANG_TARGET=""
NBEST_FILE=""
OUT_DIR=""
DEVICE="cuda"
DTYPE="float32"
NUM_PERMUTATIONS="10000"
SEED="42"
PROGRESS_EVERY="1"

usage() {
    cat <<'EOF'
Usage:
  bash experiments/rescoring/run_rescore_whisper_nbest.sh --cuda <id> --lang <jv|su> [options]

Required:
  --cuda ID        CUDA_VISIBLE_DEVICES value
  --lang LANG      Language target: jv or su

Optional:
  --nbest-file FILE  Input Whisper N-best JSONL (default: latest experiments/rescoring/nbest/<lang>_whisper_nbest_*/<lang>_nbest.jsonl)
  --out-dir DIR     Output directory (default: experiments/rescoring/rescored/<lang>_rescored)
  --device DEV      cuda|cpu for LM scoring (default: cuda)
  --dtype DTYPE     float16|bfloat16|float32 (default: float32)
  --seed N          permutation-test seed (default: 42)
  --num-permutations N  permutation count (default: 10000)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            CUDA_VISIBLE_DEVICES_VALUE="${2:-}"
            shift 2
            ;;
        --lang)
            LANG_TARGET="${2:-}"
            shift 2
            ;;
        --nbest-file)
            NBEST_FILE="${2:-}"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="${2:-}"
            shift 2
            ;;
        --device)
            DEVICE="${2:-}"
            shift 2
            ;;
        --dtype)
            DTYPE="${2:-}"
            shift 2
            ;;
        --seed)
            SEED="${2:-}"
            shift 2
            ;;
        --num-permutations)
            NUM_PERMUTATIONS="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[error] Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$LANG_TARGET" ]]; then
    echo "[error] --lang is required" >&2
    usage >&2
    exit 1
fi

case "$LANG_TARGET" in
    jv)
        DEFAULT_NBEST="$ROOT/experiments/rescoring/nbest/jv_t07_topp09_s42/jv_nbest.jsonl"
        if [[ ! -f "$DEFAULT_NBEST" ]]; then
            DEFAULT_NBEST="$(find "$ROOT/experiments/rescoring/nbest" -maxdepth 2 -type f -name 'jv_nbest.jsonl' 2>/dev/null | sort | tail -n 1 || true)"
        fi
        OUT_BASE="$ROOT/experiments/rescoring/rescored/jv_rescored"
        ;;
    su)
        DEFAULT_NBEST="$ROOT/experiments/rescoring/nbest/su_t07_topp09_s42/su_nbest.jsonl"
        if [[ ! -f "$DEFAULT_NBEST" ]]; then
            DEFAULT_NBEST="$(find "$ROOT/experiments/rescoring/nbest" -maxdepth 2 -type f -name 'su_nbest.jsonl' 2>/dev/null | sort | tail -n 1 || true)"
        fi
        OUT_BASE="$ROOT/experiments/rescoring/rescored/su_rescored"
        ;;
    *)
        echo "[error] Unsupported --lang value: $LANG_TARGET (expected jv or su)" >&2
        exit 1
        ;;
esac

if [[ -z "$NBEST_FILE" ]]; then
    NBEST_FILE="$DEFAULT_NBEST"
fi
if [[ "$NBEST_FILE" != /* ]]; then
    NBEST_FILE="$ROOT/$NBEST_FILE"
fi
if [[ -z "$NBEST_FILE" || ! -f "$NBEST_FILE" ]]; then
    echo "[error] Could not find N-best file. Pass --nbest-file explicitly." >&2
    exit 2
fi

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$OUT_BASE"
fi
if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$ROOT/$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

echo "[helper] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE"
echo "[helper] lang=$LANG_TARGET"
echo "[helper] nbest-file=$NBEST_FILE"
echo "[helper] out-dir=$OUT_DIR"
echo "[helper] device=$DEVICE"
echo "[helper] dtype=$DTYPE"
echo "[helper] python=$PYTHON_BIN"
echo "[helper] seed=$SEED"
echo "[helper] permutations=$NUM_PERMUTATIONS"
echo "[helper] progress-every=$PROGRESS_EVERY"

cd "$ROOT"
exec env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    PYTHONPATH="$ROOT/src/decoding:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "$ROOT/experiments/rescoring/rescore_whisper_nbest.py" \
    --lang "$LANG_TARGET" \
    --nbest-file "$NBEST_FILE" \
    --out-dir "$OUT_DIR" \
    --summary-dir "$ROOT/experiments/rescoring/summaries" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --num-permutations "$NUM_PERMUTATIONS" \
    --progress-every "$PROGRESS_EVERY"
