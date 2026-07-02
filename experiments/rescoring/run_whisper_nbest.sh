#!/usr/bin/env bash
# Generate N-best Whisper hypotheses for the JV/SU 460-utterance test set.
#
# Usage:
#   bash experiments/rescoring/run_whisper_nbest.sh --cuda 0 --lang jv
#   bash experiments/rescoring/run_whisper_nbest.sh --cuda 1 --lang su --out-dir experiments/rescoring/nbest/su_run1
#
# The script is a thin wrapper around experiments/rescoring/generate_whisper_nbest.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

CUDA_VISIBLE_DEVICES_VALUE="0"
LANG_TARGET=""
OUTPUT_DIR=""
MODEL="openai/whisper-large-v3"
PROCESSOR_MODEL=""
DTYPE="auto"
NUM_HYPOTHESES="10"
NUM_BEAMS="1"
DO_SAMPLE="true"
TEMPERATURE="0.7"
TOP_P="0.9"
SEED="42"
MAX_UTTERANCES=""
INITIAL_PROMPT=""
MANIFEST=""

usage() {
    cat <<'EOF'
Usage:
  bash experiments/rescoring/run_whisper_nbest.sh --cuda <id> --lang <jv|su> [options]

Required:
  --cuda ID        CUDA_VISIBLE_DEVICES value to use
  --lang LANG      Language target: jv (Javanese) or su (Sundanese)

Optional:
  --out-dir DIR        Output directory (default: experiments/rescoring/nbest/<lang>_<timestamp>)
  --model MODEL        Whisper model id (default: openai/whisper-large-v3)
  --processor-model ID Processor/tokenizer source if different from --model
  --dtype DTYPE        auto|bfloat16|float16|float32 (default: auto)
  --num-hypotheses N   Number of candidates to return (default: 10)
  --num-beams N        Beam size used to generate N-best (default: 1)
  --do-sample BOOL     Use sampling instead of beam search (default: true)
  --temperature T      Sampling temperature (default: 0.7)
  --top-p P            Nucleus sampling top-p (default: 0.9)
  --seed N             RNG seed for Whisper sampling (default: 42)
  --max-utterances N   Optional cap for debugging / smoke tests
  --initial-prompt T   Optional Whisper initial prompt
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
        --out-dir)
            OUTPUT_DIR="${2:-}"
            shift 2
            ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --processor-model)
            PROCESSOR_MODEL="${2:-}"
            shift 2
            ;;
        --dtype)
            DTYPE="${2:-}"
            shift 2
            ;;
        --num-hypotheses)
            NUM_HYPOTHESES="${2:-}"
            shift 2
            ;;
        --num-beams)
            NUM_BEAMS="${2:-}"
            shift 2
            ;;
        --do-sample)
            DO_SAMPLE="${2:-}"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="${2:-}"
            shift 2
            ;;
        --top-p)
            TOP_P="${2:-}"
            shift 2
            ;;
        --seed)
            SEED="${2:-}"
            shift 2
            ;;
        --max-utterances)
            MAX_UTTERANCES="${2:-}"
            shift 2
            ;;
        --initial-prompt)
            INITIAL_PROMPT="${2:-}"
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
    jv|jw|javanese)
        MANIFEST="$ROOT/data/speech_corpus/local_jv/test.jsonl"
        LANG_SHORT="jv"
        WHISPER_LANG="jw"
        ;;
    su|sundanese)
        MANIFEST="$ROOT/data/speech_corpus/local_su/test.jsonl"
        LANG_SHORT="su"
        WHISPER_LANG="su"
        ;;
    *)
        echo "[error] Unsupported --lang value: $LANG_TARGET (expected jv or su)" >&2
        exit 1
        ;;
esac

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$ROOT/experiments/rescoring/nbest/${LANG_SHORT}_t07_topp09_s42"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

echo "[helper] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE"
echo "[helper] lang=$LANG_TARGET -> whisper-lang=$WHISPER_LANG"
echo "[helper] manifest=$MANIFEST"
echo "[helper] output-dir=$OUTPUT_DIR"
echo "[helper] resume-dir=$(if [[ -f "$OUTPUT_DIR/${LANG_SHORT}_nbest.state.json" ]]; then echo yes; else echo no; fi)"
echo "[helper] python=$PYTHON_BIN"
echo "[helper] model=$MODEL"
echo "[helper] processor-model=${PROCESSOR_MODEL:-$MODEL}"
echo "[helper] dtype=$DTYPE"
echo "[helper] num-hypotheses=$NUM_HYPOTHESES"
echo "[helper] num-beams=$NUM_BEAMS"
echo "[helper] do-sample=$DO_SAMPLE"
echo "[helper] temperature=$TEMPERATURE"
echo "[helper] top-p=$TOP_P"
echo "[helper] seed=$SEED"

cmd=(
    "$PYTHON_BIN" "$ROOT/experiments/rescoring/generate_whisper_nbest.py"
    --model "$MODEL"
    --lang "$LANG_TARGET"
    --manifest "$MANIFEST"
    --out-dir "$OUTPUT_DIR"
    --dtype "$DTYPE"
    --num-hypotheses "$NUM_HYPOTHESES"
    --num-beams "$NUM_BEAMS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --seed "$SEED"
)

if [[ "$DO_SAMPLE" == "true" || "$DO_SAMPLE" == "1" || "$DO_SAMPLE" == "yes" ]]; then
    cmd+=(--do-sample)
else
    cmd+=(--no-do-sample)
fi

if [[ -n "$PROCESSOR_MODEL" ]]; then
    cmd+=(--processor-model "$PROCESSOR_MODEL")
fi
if [[ -n "$MAX_UTTERANCES" ]]; then
    cmd+=(--max-utterances "$MAX_UTTERANCES")
fi
if [[ -n "$INITIAL_PROMPT" ]]; then
    cmd+=(--initial-prompt "$INITIAL_PROMPT")
fi

cd "$ROOT"
exec env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    PYTHONPATH="$ROOT/src/decoding:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    "${cmd[@]}"
