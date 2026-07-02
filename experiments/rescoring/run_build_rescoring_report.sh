#!/usr/bin/env bash
# Build the rescoring comparison report after JV and SU rescoring finish.
#
# Usage:
#   bash experiments/rescoring/run_build_rescoring_report.sh
#
# Optional overrides:
#   JV_SUMMARY=/path/to/jv_summary.json SU_SUMMARY=/path/to/su_summary.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

JV_SUMMARY="${JV_SUMMARY:-$ROOT/experiments/rescoring/summaries/jv_summary.json}"
SU_SUMMARY="${SU_SUMMARY:-$ROOT/experiments/rescoring/summaries/su_summary.json}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/rescoring}"

echo "[helper] jv-summary=$JV_SUMMARY"
echo "[helper] su-summary=$SU_SUMMARY"
echo "[helper] out-dir=$OUT_DIR"
echo "[helper] python=$PYTHON_BIN"

if [[ ! -f "$JV_SUMMARY" ]]; then
    echo "[error] Missing JV summary: $JV_SUMMARY" >&2
    exit 2
fi
if [[ ! -f "$SU_SUMMARY" ]]; then
    echo "[error] Missing SU summary: $SU_SUMMARY" >&2
    exit 2
fi

cd "$ROOT"
exec "$PYTHON_BIN" "$ROOT/experiments/rescoring/build_rescoring_report.py" \
    --jv-summary "$JV_SUMMARY" \
    --su-summary "$SU_SUMMARY" \
    --out-dir "$OUT_DIR"
