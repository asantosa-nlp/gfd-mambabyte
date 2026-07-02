#!/usr/bin/env bash
# Launch full 1K-utterance grid search (curiosity run) across remote GPUs.
# Distributes 18 jobs (9 JV + 9 SU) across 5 remote GPUs sequentially per GPU.
#
# Usage:
#   bash src/search/launch_gs20k_1k_remote.sh
#
# Remote GPUs used: 2, 3, 5, 6, 7  (NEVER 0; 1 and 4 avoided by default)
# Results land in: results/gs20k_1k_{jv,su}_lm{60,70,80}_ag{15,20,25}/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-<REMOTE_USER@REMOTE_HOST>}"
IMAGE="${IMAGE:-nvidia-cuda-ffmpeg:12.8.1-python3.12}"
PYBIN="${PYBIN:-$ROOT/.venv/bin/python}"
RUNNER_DIR="$ROOT/.generated/gs1k_runners"

mkdir -p "$RUNNER_DIR"

run_remote() {
    local gpu="$1"
    local container="$2"
    local runner="$RUNNER_DIR/${container}.sh"
    shift 2
    local configs=("$@")

    # Build sequential runner script
    cat > "$runner" << RUNNER_EOF
#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=$ROOT/src/decoding
export PYTHONPATH
CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES
RUNNER_EOF

    for cfg in "${configs[@]}"; do
        local lang lm ag
        # derive result dir from config name
        local cfg_base
        cfg_base=$(basename "$cfg" .yaml)
        local result_dir="$ROOT/results"
        # extract result dir from yaml
        local rdir
        rdir=$(python3 -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c['output']['search_dir'])")
        cat >> "$runner" << STEP_EOF

echo "=== Starting $cfg_base ==="
mkdir -p $ROOT/$rdir
$PYBIN -u \\
  $ROOT/src/scoring/run_exp_mb.py $cfg --split search --resume \\
  2>&1 | tee -a $ROOT/${rdir}/run.log
echo "=== Done $cfg_base ==="
STEP_EOF
    done

    chmod +x "$runner"
    local log="$ROOT/results/_gs1k_gpu${gpu}.log"

    echo "Remote dispatch redacted in the public release."
    echo "Generated runner: $runner"
    echo "To run it, adapt the container / SSH settings for your own host."
    return 0
}

C=$ROOT/configs

# GPU 2 — JV lm60 (×3) + SU lm60_ag15
run_remote 2 gs1k_gpu2 \
  "$C/mb_jv_zs_large_v5cleanS20k_gs_lm60_ag15_1k.yaml" \
  "$C/mb_jv_zs_large_v5cleanS20k_gs_lm60_ag20_1k.yaml" \
  "$C/mb_jv_zs_large_v5cleanS20k_gs_lm60_ag25_1k.yaml" \
  "$C/mb_su_zs_large_v5cleanS20k_gs_lm60_ag15_1k_minb15.yaml"

# GPU 3 — JV lm70 (×3) + SU lm60_ag20
run_remote 3 gs1k_gpu3 \
  "$C/mb_jv_zs_large_v5cleanS20k_gs_lm70_ag15_1k.yaml" \
  "$C/mb_jv_zs_large_v5cleanS20k_gs_lm70_ag20_1k.yaml" \
  "$C/mb_jv_zs_large_v5cleanS20k_gs_lm70_ag25_1k.yaml" \
  "$C/mb_su_zs_large_v5cleanS20k_gs_lm60_ag20_1k_minb15.yaml"

# GPU 5 — JV lm80 (×3) + SU lm60_ag25
run_remote 5 gs1k_gpu5 \
  "$C/mb_jv_zs_large_v5cleanS20k_gs_lm80_ag15_1k.yaml" \
  "$C/mb_jv_zs_large_v5cleanS20k_gs_lm80_ag20_1k.yaml" \
  "$C/mb_jv_zs_large_v5cleanS20k_gs_lm80_ag25_1k.yaml" \
  "$C/mb_su_zs_large_v5cleanS20k_gs_lm60_ag25_1k_minb15.yaml"

# GPU 6 — SU lm70 (×3)
run_remote 6 gs1k_gpu6 \
  "$C/mb_su_zs_large_v5cleanS20k_gs_lm70_ag15_1k_minb15.yaml" \
  "$C/mb_su_zs_large_v5cleanS20k_gs_lm70_ag20_1k_minb15.yaml" \
  "$C/mb_su_zs_large_v5cleanS20k_gs_lm70_ag25_1k_minb15.yaml"

# GPU 7 — SU lm80 (×3)
run_remote 7 gs1k_gpu7 \
  "$C/mb_su_zs_large_v5cleanS20k_gs_lm80_ag15_1k_minb15.yaml" \
  "$C/mb_su_zs_large_v5cleanS20k_gs_lm80_ag20_1k_minb15.yaml" \
  "$C/mb_su_zs_large_v5cleanS20k_gs_lm80_ag25_1k_minb15.yaml"

echo ""
echo "All 18 runner scripts generated. Remote dispatch is redacted in the public release."
echo "Adapt the runner scripts to your own host/container layout before using them."
