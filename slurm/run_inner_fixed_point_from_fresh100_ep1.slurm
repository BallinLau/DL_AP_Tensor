#!/bin/bash
#SBATCH -J inner-fp-ep1
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/inner_fp_ep1_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/inner_fp_ep1_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
SOURCE_RUN="${SOURCE_RUN:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100_20261014_144528}"
CHECKPOINT="${CHECKPOINT:-$SOURCE_RUN/checkpoints/ep1_combined.pt}"
INNER_FP_SMOKE="${INNER_FP_SMOKE:-0}"
INNER_FP_MAX_SWEEPS="${INNER_FP_MAX_SWEEPS:-}"
INNER_FP_MODE="${INNER_FP_MODE:-coupled}"
RESUME="${RESUME:-}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-}"
SEED="${SEED:-20261014}"
STAMP="$(date +%Y%m%d_%H%M%S)"

case "$INNER_FP_MODE" in
  coupled|p_only|q_only) ;;
  *) echo "ERROR: INNER_FP_MODE must be coupled, p_only, or q_only"; exit 1 ;;
esac

if [[ "$INNER_FP_SMOKE" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/cachedir/inner_fp_block_${INNER_FP_MODE}_smoke_ep1_${STAMP}}"
  MODE_ARGS=(
    --fp-train-parents 256
    --fp-validation-parents 128
    --fp-child-shocks 4
    --fp-batch-size 64
    --fp-max-sweeps 2
    --fp-min-sweeps 2
    --fp-consecutive-passes 1
    --fp-p-epochs 1
    --fp-q-zero-epochs 1
    --fp-q-default-epochs 1
    --fp-q-survival-epochs 1
    --fp-q-polish-epochs 1
  )
else
  OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/cachedir/inner_fp_block_${INNER_FP_MODE}_ep1_10s_${STAMP}}"
  MODE_ARGS=(
    --fp-train-parents 8192
    --fp-validation-parents 2048
    --fp-child-shocks 16
    --fp-batch-size 256
    --fp-max-sweeps "${INNER_FP_MAX_SWEEPS:-30}"
    --fp-min-sweeps 3
    --fp-consecutive-passes 3
  )
fi

cd "$REPO_DIR"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "============================================================"
echo "Frozen-Macro-Law firm-side inner fixed point"
echo "Started: $(date)"
echo "Repo/branch/commit: $REPO_DIR / $(git branch --show-current) / $ACTUAL_COMMIT"
echo "Source: $SOURCE_RUN"
echo "Checkpoint: $CHECKPOINT"
echo "Smoke: $INNER_FP_SMOKE"
echo "Block mode: $INNER_FP_MODE"
echo "Output: $OUTPUT_DIR"
git status --short
echo "============================================================"

if [[ -n "$REQUIRED_COMMIT" && "$ACTUAL_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "ERROR: REQUIRED_COMMIT mismatch: expected $REQUIRED_COMMIT, got $ACTUAL_COMMIT"
  exit 1
fi
for required in \
  "$CHECKPOINT" \
  "$SOURCE_RUN/metadata/hyperparams.json" \
  "$SOURCE_RUN/metadata/config_snapshot.json" \
  "$SOURCE_RUN/metadata/policy_value_model_spec.json" \
  "$SOURCE_RUN/data/outputs/ep1_stage_modeb.pkl" \
  "$SOURCE_RUN/data/outputs/ep1_stage_modeb_macro.pkl"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing required source artifact: $required"
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" && -z "$RESUME" ]]; then
  echo "ERROR: refusing to overwrite existing OUTPUT_DIR: $OUTPUT_DIR"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORK_ROOT/.mplcache}"
mkdir -p "$MPLCONFIGDIR"

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this allocation")
print("device:", torch.cuda.get_device_name(0))
PY

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

python3 -u experiments/run_inner_fixed_point.py \
  --source-run "$SOURCE_RUN" \
  --checkpoint "$CHECKPOINT" \
  --run-root "$OUTPUT_DIR" \
  --device cuda:0 \
  --seed "$SEED" \
  --fp-mode "$INNER_FP_MODE" \
  --fp-damping 1.0 \
  "${RESUME_ARGS[@]}" \
  "${MODE_ARGS[@]}"

echo "Finished: $(date)"
echo "INNER_FP_RUN_ROOT=$OUTPUT_DIR"
