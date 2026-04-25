#!/bin/bash
#SBATCH -J dl-tensor-policy-align-80g
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_policy_alignment_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_policy_alignment_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT=/home/fit/zhuyingz/WORK/LiuHao
REPO_DIR="$WORK_ROOT/DL_AP_Tensor"
LOG_DIR="$WORK_ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)

CKPT_DIR="${CKPT_DIR:-}"
CKPT_PREFIX="${CKPT_PREFIX:-ep30}"
ALLOW_RANDOM_POLICY_INIT="${ALLOW_RANDOM_POLICY_INIT:-0}"
OUT_DIR="${OUT_DIR:-$WORK_ROOT/cachedir/policy_loss_residual_alignment_${STAMP}}"
DATAGEN_MODE="${DATAGEN_MODE:-main_branch}"
POOL_SIZE="${POOL_SIZE:-256}"
EVAL_POOL_SIZE="${EVAL_POOL_SIZE:-$POOL_SIZE}"
GROUP_SIZE="${GROUP_SIZE:-64}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-6}"
EVAL_ROLLOUT_STEPS="${EVAL_ROLLOUT_STEPS:-$ROLLOUT_STEPS}"
BRANCH_NUM="${BRANCH_NUM:-2}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
METRIC_BATCH_CAP="${METRIC_BATCH_CAP:-4}"
Q_STAGE_EPOCHS="${Q_STAGE_EPOCHS:-40}"
PVBP_STAGE_EPOCHS="${PVBP_STAGE_EPOCHS:-60}"
Q_REFRESH_STAGE_EPOCHS="${Q_REFRESH_STAGE_EPOCHS:-10}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SEED="${SEED:-42}"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$WORK_ROOT/cachedir" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

EXTRA_ARGS=()
if [[ "$ALLOW_RANDOM_POLICY_INIT" == "1" ]]; then
  EXTRA_ARGS+=(--allow-random-policy-init)
fi
if [[ -n "$CKPT_DIR" ]]; then
  EXTRA_ARGS+=(--ckpt-dir "$CKPT_DIR")
fi

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Checkpoint dir: ${CKPT_DIR:-<required unless ALLOW_RANDOM_POLICY_INIT=1>}"
echo "Checkpoint prefix: $CKPT_PREFIX"
echo "Allow random policy init: $ALLOW_RANDOM_POLICY_INIT"
echo "Output dir: $OUT_DIR"
echo "Python: $(which python3)"
echo "Torch check: $(python3 -c 'import torch; print(torch.__version__)')"

python3 -u experiments/run_policy_loss_residual_alignment_audit.py \
  "${EXTRA_ARGS[@]}" \
  --ckpt-prefix "$CKPT_PREFIX" \
  --output-dir "$OUT_DIR" \
  --device cuda:0 \
  --datagen-mode "$DATAGEN_MODE" \
  --pool-size "$POOL_SIZE" \
  --eval-pool-size "$EVAL_POOL_SIZE" \
  --group-size "$GROUP_SIZE" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --eval-rollout-steps "$EVAL_ROLLOUT_STEPS" \
  --branch-num "$BRANCH_NUM" \
  --batch-size "$BATCH_SIZE" \
  --metric-batch-cap "$METRIC_BATCH_CAP" \
  --q-stage-epochs "$Q_STAGE_EPOCHS" \
  --pvbp-stage-epochs "$PVBP_STAGE_EPOCHS" \
  --q-refresh-stage-epochs "$Q_REFRESH_STAGE_EPOCHS" \
  --eval-every "$EVAL_EVERY" \
  --seed "$SEED"

echo "Job finished on $(date)"
