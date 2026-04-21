#!/bin/bash
#SBATCH -J dl-tensor-policy-datagen-compare-80g
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_policy_datagen_compare_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_policy_datagen_compare_error.%j
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

CKPT_DIR="${CKPT_DIR:-$WORK_ROOT/cachedir/fc2_supervised_pretrain_only_80g_20260419_000000/checkpoints}"
CKPT_PREFIX="${CKPT_PREFIX:-ep30}"
OUT_DIR="${OUT_DIR:-$WORK_ROOT/cachedir/policy_datagen_compare_${STAMP}}"
POOL_SIZE="${POOL_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-64}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-6}"
EVAL_POOL_SIZE="${EVAL_POOL_SIZE:-$POOL_SIZE}"
EVAL_ROLLOUT_STEPS="${EVAL_ROLLOUT_STEPS:-$ROLLOUT_STEPS}"
BRANCH_NUM="${BRANCH_NUM:-2}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
N_EPOCHS="${N_EPOCHS:-220}"
Q_STAGE_EPOCHS="${Q_STAGE_EPOCHS:-100}"
PVBP_STAGE_EPOCHS="${PVBP_STAGE_EPOCHS:-100}"
Q_REFRESH_STAGE_EPOCHS="${Q_REFRESH_STAGE_EPOCHS:-20}"
BEFORE_EVAL_TOL="${BEFORE_EVAL_TOL:-1e-8}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SEED="${SEED:-42}"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$WORK_ROOT/cachedir" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Checkpoint dir: $CKPT_DIR"
echo "Checkpoint prefix: $CKPT_PREFIX"
echo "Output dir: $OUT_DIR"
echo "Python: $(which python3)"
echo "Torch check: $(python3 -c 'import torch; print(torch.__version__)')"

python3 -u experiments/run_policy_datagen_compare.py \
  --ckpt-dir "$CKPT_DIR" \
  --ckpt-prefix "$CKPT_PREFIX" \
  --output-dir "$OUT_DIR" \
  --device cuda:0 \
  --pool-size "$POOL_SIZE" \
  --group-size "$GROUP_SIZE" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --eval-pool-size "$EVAL_POOL_SIZE" \
  --eval-rollout-steps "$EVAL_ROLLOUT_STEPS" \
  --branch-num "$BRANCH_NUM" \
  --batch-size "$BATCH_SIZE" \
  --n-epochs "$N_EPOCHS" \
  --q-stage-epochs "$Q_STAGE_EPOCHS" \
  --pvbp-stage-epochs "$PVBP_STAGE_EPOCHS" \
  --q-refresh-stage-epochs "$Q_REFRESH_STAGE_EPOCHS" \
  --before-eval-tol "$BEFORE_EVAL_TOL" \
  --log-interval "$LOG_INTERVAL" \
  --seed "$SEED"

echo "Job finished on $(date)"
