#!/bin/bash
#SBATCH -J dl-normalized-sdf-aio-bp-logit
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_normalized_sdf_aio_bp_logit_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_normalized_sdf_aio_bp_logit_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
STAMP=$(date +%Y%m%d_%H%M%S)
REQUIRED_COMMIT="${REQUIRED_COMMIT:?REQUIRED_COMMIT must be set}"

RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_${STAMP}}"
SEED="${SEED:-12345}"
N_EPISODES="${N_EPISODES:-3}"
EPOCHS="${EPOCHS:-20}"
N_PATHS="${N_PATHS:-400}"
POST0_N_PATHS="${POST0_N_PATHS:-400}"
# Legacy fallback for entrypoints that still consume the global batch size.
BATCH_SIZE="${BATCH_SIZE:-4096}"
# Firm-level Policy/Value batch.
PV_BATCH_SIZE="${PV_BATCH_SIZE:-20480}"
# Macro parent-group FC1/SDF batch.
SDF_FC1_BATCH_SIZE="${SDF_FC1_BATCH_SIZE:-4096}"
SIMULATE_GROUP_SIZE="${SIMULATE_GROUP_SIZE:-200}"
SIMULATE_HORIZON="${SIMULATE_HORIZON:-100}"
POST0_MODE="${POST0_MODE:-modeb}"
ALTERNATE_START="${ALTERNATE_START:-modea}"
ENABLE_FC2="${ENABLE_FC2:-0}"
QUICK_TEST="${QUICK_TEST:-0}"
SDF_WEALTH_LOSS_MODE="${SDF_WEALTH_LOSS_MODE:-signed_aio}"
SDF_FRESH_PAIR_ENABLED="${SDF_FRESH_PAIR_ENABLED:-1}"
SDF_CHILD_BANK_SIZE="${SDF_CHILD_BANK_SIZE:-16}"
SDF_CHILD_BANK_REFRESH_EPOCHS="${SDF_CHILD_BANK_REFRESH_EPOCHS:-1}"
SDF_CHILD_BANK_SEED="${SDF_CHILD_BANK_SEED:-12345}"
EPISODE0_SDF_EPOCHS_PER_ROUND="${EPISODE0_SDF_EPOCHS_PER_ROUND:-20}"
EPISODE0_SDF_MAX_ROUNDS="${EPISODE0_SDF_MAX_ROUNDS:-20}"
EPISODE0_SDF_LOG_MEAN_ERROR_MAX="${EPISODE0_SDF_LOG_MEAN_ERROR_MAX:-0.25}"
EPISODE0_SDF_CLIP_LOW_RATIO_MAX="${EPISODE0_SDF_CLIP_LOW_RATIO_MAX:-0.20}"
EPISODE0_SDF_FINITE_RATIO_MIN="${EPISODE0_SDF_FINITE_RATIO_MIN:-1.0}"
SDF_TRUE_ONLY_EPOCHS="${SDF_TRUE_ONLY_EPOCHS:-100}"
SDF_RECURSIVE_ONLY_EPOCHS="${SDF_RECURSIVE_ONLY_EPOCHS:-0}"
SDF_STAGE2_LR="${SDF_STAGE2_LR:-2e-4}"
SDF_TRUE_ONLY_LR="${SDF_TRUE_ONLY_LR:-4e-5}"
SDF_TRUE_TARGET_BATCHES="${SDF_TRUE_TARGET_BATCHES:-20}"
SDF_MIN_PARENT_GROUPS_PER_BATCH="${SDF_MIN_PARENT_GROUPS_PER_BATCH:-256}"
SDF_TRUE_MOMENT_WEIGHT="${SDF_TRUE_MOMENT_WEIGHT:-5e-4}"
SDF_TRUE_ANCHOR_WEIGHT="${SDF_TRUE_ANCHOR_WEIGHT:-0.05}"
SDF_RESET_OPTIMIZER_ON_TRUE_START="${SDF_RESET_OPTIMIZER_ON_TRUE_START:-0}"
SDF_RESTORE_BEST_CHECKPOINT="${SDF_RESTORE_BEST_CHECKPOINT:-1}"
SDF_COLLAPSE_LOG_MEAN_ERROR="${SDF_COLLAPSE_LOG_MEAN_ERROR:-0.5}"
SDF_COLLAPSE_MEAN_RATIO="${SDF_COLLAPSE_MEAN_RATIO:-0.10}"
FC1_JACOBIAN_PENALTY_INTERVAL="${FC1_JACOBIAN_PENALTY_INTERVAL:-10}"
BP_GRID_POLICY_LOSS_SPACE="${BP_GRID_POLICY_LOSS_SPACE:-logit}"
BP_GRID_LOGIT_TARGET_EPS="${BP_GRID_LOGIT_TARGET_EPS:-1e-4}"
BP_GRID_LOGIT_HUBER_DELTA="${BP_GRID_LOGIT_HUBER_DELTA:-1.0}"
# Full normalized residual run: Treatment B by default.
MODEB_RESIMULATE_AFTER_PV="${MODEB_RESIMULATE_AFTER_PV:-1}"

mkdir -p "$LOG_DIR" "$RUN_ROOT" "$WORK_ROOT/cachedir" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"
export SDF_TRUE_ONLY_LR
export SDF_TRUE_TARGET_BATCHES
export SDF_MIN_PARENT_GROUPS_PER_BATCH

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Required commit: $REQUIRED_COMMIT"
if [[ "$ACTUAL_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "Expected: $REQUIRED_COMMIT"
  echo "Actual:   $ACTUAL_COMMIT"
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean"
  git status --short
  exit 1
fi
echo "Run root: $RUN_ROOT"
echo "Python: $(which python3)"
echo "Torch/CUDA check:"
python3 - <<'PY'
import torch
print("  torch:", torch.__version__)
print("  cuda_available:", torch.cuda.is_available())
print("  cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("  cuda_device_0:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available on this Slurm allocation.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

EXTRA_ARGS=()
if [[ "$ENABLE_FC2" == "1" ]]; then
  EXTRA_ARGS+=(--enable-fc2)
else
  EXTRA_ARGS+=(--no-enable-fc2)
fi
if [[ "$QUICK_TEST" == "1" ]]; then
  EXTRA_ARGS+=(--quick-test)
fi
if [[ "$MODEB_RESIMULATE_AFTER_PV" == "1" ]]; then
  EXTRA_ARGS+=(--modeb-resimulate-after-pv)
else
  EXTRA_ARGS+=(--no-modeb-resimulate-after-pv)
fi
if [[ "$SDF_FRESH_PAIR_ENABLED" == "1" ]]; then
  EXTRA_ARGS+=(--sdf-fresh-pair-enabled)
else
  EXTRA_ARGS+=(--no-sdf-fresh-pair-enabled)
fi
if [[ "$SDF_RESET_OPTIMIZER_ON_TRUE_START" == "1" ]]; then
  EXTRA_ARGS+=(--sdf-reset-optimizer-on-true-start)
else
  EXTRA_ARGS+=(--no-sdf-reset-optimizer-on-true-start)
fi
if [[ "$SDF_RESTORE_BEST_CHECKPOINT" == "1" ]]; then
  EXTRA_ARGS+=(--sdf-restore-best-checkpoint)
else
  EXTRA_ARGS+=(--no-sdf-restore-best-checkpoint)
fi

echo "Run parameters:"
echo "  SEED=$SEED"
echo "  N_EPISODES=$N_EPISODES"
echo "  EPOCHS=$EPOCHS"
echo "  N_PATHS=$N_PATHS"
echo "  POST0_N_PATHS=$POST0_N_PATHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  PV_BATCH_SIZE=$PV_BATCH_SIZE"
echo "  SDF_FC1_BATCH_SIZE=$SDF_FC1_BATCH_SIZE"
echo "  SIMULATE_GROUP_SIZE=$SIMULATE_GROUP_SIZE"
echo "  SIMULATE_HORIZON=$SIMULATE_HORIZON"
echo "  POST0_MODE=$POST0_MODE"
echo "  ALTERNATE_START=$ALTERNATE_START"
echo "  ENABLE_FC2=$ENABLE_FC2"
echo "  QUICK_TEST=$QUICK_TEST"
echo "  SDF_WEALTH_LOSS_MODE=$SDF_WEALTH_LOSS_MODE"
echo "  SDF_FRESH_PAIR_ENABLED=$SDF_FRESH_PAIR_ENABLED"
echo "  SDF_CHILD_BANK_SIZE=$SDF_CHILD_BANK_SIZE"
echo "  SDF_CHILD_BANK_REFRESH_EPOCHS=$SDF_CHILD_BANK_REFRESH_EPOCHS"
echo "  SDF_CHILD_BANK_SEED=$SDF_CHILD_BANK_SEED"
echo "  EPISODE0_SDF_EPOCHS_PER_ROUND=$EPISODE0_SDF_EPOCHS_PER_ROUND"
echo "  EPISODE0_SDF_MAX_ROUNDS=$EPISODE0_SDF_MAX_ROUNDS"
echo "  EPISODE0_SDF_LOG_MEAN_ERROR_MAX=$EPISODE0_SDF_LOG_MEAN_ERROR_MAX"
echo "  EPISODE0_SDF_CLIP_LOW_RATIO_MAX=$EPISODE0_SDF_CLIP_LOW_RATIO_MAX"
echo "  EPISODE0_SDF_FINITE_RATIO_MIN=$EPISODE0_SDF_FINITE_RATIO_MIN"
echo "  SDF_TRUE_ONLY_EPOCHS=$SDF_TRUE_ONLY_EPOCHS"
echo "  SDF_RECURSIVE_ONLY_EPOCHS=$SDF_RECURSIVE_ONLY_EPOCHS"
echo "  SDF_STAGE2_LR=$SDF_STAGE2_LR"
echo "  SDF_TRUE_ONLY_LR=$SDF_TRUE_ONLY_LR"
echo "  SDF_TRUE_TARGET_BATCHES=$SDF_TRUE_TARGET_BATCHES"
echo "  SDF_MIN_PARENT_GROUPS_PER_BATCH=$SDF_MIN_PARENT_GROUPS_PER_BATCH"
echo "  SDF_TRUE_MOMENT_WEIGHT=$SDF_TRUE_MOMENT_WEIGHT"
echo "  SDF_TRUE_ANCHOR_WEIGHT=$SDF_TRUE_ANCHOR_WEIGHT"
echo "  SDF_RESET_OPTIMIZER_ON_TRUE_START=$SDF_RESET_OPTIMIZER_ON_TRUE_START"
echo "  SDF_RESTORE_BEST_CHECKPOINT=$SDF_RESTORE_BEST_CHECKPOINT"
echo "  SDF_COLLAPSE_LOG_MEAN_ERROR=$SDF_COLLAPSE_LOG_MEAN_ERROR"
echo "  SDF_COLLAPSE_MEAN_RATIO=$SDF_COLLAPSE_MEAN_RATIO"
echo "  FC1_JACOBIAN_PENALTY_INTERVAL=$FC1_JACOBIAN_PENALTY_INTERVAL"
echo "  BP_GRID_POLICY_LOSS_SPACE=$BP_GRID_POLICY_LOSS_SPACE"
echo "  BP_GRID_LOGIT_TARGET_EPS=$BP_GRID_LOGIT_TARGET_EPS"
echo "  BP_GRID_LOGIT_HUBER_DELTA=$BP_GRID_LOGIT_HUBER_DELTA"
echo "  MODEB_RESIMULATE_AFTER_PV=$MODEB_RESIMULATE_AFTER_PV"

python3 -u experiments/run_multi_episode_job.py \
  --run-root "$RUN_ROOT" \
  --device cuda:0 \
  --seed "$SEED" \
  --n-episodes "$N_EPISODES" \
  --epochs "$EPOCHS" \
  --n-paths "$N_PATHS" \
  --post0-n-paths "$POST0_N_PATHS" \
  --batch-size "$BATCH_SIZE" \
  --pv-batch-size "$PV_BATCH_SIZE" \
  --sdf-fc1-batch-size "$SDF_FC1_BATCH_SIZE" \
  --simulate-group-size "$SIMULATE_GROUP_SIZE" \
  --simulate-horizon "$SIMULATE_HORIZON" \
  --post0-mode "$POST0_MODE" \
  --alternate-start "$ALTERNATE_START" \
  --sdf-wealth-loss-mode "$SDF_WEALTH_LOSS_MODE" \
  --sdf-child-bank-size "$SDF_CHILD_BANK_SIZE" \
  --sdf-child-bank-refresh-epochs "$SDF_CHILD_BANK_REFRESH_EPOCHS" \
  --sdf-child-bank-seed "$SDF_CHILD_BANK_SEED" \
  --episode0-sdf-epochs-per-round "$EPISODE0_SDF_EPOCHS_PER_ROUND" \
  --episode0-sdf-max-rounds "$EPISODE0_SDF_MAX_ROUNDS" \
  --episode0-sdf-log-mean-error-max "$EPISODE0_SDF_LOG_MEAN_ERROR_MAX" \
  --episode0-sdf-clip-low-ratio-max "$EPISODE0_SDF_CLIP_LOW_RATIO_MAX" \
  --episode0-sdf-finite-ratio-min "$EPISODE0_SDF_FINITE_RATIO_MIN" \
  --sdf-true-only-epochs "$SDF_TRUE_ONLY_EPOCHS" \
  --sdf-recursive-only-epochs "$SDF_RECURSIVE_ONLY_EPOCHS" \
  --sdf-stage2-lr "$SDF_STAGE2_LR" \
  --sdf-true-only-lr "$SDF_TRUE_ONLY_LR" \
  --sdf-true-target-batches "$SDF_TRUE_TARGET_BATCHES" \
  --sdf-min-parent-groups-per-batch "$SDF_MIN_PARENT_GROUPS_PER_BATCH" \
  --sdf-true-moment-weight "$SDF_TRUE_MOMENT_WEIGHT" \
  --sdf-true-anchor-weight "$SDF_TRUE_ANCHOR_WEIGHT" \
  --sdf-collapse-log-mean-error "$SDF_COLLAPSE_LOG_MEAN_ERROR" \
  --sdf-collapse-mean-ratio "$SDF_COLLAPSE_MEAN_RATIO" \
  --fc1-jacobian-penalty-interval "$FC1_JACOBIAN_PENALTY_INTERVAL" \
  --bp-grid-policy-loss-space "$BP_GRID_POLICY_LOSS_SPACE" \
  --bp-grid-logit-target-eps "$BP_GRID_LOGIT_TARGET_EPS" \
  --bp-grid-logit-huber-delta "$BP_GRID_LOGIT_HUBER_DELTA" \
  "${EXTRA_ARGS[@]}"

echo "Job finished on $(date)"
