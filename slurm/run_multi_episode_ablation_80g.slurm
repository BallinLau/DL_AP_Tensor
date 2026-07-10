#!/bin/bash
#SBATCH -J dl-tensor-ablation-80g
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_ablation_%A_%a.out
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_ablation_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
STAMP=$(date +%Y%m%d_%H%M%S)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ABLATION_MODES=(baseline bellman_only fixed_sdf fixed_policy)
ABLATION_MODE="${ABLATION_MODES[$TASK_ID]}"

RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/ablation_${ABLATION_MODE}_${STAMP}_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}}"
N_EPISODES="${N_EPISODES:-3}"
EPOCHS="${EPOCHS:-20}"
N_PATHS="${N_PATHS:-200}"
POST0_N_PATHS="${POST0_N_PATHS:-$N_PATHS}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
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
EPISODE0_SDF_EPOCHS_PER_ROUND="${EPISODE0_SDF_EPOCHS_PER_ROUND:-$EPOCHS}"
EPISODE0_SDF_MAX_ROUNDS="${EPISODE0_SDF_MAX_ROUNDS:-10}"
EPISODE0_SDF_LOG_MEAN_ERROR_MAX="${EPISODE0_SDF_LOG_MEAN_ERROR_MAX:-0.25}"
EPISODE0_SDF_CLIP_LOW_RATIO_MAX="${EPISODE0_SDF_CLIP_LOW_RATIO_MAX:-0.20}"
EPISODE0_SDF_FINITE_RATIO_MIN="${EPISODE0_SDF_FINITE_RATIO_MIN:-1.0}"
FC1_JACOBIAN_PENALTY_INTERVAL="${FC1_JACOBIAN_PENALTY_INTERVAL:-10}"
# Default to Treatment B for formal Mode B runs; set MODEB_RESIMULATE_AFTER_PV=0 for Treatment A.
MODEB_RESIMULATE_AFTER_PV="${MODEB_RESIMULATE_AFTER_PV:-1}"
MAX_FIRM_TRAIN_UNITS="${MAX_FIRM_TRAIN_UNITS:-1000000}"
POLICY_GRAD_THRESHOLD="${POLICY_GRAD_THRESHOLD:-1000}"
POLICY_ROLLING_GRAD_THRESHOLD="${POLICY_ROLLING_GRAD_THRESHOLD:-100}"
POLICY_LOSS_THRESHOLD="${POLICY_LOSS_THRESHOLD:-1000}"
PV_SDF_CLIP_RATIO_GATE="${PV_SDF_CLIP_RATIO_GATE:-1.0}"
PV_FIXED_SDF_VALUE="${PV_FIXED_SDF_VALUE:-0.98}"
PV_FIXED_POLICY_MODE="${PV_FIXED_POLICY_MODE:-parent_b}"

mkdir -p "$LOG_DIR" "$RUN_ROOT" "$WORK_ROOT/cachedir" "$WORK_ROOT/.mplcache"
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
echo "Run root: $RUN_ROOT"
echo "Array task: $TASK_ID"
echo "Ablation mode: $ABLATION_MODE"
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

echo "Run parameters:"
echo "  ABLATION_MODE=$ABLATION_MODE"
echo "  N_EPISODES=$N_EPISODES"
echo "  EPOCHS=$EPOCHS"
echo "  N_PATHS=$N_PATHS"
echo "  POST0_N_PATHS=$POST0_N_PATHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
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
echo "  FC1_JACOBIAN_PENALTY_INTERVAL=$FC1_JACOBIAN_PENALTY_INTERVAL"
echo "  MODEB_RESIMULATE_AFTER_PV=$MODEB_RESIMULATE_AFTER_PV"
echo "  MAX_FIRM_TRAIN_UNITS=$MAX_FIRM_TRAIN_UNITS"
echo "  POLICY_GRAD_THRESHOLD=$POLICY_GRAD_THRESHOLD"
echo "  POLICY_ROLLING_GRAD_THRESHOLD=$POLICY_ROLLING_GRAD_THRESHOLD"
echo "  POLICY_LOSS_THRESHOLD=$POLICY_LOSS_THRESHOLD"
echo "  PV_SDF_CLIP_RATIO_GATE=$PV_SDF_CLIP_RATIO_GATE"
echo "  PV_FIXED_SDF_VALUE=$PV_FIXED_SDF_VALUE"
echo "  PV_FIXED_POLICY_MODE=$PV_FIXED_POLICY_MODE"

python3 -u experiments/run_multi_episode_job.py \
  --run-root "$RUN_ROOT" \
  --device cuda:0 \
  --n-episodes "$N_EPISODES" \
  --epochs "$EPOCHS" \
  --n-paths "$N_PATHS" \
  --post0-n-paths "$POST0_N_PATHS" \
  --batch-size "$BATCH_SIZE" \
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
  --fc1-jacobian-penalty-interval "$FC1_JACOBIAN_PENALTY_INTERVAL" \
  --ablation-mode "$ABLATION_MODE" \
  --max-firm-train-units "$MAX_FIRM_TRAIN_UNITS" \
  --policy-grad-threshold "$POLICY_GRAD_THRESHOLD" \
  --policy-rolling-grad-threshold "$POLICY_ROLLING_GRAD_THRESHOLD" \
  --policy-loss-threshold "$POLICY_LOSS_THRESHOLD" \
  --pv-sdf-clip-ratio-gate "$PV_SDF_CLIP_RATIO_GATE" \
  --pv-fixed-sdf-value "$PV_FIXED_SDF_VALUE" \
  --pv-fixed-policy-mode "$PV_FIXED_POLICY_MODE" \
  "${EXTRA_ARGS[@]}"

echo "Job finished on $(date)"
