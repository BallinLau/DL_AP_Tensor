#!/bin/bash
#SBATCH -J dl-tensor-fc2-pretrain-80g
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_fc2_pretrain_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_fc2_pretrain_error.%j
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
RUN_ROOT="${RUN_ROOT:-/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/fc2_supervised_pretrain_only_80g_${STAMP}}"

N_EPISODES="${N_EPISODES:-31}"
N_PATHS="${N_PATHS:-15000}"
SAMPLE_GROUP_SIZE="${SAMPLE_GROUP_SIZE:-2}"
SIM_GROUP_SIZE="${SIM_GROUP_SIZE:-200}"
SIM_HORIZON="${SIM_HORIZON:-10}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-40}"
POST0_MODE="${POST0_MODE:-modea}"
FC2_MAIN_MACRO_STATE="${FC2_MAIN_MACRO_STATE:---fc2-as-main-macro-state}"

mkdir -p "$LOG_DIR" "$WORK_ROOT/cachedir"
cd "$REPO_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Run root: $RUN_ROOT"
echo "Pretrain epochs: $PRETRAIN_EPOCHS"
echo "Python: $(which python3)"
echo "Torch check: $(python3 -c 'import torch; print(torch.__version__)')"

python3 -u experiments/run_multi_episode_job.py \
  --run-root "$RUN_ROOT" \
  --device cuda:0 \
  --n-episodes "$N_EPISODES" \
  --post0-mode "$POST0_MODE" \
  --n-paths "$N_PATHS" \
  --sample-group-size "$SAMPLE_GROUP_SIZE" \
  --simulate-group-size "$SIM_GROUP_SIZE" \
  --simulate-horizon "$SIM_HORIZON" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$PRETRAIN_EPOCHS" \
  --enable-fc2 \
  --fc2-supervised-pretrain-only \
  $FC2_MAIN_MACRO_STATE \
  --no-final-sim

echo "Job finished on $(date)"
