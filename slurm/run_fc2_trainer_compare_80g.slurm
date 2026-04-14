#!/bin/bash
#SBATCH -J dl-tensor-fc2-compare-80g
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_fc2_compare_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_fc2_compare_error.%j
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

CKPT_DIR="${CKPT_DIR:-$WORK_ROOT/cachedir/fc2_supervised_pretrain_only_80g_20260413_235034/checkpoints}"
CKPT_PREFIX="${CKPT_PREFIX:-ep30}"
OUT_DIR="${OUT_DIR:-$WORK_ROOT/cachedir/fc2_trainer_compare_${STAMP}}"
N_PATHS="${N_PATHS:-2048}"
GROUP_SIZE="${GROUP_SIZE:-200}"
HORIZON="${HORIZON:-4}"
PROBE_EPOCHS="${PROBE_EPOCHS:-200}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-4096}"
PROBE_LR="${PROBE_LR:-1e-3}"
PROBE_PATIENCE="${PROBE_PATIENCE:-30}"
EPISODE_EPOCHS="${EPISODE_EPOCHS:-40}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-8192}"
EPISODE_LR="${EPISODE_LR:-1e-4}"
EPISODE_WEIGHT_DECAY="${EPISODE_WEIGHT_DECAY:-1e-4}"

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

python3 -u experiments/run_fc2_trainer_compare.py \
  --ckpt-dir "$CKPT_DIR" \
  --ckpt-prefix "$CKPT_PREFIX" \
  --output-dir "$OUT_DIR" \
  --device cuda:0 \
  --n-paths "$N_PATHS" \
  --group-size "$GROUP_SIZE" \
  --horizon "$HORIZON" \
  --probe-epochs "$PROBE_EPOCHS" \
  --probe-batch-size "$PROBE_BATCH_SIZE" \
  --probe-lr "$PROBE_LR" \
  --probe-patience "$PROBE_PATIENCE" \
  --episode-epochs "$EPISODE_EPOCHS" \
  --episode-batch-size "$EPISODE_BATCH_SIZE" \
  --episode-lr "$EPISODE_LR" \
  --episode-weight-decay "$EPISODE_WEIGHT_DECAY"

echo "Job finished on $(date)"
