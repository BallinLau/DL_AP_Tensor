#!/bin/bash
#SBATCH -J dl-tensor-fc1-summary-80g
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_fc1_summary_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_fc1_summary_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT=/home/fit/zhuyingz/WORK/LiuHao
REPO_DIR="$WORK_ROOT/DL_AP_Tensor"
LOG_DIR="$WORK_ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/fc1_cross_section_summary_80g_${STAMP}"

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
echo "Python: $(which python3)"
echo "Torch check: $(python3 -c 'import torch; print(torch.__version__)')"

python3 -u experiments/run_multi_episode_job.py \
  --run-root "$RUN_ROOT" \
  --device cuda:0 \
  --n-episodes 31 \
  --post0-mode modea \
  --n-paths 15000 \
  --sample-group-size 2 \
  --simulate-group-size 200 \
  --simulate-horizon 10 \
  --batch-size 8192 \
  --no-final-sim

echo "Job finished on $(date)"
