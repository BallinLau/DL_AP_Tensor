#!/bin/bash
#SBATCH -J dl-tensor-fc1-probe-80g
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_fc1_probe_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_tensor_fc1_probe_error.%j
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

FIRM_PKL="${FIRM_PKL:-/home/fit/zhuyingz/WORK/LiuHao/cachedir/fc1_forecast_only_80g_20260402_235621/data/outputs/ep30_stage_modea.pkl}"
MACRO_PKL="${MACRO_PKL:-/home/fit/zhuyingz/WORK/LiuHao/cachedir/fc1_forecast_only_80g_20260402_235621/data/outputs/ep30_stage_modea_macro.pkl}"
OUT_DIR="${OUT_DIR:-$(dirname "$(dirname "$MACRO_PKL")")/experiments/fc1_state_probe}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Firm PKL: $FIRM_PKL"
echo "Macro PKL: $MACRO_PKL"
echo "Output dir: $OUT_DIR"
echo "Python: $(which python3)"
echo "Torch check: $(python3 -c 'import torch; print(torch.__version__)')"

python3 -u experiments/run_fc1_state_probe.py \
  --firm-pkl "$FIRM_PKL" \
  --macro-pkl "$MACRO_PKL" \
  --output-dir "$OUT_DIR" \
  --device cuda:0 \
  --epochs 300 \
  --batch-size 2048

echo "Job finished on $(date)"
