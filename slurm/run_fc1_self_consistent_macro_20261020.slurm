#!/bin/bash
#SBATCH -J FC1_SELF_CONSISTENT_8ITER
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
set -euo pipefail
source /apps/soft/anaconda3/bin/activate
conda activate DL_HL
cd "${REPO_DIR:?}"
test "$(git rev-parse HEAD)" = "${REQUIRED_COMMIT:?}"
test -z "$(git status --porcelain)"
mkdir -p "${RUN_ROOT:?}/logs" "$RUN_ROOT/mplcache"
export MPLCONFIGDIR="$RUN_ROOT/mplcache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 30 >> "$RUN_ROOT/logs/gpu_telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
python -u analysis/run_fc1_self_consistent_macro.py \
  --checkpoint "${INITIAL_CHECKPOINT:?}" --output "$RUN_ROOT" \
  --train-parents 4096 --validation-parents 1024 --firms 200 \
  --iterations 8 --epochs 15 --parent-batch-size 128 --fit-parent-batch-size 128 \
  --dynamic-parents 128 --seed 20261020
