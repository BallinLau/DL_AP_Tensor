#!/bin/bash
#SBATCH -J bp-head-grid-benchmark
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/bp_head_grid_benchmark_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/bp_head_grid_benchmark_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR must be provided}"
EPISODE="${EPISODE:-2}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/cachedir/bp_head_vs_grid_reference_benchmark_ep${EPISODE}_${STAMP}}"

cd "$REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Episode: $EPISODE"
echo "Output dir: $OUTPUT_DIR"

python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this Slurm allocation")
print("torch:", torch.__version__)
print("device:", torch.cuda.get_device_name(0))
PY

python3 -u experiments/bp_head_vs_grid_simulation.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --episode "$EPISODE" \
  --n-paths 20 \
  --group-size 200 \
  --horizon 10 \
  --branch-num 2 \
  --teacher-child-shocks "${TEACHER_CHILD_SHOCKS:-2}" \
  --teacher-shock-seed "${TEACHER_SHOCK_SEED:-12345}" \
  --simulation-seed "${SIMULATION_SEED:-12345}" \
  --teacher-eval-batch-size "${TEACHER_EVAL_BATCH_SIZE:-512}" \
  --target-batch-size "${TARGET_BATCH_SIZE:-512}" \
  --device cuda:0 \
  --output-dir "$OUTPUT_DIR"

echo "Benchmark finished on $(date)"
