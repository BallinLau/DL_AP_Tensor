#!/bin/bash
#SBATCH -J p-inner-fixed
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/p_inner_fixed_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/p_inner_fixed_error.%j

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
FIXED_POINT_RUN="${FIXED_POINT_RUN:?FIXED_POINT_RUN must be provided}"
OUTPUT_DIR="${OUTPUT_DIR:-$FIXED_POINT_RUN/fixed_target_p_inner_solver}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:?REQUIRED_COMMIT must be provided after the diagnostic is committed}"
SEED="${SEED:-24681357}"
CYCLES="${CYCLES:-1 5 10}"
BUDGETS="${BUDGETS:-100 200 500 1000}"
MEMORIZATION_EPOCHS="${MEMORIZATION_EPOCHS:-1000}"
MEMORIZATION_BATCHES="${MEMORIZATION_BATCHES:-1 4 16 0}"
N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-64}"
SHOCK_SEED="${SHOCK_SEED:-12345}"
RUN_TARGETED_TESTS="${RUN_TARGETED_TESTS:-1}"

mkdir -p "$WORK_ROOT/logs" "$OUTPUT_DIR" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "Fixed-target P inner-solver diagnostic"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $ACTUAL_COMMIT"
echo "Required commit: $REQUIRED_COMMIT"
echo "Fixed-point run: $FIXED_POINT_RUN"
echo "Output: $OUTPUT_DIR"
echo "Cycles: $CYCLES"
echo "Independent budgets: $BUDGETS"
echo "Memorization batches: $MEMORIZATION_BATCHES"

if [[ "$ACTUAL_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "ERROR: expected commit $REQUIRED_COMMIT, found $ACTUAL_COMMIT"
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked worktree is not clean"
  git status --short
  exit 1
fi
for path in \
  "$FIXED_POINT_RUN/config.json" \
  "$FIXED_POINT_RUN/dataset_hashes.json" \
  "$FIXED_POINT_RUN/environment_hashes.json" \
  "$FIXED_POINT_RUN/cycle_checkpoint_hashes.csv" \
  "$FIXED_POINT_RUN/pq_fixed_point_cycles.csv" \
  "$FIXED_POINT_RUN/checkpoints/cycle_00.pt" \
  "$FIXED_POINT_RUN/checkpoints/cycle_04.pt" \
  "$FIXED_POINT_RUN/checkpoints/cycle_09.pt"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required fixed-point artifact not found: $path"
    exit 1
  fi
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print("cuda_device_0:", torch.cuda.get_device_name(0))
PY
nvidia-smi

python3 -m py_compile \
  evaluation/fixed_target_p_inner_solver.py \
  experiments/fixed_target_p_inner_solver.py
if [[ "$RUN_TARGETED_TESTS" == "1" ]]; then
  python3 -m pytest tests/test_fixed_target_p_inner_solver.py -q \
    -p no:cacheprovider -p no:randomly
fi

read -r -a CYCLE_ARGS <<< "$CYCLES"
read -r -a BUDGET_ARGS <<< "$BUDGETS"
read -r -a MEMORIZATION_ARGS <<< "$MEMORIZATION_BATCHES"

python3 -u experiments/fixed_target_p_inner_solver.py \
  --fixed-point-run "$FIXED_POINT_RUN" \
  --output-dir "$OUTPUT_DIR" \
  --cycles "${CYCLE_ARGS[@]}" \
  --budgets "${BUDGET_ARGS[@]}" \
  --memorization-epochs "$MEMORIZATION_EPOCHS" \
  --memorization-batches "${MEMORIZATION_ARGS[@]}" \
  --seed "$SEED" \
  --determinism-cycle 1 \
  --determinism-budget 100 \
  --determinism-tolerance 1e-7 \
  --fit-tolerance 0.05 \
  --memorization-tolerance 0.01 \
  --b-points 41 \
  --z-points 41 \
  --n-child-shocks "$N_CHILD_SHOCKS" \
  --shock-seed "$SHOCK_SEED" \
  --device cuda:0

echo "Completed: $(date)"
