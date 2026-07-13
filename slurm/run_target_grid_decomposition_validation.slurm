#!/bin/bash
#SBATCH -J dl-grid-decomp-val
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_grid_decomposition_validation_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_grid_decomposition_validation_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"

# Default to the full pytest pass after the focused diagnostics checks.
# Submit with RUN_FULL_TESTS=0 to skip the final full suite.
export RUN_FULL_TESTS="${RUN_FULL_TESTS:-1}"

mkdir -p "$LOG_DIR" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

# Current HEAD must contain the target-grid value decomposition diagnostics.
REQUIRED_COMMIT="${REQUIRED_COMMIT:-f5a0c17}"
if ! git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD; then
  echo "ERROR: HEAD does not contain required commit $REQUIRED_COMMIT"
  echo "Actual HEAD: $(git rev-parse --short HEAD)"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Required commit: $REQUIRED_COMMIT"
echo "Python: $(which python3)"
echo "RUN_FULL_TESTS=$RUN_FULL_TESTS"

python3 - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

echo "[1/4] Direct target-grid teacher decomposition tests"
python3 -m pytest tests/test_bp_grid_teacher.py -q

echo "[2/4] Policy child vectorization and decomposition reference tests"
python3 -m pytest tests/test_policy_child_vectorization.py -q

echo "[3/4] Target-grid training integration tests"
python3 -m pytest tests/test_target_grid_training_integration.py -q

echo "[4/4] Full test suite"
if [[ "$RUN_FULL_TESTS" == "1" ]]; then
  python3 -m pytest -q
else
  echo "Full pytest skipped because RUN_FULL_TESTS=$RUN_FULL_TESTS"
fi

echo "Final acceptance summary:"
echo "  tests/test_bp_grid_teacher.py passed"
echo "  tests/test_policy_child_vectorization.py passed"
echo "  tests/test_target_grid_training_integration.py passed"

if [[ "$RUN_FULL_TESTS" == "1" ]]; then
  echo "  full pytest passed"
else
  echo "  full pytest was intentionally skipped"
fi

echo "Target-grid value decomposition validation passed."
echo "Job finished on $(date)"
