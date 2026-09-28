#!/bin/bash
#SBATCH -J p-teacher-drift-5ep
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/p_teacher_drift_5ep_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/p_teacher_drift_5ep_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --exclude=g85,g05

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/hybrid_q_solver_stability_5ep_ab5c855_20260927_141640}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/cachedir/p_teacher_drift_5ep_ab5c855}"
EPISODES="${EPISODES:-0:4}"
N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-64}"
SHOCK_SEED="${SHOCK_SEED:-12345}"
EXPECTED_BASE_COMMIT="${EXPECTED_BASE_COMMIT:-ab5c85541c6ebeab70f8eb7e55b5a9b065050cd1}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "============================================================"
echo "P-teacher drift evaluator"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $ACTUAL_COMMIT"
echo "Run root: $RUN_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Episodes: $EPISODES"
echo "N child shocks: $N_CHILD_SHOCKS"
echo "Shock seed: $SHOCK_SEED"
echo "============================================================"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean"
  git status --short
  exit 1
fi
if ! git merge-base --is-ancestor "$EXPECTED_BASE_COMMIT" HEAD; then
  echo "ERROR: HEAD does not contain expected base $EXPECTED_BASE_COMMIT"
  exit 1
fi
if [[ -n "$REQUIRED_COMMIT" && "$ACTUAL_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "ERROR: REQUIRED_COMMIT mismatch"
  echo "Expected: $REQUIRED_COMMIT"
  echo "Actual:   $ACTUAL_COMMIT"
  exit 1
fi
if [[ ! -d "$RUN_ROOT" ]]; then
  echo "ERROR: run root does not exist: $RUN_ROOT"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this Slurm allocation")
print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

ARGS=(
  --run-root "$RUN_ROOT"
  --episodes "$EPISODES"
  --output-dir "$OUTPUT_DIR"
  --device cuda:0
  --b-values 0.01 0.05 0.20
  --z-values 0 2 3 4
  --eta-values 0 1
  --n-child-shocks "$N_CHILD_SHOCKS"
  --shock-seed "$SHOCK_SEED"
)
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python3 -u experiments/evaluate_p_teacher_drift.py "${ARGS[@]}"

echo "Finished: $(date)"
