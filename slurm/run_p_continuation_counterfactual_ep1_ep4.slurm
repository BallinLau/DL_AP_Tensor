#!/bin/bash
#SBATCH -J p-cont-cf-ep1-ep4
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/p_continuation_counterfactual_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/p_continuation_counterfactual_error.%j
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
EXISTING_ANALYSIS="${EXISTING_ANALYSIS:-$WORK_ROOT/cachedir/p_teacher_drift_5ep_ab5c855}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/cachedir/p_continuation_counterfactual_ep1_ep4}"
OLD_EPISODE="${OLD_EPISODE:-1}"
NEW_EPISODE="${NEW_EPISODE:-4}"
B_VALUE="${B_VALUE:-0.05}"
Z_VALUES="${Z_VALUES:-2 3 4}"
ETA_VALUE="${ETA_VALUE:-1}"
N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-64}"
SHOCK_SEED="${SHOCK_SEED:-12345}"
OVERWRITE="${OVERWRITE:-1}"
EXPECTED_BASE_COMMIT="${EXPECTED_BASE_COMMIT:-e695671394c0fb3c979417f82e12df526b49e9f5}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-}"

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "============================================================"
echo "P continuation counterfactual EP1 vs EP4"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $ACTUAL_COMMIT"
echo "Run root: $RUN_ROOT"
echo "Existing analysis: $EXISTING_ANALYSIS"
echo "Output dir: $OUTPUT_DIR"
echo "Episodes: $OLD_EPISODE -> $NEW_EPISODE"
echo "Focus: b=$B_VALUE z=[$Z_VALUES] eta=$ETA_VALUE"
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
if [[ ! -f "$EXISTING_ANALYSIS/tables/p_teacher_components_by_i.csv" ]]; then
  echo "ERROR: missing existing components table"
  exit 1
fi
if [[ ! -f "$EXISTING_ANALYSIS/tables/p_teacher_selected_branch.csv" ]]; then
  echo "ERROR: missing existing selected-branch table"
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
  --existing-analysis "$EXISTING_ANALYSIS"
  --output-dir "$OUTPUT_DIR"
  --old-episode "$OLD_EPISODE"
  --new-episode "$NEW_EPISODE"
  --b-value "$B_VALUE"
  --z-values $Z_VALUES
  --eta-value "$ETA_VALUE"
  --n-child-shocks "$N_CHILD_SHOCKS"
  --shock-seed "$SHOCK_SEED"
  --device cuda:0
)
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python3 -u experiments/evaluate_p_continuation_counterfactual.py "${ARGS[@]}"

echo "Finished: $(date)"
