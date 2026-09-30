#!/bin/bash
#SBATCH -J bp-teacher-fit-ep2
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/bp_teacher_fit_ep2_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/bp_teacher_fit_ep2_error.%j
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
EPISODE="${EPISODE:-2}"
CHECKPOINT_STAGE="${CHECKPOINT_STAGE:-post_bp}"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/episode_diagnostics/ep_$(printf '%03d' "$EPISODE")/${CHECKPOINT_STAGE}.pt}"
COMBINED_CHECKPOINT="${COMBINED_CHECKPOINT:-}"
if [[ -z "$COMBINED_CHECKPOINT" ]]; then
  if [[ -f "$RUN_ROOT/checkpoints_analysis/ep${EPISODE}_combined.pt" ]]; then
    COMBINED_CHECKPOINT="$RUN_ROOT/checkpoints_analysis/ep${EPISODE}_combined.pt"
  else
    COMBINED_CHECKPOINT="$RUN_ROOT/checkpoints/ep${EPISODE}_combined.pt"
  fi
fi
FIRM_DATA="${FIRM_DATA:-$RUN_ROOT/data/outputs/final_simulate_firm.pkl}"
MACRO_DATA="${MACRO_DATA:-$RUN_ROOT/data/outputs/final_simulate_macro.pkl}"
ONDIST_FIRM_DATA="${ONDIST_FIRM_DATA:-$RUN_ROOT/data/outputs/ep${EPISODE}_stage_modeb.pkl}"
ONDIST_MACRO_DATA="${ONDIST_MACRO_DATA:-$RUN_ROOT/data/outputs/ep${EPISODE}_stage_modeb_macro.pkl}"
PRESENTATION_FOCUS_CSV="${PRESENTATION_FOCUS_CSV:-$WORK_ROOT/cachedir/presentation_figures_ep2_e695671/tables/bp_teacher_focus_ep2.csv}"
EVAL_MODE="${EVAL_MODE:-smoke}"
if [[ "$EVAL_MODE" == "smoke" ]]; then
  N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-8}"
  DENSE_POINTS="${DENSE_POINTS:-11}"
  ONDIST_MAX_STATES="${ONDIST_MAX_STATES:-100}"
  BRANCHES=(pi_mid)
elif [[ "$EVAL_MODE" == "formal" ]]; then
  N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-64}"
  DENSE_POINTS="${DENSE_POINTS:-41}"
  ONDIST_MAX_STATES="${ONDIST_MAX_STATES:-2000}"
  BRANCHES=(p0 pi_low pi_mid pi_high)
else
  echo "ERROR: EVAL_MODE must be smoke or formal, got $EVAL_MODE"
  exit 1
fi
SAMPLE_SEED="${SAMPLE_SEED:-12345}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/cachedir/bp_teacher_fit_ep2_f8e19ea_${EVAL_MODE}_$(date +%Y%m%d_%H%M%S)}"
SHOCK_SEED="${SHOCK_SEED:-12345}"
OVERWRITE="${OVERWRITE:-1}"
EXPECTED_BASE_COMMIT="${EXPECTED_BASE_COMMIT:-f8e19ea2046566504884e134dd149b90630fd1cc}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-}"

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "============================================================"
echo "BP teacher-fit read-only diagnostic"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $ACTUAL_COMMIT"
echo "Checkpoint: $CHECKPOINT"
echo "Combined checkpoint: $COMBINED_CHECKPOINT"
echo "Firm data: $FIRM_DATA"
echo "Macro data: $MACRO_DATA"
echo "On-distribution firm data: $ONDIST_FIRM_DATA"
echo "On-distribution macro data: $ONDIST_MACRO_DATA"
echo "Presentation focus CSV: $PRESENTATION_FOCUS_CSV"
echo "Output: $OUTPUT_DIR"
echo "Episode/stage: $EPISODE/$CHECKPOINT_STAGE"
echo "Evaluation mode: $EVAL_MODE"
echo "Dense points: ${DENSE_POINTS}x${DENSE_POINTS}"
echo "On-distribution N: $ONDIST_MAX_STATES"
echo "Branches: ${BRANCHES[*]}"
echo "J/shock seed/sample seed: $N_CHILD_SHOCKS/$SHOCK_SEED/$SAMPLE_SEED"
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
  echo "ERROR: REQUIRED_COMMIT mismatch; expected $REQUIRED_COMMIT, actual $ACTUAL_COMMIT"
  exit 1
fi
for path in "$CHECKPOINT" "$COMBINED_CHECKPOINT" "$FIRM_DATA" "$MACRO_DATA" "$ONDIST_FIRM_DATA" "$ONDIST_MACRO_DATA"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required input missing: $path"
    exit 1
  fi
done

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
  --checkpoint "$CHECKPOINT"
  --combined-checkpoint "$COMBINED_CHECKPOINT"
  --firm-data "$FIRM_DATA"
  --macro-data "$MACRO_DATA"
  --ondist-firm-data "$ONDIST_FIRM_DATA"
  --ondist-macro-data "$ONDIST_MACRO_DATA"
  --ondist-max-states "$ONDIST_MAX_STATES"
  --sample-seed "$SAMPLE_SEED"
  --episode "$EPISODE"
  --checkpoint-stage "$CHECKPOINT_STAGE"
  --dense-points "$DENSE_POINTS"
  --branches "${BRANCHES[@]}"
  --n-child-shocks "$N_CHILD_SHOCKS"
  --shock-seed "$SHOCK_SEED"
  --device cuda:0
  --output-dir "$OUTPUT_DIR"
)
if [[ -f "$PRESENTATION_FOCUS_CSV" ]]; then
  ARGS+=(--presentation-focus-csv "$PRESENTATION_FOCUS_CSV")
else
  echo "WARNING: presentation focus CSV not found; exact presentation bank will be skipped"
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python3 -u experiments/evaluate_bp_teacher_fit.py "${ARGS[@]}"

echo "Finished: $(date)"
