#!/bin/bash
#SBATCH -J full-run-eval
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/full_run_evaluator_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/full_run_evaluator_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT must point to a completed multi-episode run}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/data/outputs/full_run_evaluation}"
DEVICE="${DEVICE:-cuda:0}"
EPISODES="${EPISODES:-}"
TRAINING_LOG="${TRAINING_LOG:-}"
REFERENCE_FIRM_DATA="${REFERENCE_FIRM_DATA:-}"
REFERENCE_MACRO_DATA="${REFERENCE_MACRO_DATA:-}"
N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-64}"
ROBUSTNESS_CHILD_SHOCKS="${ROBUSTNESS_CHILD_SHOCKS:-32 64 128}"
ROBUSTNESS_SCOPE="${ROBUSTNESS_SCOPE:-representative}"
SHOCK_SEED="${SHOCK_SEED:-12345}"
MAX_SDF_PARENTS="${MAX_SDF_PARENTS:-512}"
ETA_VALUES="${ETA_VALUES:-0 1}"
B_MIN="${B_MIN:-0.0}"
B_MAX="${B_MAX:-1.0}"
B_POINTS="${B_POINTS:-101}"
Z_MIN="${Z_MIN:--2.0}"
Z_MAX="${Z_MAX:-2.0}"
Z_POINTS="${Z_POINTS:-101}"
I_POINTS="${I_POINTS:-101}"
FORWARD_CHUNK_SIZE="${FORWARD_CHUNK_SIZE:-8192}"
BP_TEACHER_MARGIN_TOL="${BP_TEACHER_MARGIN_TOL:-1e-8}"
# Evaluator-only BP chunk budget (65536/131072/262144/524288). Empty means "auto":
# >=70GiB -> 262144, >=35GiB -> 131072, otherwise 65536. An explicit value always wins.
# This only changes read-only evaluator chunking; training semantics and model state are untouched.
BP_EVAL_MAX_EXPANDED_STATES="${BP_EVAL_MAX_EXPANDED_STATES:-}"
RUN_EVALUATOR_TESTS="${RUN_EVALUATOR_TESTS:-1}"
ALLOW_DIRTY_WORKTREE="${ALLOW_DIRTY_WORKTREE:-0}"
OVERWRITE="${OVERWRITE:-0}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "============================================================"
echo "Full-run checkpoint evaluator (read-only, GPU)"
echo "============================================================"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $ACTUAL_COMMIT"
echo "Required commit: ${REQUIRED_COMMIT:-ancestor check disabled}"
echo "Run root: $RUN_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Episodes: ${EPISODES:-auto}"
echo "Training log: ${TRAINING_LOG:-auto/unavailable}"
echo "Device: $DEVICE"
echo "Child shocks: primary=$N_CHILD_SHOCKS robustness=$ROBUSTNESS_CHILD_SHOCKS"
echo "Robustness scope: $ROBUSTNESS_SCOPE"
echo "BP evaluator budget (BP_EVAL_MAX_EXPANDED_STATES): ${BP_EVAL_MAX_EXPANDED_STATES:-auto}"
echo "Overwrite evaluator output: $OVERWRITE"
echo "Grid: b=[$B_MIN,$B_MAX]x$B_POINTS z=[$Z_MIN,$Z_MAX]x$Z_POINTS"
echo "============================================================"

if [[ -n "$REQUIRED_COMMIT" ]] && ! git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD; then
  echo "ERROR: HEAD does not contain REQUIRED_COMMIT=$REQUIRED_COMMIT"
  exit 1
fi
if [[ -n "$(git status --porcelain)" && "$ALLOW_DIRTY_WORKTREE" != "1" ]]; then
  echo "ERROR: working tree is not clean"
  git status --short
  exit 1
fi
if [[ ! -d "$RUN_ROOT" ]]; then
  echo "ERROR: RUN_ROOT does not exist: $RUN_ROOT"
  exit 1
fi

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this allocation")
print("cuda_device_0:", torch.cuda.get_device_name(0))
total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"cuda_device_0_total_memory_gib: {total_gib:.2f}")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python3 -m py_compile \
  experiments/evaluate_full_run.py \
  experiments/evaluate_checkpoints.py \
  evaluation/full_run_diagnostics.py \
  evaluation/bellman_diagnostics.py \
  evaluation/bp_diagnostics.py \
  training/bp_grid_teacher.py \
  analysis/checkpoint_loader.py

if [[ "$RUN_EVALUATOR_TESTS" == "1" ]]; then
  python3 -m pytest \
    tests/test_full_run_evaluator.py \
    tests/test_checkpoint_evaluator.py \
    tests/test_policy_child_vectorization.py -q
fi

ARGS=(
  --run-root "$RUN_ROOT"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --n-child-shocks "$N_CHILD_SHOCKS"
  --robustness-scope "$ROBUSTNESS_SCOPE"
  --shock-seed "$SHOCK_SEED"
  --max-sdf-parents "$MAX_SDF_PARENTS"
  --b-min "$B_MIN" --b-max "$B_MAX" --b-points "$B_POINTS"
  --z-min "$Z_MIN" --z-max "$Z_MAX" --z-points "$Z_POINTS"
  --i-points "$I_POINTS"
  --forward-chunk-size "$FORWARD_CHUNK_SIZE"
  --bp-teacher-margin-tol "$BP_TEACHER_MARGIN_TOL"
)
read -r -a ROBUSTNESS_ARGS <<< "$ROBUSTNESS_CHILD_SHOCKS"
read -r -a ETA_ARGS <<< "$ETA_VALUES"
ARGS+=(--robustness-child-shocks "${ROBUSTNESS_ARGS[@]}" --eta-values "${ETA_ARGS[@]}")
[[ -n "$EPISODES" ]] && ARGS+=(--episodes "$EPISODES")
[[ -n "$TRAINING_LOG" ]] && ARGS+=(--training-log "$TRAINING_LOG")
[[ -n "$REFERENCE_FIRM_DATA" ]] && ARGS+=(--reference-firm-data "$REFERENCE_FIRM_DATA")
[[ -n "$REFERENCE_MACRO_DATA" ]] && ARGS+=(--reference-macro-data "$REFERENCE_MACRO_DATA")
[[ "$ALLOW_DIRTY_WORKTREE" == "1" ]] && ARGS+=(--allow-dirty-worktree)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)
[[ -n "$BP_EVAL_MAX_EXPANDED_STATES" ]] && ARGS+=(--bp-eval-max-expanded-states "$BP_EVAL_MAX_EXPANDED_STATES")

python3 -u experiments/evaluate_full_run.py "${ARGS[@]}"

python3 - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["OUTPUT_DIR"])
metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
timing = json.loads((output / "evaluation_timing.json").read_text(encoding="utf-8"))
bank = metadata.get("common_shock_bank", {})
print("------------------------------------------------------------")
print("Evaluator BP summary")
print("canonical_shock_bank_max_children:", bank.get("canonical_shock_bank_max_children"))
print("evaluated_child_counts:", bank.get("evaluated_child_counts"))
print("max_evaluated_children:", bank.get("max_evaluated_children"))
print("primary_n_child_shocks:", metadata.get("primary_n_child_shocks"))
print("robustness_child_shocks:", metadata.get("robustness_child_shocks"))
print("robustness_scope:", metadata.get("robustness_scope"))
print("bp_max_expanded_states:", timing.get("bp_max_expanded_states"))
print("bp_max_actual_expanded_states:", timing.get("bp_max_actual_expanded_states"))
print("bp_multi_j_reuse_enabled:", timing.get("bp_multi_j_reuse_enabled"))
print("bp_branch_reuse_enabled:", timing.get("bp_branch_reuse_enabled"))
print("bp_model_forward_calls:", timing.get("bp_model_forward_calls"))
print("bp_child_equity_forward_calls:", timing.get("bp_child_equity_forward_calls"))
print("bp_q_forward_calls:", timing.get("bp_q_forward_calls"))
print("bp_seconds:", timing.get("bp_seconds"))
print("total_seconds:", timing.get("total_seconds"))
print("cuda_peak_memory_mb:", timing.get("cuda_peak_memory_mb"))
print("------------------------------------------------------------")
PY

for required in \
  headline_metrics.csv \
  equation_residual_trajectories.csv \
  metadata.json \
  missing_artifacts.md \
  README.md \
  convergence_dashboard.png \
  tables/headline_metrics_by_episode.csv \
  figures/equilibrium_convergence_dashboard.png; do
  if [[ ! -f "$OUTPUT_DIR/$required" ]]; then
    echo "ERROR: evaluator did not create $OUTPUT_DIR/$required"
    exit 1
  fi
done
echo "Full-run evaluation completed: $OUTPUT_DIR"
