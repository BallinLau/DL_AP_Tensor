#!/bin/bash
#SBATCH -J firm-ckpt-eval
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/firm_checkpoint_evaluator_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/firm_checkpoint_evaluator_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT must point to the completed training run}"
EPISODE="${EPISODE:?EPISODE must identify the checkpoint to evaluate}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# The submitted revision may be newer, but it must contain the evaluator fixes.
REQUIRED_EVALUATOR_COMMIT="${REQUIRED_EVALUATOR_COMMIT:-0936c6ae2f25921618bd4c275d99eeb4617bcaf1}"
ALLOW_DIRTY_WORKTREE="${ALLOW_DIRTY_WORKTREE:-0}"
STRICT_UNTRACKED_WORKTREE="${STRICT_UNTRACKED_WORKTREE:-0}"
RUN_EVALUATOR_TESTS="${RUN_EVALUATOR_TESTS:-1}"

CHECKPOINT_MODE="${CHECKPOINT_MODE:-combined}"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/checkpoints_analysis/ep${EPISODE}_combined.pt}"
PV_CKPT="${PV_CKPT:-$RUN_ROOT/checkpoints/ep${EPISODE}_policy_value.pt}"
SDF_CKPT="${SDF_CKPT:-$RUN_ROOT/checkpoints/ep${EPISODE}_sdf_fc1.pt}"
HYPERPARAMS_JSON="${HYPERPARAMS_JSON:-}"
CONFIG_JSON="${CONFIG_JSON:-}"
MODEL_SPEC_JSON="${MODEL_SPEC_JSON:-}"

FIRM_DATA="${FIRM_DATA:-$RUN_ROOT/data/outputs/final_simulate_firm.pkl}"
MACRO_DATA="${MACRO_DATA:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/data/outputs/firm_checkpoint_evaluator/ep${EPISODE}_eta_matrix_${STAMP}}"

DEVICE="${DEVICE:-cuda:0}"
ETA="${ETA:-1.0}"
ETA_VALUES="${ETA_VALUES:-0 1}"
B_MIN="${B_MIN:-0.0}"
B_MAX="${B_MAX:-1.0}"
B_POINTS="${B_POINTS:-101}"
Z_MIN="${Z_MIN:--3.0}"
Z_MAX="${Z_MAX:-3.0}"
Z_POINTS="${Z_POINTS:-101}"
I_POINTS="${I_POINTS:-101}"
FORWARD_CHUNK_SIZE="${FORWARD_CHUNK_SIZE:-8192}"
N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-32}"
ROBUSTNESS_CHILD_SHOCKS="${ROBUSTNESS_CHILD_SHOCKS:-2 32}"
SHOCK_SEED="${SHOCK_SEED:-12345}"
BP_TEACHER_MARGIN_TOL="${BP_TEACHER_MARGIN_TOL:-1e-8}"
REQUIRE_PRIMARY_ETA_NEXT_SUPPORT="${REQUIRE_PRIMARY_ETA_NEXT_SUPPORT:-1}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "============================================================"
echo "Firm-side checkpoint evaluator (GPU)"
echo "============================================================"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $ACTUAL_COMMIT"
echo "Required evaluator ancestor: $REQUIRED_EVALUATOR_COMMIT"
echo "Run root: $RUN_ROOT"
echo "Episode: $EPISODE"
echo "Checkpoint mode: $CHECKPOINT_MODE"
echo "Firm data: $FIRM_DATA"
echo "Macro data: $MACRO_DATA"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Grid: b=[$B_MIN,$B_MAX]x$B_POINTS, z=[$Z_MIN,$Z_MAX]x$Z_POINTS, i=$I_POINTS"
echo "Eta fallback: $ETA"
echo "Eta matrix: $ETA_VALUES"
echo "Primary child shocks: $N_CHILD_SHOCKS"
echo "Robustness child shocks: $ROBUSTNESS_CHILD_SHOCKS (seed=$SHOCK_SEED)"
echo "BP teacher margin tolerance: $BP_TEACHER_MARGIN_TOL"
echo "============================================================"

if ! git merge-base --is-ancestor "$REQUIRED_EVALUATOR_COMMIT" HEAD; then
  echo "ERROR: HEAD does not contain required evaluator commit $REQUIRED_EVALUATOR_COMMIT"
  exit 1
fi

TRACKED_CHANGES="$(git status --porcelain --untracked-files=no)"
UNTRACKED_CHANGES="$(git status --porcelain --untracked-files=normal | sed -n '/^??/p')"
if [[ -n "$TRACKED_CHANGES" ]]; then
  printf '%s\n' "$TRACKED_CHANGES"
  if [[ "$ALLOW_DIRTY_WORKTREE" != "1" ]]; then
    echo "ERROR: tracked files differ from HEAD. Commit/stash changes, or set ALLOW_DIRTY_WORKTREE=1 for an intentional diagnostic run."
    exit 1
  fi
  echo "WARNING: continuing with modified tracked files because ALLOW_DIRTY_WORKTREE=1."
fi
if [[ -n "$UNTRACKED_CHANGES" ]]; then
  printf '%s\n' "$UNTRACKED_CHANGES"
  if [[ "$STRICT_UNTRACKED_WORKTREE" == "1" ]]; then
    echo "ERROR: untracked files exist and STRICT_UNTRACKED_WORKTREE=1."
    exit 1
  fi
  echo "WARNING: untracked files are present; tracked source files still match HEAD."
fi

python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this Slurm allocation")
print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python3 -m py_compile \
  experiments/evaluate_checkpoints.py \
  evaluation/grids.py \
  evaluation/firm_surfaces.py \
  evaluation/boundaries.py \
  evaluation/bellman_diagnostics.py \
  evaluation/bp_diagnostics.py \
  evaluation/plotting.py

if [[ "$RUN_EVALUATOR_TESTS" == "1" ]]; then
  echo "[Preflight] Running checkpoint evaluator tests"
  python3 -m pytest tests/test_checkpoint_evaluator.py -q
fi

for input_path in "$FIRM_DATA"; do
  if [[ ! -f "$input_path" ]]; then
    echo "ERROR: required input does not exist: $input_path"
    exit 1
  fi
done

CHECKPOINT_ARGS=()
case "$CHECKPOINT_MODE" in
  combined)
    if [[ ! -f "$CHECKPOINT" ]]; then
      echo "ERROR: combined checkpoint does not exist: $CHECKPOINT"
      echo "Use CHECKPOINT=/path/to/epN_combined.pt, or CHECKPOINT_MODE=raw with all JSON sidecars."
      exit 1
    fi
    CHECKPOINT_ARGS+=(--checkpoint "$CHECKPOINT")
    ;;
  raw)
    for input_path in "$PV_CKPT" "$SDF_CKPT" "$HYPERPARAMS_JSON" "$CONFIG_JSON" "$MODEL_SPEC_JSON"; do
      if [[ -z "$input_path" || ! -f "$input_path" ]]; then
        echo "ERROR: raw mode requires existing PV_CKPT, SDF_CKPT, HYPERPARAMS_JSON, CONFIG_JSON, and MODEL_SPEC_JSON."
        echo "Missing: ${input_path:-<empty>}"
        exit 1
      fi
    done
    CHECKPOINT_ARGS+=(
      --pv-ckpt "$PV_CKPT"
      --sdf-ckpt "$SDF_CKPT"
      --hyperparams-json "$HYPERPARAMS_JSON"
      --config-json "$CONFIG_JSON"
      --model-spec-json "$MODEL_SPEC_JSON"
    )
    ;;
  *)
    echo "ERROR: CHECKPOINT_MODE must be combined or raw, got: $CHECKPOINT_MODE"
    exit 1
    ;;
esac

MACRO_ARGS=()
if [[ "$MACRO_DATA" != "auto" ]]; then
  if [[ ! -f "$MACRO_DATA" ]]; then
    echo "ERROR: explicit macro data does not exist: $MACRO_DATA"
    echo "Set MACRO_DATA=auto to use evaluator auto-discovery."
    exit 1
  fi
  MACRO_ARGS+=(--macro-data "$MACRO_DATA")
fi

read -r -a ETA_VALUE_ARGS <<< "$ETA_VALUES"
read -r -a ROBUSTNESS_J_ARGS <<< "$ROBUSTNESS_CHILD_SHOCKS"

python3 -u experiments/evaluate_checkpoints.py \
  "${CHECKPOINT_ARGS[@]}" \
  --firm-data "$FIRM_DATA" \
  "${MACRO_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --eta "$ETA" \
  --eta-values "${ETA_VALUE_ARGS[@]}" \
  --b-min "$B_MIN" \
  --b-max "$B_MAX" \
  --b-points "$B_POINTS" \
  --z-min "$Z_MIN" \
  --z-max "$Z_MAX" \
  --z-points "$Z_POINTS" \
  --i-points "$I_POINTS" \
  --forward-chunk-size "$FORWARD_CHUNK_SIZE" \
  --n-child-shocks "$N_CHILD_SHOCKS" \
  --robustness-child-shocks "${ROBUSTNESS_J_ARGS[@]}" \
  --shock-seed "$SHOCK_SEED" \
  --bp-teacher-margin-tol "$BP_TEACHER_MARGIN_TOL"

export OUTPUT_DIR N_CHILD_SHOCKS B_POINTS Z_POINTS ETA_VALUES ROBUSTNESS_CHILD_SHOCKS REQUIRE_PRIMARY_ETA_NEXT_SUPPORT
python3 - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["OUTPUT_DIR"])
required_root = ["summary.csv", "metadata.json", "robustness/j_comparison.csv"]
missing = [name for name in required_root if not (root / name).is_file()]
if missing:
    raise SystemExit(f"Missing required evaluator matrix outputs: {missing}")

summary = pd.read_csv(root / "summary.csv")
if len(summary) < 1 or summary.shape[1] == 0:
    raise SystemExit(f"Invalid summary.csv shape: {summary.shape}")
required_summary = {
    "eta_parent", "n_child_shocks", "p0_residual_abs_mean", "p0_residual_abs_p90",
    "pi_residual_abs_mean", "pi_residual_abs_p90", "bp_mae_raw",
    "bp_mae_survival", "bp_mae_survival_identified", "bp_grid_star_mean",
    "bp_continuation_at_coarse_star_mean", "bp_regret_mean",
    "bp_regret_median", "bp_regret_p90", "bp_regret_p99", "bp_regret_max",
    "teacher_identified_share", "eta_next_active_share",
    "investment_i_monotonicity_violation_share",
    "investment_i_monotonicity_violation_share_survival", "hard_default_share",
    "soft_default_mean", "q_unit_mean_survival",
}
missing_columns = sorted(required_summary - set(summary.columns))
if missing_columns:
    raise SystemExit(f"summary.csv missing required diagnostics: {missing_columns}")

metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
if metadata.get("evaluator") != "firm_side_checkpoint_evaluator_v2_matrix":
    raise SystemExit("Unexpected evaluator matrix metadata")
if metadata.get("model_state_unchanged") is not True:
    raise SystemExit("At least one evaluator case changed checkpoint model state")

full_cases = [case for case in metadata.get("cases", []) if case.get("full_output")]
if not full_cases:
    raise SystemExit("Matrix metadata contains no full-output ETA cases")
for case_record in full_cases:
    case = Path(case_record["output"])
    eta_label = case.name
    required_case = [
        "summary.csv", "metadata.json", "default/default_boundary.csv",
        "default/boundary_comparison.csv", "default/boundary_comparison.png",
        "investment/investment_boundary.csv", "investment/D_VI_minus_V0_mid.csv",
        "q/Q_b_slices.png", "q/q_unit.csv", "bellman/R0_signed.csv",
        "bellman/RI_signed.csv", "bp/p0_bp_regret_survival_identified.csv",
        "audits/child_continuation_audit.csv",
    ]
    missing_case = [name for name in required_case if not (case / name).is_file()]
    if missing_case:
        raise SystemExit(f"{eta_label} missing outputs: {missing_case}")
    case_meta = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
    if case_meta.get("model_state_unchanged") is not True:
        raise SystemExit(f"{eta_label}: evaluator changed checkpoint model state")
    timing = case_meta.get("semantics", {}).get("child_leverage_timing")
    if timing != "b_next = eta_next * bp_current + (1-eta_next) * b_current":
        raise SystemExit(f"{eta_label}: leverage timing metadata mismatch")
    audit = pd.read_csv(case / "audits" / "child_continuation_audit.csv")
    if audit["child_b_identity_error"].max() > 1e-6:
        raise SystemExit(f"{eta_label}: child leverage identity failed")
    if not {0.0, 1.0}.issubset(set(audit["eta_next"].unique())):
        message = f"{eta_label} J={os.environ['N_CHILD_SHOCKS']} did not draw both eta_next states"
        if os.environ["REQUIRE_PRIMARY_ETA_NEXT_SUPPORT"] == "1":
            raise SystemExit(message)
        print(f"WARNING: {message}")
    expected_weighted = audit["branch_weight"] * audit["raw_continuation_term"]
    if (audit["weighted_continuation_contribution"] - expected_weighted).abs().max() > 1e-8:
        raise SystemExit(f"{eta_label}: weighted continuation contribution identity failed")

print("Evaluator output validation passed")
print(f"summary_shape={summary.shape}")
print(summary[["eta_parent", "n_child_shocks", "eta_next_active_share", "bp_regret_mean"]].to_string(index=False))
PY

echo "============================================================"
echo "Firm-side checkpoint evaluation completed"
echo "Finished: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
