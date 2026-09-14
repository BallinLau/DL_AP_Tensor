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
REQUIRED_EVALUATOR_COMMIT="${REQUIRED_EVALUATOR_COMMIT:-6fbfc9f47ef8c7f07c9a16c3ec76209295c19d19}"
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
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/data/outputs/firm_checkpoint_evaluator/ep${EPISODE}_J${N_CHILD_SHOCKS:-2}_${STAMP}}"

DEVICE="${DEVICE:-cuda:0}"
B_MIN="${B_MIN:-0.0}"
B_MAX="${B_MAX:-1.0}"
B_POINTS="${B_POINTS:-101}"
Z_MIN="${Z_MIN:--4.0}"
Z_MAX="${Z_MAX:-4.0}"
Z_POINTS="${Z_POINTS:-101}"
I_POINTS="${I_POINTS:-101}"
FORWARD_CHUNK_SIZE="${FORWARD_CHUNK_SIZE:-8192}"
N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-2}"
SHOCK_SEED="${SHOCK_SEED:-12345}"

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
echo "Child shocks: $N_CHILD_SHOCKS (seed=$SHOCK_SEED)"
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

python3 -u experiments/evaluate_checkpoints.py \
  "${CHECKPOINT_ARGS[@]}" \
  --firm-data "$FIRM_DATA" \
  "${MACRO_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --b-min "$B_MIN" \
  --b-max "$B_MAX" \
  --b-points "$B_POINTS" \
  --z-min "$Z_MIN" \
  --z-max "$Z_MAX" \
  --z-points "$Z_POINTS" \
  --i-points "$I_POINTS" \
  --forward-chunk-size "$FORWARD_CHUNK_SIZE" \
  --n-child-shocks "$N_CHILD_SHOCKS" \
  --shock-seed "$SHOCK_SEED"

export OUTPUT_DIR N_CHILD_SHOCKS B_POINTS Z_POINTS
python3 - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["OUTPUT_DIR"])
required = [
    "summary.csv",
    "metadata.json",
    "default/default_boundary.csv",
    "default/default_boundary.png",
    "investment/investment_boundary.csv",
    "q/Q_b_slices.png",
    "q/Q_peak_by_z.csv",
    "q/Q_peak_by_z.png",
    "q/q_unit_b_slices.png",
]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"Missing required evaluator outputs: {missing}")

summary = pd.read_csv(root / "summary.csv")
if len(summary) != 1 or summary.shape[1] == 0:
    raise SystemExit(f"Invalid summary.csv shape: {summary.shape}")

metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
if metadata.get("model_state_unchanged") is not True:
    raise SystemExit("Evaluator changed checkpoint model state")
if metadata.get("model_state_hash_before") != metadata.get("model_state_hash_after"):
    raise SystemExit("Before/after model hashes differ")
if metadata.get("bp_teacher_model") != "policy_value":
    raise SystemExit("Unexpected BP teacher model semantics")
transition = metadata.get("reference_transition_bank", {})
if transition.get("builder") != "ConvergenceShockBank+build_child_exogenous_bundle":
    raise SystemExit(f"Unexpected transition builder: {transition.get('builder')}")
if int(transition.get("n_child_shocks", -1)) != int(os.environ["N_CHILD_SHOCKS"]):
    raise SystemExit("Metadata child-shock count does not match the submitted run")

boundary = pd.read_csv(root / "default" / "default_boundary.csv")
valid_default_status = {
    "all_default", "all_survival", "single_crossing", "multiple_crossings",
    "no_identified_crossing", "nonfinite"
}
if not set(boundary["boundary_status"].dropna().unique()).issubset(valid_default_status):
    raise SystemExit("default_boundary.csv contains an unknown boundary_status")
single_default = boundary["boundary_status"].eq("single_crossing")
if not boundary.loc[single_default, "z_default"].notna().all():
    raise SystemExit("Single-crossing default rows must have finite z_default")
if boundary.loc[~single_default, "z_default"].notna().any():
    raise SystemExit("Non-single-crossing default rows must keep z_default as NaN")

investment = pd.read_csv(root / "investment" / "investment_boundary.csv")
valid_investment_status = {
    "all_invest", "all_no_invest", "single_crossing", "multiple_crossings", "nonfinite"
}
if not set(investment["investment_status"].dropna().unique()).issubset(valid_investment_status):
    raise SystemExit("investment_boundary.csv contains an unknown investment_status")
if investment.loc[investment["investment_status"].ne("single_crossing"), "i_star"].notna().any():
    raise SystemExit("Non-single-crossing investment rows must keep i_star as NaN")

survival_plots = list((root / "bp").glob("*survival*.png"))
slice_csvs = list((root / "objective_slices").glob("*.csv"))
slice_plots = list((root / "objective_slices").glob("*.png"))
if not survival_plots:
    raise SystemExit("No survival-region BP plots were generated")
if len(slice_csvs) != 18 or len(slice_plots) != 18:
    raise SystemExit(
        f"Expected 18 objective CSVs and 18 plots, got {len(slice_csvs)} and {len(slice_plots)}"
    )

print("Evaluator output validation passed")
print(f"summary_shape={summary.shape}")
print(f"default_status_counts={boundary['boundary_status'].value_counts(dropna=False).to_dict()}")
print(f"investment_status_counts={investment['investment_status'].value_counts(dropna=False).to_dict()}")
print(f"survival_bp_plots={len(survival_plots)}")
print(f"objective_slice_csvs={len(slice_csvs)}")
print(f"objective_slice_plots={len(slice_plots)}")
PY

echo "============================================================"
echo "Firm-side checkpoint evaluation completed"
echo "Finished: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
