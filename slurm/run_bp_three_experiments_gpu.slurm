#!/bin/bash
#SBATCH -J bp-3exp-gpu
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/bp_three_experiments_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/bp_three_experiments_error.%j

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_full_20260712_174101}"
EPISODES="${EPISODES:-0 1 2}"
EP0_FIRM_PKL="${EP0_FIRM_PKL:-$RUN_ROOT/data/outputs/ep0_stage_mode0.pkl}"
POST0_FIRM_TEMPLATE="${POST0_FIRM_TEMPLATE:-$RUN_ROOT/data/outputs/ep{ep}_stage_modeb.pkl}"
FIRM_TEMPLATE="${FIRM_TEMPLATE:-$POST0_FIRM_TEMPLATE}"
N_STATES="${N_STATES:-20}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
N_BRANCHES="${N_BRANCHES:-2}"
REFIT_STEPS="${REFIT_STEPS:-1000}"
REFIT_LR="${REFIT_LR:-}"
REFERENCE_EPISODE="${REFERENCE_EPISODE:-2}"
STATES_PER_GROUP="${STATES_PER_GROUP:-20}"
FLIP_THRESHOLD="${FLIP_THRESHOLD:-0.2}"
B_GRID_SIZE="${B_GRID_SIZE:-21}"
Z_GRID_SIZE="${Z_GRID_SIZE:-41}"
BASE_STATE_SOURCE_INDEX="${BASE_STATE_SOURCE_INDEX:-}"
BAR_I_MODE="${BAR_I_MODE:-fixed_parent}"
RUN_RELATED_TESTS="${RUN_RELATED_TESTS:-1}"
ALLOW_DIRTY_WORKTREE="${ALLOW_DIRTY_WORKTREE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/data/outputs/bp_three_experiments}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-84dd4e8}"

mkdir -p "$LOG_DIR" "$WORK_ROOT/.mplcache" "$OUTPUT_ROOT"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

echo "============================================================"
echo "Three BP experiments with GPU"
echo "============================================================"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "git rev-parse HEAD:"
git rev-parse HEAD
echo "git status --short:"
git status --short
echo "Required commit: $REQUIRED_COMMIT"
echo "Run root: $RUN_ROOT"
echo "Episodes: $EPISODES"
echo "Episode 0 firm data: $EP0_FIRM_PKL"
echo "Post-0 firm template: $POST0_FIRM_TEMPLATE"
echo "N states: $N_STATES"
echo "Device: $DEVICE"
echo "Bar i mode: $BAR_I_MODE"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

if ! git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD; then
    echo "ERROR: HEAD does not contain required commit $REQUIRED_COMMIT"
    echo "Actual HEAD: $(git rev-parse --short HEAD)"
    exit 1
fi

if [[ "$ALLOW_DIRTY_WORKTREE" != "1" ]] && [[ -n "$(git status --short)" ]]; then
    echo "ERROR: working tree is not clean. Set ALLOW_DIRTY_WORKTREE=1 only for intentional local smoke runs."
    git status --short
    exit 1
fi

if [[ ! -f experiments/export_target_grid_decomposition.py ]]; then
    echo "ERROR: missing experiments/export_target_grid_decomposition.py"
    exit 1
fi

if [[ ! -f experiments/export_bp_deep_diagnostics.py ]]; then
    echo "ERROR: missing experiments/export_bp_deep_diagnostics.py"
    exit 1
fi

python3 -m py_compile \
    experiments/export_target_grid_decomposition.py \
    experiments/export_bp_deep_diagnostics.py \
    experiments/run_bp_fixed_teacher_refit_probe.py \
    experiments/export_bp_fixed_state_cross_episode.py \
    experiments/export_continuation_surface.py

if [[ "$RUN_RELATED_TESTS" == "1" ]]; then
    echo "[Preflight] Running exporter integration test"
    python3 -m pytest tests/test_target_grid_exporter.py -q

    echo "[Preflight] Running Q residual sign, bar_i mode, and boundary censoring tests"
    python3 -m pytest tests/test_continuation_surface.py -q -k \
        "q_residual_signs_cover_positive_and_negative_cases or q_csv_validator_checks_finiteness_and_sign_identity or bar_i_mode_fixed_parent_vs_recompute_issue_state or default_boundary_censoring_and_monotonicity_use_observed_only"
fi

nvidia-smi

python3 - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("This job requested GPU diagnostics, but CUDA is unavailable.")
print("CUDA device 0:", torch.cuda.get_device_name(0))
print("CUDA capability:", torch.cuda.get_device_capability(0))
print("Initial allocated memory:", torch.cuda.memory_allocated(0))
print("Initial reserved memory:", torch.cuda.memory_reserved(0))
PY

expand_episode_path() {
    local template="$1"
    local episode="$2"
    echo "${template//\{ep\}/$episode}"
}

firm_path_for_episode() {
    local episode="$1"
    if [[ "$episode" == "0" ]]; then
        echo "$EP0_FIRM_PKL"
    else
        expand_episode_path "$POST0_FIRM_TEMPLATE" "$episode"
    fi
}

for EP in $EPISODES; do
    echo
    echo "============================================================"
    echo "Processing episode $EP"
    echo "============================================================"

    EP_DIR="$OUTPUT_ROOT/ep${EP}"
    mkdir -p "$EP_DIR"

    POLICY_CKPT="$RUN_ROOT/checkpoints/ep${EP}_policy_value.pt"
    FIRM_PKL="$(firm_path_for_episode "$EP")"

    if [[ ! -f "$POLICY_CKPT" ]]; then
        echo "ERROR: missing policy checkpoint:"
        echo "  $POLICY_CKPT"
        exit 1
    fi

    if [[ ! -f "$FIRM_PKL" ]]; then
        echo "ERROR: missing matching firm data:"
        echo "  $FIRM_PKL"
        exit 1
    fi

    echo "Episode: $EP"
    echo "Policy checkpoint: $POLICY_CKPT"
    echo "Firm data: $FIRM_PKL"
    echo "Episode output: $EP_DIR"

    echo
    echo "[EP$EP][Experiment 1] Policy-teacher comparison"

    python3 experiments/export_target_grid_decomposition.py \
        --run-root "$RUN_ROOT" \
        --episode "$EP" \
        --firm-pkl "$FIRM_PKL" \
        --n-states "$N_STATES" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --n-branches "$N_BRANCHES" \
        --output-dir "$EP_DIR" \
        2>&1 | tee "$EP_DIR/experiment1_decomposition.log"

    LONG_CSV="$EP_DIR/ep${EP}_target_grid_decomposition_long.csv"
    SUMMARY_CSV="$EP_DIR/ep${EP}_target_grid_decomposition_summary.csv"

    test -f "$LONG_CSV"
    test -f "$SUMMARY_CSV"

    echo
    echo "[EP$EP][Experiments 2-3] Deep BP diagnostics"

    python3 experiments/export_bp_deep_diagnostics.py \
        --run-root "$RUN_ROOT" \
        --episode "$EP" \
        --firm-pkl "$FIRM_PKL" \
        --decomposition-long "$LONG_CSV" \
        --decomposition-summary "$SUMMARY_CSV" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --n-branches "$N_BRANCHES" \
        --output-dir "$EP_DIR" \
        2>&1 | tee "$EP_DIR/experiments2_3_deep_diagnostics.log"

    export EP
    export EP_DIR
    export LONG_CSV
    export SUMMARY_CSV
    export N_STATES

    python3 - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ep = int(os.environ["EP"])
ep_dir = Path(os.environ["EP_DIR"])
long_path = Path(os.environ["LONG_CSV"])
summary_path = Path(os.environ["SUMMARY_CSV"])
requested_states = int(os.environ["N_STATES"])

logit_states_path = ep_dir / f"ep{ep}_bp_logits_states.csv"
gradient_path = ep_dir / f"ep{ep}_bp_gradient_summary.csv"
cashflow_long_path = ep_dir / f"ep{ep}_cashflow_components_long.csv"
cashflow_summary_path = ep_dir / f"ep{ep}_cashflow_components_summary.csv"

for path in [
    long_path,
    summary_path,
    logit_states_path,
    gradient_path,
    cashflow_long_path,
    cashflow_summary_path,
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing diagnostic output: {path}")

long_df = pd.read_csv(long_path)
summary_df = pd.read_csv(summary_path)

if long_df.empty or summary_df.empty:
    raise AssertionError(f"EP{ep}: decomposition output is empty")

expected_branches = {"p0", "pi", "mix"}
if set(summary_df["branch"]) != expected_branches:
    raise AssertionError(
        f"EP{ep}: unexpected summary branches {sorted(summary_df['branch'].unique())}"
    )

n_states = int(summary_df["state_pos"].nunique())
if n_states <= 0 or n_states > requested_states:
    raise AssertionError(f"EP{ep}: invalid selected state count {n_states}")

identity_error = np.max(
    np.abs(
        long_df["value"].to_numpy()
        - long_df["cashflow"].to_numpy()
        - long_df["continuation"].to_numpy()
    )
)
if identity_error >= 1e-5:
    raise AssertionError(f"EP{ep}: Bellman decomposition error {identity_error}")

for column in ["bp_pred", "bp_star_coarse", "bp_star_teacher", "regret_at_pred"]:
    values = pd.to_numeric(summary_df[column], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"EP{ep}: non-finite values in {column}")

for column in ["confidence_weight", "mix_survival_weight", "value_margin"]:
    if column not in summary_df.columns:
        raise AssertionError(f"EP{ep}: summary missing {column}")
    values = pd.to_numeric(summary_df[column], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"EP{ep}: non-finite values in {column}")
    if column == "mix_survival_weight" and ((values < -1e-8).any() or (values > 1.0 + 1e-8).any()):
        raise AssertionError(f"EP{ep}: mix_survival_weight outside [0, 1]")

for column in ["value_star_coarse", "cashflow_coarse_star", "continuation_coarse_star"]:
    if column not in summary_df.columns:
        raise AssertionError(f"EP{ep}: summary missing {column}")
    values = pd.to_numeric(summary_df[column], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"EP{ep}: non-finite values in {column}")
coarse_star_identity_error = np.max(
    np.abs(
        summary_df["value_star_coarse"].to_numpy()
        - summary_df["cashflow_coarse_star"].to_numpy()
        - summary_df["continuation_coarse_star"].to_numpy()
    )
)
if coarse_star_identity_error >= 1e-5:
    raise AssertionError(f"EP{ep}: coarse value-star cashflow identity error {coarse_star_identity_error}")

logit_df = pd.read_csv(logit_states_path)
grad_df = pd.read_csv(gradient_path)

required_logit_columns = {
    "episode",
    "source_index",
    "branch",
    "bp0_pred",
    "bpI_pred",
    "bp_mix_pred",
    "bp_star_teacher",
    "bp0_logit",
    "bpI_logit",
    "bp0_sigmoid_derivative",
    "bpI_sigmoid_derivative",
    "bp0_logit_censored_low",
    "bpI_logit_censored_low",
    "bp0_logit_censored_high",
    "bpI_logit_censored_high",
}
required_gradient_columns = {
    "episode",
    "branch",
    "n_states",
    "diagnostic_target_loss",
    "bp0_head_grad_norm",
    "bpI_head_grad_norm",
    "policy_encoder_grad_norm",
    "bp0_logit_mean",
    "bp0_logit_p01",
    "bp0_logit_p50",
    "bp0_logit_p99",
    "bpI_logit_mean",
    "bpI_logit_p01",
    "bpI_logit_p50",
    "bpI_logit_p99",
    "bp0_sigmoid_derivative_mean",
    "bpI_sigmoid_derivative_mean",
    "gradient_semantics",
    "posthoc_mse_loss",
    "posthoc_mse_bp0_head_grad_norm",
    "posthoc_mse_bpI_head_grad_norm",
    "posthoc_mse_policy_encoder_grad_norm",
    "training_equivalent_loss",
    "training_equivalent_bp0_head_grad_norm",
    "training_equivalent_bpI_head_grad_norm",
    "training_equivalent_policy_encoder_grad_norm",
}
missing = required_logit_columns.difference(logit_df.columns)
if missing:
    raise AssertionError(f"EP{ep}: logit CSV missing {sorted(missing)}")
missing = required_gradient_columns.difference(grad_df.columns)
if missing:
    raise AssertionError(f"EP{ep}: gradient CSV missing {sorted(missing)}")

for column in [
    "bp0_pred",
    "bpI_pred",
    "bp_mix_pred",
    "bp_star_teacher",
    "bp0_logit",
    "bpI_logit",
    "bp0_sigmoid_derivative",
    "bpI_sigmoid_derivative",
]:
    values = pd.to_numeric(logit_df[column], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"EP{ep}: non-finite logit diagnostic {column}")

for column in [
    "diagnostic_target_loss",
    "bp0_head_grad_norm",
    "bpI_head_grad_norm",
    "policy_encoder_grad_norm",
    "posthoc_mse_loss",
    "posthoc_mse_bp0_head_grad_norm",
    "posthoc_mse_bpI_head_grad_norm",
    "posthoc_mse_policy_encoder_grad_norm",
    "training_equivalent_loss",
    "training_equivalent_bp0_head_grad_norm",
    "training_equivalent_bpI_head_grad_norm",
    "training_equivalent_policy_encoder_grad_norm",
]:
    values = pd.to_numeric(grad_df[column], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"EP{ep}: non-finite gradient diagnostic {column}")
    if (values < 0).any():
        raise AssertionError(f"EP{ep}: negative gradient metric {column}")

cf_long = pd.read_csv(cashflow_long_path)
cf_summary = pd.read_csv(cashflow_summary_path)

required_cashflow_columns = {
    "episode",
    "source_index",
    "branch",
    "bp_candidate",
    "cashflow",
    "production_cashflow",
    "debt_adjustment",
    "equity_financing_cost",
    "investment_adjustment",
    "cashflow_reconstructed",
    "cashflow_identity_error",
    "q_current",
    "q_issue",
    "q_unit",
    "p_child_mean",
    "default_mean",
    "recovery_grid_mean",
    "q_minus_raw_recovery_mean",
}
missing = required_cashflow_columns.difference(cf_long.columns)
if missing:
    raise AssertionError(f"EP{ep}: cashflow CSV missing {sorted(missing)}")

for column in [
    "cashflow",
    "production_cashflow",
    "debt_adjustment",
    "equity_financing_cost",
    "investment_adjustment",
    "cashflow_reconstructed",
    "cashflow_identity_error",
    "q_current",
    "q_issue",
    "p_child_mean",
    "default_mean",
    "recovery_grid_mean",
    "q_minus_raw_recovery_mean",
]:
    values = pd.to_numeric(cf_long[column], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"EP{ep}: non-finite cashflow diagnostic {column}")

max_cf_error = float(np.abs(cf_long["cashflow_identity_error"]).max())
if max_cf_error >= 1e-5:
    raise AssertionError(f"EP{ep}: cashflow identity error {max_cf_error}")

print()
print(f"EP{ep} validation passed")
print("Selected states:", n_states)
print("Bellman identity error:", identity_error)
print("Coarse value-star cashflow identity error:", coarse_star_identity_error)
print("Cashflow identity error:", max_cf_error)

print("\nPolicy-teacher summary:")
print(
    summary_df.groupby("branch")[
        ["bp_pred", "bp_star_teacher", "regret_at_pred"]
    ].agg(["mean", "min", "max"])
)

print("\nGradient summary:")
print(
    grad_df[
        [
            "branch",
            "diagnostic_target_loss",
            "bp0_head_grad_norm",
            "bpI_head_grad_norm",
            "policy_encoder_grad_norm",
            "bp0_logit_mean",
            "bpI_logit_mean",
            "bp0_sigmoid_derivative_mean",
            "bpI_sigmoid_derivative_mean",
        ]
    ].to_string(index=False)
)

print("\nCashflow component summary:")
print(cf_summary.to_string(index=False))

if "delta_cashflow_identity_error" not in cf_summary.columns:
    raise AssertionError(f"EP{ep}: cashflow summary missing delta_cashflow_identity_error")
max_delta_cf_error = float(np.abs(cf_summary["delta_cashflow_identity_error"]).max())
if max_delta_cf_error >= 1e-5:
    raise AssertionError(f"EP{ep}: cashflow delta identity error {max_delta_cf_error}")
PY

    echo "EP$EP passed all three diagnostic experiments."

    echo
    echo "[EP$EP][Stage 3] Fixed-teacher refit probe"

    REFIT_ARGS=()
    if [[ -n "$REFIT_LR" ]]; then
        REFIT_ARGS+=(--learning-rate "$REFIT_LR")
    fi
    python3 experiments/run_bp_fixed_teacher_refit_probe.py \
        --run-root "$RUN_ROOT" \
        --episode "$EP" \
        --firm-pkl "$FIRM_PKL" \
        --decomposition-summary "$SUMMARY_CSV" \
        --device "$DEVICE" \
        --steps "$REFIT_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --n-branches "$N_BRANCHES" \
        --output-dir "$EP_DIR" \
        "${REFIT_ARGS[@]}" \
        2>&1 | tee "$EP_DIR/fixed_teacher_refit_probe.log"

    nvidia-smi
done

echo
echo "============================================================"
echo "[Stage 4] Fixed-state cross-episode probe"
echo "============================================================"

REFERENCE_FIRM_PKL="$(firm_path_for_episode "$REFERENCE_EPISODE")"
python3 experiments/export_bp_fixed_state_cross_episode.py \
    --run-root "$RUN_ROOT" \
    --episodes "$EPISODES" \
    --reference-episode "$REFERENCE_EPISODE" \
    --reference-firm-pkl "$REFERENCE_FIRM_PKL" \
    --device "$DEVICE" \
    --states-per-group "$STATES_PER_GROUP" \
    --flip-threshold "$FLIP_THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    --n-branches "$N_BRANCHES" \
    --output-dir "$OUTPUT_ROOT/fixed_state_cross_episode" \
    2>&1 | tee "$OUTPUT_ROOT/fixed_state_cross_episode.log"

if [[ -z "$BASE_STATE_SOURCE_INDEX" ]]; then
    BASE_STATE_SOURCE_INDEX="$(
        python3 - <<'PY'
from pathlib import Path
import os
import pandas as pd

path = Path(os.environ["OUTPUT_ROOT"]) / "fixed_state_cross_episode" / "fixed_probe_states.csv"
df = pd.read_csv(path)
print(int(df["source_index"].iloc[0]))
PY
    )"
fi

echo
echo "============================================================"
echo "[Stage 4] Continuation and bond-value surface"
echo "============================================================"

python3 experiments/export_continuation_surface.py \
    --run-root "$RUN_ROOT" \
    --episode "$REFERENCE_EPISODE" \
    --base-state-source-index "$BASE_STATE_SOURCE_INDEX" \
    --firm-pkl "$REFERENCE_FIRM_PKL" \
    --b-grid-size "$B_GRID_SIZE" \
    --z-grid-size "$Z_GRID_SIZE" \
    --m-mode observed_child_mean \
    --bar-i-mode "$BAR_I_MODE" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --n-branches "$N_BRANCHES" \
    --output-dir "$OUTPUT_ROOT/continuation_surface_ep${REFERENCE_EPISODE}" \
    2>&1 | tee "$OUTPUT_ROOT/continuation_surface_ep${REFERENCE_EPISODE}.log"

export SURFACE_DIR="$OUTPUT_ROOT/continuation_surface_ep${REFERENCE_EPISODE}"
export REFERENCE_EPISODE
python3 - <<'PY'
from pathlib import Path
import os
import numpy as np
import pandas as pd

surface_dir = Path(os.environ["SURFACE_DIR"])
ep = int(os.environ["REFERENCE_EPISODE"])
paths = [
    surface_dir / f"ep{ep}_continuation_surface.csv",
    surface_dir / f"ep{ep}_default_boundary.csv",
    surface_dir / f"ep{ep}_bond_value_decomposition_long.csv",
    surface_dir / f"ep{ep}_bond_value_decomposition_summary.csv",
]
for path in paths:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise AssertionError(f"{path} is empty")
    if path.name.endswith("_default_boundary.csv"):
        required = {
            "boundary_censoring",
            "z_survival_boundary",
            "z_boundary_lower_bound",
            "z_boundary_upper_bound",
        }
        missing = required.difference(df.columns)
        if missing:
            raise AssertionError(f"{path} missing columns: {sorted(missing)}")
        allowed = {"observed", "left_censored", "right_censored"}
        invalid = set(df["boundary_censoring"].dropna().unique()).difference(allowed)
        if invalid:
            raise AssertionError(f"{path} invalid boundary_censoring values: {sorted(invalid)}")
        observed = df["boundary_censoring"] == "observed"
        if not np.isfinite(df.loc[observed, "z_survival_boundary"].to_numpy()).all():
            raise AssertionError(f"{path} observed rows contain non-finite z_survival_boundary")
        censored = ~observed
        if not df.loc[censored, "z_survival_boundary"].isna().all():
            raise AssertionError(f"{path} censored rows must have NaN z_survival_boundary")
        counts = df["boundary_censoring"].value_counts().reindex(
            ["observed", "left_censored", "right_censored"],
            fill_value=0,
        )
        print(
            f"{path.name} censoring counts: "
            f"observed={int(counts['observed'])}, "
            f"left_censored={int(counts['left_censored'])}, "
            f"right_censored={int(counts['right_censored'])}"
        )
        numeric = df.select_dtypes(include=["number"]).drop(
            columns=[
                "z_survival_boundary",
                "z_boundary_lower_bound",
                "z_boundary_upper_bound",
            ],
            errors="ignore",
        )
        finite = np.isfinite(numeric.to_numpy()).all()
        if not finite:
            raise AssertionError(f"{path} contains NaN/Inf outside censoring boundary columns")
        continue
    if path.name.endswith("_bond_value_decomposition_long.csv"):
        required = {
            "q_issue",
            "q_target_total",
            "q_training_residual",
            "q_issue_minus_target",
            "bar_i",
            "bar_i_multiplier",
            "bar_i_mode",
        }
        missing = required.difference(df.columns)
        if missing:
            raise AssertionError(f"{path} missing columns: {sorted(missing)}")
        allowed_modes = {"fixed_parent", "recompute_issue_state"}
        invalid_modes = set(df["bar_i_mode"].dropna().unique()).difference(allowed_modes)
        if invalid_modes:
            raise AssertionError(f"{path} invalid bar_i_mode values: {sorted(invalid_modes)}")
        numeric_cols = [
            "q_issue",
            "q_target_total",
            "q_training_residual",
            "q_issue_minus_target",
            "bar_i",
            "bar_i_multiplier",
        ]
        numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all():
            raise AssertionError(f"{path} contains non-finite Q residual/bar_i values")
        q_training_error = np.max(
            np.abs(
                numeric["q_training_residual"].to_numpy()
                - (numeric["q_target_total"].to_numpy() - numeric["q_issue"].to_numpy())
            )
        )
        q_issue_minus_error = np.max(
            np.abs(
                numeric["q_issue_minus_target"].to_numpy()
                + numeric["q_training_residual"].to_numpy()
            )
        )
        if q_training_error >= 1e-7 or q_issue_minus_error >= 1e-7:
            raise AssertionError(
                f"{path} Q residual sign identity failed: "
                f"training={q_training_error}, issue_minus={q_issue_minus_error}"
            )
        print(
            f"{path.name} Q residual identities: "
            f"training_error={q_training_error:.3e}, "
            f"issue_minus_error={q_issue_minus_error:.3e}, "
            f"bar_i_modes={sorted(df['bar_i_mode'].dropna().unique())}"
        )
        continue
    if path.name.endswith("_bond_value_decomposition_summary.csv"):
        if "bar_i_mode" not in df.columns:
            raise AssertionError(f"{path} missing bar_i_mode")
        invalid_modes = set(df["bar_i_mode"].dropna().unique()).difference(
            {"fixed_parent", "recompute_issue_state"}
        )
        if invalid_modes:
            raise AssertionError(f"{path} invalid bar_i_mode values: {sorted(invalid_modes)}")
    numeric = df.select_dtypes(include=["number"])
    finite = np.isfinite(numeric.to_numpy()).all()
    if not finite:
        raise AssertionError(f"{path} contains NaN/Inf")
print("Continuation/Q surface validation passed.")
PY

export OUTPUT_ROOT
export EPISODES

python3 - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["OUTPUT_ROOT"])
episodes = [int(x) for x in os.environ["EPISODES"].split()]

policy_rows = []
gradient_rows = []
cashflow_rows = []

for ep in episodes:
    ep_dir = root / f"ep{ep}"
    summary = pd.read_csv(ep_dir / f"ep{ep}_target_grid_decomposition_summary.csv")
    summary.insert(0, "episode", ep)
    policy_rows.append(summary)
    gradient_rows.append(pd.read_csv(ep_dir / f"ep{ep}_bp_gradient_summary.csv"))
    cashflow_rows.append(pd.read_csv(ep_dir / f"ep{ep}_cashflow_components_summary.csv"))

policy = pd.concat(policy_rows, ignore_index=True)
gradients = pd.concat(gradient_rows, ignore_index=True)
cashflow = pd.concat(cashflow_rows, ignore_index=True)

policy_agg = (
    policy.groupby(["episode", "branch"])
    .agg(
        n_states=("source_index", "nunique"),
        bp_pred_mean=("bp_pred", "mean"),
        bp_pred_min=("bp_pred", "min"),
        bp_pred_max=("bp_pred", "max"),
        bp_teacher_mean=("bp_star_teacher", "mean"),
        bp_teacher_min=("bp_star_teacher", "min"),
        bp_teacher_max=("bp_star_teacher", "max"),
        regret_mean=("regret_at_pred", "mean"),
        regret_max=("regret_at_pred", "max"),
        cashflow_delta_mean=("delta_cashflow_low_to_coarse_star", "mean"),
        continuation_delta_mean=("delta_continuation_low_to_coarse_star", "mean"),
    )
    .reset_index()
)

policy_path = root / "episode_policy_teacher_comparison.csv"
gradient_path = root / "episode_logits_gradients.csv"
cashflow_path = root / "episode_cashflow_components.csv"

policy_agg.to_csv(policy_path, index=False)
gradients.to_csv(gradient_path, index=False)
cashflow.to_csv(cashflow_path, index=False)

print("============================================================")
print("Cross-episode policy/teacher comparison")
print("============================================================")
print(policy_agg.to_string(index=False))

print()
print("============================================================")
print("Cross-episode logits and gradients")
print("============================================================")
print(gradients.to_string(index=False))

print()
print("============================================================")
print("Cross-episode cashflow components")
print("============================================================")
print(cashflow.to_string(index=False))

print()
print("Saved:")
print(policy_path)
print(gradient_path)
print(cashflow_path)
PY

if [[ "$RUN_RELATED_TESTS" == "1" ]]; then
    echo "[Tests] Running related test files"
    python3 -m pytest \
        tests/test_bp_deep_diagnostics.py \
        tests/test_bp_fixed_teacher_refit.py \
        tests/test_bp_fixed_state_cross_episode.py \
        tests/test_continuation_surface.py \
        tests/test_bp_grid_teacher.py \
        tests/test_target_grid_training_integration.py \
        -q
else
    echo "Related pytest skipped because RUN_RELATED_TESTS=$RUN_RELATED_TESTS"
fi

echo "============================================================"
echo "All BP GPU diagnostic stages passed."
echo "Output root: $OUTPUT_ROOT"
echo "Finished: $(date)"
echo "============================================================"
