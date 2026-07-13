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
FIRM_TEMPLATE="${FIRM_TEMPLATE:-$RUN_ROOT/data/outputs/ep{ep}_stage_modeb.pkl}"
N_STATES="${N_STATES:-20}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
N_BRANCHES="${N_BRANCHES:-2}"
RUN_RELATED_TESTS="${RUN_RELATED_TESTS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/data/outputs/bp_three_experiments}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-85a9d49}"

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
echo "Required commit: $REQUIRED_COMMIT"
echo "Run root: $RUN_ROOT"
echo "Episodes: $EPISODES"
echo "Firm template: $FIRM_TEMPLATE"
echo "N states: $N_STATES"
echo "Device: $DEVICE"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

if ! git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD; then
    echo "ERROR: HEAD does not contain required commit $REQUIRED_COMMIT"
    echo "Actual HEAD: $(git rev-parse --short HEAD)"
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
    experiments/export_bp_deep_diagnostics.py

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

for EP in $EPISODES; do
    echo
    echo "============================================================"
    echo "Processing episode $EP"
    echo "============================================================"

    EP_DIR="$OUTPUT_ROOT/ep${EP}"
    mkdir -p "$EP_DIR"

    POLICY_CKPT="$RUN_ROOT/checkpoints/ep${EP}_policy_value.pt"
    FIRM_PKL="$(expand_episode_path "$FIRM_TEMPLATE" "$EP")"

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
    "q_minus_recovery",
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
    "q_minus_recovery",
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
PY

    echo "EP$EP passed all three diagnostic experiments."
    nvidia-smi
done

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
        tests/test_bp_grid_teacher.py \
        tests/test_target_grid_training_integration.py \
        -q
else
    echo "Related pytest skipped because RUN_RELATED_TESTS=$RUN_RELATED_TESTS"
fi

echo "============================================================"
echo "All three GPU experiments passed."
echo "Output root: $OUTPUT_ROOT"
echo "Finished: $(date)"
echo "============================================================"
