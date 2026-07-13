#!/bin/bash
#SBATCH -J dl-grid-decomp-smoke
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_grid_decomposition_smoke_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_grid_decomposition_smoke_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"

RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_full_20260712_174101}"
EP="${EP:-2}"
FIRM_PKL="${FIRM_PKL:-$RUN_ROOT/data/outputs/ep2_stage_modeb.pkl}"
SDF_FC1_CKPT="${SDF_FC1_CKPT:-$RUN_ROOT/checkpoints/ep2_sdf_fc1.pt}"
POLICY_CKPT="${POLICY_CKPT:-$RUN_ROOT/checkpoints/ep2_policy_value.pt}"
FC2_CKPT="${FC2_CKPT:-$RUN_ROOT/checkpoints/ep2_fc2.pt}"

N_STATES="${N_STATES:-5}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
N_BRANCHES="${N_BRANCHES:-2}"
RUN_EXPORTER_PYTEST="${RUN_EXPORTER_PYTEST:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/data/outputs/grid_decomposition_smoke_ep${EP}}"

mkdir -p "$LOG_DIR" "$WORK_ROOT/.mplcache" "$OUTPUT_DIR"
cd "$REPO_DIR"

REQUIRED_COMMIT="${REQUIRED_COMMIT:-19fa9cc}"
if ! git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD; then
    echo "ERROR: HEAD does not contain required commit $REQUIRED_COMMIT"
    echo "Actual HEAD: $(git rev-parse --short HEAD)"
    exit 1
fi

for required_file in "$FIRM_PKL" "$SDF_FC1_CKPT" "$POLICY_CKPT" "$FC2_CKPT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file does not exist:"
        echo "  $required_file"
        exit 1
    fi
done

EXPECTED_POLICY_CKPT="$RUN_ROOT/checkpoints/ep${EP}_policy_value.pt"
if [[ "$POLICY_CKPT" != "$EXPECTED_POLICY_CKPT" ]]; then
    echo "ERROR: POLICY_CKPT must match exporter episode path:"
    echo "  POLICY_CKPT=$POLICY_CKPT"
    echo "  expected=$EXPECTED_POLICY_CKPT"
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

echo "============================================================"
echo "Target-grid decomposition real-checkpoint smoke"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Required commit: $REQUIRED_COMMIT"
echo "Python: $(which python3)"
echo "Run root: $RUN_ROOT"
echo "Episode: $EP"
echo "Firm pickle: $FIRM_PKL"
echo "SDF/FC1 checkpoint: $SDF_FC1_CKPT"
echo "Policy/value checkpoint: $POLICY_CKPT"
echo "FC2 checkpoint: $FC2_CKPT"
echo "Output directory: $OUTPUT_DIR"
echo "Requested states: $N_STATES"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Number of branches: $N_BRANCHES"
echo "============================================================"

python3 - <<'PY'
import pandas as pd
import torch

print("Torch:", torch.__version__)
print("Pandas:", pd.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

echo "[1/4] Syntax and CLI validation"

python3 -m py_compile \
    experiments/export_target_grid_decomposition.py

python3 experiments/export_target_grid_decomposition.py \
    --help > "$OUTPUT_DIR/exporter_help.txt"

echo "[2/4] Run real-checkpoint decomposition export"

python3 experiments/export_target_grid_decomposition.py \
    --run-root "$RUN_ROOT" \
    --episode "$EP" \
    --firm-pkl "$FIRM_PKL" \
    --n-states "$N_STATES" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --n-branches "$N_BRANCHES" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/exporter.log"

LONG_CSV="$OUTPUT_DIR/ep${EP}_target_grid_decomposition_long.csv"
SUMMARY_CSV="$OUTPUT_DIR/ep${EP}_target_grid_decomposition_summary.csv"

echo "[3/4] Validate exported CSV files"

export LONG_CSV
export SUMMARY_CSV
export N_STATES

python3 - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

long_path = Path(os.environ["LONG_CSV"])
summary_path = Path(os.environ["SUMMARY_CSV"])
requested_states = int(os.environ["N_STATES"])

if not long_path.exists():
    raise FileNotFoundError(f"Missing long decomposition CSV: {long_path}")

if not summary_path.exists():
    raise FileNotFoundError(f"Missing summary decomposition CSV: {summary_path}")

long_df = pd.read_csv(long_path)
summary_df = pd.read_csv(summary_path)

print("Long CSV:", long_path)
print("Summary CSV:", summary_path)
print("Long shape:", long_df.shape)
print("Summary shape:", summary_df.shape)

required_long_columns = {
    "state_pos",
    "source_index",
    "branch",
    "grid_index",
    "b",
    "z",
    "eta",
    "i",
    "x",
    "hatcf",
    "lnkf",
    "bp_pred",
    "bp_candidate",
    "cashflow",
    "continuation",
    "value",
    "q_issue",
    "p_child_mean",
    "default_mean",
}

required_summary_columns = {
    "state_pos",
    "source_index",
    "branch",
    "b",
    "z",
    "eta",
    "i",
    "x",
    "hatcf",
    "lnkf",
    "bp_pred",
    "bp_star_coarse",
    "bp_star_teacher",
    "argmax_index",
    "value_low",
    "value_star_coarse",
    "value_star_teacher",
    "regret_at_pred",
    "cashflow_low",
    "cashflow_coarse_star",
    "continuation_low",
    "continuation_coarse_star",
    "delta_cashflow_low_to_coarse_star",
    "delta_continuation_low_to_coarse_star",
    "delta_value_low_to_coarse_star",
    "q_issue_coarse_star",
    "p_child_coarse_star",
    "default_coarse_star",
    "q_issue_teacher_star",
    "p_child_teacher_star",
    "default_teacher_star",
    "identity_error_max",
    "dominant_component",
}

missing_long = required_long_columns.difference(long_df.columns)
missing_summary = required_summary_columns.difference(summary_df.columns)

if missing_long:
    raise AssertionError(f"Long CSV missing columns: {sorted(missing_long)}")

if missing_summary:
    raise AssertionError(f"Summary CSV missing columns: {sorted(missing_summary)}")

if long_df.empty:
    raise AssertionError("Long CSV is empty")

if summary_df.empty:
    raise AssertionError("Summary CSV is empty")

branches = {"p0", "pi", "mix"}

if set(long_df["branch"].unique()) != branches:
    raise AssertionError(
        f"Unexpected long branches: {sorted(long_df['branch'].unique())}"
    )

if set(summary_df["branch"].unique()) != branches:
    raise AssertionError(
        "Unexpected summary branches: "
        f"{sorted(summary_df['branch'].unique())}"
    )

n_selected = int(summary_df["state_pos"].nunique())

if n_selected <= 0:
    raise AssertionError("No selected states were exported")

if n_selected > requested_states:
    raise AssertionError(
        f"Exported {n_selected} states, requested only {requested_states}"
    )

expected_summary_rows = n_selected * 3
if len(summary_df) != expected_summary_rows:
    raise AssertionError(
        f"Expected {expected_summary_rows} summary rows, got {len(summary_df)}"
    )

if not (summary_df["eta"] > 0.5).all():
    bad = summary_df.loc[
        summary_df["eta"] <= 0.5,
        ["state_pos", "source_index", "branch", "eta"],
    ]
    raise AssertionError(
        "Inactive parent states were exported:\n"
        + bad.to_string(index=False)
    )

group_sizes = long_df.groupby(["state_pos", "branch"], sort=True).size()

if group_sizes.nunique() != 1:
    raise AssertionError(
        "Grid sizes differ across state/branch groups:\n"
        + group_sizes.to_string()
    )

n_grid = int(group_sizes.iloc[0])
expected_long_rows = n_selected * 3 * n_grid

if len(long_df) != expected_long_rows:
    raise AssertionError(
        f"Expected {expected_long_rows} long rows, got {len(long_df)}"
    )

long_finite_columns = [
    "b",
    "z",
    "eta",
    "i",
    "x",
    "hatcf",
    "lnkf",
    "bp_pred",
    "bp_candidate",
    "cashflow",
    "continuation",
    "value",
    "q_issue",
    "p_child_mean",
    "default_mean",
]

summary_finite_columns = [
    "bp_pred",
    "bp_star_coarse",
    "bp_star_teacher",
    "value_star_coarse",
    "value_star_teacher",
    "regret_at_pred",
    "q_issue_coarse_star",
    "p_child_coarse_star",
    "default_coarse_star",
    "q_issue_teacher_star",
    "p_child_teacher_star",
    "default_teacher_star",
    "identity_error_max",
]

for column in long_finite_columns:
    values = pd.to_numeric(long_df[column], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"Non-finite values found in long.{column}")

for column in summary_finite_columns:
    values = pd.to_numeric(summary_df[column], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"Non-finite values found in summary.{column}")

identity_error = np.max(
    np.abs(
        long_df["value"].to_numpy()
        - long_df["cashflow"].to_numpy()
        - long_df["continuation"].to_numpy()
    )
)

summary_identity_error = float(summary_df["identity_error_max"].max())

print("Selected states:", n_selected)
print("Grid points per state/branch:", n_grid)
print("Maximum long identity error:", identity_error)
print("Maximum summary identity error:", summary_identity_error)

if identity_error >= 1e-5:
    raise AssertionError(
        f"Long decomposition identity error too large: {identity_error}"
    )

if summary_identity_error >= 1e-5:
    raise AssertionError(
        "Summary decomposition identity error too large: "
        f"{summary_identity_error}"
    )

for column in ["bp_pred", "bp_star_coarse", "bp_star_teacher"]:
    if not summary_df[column].between(0.0, 1.0).all():
        raise AssertionError(f"{column} contains values outside [0, 1]")

if not long_df["bp_candidate"].between(0.0, 1.0).all():
    raise AssertionError("bp_candidate contains values outside [0, 1]")

winner_index = long_df.groupby(["state_pos", "branch"])["value"].idxmax()

winner = long_df.loc[
    winner_index,
    ["state_pos", "branch", "bp_candidate", "value"],
].rename(
    columns={
        "bp_candidate": "bp_star_from_long",
        "value": "value_star_from_long",
    }
)

merged = summary_df.merge(
    winner,
    on=["state_pos", "branch"],
    how="left",
    validate="one_to_one",
)

if not np.allclose(
    merged["bp_star_coarse"],
    merged["bp_star_from_long"],
    rtol=0.0,
    atol=1e-7,
):
    raise AssertionError("bp_star_coarse does not match long-grid argmax")

if not np.allclose(
    merged["value_star_coarse"],
    merged["value_star_from_long"],
    rtol=1e-6,
    atol=1e-7,
):
    raise AssertionError("value_star_coarse does not match long-grid maximum")

valid_components = {"cashflow", "continuation"}
if not set(summary_df["dominant_component"]).issubset(valid_components):
    raise AssertionError(
        "Unexpected dominant_component values: "
        f"{sorted(summary_df['dominant_component'].unique())}"
    )

print("\nDominant-component counts:")
print(summary_df.groupby(["branch", "dominant_component"]).size())

print("\nTeacher/network comparison:")
print(
    summary_df[
        [
            "source_index",
            "branch",
            "b",
            "bp_pred",
            "bp_star_coarse",
            "bp_star_teacher",
            "regret_at_pred",
            "q_issue_teacher_star",
            "p_child_teacher_star",
            "default_teacher_star",
            "dominant_component",
        ]
    ].to_string(index=False)
)

print("\nCSV validation passed.")
PY

echo "[4/4] Optional exporter-related pytest"

if [[ "$RUN_EXPORTER_PYTEST" == "1" ]]; then
    python3 -m pytest tests/test_bp_grid_teacher.py -q
    python3 -m pytest tests/test_target_grid_training_integration.py -q
else
    echo "Exporter-related pytest skipped because RUN_EXPORTER_PYTEST=$RUN_EXPORTER_PYTEST"
fi

echo "============================================================"
echo "Real-checkpoint target-grid decomposition smoke passed."
echo "Long CSV: $LONG_CSV"
echo "Summary CSV: $SUMMARY_CSV"
echo "Job finished: $(date)"
echo "============================================================"
