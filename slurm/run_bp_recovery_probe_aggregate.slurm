#!/bin/bash
#SBATCH --job-name=bp_recovery_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=a01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
CONDA_ENV="${CONDA_ENV:-DL_HL}"
conda activate "$CONDA_ENV"

REPO_ROOT="${REPO_ROOT:-/home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor}"
RUN_ROOT="${RUN_ROOT:-/home/fit/zhuyingz/WORK/LiuHao/cachedir/normalized_sdf_aio_full_20260712_174101}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/data/outputs/bp_recovery_probes}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-}"

cd "$REPO_ROOT"
mkdir -p logs "$OUTPUT_ROOT"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ -z "$REQUIRED_COMMIT" ]]; then
  REQUIRED_COMMIT="$ACTUAL_COMMIT"
fi
SHORT_SHA="$(git rev-parse --short HEAD)"
INPUT_ROOT="$OUTPUT_ROOT/full/commit_${SHORT_SHA}"
AGG_OUTPUT_DIR="$OUTPUT_ROOT/aggregate/commit_${SHORT_SHA}"

echo "============================================================"
echo "BP recovery probe aggregate"
echo "============================================================"
date
hostname
pwd
echo "git rev-parse HEAD:"
git rev-parse HEAD
echo "git status --short:"
git status --short
python3 --version
echo "input_root=$INPUT_ROOT"
echo "output_dir=$AGG_OUTPUT_DIR"
echo "required_commit=$REQUIRED_COMMIT"
echo "actual_commit=$ACTUAL_COMMIT"
echo "============================================================"

if [[ "$ACTUAL_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "Expected commit: $REQUIRED_COMMIT"
  echo "Actual commit:   $ACTUAL_COMMIT"
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean."
  git status --short
  exit 1
fi
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "ERROR: input root missing: $INPUT_ROOT"
  exit 1
fi

python3 -m py_compile experiments/summarize_bp_recovery_probes.py

mkdir -p "$AGG_OUTPUT_DIR"
cat > "$AGG_OUTPUT_DIR/run_metadata.txt" <<EOF
started_at=$(date)
host=$(hostname)
repo=$REPO_ROOT
commit=$ACTUAL_COMMIT
input_root=$INPUT_ROOT
output_dir=$AGG_OUTPUT_DIR
episodes=1,2
seeds=12345,23456,34567
EOF

python3 experiments/summarize_bp_recovery_probes.py \
  --input-root "$INPUT_ROOT" \
  --output-dir "$AGG_OUTPUT_DIR" \
  --episodes 1,2 \
  --seeds 12345,23456,34567

python3 - "$AGG_OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(sys.argv[1])
files = [
    "bp_recovery_all_runs.csv",
    "bp_recovery_multiseed_summary.csv",
    "bp_recovery_pairwise_comparisons.csv",
    "bp_recovery_decision_table.csv",
]
for name in files:
    path = out / name
    if not path.exists():
        raise SystemExit(f"missing aggregate output: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"empty aggregate output: {path}")
    numeric = df.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise SystemExit(f"non-finite values in {path}")
comparisons = pd.read_csv(out / "bp_recovery_pairwise_comparisons.csv")
expected = {
    "H002_bias_recenter_minus_baseline",
    "H003_logit_minus_branch_output",
    "mix_branch_minus_full_output",
}
if set(comparisons["comparison"].unique()) != expected:
    raise SystemExit("unexpected comparison labels")
print("BP recovery aggregate validator passed")
PY

echo "BP recovery aggregate completed successfully"
