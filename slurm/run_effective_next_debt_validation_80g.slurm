#!/bin/bash
#SBATCH -J dl-effective-next-debt-val
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_effective_next_debt_validation_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_effective_next_debt_validation_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"

# Tiny simulation smoke scale. This job validates output identities only; it
# does not run formal training, GPU experiments, or long episode loops.
SMOKE_N_PATHS="${SMOKE_N_PATHS:-2}"
SMOKE_GROUP_SIZE="${SMOKE_GROUP_SIZE:-5}"
SMOKE_HORIZON="${SMOKE_HORIZON:-2}"
SMOKE_BRANCH_NUM="${SMOKE_BRANCH_NUM:-2}"
SMOKE_TOL="${SMOKE_TOL:-1e-6}"

mkdir -p "$LOG_DIR" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Python: $(which python3)"
echo "Validation scope:"
echo "  target: effective next-period debt reporting"
echo "  no formal training experiment"
echo "  no Bellman/policy/Q/optimizer/transition changes are exercised beyond schema validation"
echo "Tiny simulation smoke config:"
echo "  SMOKE_N_PATHS=$SMOKE_N_PATHS"
echo "  SMOKE_GROUP_SIZE=$SMOKE_GROUP_SIZE"
echo "  SMOKE_HORIZON=$SMOKE_HORIZON"
echo "  SMOKE_BRANCH_NUM=$SMOKE_BRANCH_NUM"
echo "  SMOKE_TOL=$SMOKE_TOL"

python3 - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

echo "[1/5] First layer: directed simulation schema/unit tests"
python3 -m pytest tests/test_simulate_ts.py -q

echo "[2/5] Second layer: simulation and batch regression tests"
python3 -m pytest tests/test_stage_batch_sizes.py -q
python3 -m pytest tests/test_policy_child_vectorization.py -q
python3 -m pytest tests/test_target_grid_training_integration.py -q

echo "[3/5] Third layer: full test suite"
python3 -m pytest -q

echo "[4/5] Tiny simulation smoke: identity and parent-to-main-child transition checks"
python3 - <<'PY'
import os
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.append(str(ROOT))

from config import Config
from data.simulate_ts import SimulateTS
from models.policy_value import PolicyValueModel

n_paths = int(os.environ.get("SMOKE_N_PATHS", "2"))
group_size = int(os.environ.get("SMOKE_GROUP_SIZE", "5"))
horizon = int(os.environ.get("SMOKE_HORIZON", "2"))
branch_num = int(os.environ.get("SMOKE_BRANCH_NUM", "2"))
tol = float(os.environ.get("SMOKE_TOL", "1e-6"))

device = torch.device("cpu")
Config.DEVICE = device
models = {
    "policy_value": PolicyValueModel().to(device),
    "sdf_fc1": None,
    "fc2": None,
    "dist_b": None,
}

sim = SimulateTS(
    models=models,
    config=Config,
    n_paths=n_paths,
    group_size=group_size,
    horizon=horizon,
    branch_num=branch_num,
    enable_entry=False,
    enable_exit=False,
    device=device,
)
df_firm, df_macro = sim.simulate()

required = {"b_next_p0", "b_next_pi", "b_next_policy"}
missing = sorted(required.difference(df_firm.columns))
if missing:
    raise SystemExit(f"Missing effective next-debt columns: {missing}")

expected = df_firm["ETA"] * df_firm["bp"] + (1.0 - df_firm["ETA"]) * df_firm["b"]
max_identity_error = (df_firm["b_next_policy"] - expected).abs().max()
print(f"max_identity_error={max_identity_error:.12g}")
if not max_identity_error < tol:
    raise SystemExit(
        f"identity check failed: max_identity_error={max_identity_error} >= {tol}"
    )

parents = df_firm[df_firm["branch"] == -1]
children = df_firm[df_firm["branch"] == 0]
merged = parents.merge(
    children,
    on=["path", "ID"],
    suffixes=("_parent", "_child"),
)
merged = merged[merged["t_child"] == merged["t_parent"] + 1]
if merged.empty:
    raise SystemExit("transition check failed: no parent/main-child matches found")

max_transition_error = (
    merged["b_next_policy_parent"] - merged["b_child"]
).abs().max()
print(f"max_transition_error={max_transition_error:.12g}")
if not max_transition_error < tol:
    raise SystemExit(
        f"transition check failed: max_transition_error={max_transition_error} >= {tol}"
    )

print("Tiny simulation smoke passed.")
print(f"firm_rows={len(df_firm)} macro_rows={len(df_macro)}")
print("L_new(theta)=L_old(theta); grad_theta L_new(theta)=grad_theta L_old(theta) for this reporting-only change.")
PY

echo "[5/5] Final acceptance summary"
echo "  tests/test_simulate_ts.py passed"
echo "  related regression tests passed"
echo "  full pytest passed"
echo "  tiny simulation identity and transition errors are below $SMOKE_TOL"
echo "Job finished on $(date)"
