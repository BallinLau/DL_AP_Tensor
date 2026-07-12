#!/bin/bash
#SBATCH -J dl-target-grid-diag-smoke
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_target_grid_diag_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_target_grid_diag_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
RUN_FULL_TESTS="${RUN_FULL_TESTS:-0}"

# Documentation-only smoke scale for this diagnostic job. The field checks
# below construct an even smaller synthetic batch and do not train to
# convergence.
N_PATHS="${N_PATHS:-16}"
SIMULATE_HORIZON="${SIMULATE_HORIZON:-3}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BP_GRID_COARSE_SIZE="${BP_GRID_COARSE_SIZE:-3}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Python: $(which python3)"
echo "Smoke scale:"
echo "  N_PATHS=$N_PATHS"
echo "  SIMULATE_HORIZON=$SIMULATE_HORIZON"
echo "  EPOCHS=$EPOCHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  BP_GRID_COARSE_SIZE=$BP_GRID_COARSE_SIZE"
echo "  RUN_FULL_TESTS=$RUN_FULL_TESTS"

python3 - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device 0:", torch.cuda.get_device_name(0))
PY

echo "Running directed target-grid diagnostics test..."
python3 -m pytest tests/test_target_grid_training_integration.py -q

echo "Running related regression tests..."
python3 -m pytest tests/test_bp_grid_teacher.py -q
python3 -m pytest tests/test_policy_child_vectorization.py -q
python3 -m pytest tests/test_stage_batch_sizes.py -q

echo "Checking explicit P0/PI/mix active-refinancing diagnostic fields..."
python3 - <<'PY'
import math
import torch

from config import Config
from config.hyperparams import HyperParams
from models.policy_value import PolicyValueModel
from training.episode import Episode


def small_hyperparams() -> HyperParams:
    hp = HyperParams()
    hp.pv_bp_training_mode = "target_grid"
    hp.bp_grid_coarse_size = 3
    hp.bp_grid_refine_enabled = False
    hp.bp_grid_candidate_chunk_size = 2
    hp.bp_grid_parent_chunk_size = 2
    hp.bp_grid_max_expanded_states = 16
    hp.bp_grid_mix_policy_weight = 1.0
    hp.bp_grid_margin_scale = 1e-4
    hp.bp_grid_confidence_relative = True
    hp.pv_use_clipped_m = False
    hp.firm_target_update = "epoch_hard"
    hp.epochs = 1
    hp.batch_size = 2
    return hp


def small_batch(device: torch.device):
    parent_state = torch.tensor(
        [
            [0.10, 0.20, 1.00, 0.20, 0.10, -2.00, 4.00],
            [0.30, 0.10, 0.00, 0.40, 0.00, -1.80, 4.20],
        ],
        device=device,
        dtype=torch.float32,
    )
    child0 = parent_state.clone()
    child1 = parent_state.clone()
    child0[:, 1:2] = child0[:, 1:2] + 0.05
    child1[:, 1:2] = child1[:, 1:2] - 0.03
    child0[:, 2:3] = torch.tensor([[1.0], [0.0]], device=device)
    child1[:, 2:3] = torch.tensor([[0.0], [1.0]], device=device)
    m0 = torch.full((2, 1), 0.98, device=device)
    m1 = torch.full((2, 1), 1.02, device=device)
    return {
        "parent": torch.cat([parent_state, torch.ones(2, 1, device=device)], dim=1),
        "children": [
            torch.cat([child0, m0], dim=1),
            torch.cat([child1, m1], dim=1),
        ],
    }


device = torch.device("cpu")
Config.DEVICE = device
online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
target = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
episode = Episode(
    models={"policy_value": online},
    optimizers={"policy_value": torch.optim.Adam(online.parameters(), lr=1e-3)},
    config=Config,
    hyperparams=small_hyperparams(),
    device=device,
    firm_target=target,
)

batch = small_batch(device)
loss = episode._compute_p0_loss(batch) + episode._compute_pi_loss(batch)
if not torch.isfinite(loss):
    raise SystemExit("target-grid smoke loss is not finite")

terms = {}
terms.update(episode._latest_p0_terms)
terms.update(episode._latest_pi_terms)

required_keys = [
    "p0_grid_refi_active_n",
    "pi_grid_refi_active_n",
    "mix_grid_refi_active_n",
    "p0_grid_bp_mae_active",
    "pi_grid_bp_mae_active",
    "mix_grid_bp_mae_active",
    "p0_grid_bp_star_mean_active",
    "pi_grid_bp_star_mean_active",
    "mix_grid_bp_star_mean_active",
    "p0_grid_bp_candidate_mean_active",
    "pi_grid_bp_candidate_mean_active",
    "mix_grid_bp_candidate_mean_active",
    "p0_grid_default_at_star_mean_active",
    "p0_grid_p_child_at_star_mean_active",
    "p0_grid_q_issue_at_star_mean_active",
]

for key in required_keys:
    if key not in terms:
        raise SystemExit(f"missing diagnostic field: {key}")
    value = float(terms[key])
    if not math.isfinite(value):
        raise SystemExit(f"non-finite diagnostic field: {key}={value}")
    print(f"{key}={value:.8g}")

if terms["p0_grid_refi_active_n"] <= 0.0:
    raise SystemExit("p0 active refinancing diagnostic did not observe active states")
if terms["pi_grid_refi_active_n"] <= 0.0:
    raise SystemExit("pi active refinancing diagnostic did not observe active states")
if terms["mix_grid_refi_active_n"] <= 0.0:
    raise SystemExit("mix active refinancing diagnostic did not observe active states")

print("Target-grid active diagnostic smoke check passed.")
PY

if [[ "$RUN_FULL_TESTS" == "1" ]]; then
  echo "Running full test suite..."
  python3 -m pytest -q
else
  echo "Skipping full test suite. Set RUN_FULL_TESTS=1 to run python3 -m pytest -q."
fi

echo "Job finished on $(date)"
