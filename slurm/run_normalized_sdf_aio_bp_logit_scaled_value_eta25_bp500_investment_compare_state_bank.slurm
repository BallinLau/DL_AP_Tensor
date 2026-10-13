#!/bin/bash
#SBATCH -J dl-entry-state-bank
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_entry_state_bank_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_entry_state_bank_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
STAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_SCRIPT="slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_state_bank.slurm"
DEFAULT_REQUIRED_COMMIT="$(
  git -C "$REPO_DIR" log -1 --format=%H -- "$EXPERIMENT_SCRIPT"
)"
if [[ -z "$DEFAULT_REQUIRED_COMMIT" ]]; then
  echo "ERROR: cannot resolve committed experiment revision for $EXPERIMENT_SCRIPT"
  exit 1
fi
export REQUIRED_COMMIT="${REQUIRED_COMMIT:-$DEFAULT_REQUIRED_COMMIT}"

export RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_state_bank_${STAMP}}"
export N_EPISODES="${N_EPISODES:-10}"

# Initialize model weights from the completed scaled-value eta25/bp500 run,
# while keeping this experiment's simulation state and optimizers fresh from
# Episode 0.  `model_init` is intentional: continuation mode would resume at
# Episode 10 and would not run the new State Bank experiment from Episode 0.
WARMSTART_RUN_ROOT="${WARMSTART_RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_20261008_184305}"
export INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$WARMSTART_RUN_ROOT/checkpoints/ep9_combined.pt}"
export INITIAL_CHECKPOINT_EPISODE="${INITIAL_CHECKPOINT_EPISODE:-9}"
export INITIAL_CHECKPOINT_MODE="${INITIAL_CHECKPOINT_MODE:-model_init}"
export START_EPISODE="${START_EPISODE:-0}"

# Episode 0 always uses the runner's fixed two-firm bootstrap.  Keep the fresh
# post-0 simulation small for this retry so Episode 1 starts with 20 firms per
# path instead of the inherited baseline setting.
export SIMULATE_GROUP_SIZE="${SIMULATE_GROUP_SIZE:-20}"

if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "ERROR: warm-start checkpoint not found: $INITIAL_CHECKPOINT"
  exit 1
fi

# State Bank publication requires safe SDF progress, while final SDF convergence
# remains diagnostic. Keep this wrapper in each formal State Bank code revision
# so its self-pin resolves to the exact implementation under test. Economic
# entry settings are unchanged.
export STATE_BANK_ENABLED=1
export STATE_BANK_SHORT_HORIZON="${STATE_BANK_SHORT_HORIZON:-10}"
export STATE_BANK_BOOTSTRAP_BURN_IN="${STATE_BANK_BOOTSTRAP_BURN_IN:-20}"
export STATE_BANK_BOOTSTRAP_CAPTURE_STRIDE="${STATE_BANK_BOOTSTRAP_CAPTURE_STRIDE:-5}"
export STATE_BANK_REFRESH_BURN_IN="${STATE_BANK_REFRESH_BURN_IN:-2}"
export STATE_BANK_REFRESH_CAPTURE_STRIDE="${STATE_BANK_REFRESH_CAPTURE_STRIDE:-1}"
export STATE_BANK_PARENT_TRANSITION_BUDGET="${STATE_BANK_PARENT_TRANSITION_BUDGET:-40000}"
export STATE_BANK_MAX_ENTRIES="${STATE_BANK_MAX_ENTRIES:-5000}"
export STATE_BANK_SAMPLING_MODE="${STATE_BANK_SAMPLING_MODE:-uniform_lineage}"
export STATE_BANK_ACCEPTANCE_MODE="${STATE_BANK_ACCEPTANCE_MODE:-strict}"
export STATE_BANK_MIN_UNIQUE_LINEAGES="${STATE_BANK_MIN_UNIQUE_LINEAGES:-200}"
export STATE_BANK_SAMPLE_SEED="${STATE_BANK_SAMPLE_SEED:-271828}"
export STATE_BANK_REFRESH_SEED="${STATE_BANK_REFRESH_SEED:-314159}"
export STATE_BANK_RESERVOIR_SEED="${STATE_BANK_RESERVOIR_SEED:-161803}"

# This experiment intentionally allows wider SDF moment deviations between
# outer episodes; final SDF convergence tolerances remain unchanged.
export SDF_CONTINUE_CONSTRAINT_TOL="${SDF_CONTINUE_CONSTRAINT_TOL:-1.0}"

export TRANSITION_RNG_MODE=stable_firm_identity
export COMMON_TRANSITION_SEED="${COMMON_TRANSITION_SEED:-424242}"
export ENTRY_RNG_MODE=stable_economic_identity
export PRESERVE_GLOBAL_RNG_AROUND_ENTRY=1

export ENTRY_MODE=investment_compare
export ENTRY_CAPITAL_RATIO="${ENTRY_CAPITAL_RATIO:-0.10}"
export ENTRY_SIZE_RATIO=1.0
export ENTRY_COST_MAX=""
export ENTRY_DUMMY_I=0.0
export ENTRY_INFERENCE_CHUNK_SIZE="${ENTRY_INFERENCE_CHUNK_SIZE:-65536}"
export ENTRY_RNG_SEED="${ENTRY_RNG_SEED:-86420}"
export CONSUMPTION_AGGREGATION_MODE=raw
export CONSUMPTION_FLOOR="${CONSUMPTION_FLOOR:-1e-4}"
export NODE_ACCOUNTING_MODE=economic_node_ledger

echo "Dispatching State Bank v2 investment-compare full-equilibrium run"
echo "RUN_ROOT=$RUN_ROOT"
echo "REQUIRED_COMMIT=$REQUIRED_COMMIT"
echo "INITIAL_CHECKPOINT=$INITIAL_CHECKPOINT"
echo "INITIAL_CHECKPOINT_EPISODE=$INITIAL_CHECKPOINT_EPISODE"
echo "INITIAL_CHECKPOINT_MODE=$INITIAL_CHECKPOINT_MODE"
echo "START_EPISODE=$START_EPISODE"
echo "SIMULATE_GROUP_SIZE=$SIMULATE_GROUP_SIZE"
echo "STATE_BANK_PARENT_TRANSITION_BUDGET=$STATE_BANK_PARENT_TRANSITION_BUDGET"
echo "STATE_BANK_SAMPLING_MODE=$STATE_BANK_SAMPLING_MODE"
echo "STATE_BANK_ACCEPTANCE_MODE=$STATE_BANK_ACCEPTANCE_MODE"
echo "SDF_CONTINUE_CONSTRAINT_TOL=$SDF_CONTINUE_CONSTRAINT_TOL"
echo "TRANSITION_RNG_MODE=$TRANSITION_RNG_MODE"
echo "ENTRY_RNG_MODE=$ENTRY_RNG_MODE"

exec bash "$REPO_DIR/slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare.slurm"
