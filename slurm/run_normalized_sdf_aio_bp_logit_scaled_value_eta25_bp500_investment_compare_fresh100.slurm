#!/bin/bash
#SBATCH -J dl-entry-fresh100
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_entry_fresh100_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_entry_fresh100_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
STAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_SCRIPT="slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100.slurm"
DEFAULT_REQUIRED_COMMIT="$(
  git -C "$REPO_DIR" log -1 --format=%H -- "$EXPERIMENT_SCRIPT"
)"
if [[ -z "$DEFAULT_REQUIRED_COMMIT" ]]; then
  echo "ERROR: cannot resolve committed experiment revision for $EXPERIMENT_SCRIPT"
  exit 1
fi
export REQUIRED_COMMIT="${REQUIRED_COMMIT:-$DEFAULT_REQUIRED_COMMIT}"

export RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100_${STAMP}}"
# The first run is intentionally the Episode 0-3 smoke. Override explicitly
# after that comparison is accepted.
export N_EPISODES="${N_EPISODES:-4}"

# Matched model-only initialization: reuse the same weights as the State Bank
# experiment while recreating optimizers, RNG, simulation state, and Episode 0.
WARMSTART_RUN_ROOT="${WARMSTART_RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_20261008_184305}"
export INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$WARMSTART_RUN_ROOT/checkpoints/ep9_combined.pt}"
export INITIAL_CHECKPOINT_EPISODE="${INITIAL_CHECKPOINT_EPISODE:-9}"
export INITIAL_CHECKPOINT_MODE="${INITIAL_CHECKPOINT_MODE:-model_init}"
export START_EPISODE="${START_EPISODE:-0}"

if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "ERROR: warm-start checkpoint not found: $INITIAL_CHECKPOINT"
  exit 1
fi

# Dedicated Fresh100 ablation. Every post-bootstrap episode starts from a new
# broad cross-section and runs the ordinary Mode-B simulator for 100 periods.
export SIMULATE_GROUP_SIZE=200
export SIMULATE_HORIZON="${SIMULATE_HORIZON:-100}"
export POST0_MODE=modeb
export STATE_BANK_ENABLED=0
export STATE_BANK_SDF_PUBLICATION_GATE_ENABLED=0
export STATE_BANK_PQ_DECOUPLING_ENABLED=0
export STATE_BANK_ECONOMIC_FEASIBILITY_FILTER_ENABLED=0
export STATE_BANK_MIN_UNIQUE_LINEAGES=0

# Restore the historical long-simulation mixture: 80% simulated/natural rows
# plus 20% generic coverage. The State Bank-specific 80% coverage ratio is
# inert because both State Bank and its P/Q decoupling are disabled.
export PV_MIXTURE_ENABLED=1
export PV_MIXTURE_RATIO=0.20

# Allow numerically safe SDF states to continue across outer episodes while
# keeping the separate final-convergence tolerances unchanged.
export SDF_CONTINUE_CONSTRAINT_TOL="${SDF_CONTINUE_CONSTRAINT_TOL:-1}"

# These identity-based RNG modes are simulator features, not State Bank-only
# features. Match the formal State Bank experiment to remove RNG confounds.
export TRANSITION_RNG_MODE=stable_firm_identity
export COMMON_TRANSITION_SEED=424242
export ENTRY_RNG_MODE=stable_economic_identity
export PRESERVE_GLOBAL_RNG_AROUND_ENTRY=1

# Preserve the current investment-compare entry and accounting specification.
export ENTRY_MODE=investment_compare
export ENTRY_CAPITAL_RATIO=0.10
export ENTRY_SIZE_RATIO=1.0
export ENTRY_COST_MAX=""
export ENTRY_DUMMY_I=0.0
export ENTRY_INFERENCE_CHUNK_SIZE="${ENTRY_INFERENCE_CHUNK_SIZE:-65536}"
export ENTRY_RNG_SEED="${ENTRY_RNG_SEED:-86420}"
export CONSUMPTION_AGGREGATION_MODE=raw
export CONSUMPTION_FLOOR=1e-4
export NODE_ACCOUNTING_MODE=economic_node_ledger
# Keep the authoritative first-pass ledger and all emitted accounting fields;
# only the optional rebuilt second-pass validator remains disabled.
export ENTRY_INDEPENDENT_RESOURCE_VALIDATION_ENABLED=0

echo "Dispatching Fresh100 no-State-Bank investment-compare baseline"
echo "RUN_ROOT=$RUN_ROOT"
echo "REQUIRED_COMMIT=$REQUIRED_COMMIT"
echo "N_EPISODES=$N_EPISODES"
echo "SIMULATE_GROUP_SIZE=$SIMULATE_GROUP_SIZE"
echo "SIMULATE_HORIZON=$SIMULATE_HORIZON"
echo "POST0_MODE=$POST0_MODE"
echo "STATE_BANK_ENABLED=$STATE_BANK_ENABLED"
echo "STATE_BANK_PQ_DECOUPLING_ENABLED=$STATE_BANK_PQ_DECOUPLING_ENABLED"
echo "STATE_BANK_ECONOMIC_FEASIBILITY_FILTER_ENABLED=$STATE_BANK_ECONOMIC_FEASIBILITY_FILTER_ENABLED"
echo "PV_MIXTURE_ENABLED=$PV_MIXTURE_ENABLED"
echo "PV_MIXTURE_RATIO=$PV_MIXTURE_RATIO"
echo "ENTRY_MODE=$ENTRY_MODE"
echo "TRANSITION_RNG_MODE=$TRANSITION_RNG_MODE"
echo "COMMON_TRANSITION_SEED=$COMMON_TRANSITION_SEED"
echo "ENTRY_RNG_MODE=$ENTRY_RNG_MODE"
echo "PRESERVE_GLOBAL_RNG_AROUND_ENTRY=$PRESERVE_GLOBAL_RNG_AROUND_ENTRY"
echo "SDF_CONTINUE_CONSTRAINT_TOL=$SDF_CONTINUE_CONSTRAINT_TOL"
echo "INITIAL_CHECKPOINT=$INITIAL_CHECKPOINT"
echo "INITIAL_CHECKPOINT_EPISODE=$INITIAL_CHECKPOINT_EPISODE"
echo "INITIAL_CHECKPOINT_MODE=$INITIAL_CHECKPOINT_MODE"
echo "START_EPISODE=$START_EPISODE"

exec bash "$REPO_DIR/slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare.slurm"
