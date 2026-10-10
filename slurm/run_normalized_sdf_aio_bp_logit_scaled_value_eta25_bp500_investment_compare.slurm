#!/bin/bash
#SBATCH -J dl-entry-invest-compare
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_entry_invest_compare_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_entry_invest_compare_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
STAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_SCRIPT="slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare.slurm"
DEFAULT_REQUIRED_COMMIT="$(
  git -C "$REPO_DIR" log -1 --format=%H -- "$EXPERIMENT_SCRIPT"
)"
if [[ -z "$DEFAULT_REQUIRED_COMMIT" ]]; then
  echo "ERROR: cannot resolve the committed experiment revision for $EXPERIMENT_SCRIPT"
  exit 1
fi
export REQUIRED_COMMIT="${REQUIRED_COMMIT:-$DEFAULT_REQUIRED_COMMIT}"

export RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_${STAMP}}"
export N_EPISODES="${N_EPISODES:-10}"

# Entry approximation: screen at b=0, eta=0, i=e using physical PI >= P0.
export ENTRY_MODE=investment_compare
export ENTRY_CAPITAL_RATIO="${ENTRY_CAPITAL_RATIO:-0.10}"
export ENTRY_SIZE_RATIO="${ENTRY_SIZE_RATIO:-0.10}"
export ENTRY_COST_MAX=""
export ENTRY_DUMMY_I=0.0
export ENTRY_INFERENCE_CHUNK_SIZE="${ENTRY_INFERENCE_CHUNK_SIZE:-65536}"
export ENTRY_RNG_SEED="${ENTRY_RNG_SEED:-86420}"
export CONSUMPTION_AGGREGATION_MODE=raw
export NODE_ACCOUNTING_MODE=economic_node_ledger

echo "Dispatching 10-episode investment-compare full-equilibrium run"
echo "RUN_ROOT=$RUN_ROOT"
echo "REQUIRED_COMMIT=$REQUIRED_COMMIT"
echo "N_EPISODES=$N_EPISODES"
echo "ENTRY_MODE=$ENTRY_MODE"
echo "ENTRY_COST_MAX=ordinary I_THRESHOLD"
echo "CONSUMPTION_AGGREGATION_MODE=$CONSUMPTION_AGGREGATION_MODE"
echo "NODE_ACCOUNTING_MODE=$NODE_ACCOUNTING_MODE"

exec bash "$REPO_DIR/slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500.slurm"
