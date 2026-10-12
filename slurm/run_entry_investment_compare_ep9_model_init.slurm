#!/bin/bash
#SBATCH -J dl-entry-init-ep9
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_entry_init_ep9_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_entry_init_ep9_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
STAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_SCRIPT="slurm/run_entry_investment_compare_ep9_model_init.slurm"
DEFAULT_REQUIRED_COMMIT="$(
  git -C "$REPO_DIR" log -1 --format=%H -- "$EXPERIMENT_SCRIPT"
)"
if [[ -z "$DEFAULT_REQUIRED_COMMIT" ]]; then
  echo "ERROR: cannot resolve the committed experiment revision for $EXPERIMENT_SCRIPT"
  exit 1
fi
export REQUIRED_COMMIT="${REQUIRED_COMMIT:-$DEFAULT_REQUIRED_COMMIT}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_20261008_184305}"
export INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$SOURCE_RUN_ROOT/checkpoints/ep9_combined.pt}"
export INITIAL_CHECKPOINT_EPISODE="${INITIAL_CHECKPOINT_EPISODE:-9}"
export INITIAL_CHECKPOINT_MODE="model_init"
export START_EPISODE="0"
export N_EPISODES="${N_EPISODES:-10}"
export RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/investment_compare_ep9_model_init_${STAMP}}"
export ENTRY_INDEPENDENT_RESOURCE_VALIDATION_ENABLED=0

echo "Dispatching fresh Episode-0 training with checkpoint model initialization"
echo "SOURCE_RUN_ROOT=$SOURCE_RUN_ROOT"
echo "INITIAL_CHECKPOINT=$INITIAL_CHECKPOINT"
echo "INITIAL_CHECKPOINT_EPISODE=$INITIAL_CHECKPOINT_EPISODE"
echo "INITIAL_CHECKPOINT_MODE=$INITIAL_CHECKPOINT_MODE"
echo "START_EPISODE=$START_EPISODE"
echo "N_EPISODES=$N_EPISODES"
echo "RUN_ROOT=$RUN_ROOT"
echo "REQUIRED_COMMIT=$REQUIRED_COMMIT"
echo "ENTRY_INDEPENDENT_RESOURCE_VALIDATION_ENABLED=$ENTRY_INDEPENDENT_RESOURCE_VALIDATION_ENABLED"
echo "Restored: model weights only (policy/value, SDF/FC1, FC2, firm target)"
echo "Fresh: Episode numbering, simulation/data, optimizers, and RNG"

exec bash "$REPO_DIR/slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare.slurm"
