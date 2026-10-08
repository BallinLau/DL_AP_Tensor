#!/bin/bash
#SBATCH -J dl-scaled-grid-reference
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_scaled_grid_reference_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_scaled_grid_reference_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
STAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_SCRIPT="slurm/run_normalized_sdf_aio_bp_logit_scaled_value_grid_reference.slurm"
DEFAULT_REQUIRED_COMMIT="$(
  git -C "$REPO_DIR" log -1 --format=%H -- "$EXPERIMENT_SCRIPT"
)"
if [[ -z "$DEFAULT_REQUIRED_COMMIT" ]]; then
  echo "ERROR: cannot resolve the committed experiment revision for $EXPERIMENT_SCRIPT"
  exit 1
fi
export REQUIRED_COMMIT="${REQUIRED_COMMIT:-$DEFAULT_REQUIRED_COMMIT}"

export RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_grid_reference_${STAMP}}"
export PV_BP_HEAD_TRAINING_ENABLED=0
export SIMULATION_BP_ACTION_SOURCE=grid
export SIMULATION_BP_GRID_N_CHILD_SHOCKS="${SIMULATION_BP_GRID_N_CHILD_SHOCKS:-2}"
export SIMULATION_BP_GRID_SHOCK_SEED="${SIMULATION_BP_GRID_SHOCK_SEED:-314159}"

echo "Dispatching scaled-value GRID reference solver"
echo "RUN_ROOT=$RUN_ROOT"
echo "REQUIRED_COMMIT=$REQUIRED_COMMIT"
echo "PV_BP_HEAD_TRAINING_ENABLED=$PV_BP_HEAD_TRAINING_ENABLED"
echo "SIMULATION_BP_ACTION_SOURCE=$SIMULATION_BP_ACTION_SOURCE"
echo "SIMULATION_BP_GRID_N_CHILD_SHOCKS=$SIMULATION_BP_GRID_N_CHILD_SHOCKS"
echo "SIMULATION_BP_GRID_SHOCK_SEED=$SIMULATION_BP_GRID_SHOCK_SEED"
echo "BP head diagnostics remain enabled; BP head optimization is disabled."

exec bash "$REPO_DIR/slurm/run_normalized_sdf_aio_bp_logit_scaled_value_pilot.slurm"
