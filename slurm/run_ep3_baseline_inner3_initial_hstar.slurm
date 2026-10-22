#!/bin/bash
#SBATCH -J ep3-inner3-hstar
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/ep3_inner3_hstar_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/ep3_inner3_hstar_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

# Episode-4 continuation of the 2026-10-14 ep3 baseline.
#
# BASE_SOURCE_COMMIT=004a2ad
# INITIAL_SOURCE_EPISODE=3
# TRAINING_START_EPISODE=4
# FIRM_INNER_CYCLES=3
# INITIAL_HSTAR_ONLY_T0=true
# BASELINE_MODEB_FRESH100=true
# STATE_BANK_ENABLED=false
# NEW_JOINT_COVERAGE=false
# NEW_H10_H100=false
# NEW_DYNAMIC_EMPIRICAL=false
# NEW_REFERENCE_MACRO=false
#
# Everything except the two added features is inherited from the committed
# ep3 Fresh100 chain script, so no ep3 parameter can silently drift here.

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/worktrees/ep3_baseline_004a2ad}"
STAMP=$(date +%Y%m%d_%H%M%S)

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100_20261014_105652}"
SOURCE_EP3_CHECKPOINT="${SOURCE_EP3_CHECKPOINT:-$SOURCE_RUN_ROOT/checkpoints/ep3_combined.pt}"

export REPO_DIR
export REQUIRED_COMMIT="${REQUIRED_COMMIT:-$(git -C "$REPO_DIR" rev-parse HEAD)}"
export RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/ep3_fresh100_inner3_initial_hstar_${STAMP}}"

export N_EPISODES="${N_EPISODES:-1}"
export INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$SOURCE_EP3_CHECKPOINT}"
export INITIAL_CHECKPOINT_EPISODE="${INITIAL_CHECKPOINT_EPISODE:-3}"
export INITIAL_CHECKPOINT_MODE="${INITIAL_CHECKPOINT_MODE:-continuation}"
export START_EPISODE="${START_EPISODE:-4}"
export RESUME_CLASSIFICATION="${RESUME_CLASSIFICATION:-WEIGHTS_ONLY_CONTINUATION}"

# Feature A: three frozen-macro Firm inner value cycles per Episode.
export FIRM_INNER_CYCLES="${FIRM_INNER_CYCLES:-3}"
# Feature B: strict t=0 macro clearing; t>=1 keeps the FC1 forecast.
export INITIAL_HSTAR_ENABLED="${INITIAL_HSTAR_ENABLED:-1}"
export INITIAL_HSTAR_RESIDUAL_TOLERANCE="${INITIAL_HSTAR_RESIDUAL_TOLERANCE:-1e-5}"
export INITIAL_HSTAR_PARITY_TOLERANCE="${INITIAL_HSTAR_PARITY_TOLERANCE:-1e-5}"
export INITIAL_HSTAR_MAX_ITERATIONS="${INITIAL_HSTAR_MAX_ITERATIONS:-40}"

if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "ERROR: ep3 source checkpoint not found: $INITIAL_CHECKPOINT"
  exit 1
fi

echo "=== ep3 baseline + inner3 + initial h-star ==="
echo "BASE_SOURCE_COMMIT=004a2ad"
echo "REPO_DIR=$REPO_DIR"
echo "REQUIRED_COMMIT=$REQUIRED_COMMIT"
echo "INITIAL_CHECKPOINT=$INITIAL_CHECKPOINT"
echo "INITIAL_CHECKPOINT_EPISODE=$INITIAL_CHECKPOINT_EPISODE"
echo "INITIAL_CHECKPOINT_MODE=$INITIAL_CHECKPOINT_MODE"
echo "START_EPISODE=$START_EPISODE"
echo "N_EPISODES=$N_EPISODES"
echo "FIRM_INNER_CYCLES=$FIRM_INNER_CYCLES"
echo "INITIAL_HSTAR_ENABLED=$INITIAL_HSTAR_ENABLED"
echo "RESUME_CLASSIFICATION=$RESUME_CLASSIFICATION"

exec bash "$REPO_DIR/slurm/run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100.slurm"
