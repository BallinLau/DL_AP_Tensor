#!/bin/bash
#SBATCH -J bp-phase6-full-recompute
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/bp_phase6_full_recompute_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/bp_phase6_full_recompute_error.%j

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_full_20260712_174101}"

export WORK_ROOT
export REPO_DIR
export RUN_ROOT
export FIRM_TEMPLATE="$RUN_ROOT/data/outputs/ep{ep}_stage_modeb.pkl"
export EPISODES="${EPISODES:-0 1 2}"
export REFERENCE_EPISODE="${REFERENCE_EPISODE:-2}"
export N_STATES="${N_STATES:-100}"
export REFIT_STEPS="${REFIT_STEPS:-1000}"
export B_GRID_SIZE="${B_GRID_SIZE:-41}"
export Z_GRID_SIZE="${Z_GRID_SIZE:-81}"
export RUN_RELATED_TESTS="${RUN_RELATED_TESTS:-1}"
export ALLOW_DIRTY_WORKTREE="${ALLOW_DIRTY_WORKTREE:-1}"
export BAR_I_MODE="${BAR_I_MODE:-recompute_issue_state}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/data/outputs/bp_phase6_full_recompute_issue_state}"

exec bash "$REPO_DIR/slurm/run_bp_three_experiments_gpu.slurm"
