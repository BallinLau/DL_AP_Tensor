#!/bin/bash
#SBATCH -J bp-phase6-smoke-fixed
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/bp_phase6_smoke_fixed_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/bp_phase6_smoke_fixed_error.%j

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_full_20260712_174101}"

export WORK_ROOT
export REPO_DIR
export RUN_ROOT
export EPISODES="${EPISODES:-2}"
export REFERENCE_EPISODE="${REFERENCE_EPISODE:-2}"
export N_STATES="${N_STATES:-5}"
export REFIT_STEPS="${REFIT_STEPS:-20}"
export B_GRID_SIZE="${B_GRID_SIZE:-11}"
export Z_GRID_SIZE="${Z_GRID_SIZE:-21}"
export RUN_RELATED_TESTS="${RUN_RELATED_TESTS:-1}"
export BAR_I_MODE="${BAR_I_MODE:-fixed_parent}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/data/outputs/bp_phase6_smoke_fixed_parent}"

exec bash "$REPO_DIR/slurm/run_bp_three_experiments_gpu.slurm"
