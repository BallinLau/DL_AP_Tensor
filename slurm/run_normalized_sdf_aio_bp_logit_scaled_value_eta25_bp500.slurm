#!/bin/bash
#SBATCH -J dl-scaled-eta25-bp500
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/dl_scaled_eta25_bp500_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/dl_scaled_eta25_bp500_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
STAMP=$(date +%Y%m%d_%H%M%S)

# Preserve the scaled-value full-equilibrium baseline and change only the BP
# current-parent-eta training distribution and successful-step budget.
export RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_${STAMP}}"
export BP_CURRENT_ETA_RESAMPLE_ENABLED=1
export BP_CURRENT_ETA1_TRAIN_SHARE=0.25
export BP_DISTILL_MAX_OPTIMIZER_STEPS=500
export BP_DISTILL_PATIENCE=1000000
export BP_DISTILL_TRAINABLE_SCOPE=heads_only
export PV_ETA_RESAMPLE_ENABLED=0
export PV_EXACT_ETA_INTEGRATION_ENABLED=1
export PV_MIXTURE_ENABLED=1

echo "Dispatching full scaled-value equilibrium training with BP current-eta sampling"
echo "RUN_ROOT=$RUN_ROOT"
echo "BP_CURRENT_ETA_RESAMPLE_ENABLED=$BP_CURRENT_ETA_RESAMPLE_ENABLED"
echo "BP_CURRENT_ETA1_TRAIN_SHARE=$BP_CURRENT_ETA1_TRAIN_SHARE"
echo "BP_DISTILL_MAX_OPTIMIZER_STEPS=$BP_DISTILL_MAX_OPTIMIZER_STEPS"
echo "BP_DISTILL_TRAINABLE_SCOPE=$BP_DISTILL_TRAINABLE_SCOPE"
echo "PV_EXACT_ETA_INTEGRATION_ENABLED=$PV_EXACT_ETA_INTEGRATION_ENABLED"
echo "PV_MIXTURE_ENABLED=$PV_MIXTURE_ENABLED"
echo "current eta_t train sampling share = 0.25"
echo "future eta_{t+1} economic probability remains ZETA = 0.03"

exec bash "$REPO_DIR/slurm/run_normalized_sdf_aio_bp_logit_scaled_value_pilot.slurm"
