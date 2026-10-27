#!/bin/bash
#SBATCH -J ep32-42-eta1baug
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/ep32to42_eta1baug_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/ep32to42_eta1baug_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

# EP32-EP42 two-arm validation of the P-stage eta=1 b-uniform augmentation.
#
# Arm selection (env AUGMENT_ARM):
#   control    -> PV_ETA1_B_AUGMENT_MODE=none   (exact EP19-28 continuation)
#   treatment  -> PV_ETA1_B_AUGMENT_MODE=uniform, share=0.5
#                (half of the eta=1 collocation half drawn from the global
#                 train pool with inverse-density weights over parent b)
#
# Everything else is the EP19-28 configuration verbatim: hybrid formal runner
# (frozen-policy dynamic-residual P + EP3 staged Q), P-only counterfactual
# parent eta balance 50/50, Adam-only P root, h-star recalibration every 10
# steps (HSTAR_RECALIB_COUNT=4), 200 paths x 50 periods, Grid BP policy.
# Initial checkpoint = ep31_combined.pt of the EP29-31 eta1baug treatment run
# (START_EPISODE=32).
#
# Usage:
#   AUGMENT_ARM=control   sbatch slurm/run_ep32to42_eta1_baugment.slurm
#   AUGMENT_ARM=treatment sbatch slurm/run_ep32to42_eta1_baugment.slurm

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/worktrees/train_hstar4}"
export WORK_ROOT REPO_DIR

AUGMENT_ARM="${AUGMENT_ARM:-control}"
if [[ "$AUGMENT_ARM" == "treatment" ]]; then
  export PV_ETA1_B_AUGMENT_MODE="${PV_ETA1_B_AUGMENT_MODE:-uniform}"
  export PV_ETA1_B_AUGMENT_SHARE="${PV_ETA1_B_AUGMENT_SHARE:-0.5}"
elif [[ "$AUGMENT_ARM" == "control" ]]; then
  export PV_ETA1_B_AUGMENT_MODE="${PV_ETA1_B_AUGMENT_MODE:-none}"
else
  echo "ERROR: AUGMENT_ARM must be 'control' or 'treatment' (got: $AUGMENT_ARM)"
  exit 1
fi
export PV_ETA1_B_AUGMENT_BINS="${PV_ETA1_B_AUGMENT_BINS:-10}"
export PV_ETA1_B_AUGMENT_SEED="${PV_ETA1_B_AUGMENT_SEED:-20261101}"

STAMP=$(date +%Y%m%d_%H%M%S)
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/ep29to31_eta1baug_treatment_20261027_195039}"
export RUN_ROOT="${RUN_ROOT:-/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/ep32to42_eta1baug_${AUGMENT_ARM}_${STAMP}}"

# --- EP19-31 settings inherited verbatim (do not edit without a new arm) ---
export N_EPISODES="${N_EPISODES:-11}"
export INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$SOURCE_RUN_ROOT/checkpoints/ep31_combined.pt}"
export INITIAL_CHECKPOINT_EPISODE="${INITIAL_CHECKPOINT_EPISODE:-31}"
export INITIAL_CHECKPOINT_MODE="${INITIAL_CHECKPOINT_MODE:-continuation}"
export START_EPISODE="${START_EPISODE:-32}"
export RESUME_CLASSIFICATION="${RESUME_CLASSIFICATION:-WEIGHTS_ONLY_CONTINUATION}"
export HSTAR_RECALIB_COUNT="${HSTAR_RECALIB_COUNT:-4}"
# P-only counterfactual parent eta balance (50/50), exactly as EP19-28.
export PV_CURRENT_ETA_BALANCE_ENABLED="${PV_CURRENT_ETA_BALANCE_ENABLED:-1}"
export PV_CURRENT_ETA1_TRAIN_SHARE="${PV_CURRENT_ETA1_TRAIN_SHARE:-0.50}"
export PV_CURRENT_ETA_BALANCE_VALIDATION="${PV_CURRENT_ETA_BALANCE_VALIDATION:-1}"
export PV_CURRENT_ETA_BALANCE_SEED="${PV_CURRENT_ETA_BALANCE_SEED:-97531}"
export P_ETA_NATURAL_DIAG_ENABLED="${P_ETA_NATURAL_DIAG_ENABLED:-1}"
export P_ETA_NATURAL_DIAG_MAX_PARENTS="${P_ETA_NATURAL_DIAG_MAX_PARENTS:-4096}"
export P_ETA_NATURAL_DIAG_CHUNK_SIZE="${P_ETA_NATURAL_DIAG_CHUNK_SIZE:-512}"
export P_ETA_NATURAL_DIAG_FREQUENCY="${P_ETA_NATURAL_DIAG_FREQUENCY:-firm_inner_end}"
export P_ETA_NATURAL_DIAG_SEED="${P_ETA_NATURAL_DIAG_SEED:-97531}"
export PV_ETA_RESAMPLE_ENABLED="${PV_ETA_RESAMPLE_ENABLED:-0}"
export PV_EXACT_ETA_INTEGRATION_ENABLED="${PV_EXACT_ETA_INTEGRATION_ENABLED:-1}"
export BP_CURRENT_ETA_RESAMPLE_ENABLED="${BP_CURRENT_ETA_RESAMPLE_ENABLED:-0}"
export PV_BP_HEAD_TRAINING_ENABLED="${PV_BP_HEAD_TRAINING_ENABLED:-0}"
export SIMULATION_BP_ACTION_SOURCE="${SIMULATION_BP_ACTION_SOURCE:-grid}"
export PV_BP_DRIFT_MODE="${PV_BP_DRIFT_MODE:-full}"
export P_BP_DRIFT_SUBSET_ROWS="${P_BP_DRIFT_SUBSET_ROWS:-4096}"
export PV_FRESH_INITIAL_SOURCE="${PV_FRESH_INITIAL_SOURCE:-resolve}"

if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "ERROR: ep31 source checkpoint not found: $INITIAL_CHECKPOINT"
  exit 1
fi

echo "=== EP32-42 eta=1 b-augmentation two-arm run ==="
echo "AUGMENT_ARM=$AUGMENT_ARM"
echo "PV_ETA1_B_AUGMENT_MODE=$PV_ETA1_B_AUGMENT_MODE"
echo "PV_ETA1_B_AUGMENT_SHARE=$PV_ETA1_B_AUGMENT_SHARE"
echo "PV_ETA1_B_AUGMENT_BINS=$PV_ETA1_B_AUGMENT_BINS"
echo "PV_ETA1_B_AUGMENT_SEED=$PV_ETA1_B_AUGMENT_SEED"
echo "INITIAL_CHECKPOINT=$INITIAL_CHECKPOINT"
echo "START_EPISODE=$START_EPISODE N_EPISODES=$N_EPISODES"
echo "HSTAR_RECALIB_COUNT=$HSTAR_RECALIB_COUNT"
echo "RUN_ROOT=$RUN_ROOT"
echo "REPO_DIR=$REPO_DIR"

# Delegate to the EP3 Hybrid Formal Runner chain (Inner3 + h-star + Grid BP).
exec bash "$REPO_DIR/slurm/run_ep3_baseline_inner3_initial_hstar.slurm"
