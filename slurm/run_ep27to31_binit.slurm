#!/bin/bash
#SBATCH -J binit
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/binit_%x_%j.out
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/binit_%x_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

# EP27-31 initial-b distribution experiment: 12 distribution versions + 1
# uniform control, one slurm job each (sbatch parallel).
#
# Every run continues from ep26_combined.pt (EP19-28 run), START_EPISODE=27,
# N_EPISODES=5, with the EP19-28 configuration verbatim (augmentation OFF).
# The ONLY difference between the 12 versions is the initial-b distribution:
# a frozen conditional-quantile generator maps common uniforms U through
# Q(u; c_v), where c_v is the fixed distribution code of version v
# (b_distribution_network_preview_20261109_v1/selected_codes.csv rows
# V01..V12, the max-min W1 diverse subset of 512 candidates).
#
# Paired design: all versions AND the control share SIM_B_INIT_SEED, so the
# same firm index sits at the same quantile of its own distribution across
# versions.  Per-episode U seed = SIM_B_INIT_SEED + episode (handled inside
# run_multi_episode_job).  The generator is frozen and never optimized.
#
# Arm selection (env BINIT_ARM):
#   V01..V12 -> SIM_B_INIT_SOURCE=conditional_quantile_network + code c_v
#   control  -> SIM_B_INIT_SOURCE=uniform (exact legacy SIM_B_INIT_MIN/MAX
#               fallback; the single-variable baseline)
#
# Usage (normally launched by launch_binit12_ep27to31.ipynb, which passes
# BINIT_ARM / SIM_B_INIT_* / RUN_ROOT explicitly):
#   sbatch --export=ALL,BINIT_ARM=V01,.. slurm/run_ep27to31_binit.slurm

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/worktrees/train_hstar4}"
export WORK_ROOT REPO_DIR

BINIT_ARM="${BINIT_ARM:?BINIT_ARM is required (V01..V12 or control)}"
if [[ "$BINIT_ARM" == "control" ]]; then
  export SIM_B_INIT_SOURCE="uniform"
else
  export SIM_B_INIT_SOURCE="conditional_quantile_network"
  : "${SIM_B_INIT_GENERATOR_CHECKPOINT:?SIM_B_INIT_GENERATOR_CHECKPOINT is required for $BINIT_ARM}"
  : "${SIM_B_INIT_DISTRIBUTION_CODE:?SIM_B_INIT_DISTRIBUTION_CODE is required for $BINIT_ARM}"
fi
export SIM_B_INIT_SEED="${SIM_B_INIT_SEED:-20261109}"

# eta=1 b-augmentation OFF for every arm: single-variable experiment where
# the only treatment is the initial-b distribution.
export PV_ETA1_B_AUGMENT_MODE="${PV_ETA1_B_AUGMENT_MODE:-none}"
export PV_ETA1_B_AUGMENT_SHARE="${PV_ETA1_B_AUGMENT_SHARE:-0.0}"
export PV_ETA1_B_AUGMENT_BINS="${PV_ETA1_B_AUGMENT_BINS:-10}"
export PV_ETA1_B_AUGMENT_SEED="${PV_ETA1_B_AUGMENT_SEED:-20261101}"

STAMP=$(date +%Y%m%d_%H%M%S)
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/ep19to28_p_fast_hstar4_20261026_023457}"
export RUN_ROOT="${RUN_ROOT:-/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/ep27to31_binit_${BINIT_ARM}_${STAMP}}"

# --- EP19-28 settings inherited verbatim (do not edit without a new arm) ---
export N_EPISODES="${N_EPISODES:-5}"
export INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-$SOURCE_RUN_ROOT/checkpoints/ep26_combined.pt}"
export INITIAL_CHECKPOINT_EPISODE="${INITIAL_CHECKPOINT_EPISODE:-26}"
export INITIAL_CHECKPOINT_MODE="${INITIAL_CHECKPOINT_MODE:-continuation}"
export START_EPISODE="${START_EPISODE:-27}"
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
export PV_BP_DRIFT_MODE="${PV_BP_DRIFT_MODE:-subset}"
export P_BP_DRIFT_SUBSET_ROWS="${P_BP_DRIFT_SUBSET_ROWS:-4096}"
export PV_FRESH_INITIAL_SOURCE="${PV_FRESH_INITIAL_SOURCE:-reuse_build}"
# Simulation budget / Adam-only P root (as run by the EP29+ launches).
export N_PATHS="${N_PATHS:-200}"
export POST0_N_PATHS="${POST0_N_PATHS:-200}"
export SIMULATE_HORIZON="${SIMULATE_HORIZON:-50}"
export P_ROOT_LBFGS_MAX_ITER="${P_ROOT_LBFGS_MAX_ITER:-0}"

if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "ERROR: ep26 source checkpoint not found: $INITIAL_CHECKPOINT"
  exit 1
fi

echo "=== EP27-31 initial-b distribution run ==="
echo "BINIT_ARM=$BINIT_ARM"
echo "SIM_B_INIT_SOURCE=$SIM_B_INIT_SOURCE"
echo "SIM_B_INIT_SEED=$SIM_B_INIT_SEED"
echo "SIM_B_INIT_GENERATOR_CHECKPOINT=$SIM_B_INIT_GENERATOR_CHECKPOINT"
echo "SIM_B_INIT_DISTRIBUTION_CODE=$SIM_B_INIT_DISTRIBUTION_CODE"
echo "PV_ETA1_B_AUGMENT_MODE=$PV_ETA1_B_AUGMENT_MODE"
echo "INITIAL_CHECKPOINT=$INITIAL_CHECKPOINT"
echo "START_EPISODE=$START_EPISODE N_EPISODES=$N_EPISODES"
echo "HSTAR_RECALIB_COUNT=$HSTAR_RECALIB_COUNT"
echo "N_PATHS=$N_PATHS POST0_N_PATHS=$POST0_N_PATHS SIMULATE_HORIZON=$SIMULATE_HORIZON"
echo "PV_BP_DRIFT_MODE=$PV_BP_DRIFT_MODE PV_FRESH_INITIAL_SOURCE=$PV_FRESH_INITIAL_SOURCE"
echo "P_ROOT_LBFGS_MAX_ITER=$P_ROOT_LBFGS_MAX_ITER"
echo "RUN_ROOT=$RUN_ROOT"
echo "REPO_DIR=$REPO_DIR"

# Delegate to the EP3 Hybrid Formal Runner chain (Inner3 + h-star + Grid BP).
exec bash "$REPO_DIR/slurm/run_ep3_baseline_inner3_initial_hstar.slurm"
