#!/bin/bash
#SBATCH -J ep4-hybrid-p-eta50
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/ep4_hybrid_p_eta50_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/ep4_hybrid_p_eta50_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

# EP4 Hybrid Formal Runner + P-only Parent η Balance (50/50)
#
# 本启动器只做一件事：为 P0/PI 的数值求解启用既有
# Episode._balance_p_current_eta_batches() 的反事实 Parent η 平衡
# （counterfactual_parent_eta_balance），使 P 训练样本中 η=0/η=1 各约 50%，
# 然后委托现有 EP3 Hybrid Formal Runner（Inner3 + t0 h-star + Grid BP）。
#
# 这不是修改经济模型中的再融资概率：Config.ZETA=0.03 与 Child η_next 的
# 精确 Bernoulli 积分（PV_EXACT_ETA_INTEGRATION_ENABLED=1）保持不变；
# Q 训练分布、FC1/SDF、正式模拟的 η 转移、Bellman 方程均不变。
# 该采样是覆盖 Bellman 状态空间的 Collocation Augmentation，
# 不是对真实 η=1 企业的 oversampling，也不是对真实 η 分布的重新估计。
#
# 共享的原始 EP3 Baseline 启动器默认参数保持不变；本文件的所有配置
# 均允许环境变量覆盖；生效配置写入训练日志（下方 echo）与
# Checkpoint Metadata（p_eta_sampling_method / hyperparams.json）。
#
# Usage: sbatch slurm/run_ep4_hybrid_p_eta50.slurm

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/worktrees/ep3_baseline_004a2ad}"
export WORK_ROOT REPO_DIR
STAMP=$(date +%Y%m%d_%H%M%S)
# 独立 RUN_ROOT，绝不覆盖任何既有实验目录。
export RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/ep4_hybrid_p_eta50_${STAMP}}"

# --- P-only Parent η Balance（Collocation Augmentation）正式配置 ---
export PV_CURRENT_ETA_BALANCE_ENABLED="${PV_CURRENT_ETA_BALANCE_ENABLED:-1}"
export PV_CURRENT_ETA1_TRAIN_SHARE="${PV_CURRENT_ETA1_TRAIN_SHARE:-0.50}"
export PV_CURRENT_ETA_BALANCE_VALIDATION="${PV_CURRENT_ETA_BALANCE_VALIDATION:-1}"
export PV_CURRENT_ETA_BALANCE_SEED="${PV_CURRENT_ETA_BALANCE_SEED:-97531}"
# --- Natural Validation Diagnostic（纯诊断，不参与 P Acceptance）---
export P_ETA_NATURAL_DIAG_ENABLED="${P_ETA_NATURAL_DIAG_ENABLED:-1}"
export P_ETA_NATURAL_DIAG_MAX_PARENTS="${P_ETA_NATURAL_DIAG_MAX_PARENTS:-4096}"
export P_ETA_NATURAL_DIAG_CHUNK_SIZE="${P_ETA_NATURAL_DIAG_CHUNK_SIZE:-512}"
export P_ETA_NATURAL_DIAG_FREQUENCY="${P_ETA_NATURAL_DIAG_FREQUENCY:-firm_inner_end}"
export P_ETA_NATURAL_DIAG_SEED="${P_ETA_NATURAL_DIAG_SEED:-97531}"
# --- 明确固定的相关口径（防止隐式漂移）---
export PV_ETA_RESAMPLE_ENABLED="${PV_ETA_RESAMPLE_ENABLED:-0}"
export PV_EXACT_ETA_INTEGRATION_ENABLED="${PV_EXACT_ETA_INTEGRATION_ENABLED:-1}"
export BP_CURRENT_ETA_RESAMPLE_ENABLED="${BP_CURRENT_ETA_RESAMPLE_ENABLED:-0}"
export PV_BP_HEAD_TRAINING_ENABLED="${PV_BP_HEAD_TRAINING_ENABLED:-0}"
export SIMULATION_BP_ACTION_SOURCE="${SIMULATION_BP_ACTION_SOURCE:-grid}"
# 执行层 P 性能开关（默认 = 旧行为；只裁剪纯观测冗余计算）。
export PV_BP_DRIFT_MODE="${PV_BP_DRIFT_MODE:-full}"
export P_BP_DRIFT_SUBSET_ROWS="${P_BP_DRIFT_SUBSET_ROWS:-4096}"

echo "=== ep4 hybrid + P-only parent eta balance (50/50) ==="
echo "P_ETA_SAMPLING_METHOD=counterfactual_parent_eta_balance"
echo "PV_CURRENT_ETA_BALANCE_ENABLED=$PV_CURRENT_ETA_BALANCE_ENABLED"
echo "PV_CURRENT_ETA1_TRAIN_SHARE=$PV_CURRENT_ETA1_TRAIN_SHARE"
echo "PV_CURRENT_ETA_BALANCE_VALIDATION=$PV_CURRENT_ETA_BALANCE_VALIDATION"
echo "PV_CURRENT_ETA_BALANCE_SEED=$PV_CURRENT_ETA_BALANCE_SEED"
echo "P_ETA_NATURAL_DIAG_ENABLED=$P_ETA_NATURAL_DIAG_ENABLED"
echo "P_ETA_NATURAL_DIAG_MAX_PARENTS=$P_ETA_NATURAL_DIAG_MAX_PARENTS"
echo "P_ETA_NATURAL_DIAG_CHUNK_SIZE=$P_ETA_NATURAL_DIAG_CHUNK_SIZE"
echo "P_ETA_NATURAL_DIAG_FREQUENCY=$P_ETA_NATURAL_DIAG_FREQUENCY"
echo "P_ETA_NATURAL_DIAG_SEED=$P_ETA_NATURAL_DIAG_SEED"
echo "PV_ETA_RESAMPLE_ENABLED=$PV_ETA_RESAMPLE_ENABLED"
echo "PV_EXACT_ETA_INTEGRATION_ENABLED=$PV_EXACT_ETA_INTEGRATION_ENABLED"
echo "BP_CURRENT_ETA_RESAMPLE_ENABLED=$BP_CURRENT_ETA_RESAMPLE_ENABLED"
echo "PV_BP_HEAD_TRAINING_ENABLED=$PV_BP_HEAD_TRAINING_ENABLED"
echo "SIMULATION_BP_ACTION_SOURCE=$SIMULATION_BP_ACTION_SOURCE"
echo "PV_BP_DRIFT_MODE=$PV_BP_DRIFT_MODE P_BP_DRIFT_SUBSET_ROWS=$P_BP_DRIFT_SUBSET_ROWS"
echo "RUN_ROOT=$RUN_ROOT"

# 委托现有 EP3 Hybrid Formal Runner（Inner3 + t0 h-star + Grid BP）。
exec bash "$REPO_DIR/slurm/run_ep3_baseline_inner3_initial_hstar.slurm"
