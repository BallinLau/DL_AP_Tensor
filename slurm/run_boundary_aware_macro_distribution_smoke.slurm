#!/bin/bash
#SBATCH -J boundary-aware-macro-smoke
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/cachedir/boundary_aware_macro_distribution_20261019/logs/smoke.%j.out
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/cachedir/boundary_aware_macro_distribution_20261019/logs/smoke.%j.err

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

REPO_DIR="${REPO_DIR:-/home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor_boundary_aware_20261019_9a2468f}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/fit/zhuyingz/WORK/LiuHao/cachedir/boundary_aware_macro_distribution_20261019/smoke}"
OLD_COMMAND="${OLD_COMMAND:-/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/market_cleared_full_nested_equilibrium_20261019_r8/logs/small_command.json}"
CHECKPOINT="${CHECKPOINT:-/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/market_cleared_full_nested_equilibrium_20261019_r8/small/equilibrium_outer_000/firm/policy_distill/policy_distill_complete.pt}"
TRAIN_PATHS="${TRAIN_PATHS:-1}"
VALIDATION_PATHS="${VALIDATION_PATHS:-1}"
FIRMS="${FIRMS:-1}"
HORIZON="${HORIZON:-1}"
Z_SCAN_POINTS="${Z_SCAN_POINTS:-17}"
Z_QUADRATURE_POINTS="${Z_QUADRATURE_POINTS:-2}"
I_SCAN_POINTS="${I_SCAN_POINTS:-9}"
I_QUADRATURE_POINTS="${I_QUADRATURE_POINTS:-2}"
PROJECTION_AFTER_T="${PROJECTION_AFTER_T:--1}"
PROJECTION_B_POINTS="${PROJECTION_B_POINTS:-6}"
PROJECTION_K_POINTS="${PROJECTION_K_POINTS:-6}"
PROJECTION_U_POINTS="${PROJECTION_U_POINTS:-12}"

mkdir -p "$(dirname "$OUTPUT_DIR")/logs" "$OUTPUT_DIR" /home/fit/zhuyingz/WORK/LiuHao/.mplcache
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/fit/zhuyingz/WORK/LiuHao/.mplcache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "host=$(hostname)"
echo "commit=$(git rev-parse HEAD)"
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

python3 analysis/run_boundary_aware_macro_distribution.py \
  --output-dir "$OUTPUT_DIR" \
  --old-command "$OLD_COMMAND" \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 \
  --train-paths "$TRAIN_PATHS" \
  --validation-paths "$VALIDATION_PATHS" \
  --firms "$FIRMS" \
  --horizon "$HORIZON" \
  --z-scan-points "$Z_SCAN_POINTS" \
  --z-quadrature-points "$Z_QUADRATURE_POINTS" \
  --i-scan-points "$I_SCAN_POINTS" \
  --i-quadrature-points "$I_QUADRATURE_POINTS" \
  --projection-after-t "$PROJECTION_AFTER_T" \
  --projection-b-points "$PROJECTION_B_POINTS" \
  --projection-k-points "$PROJECTION_K_POINTS" \
  --projection-u-points "$PROJECTION_U_POINTS"

