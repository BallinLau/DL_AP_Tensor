#!/bin/bash
#SBATCH -J boundary-integration-audit
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/cachedir/boundary_aware_macro_distribution_20261019/logs/integration.%j.out
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/cachedir/boundary_aware_macro_distribution_20261019/logs/integration.%j.err

set -euo pipefail
source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

REPO_DIR="${REPO_DIR:-/home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor_boundary_aware_20261019_9a2468f}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/fit/zhuyingz/WORK/LiuHao/cachedir/boundary_aware_macro_distribution_20261019}"
OLD_COMMAND="${OLD_COMMAND:-/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/market_cleared_full_nested_equilibrium_20261019_r8/logs/small_command.json}"
CHECKPOINT="${CHECKPOINT:-/WORK/PUBLIC/zhuyingz_work/LiuHao/cachedir/market_cleared_full_nested_equilibrium_20261019_r8/small/equilibrium_outer_000/firm/policy_distill/policy_distill_complete.pt}"
FIRMS="${FIRMS:-200}"
OPERATOR_SCAN_POINTS="${OPERATOR_SCAN_POINTS:-0}"

mkdir -p "$OUTPUT_DIR" /home/fit/zhuyingz/WORK/LiuHao/.mplcache
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/fit/zhuyingz/WORK/LiuHao/.mplcache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "host=$(hostname)"
echo "commit=$(git rev-parse HEAD)"
python3 analysis/run_boundary_aware_integration_audit.py \
  --output-dir "$OUTPUT_DIR" \
  --old-command "$OLD_COMMAND" \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 \
  --firms "$FIRMS" \
  --operator-scan-points "$OPERATOR_SCAN_POINTS"
