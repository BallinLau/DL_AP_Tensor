#!/bin/bash
#SBATCH -J diverse-macro-bank
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

REPO_DIR="${REPO_DIR:?Set REPO_DIR}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:?Set REQUIRED_COMMIT}"
FIRM_CHECKPOINT="${FIRM_CHECKPOINT:?Set FIRM_CHECKPOINT}"
MACRO_CHECKPOINT="${MACRO_CHECKPOINT:?Set MACRO_CHECKPOINT}"
TRAIN_PATHS="${TRAIN_PATHS:-8}"
VALIDATION_PATHS="${VALIDATION_PATHS:-8}"
FIRMS_PER_DISTRIBUTION="${FIRMS_PER_DISTRIBUTION:-200}"
HORIZON="${HORIZON:-5}"
BURN_IN="${BURN_IN:-1}"
BANK_TRAIN_ROWS="${BANK_TRAIN_ROWS:-512}"
BANK_VALIDATION_ROWS="${BANK_VALIDATION_ROWS:-256}"
RUN_TESTS="${RUN_TESTS:-1}"

mkdir -p "$RUN_ROOT" /home/fit/zhuyingz/WORK/LiuHao/.mplcache
cd "$REPO_DIR"
test "$(git rev-parse HEAD)" = "$REQUIRED_COMMIT"
test -z "$(git status --porcelain)"
test -f "$FIRM_CHECKPOINT"
test -f "$MACRO_CHECKPOINT"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/fit/zhuyingz/WORK/LiuHao/.mplcache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "commit=$(git rev-parse HEAD)"
echo "run_root=$RUN_ROOT"
echo "train_paths=$TRAIN_PATHS validation_paths=$VALIDATION_PATHS horizon=$HORIZON"
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
PY

if [[ "$RUN_TESTS" = "1" ]]; then
  python3 -m pytest -p no:cacheprovider \
    tests/test_diverse_short_horizon.py \
    tests/test_macro_outer_inner_pq.py \
    tests/test_inner_fixed_point.py -q
fi

python3 -u analysis/run_diverse_macro_unified_bank.py \
  --firm-checkpoint "$FIRM_CHECKPOINT" \
  --macro-checkpoint "$MACRO_CHECKPOINT" \
  --output-dir "$RUN_ROOT" \
  --device cuda:0 \
  --train-paths "$TRAIN_PATHS" \
  --validation-paths "$VALIDATION_PATHS" \
  --firms-per-distribution "$FIRMS_PER_DISTRIBUTION" \
  --horizon "$HORIZON" \
  --burn-in "$BURN_IN" \
  --branch-num 2 \
  --bank-train-rows "$BANK_TRAIN_ROWS" \
  --bank-validation-rows "$BANK_VALIDATION_ROWS" \
  --entry-mode investment_compare \
  --entry-spec-version investment_compare_v1 \
  --entry-capital-ratio 0.05 \
  --entry-size-ratio 1.0 \
  --entry-cost-max 0.5 \
  --consumption-aggregation-mode raw \
  --node-accounting-mode economic_node_ledger \
  --bp-action-source grid

echo "completed=$(date --iso-8601=seconds)"
