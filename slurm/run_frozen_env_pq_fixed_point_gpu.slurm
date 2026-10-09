#!/bin/bash
#SBATCH -J pq-fixed-ep2
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/pq_fixed_point_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/pq_fixed_point_error.%j

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
GRID_RUN="${GRID_RUN:-$WORK_ROOT/cachedir/scaled_value_grid_reference_ep0_ep4_4b236ea_20261008_210439}"
CHECKPOINT="${CHECKPOINT:-$GRID_RUN/checkpoints/ep2_combined.pt}"
FROZEN_BATCH_BANK="${FROZEN_BATCH_BANK:?FROZEN_BATCH_BANK must point to the exact EP2 post-SDF-refresh pre-PV batch bank}"
REFERENCE_FIRM_DATA="${REFERENCE_FIRM_DATA:-$GRID_RUN/data/outputs/ep2_stage_modeb.pkl}"
REFERENCE_MACRO_DATA="${REFERENCE_MACRO_DATA:-$GRID_RUN/data/outputs/ep2_stage_modeb_macro.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-$GRID_RUN/frozen_env_pq_fixed_point_ep2}"
DELIVERY_BASE_COMMIT="${DELIVERY_BASE_COMMIT:-5195ad058077f5fd45865c0ce8ed050be0f4fb6a}"
CYCLES="${CYCLES:-10}"
FIXED_POINT_SEED="${FIXED_POINT_SEED:-24681357}"
N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-64}"
SHOCK_SEED="${SHOCK_SEED:-12345}"
FIT_NORMALIZED_TOLERANCE="${FIT_NORMALIZED_TOLERANCE:-0.05}"
DISTANCE_TOLERANCE="${DISTANCE_TOLERANCE:-0.001}"
RESIDUAL_IMPROVEMENT_RATIO="${RESIDUAL_IMPROVEMENT_RATIO:-0.8}"
BOUNDARY_DRIFT_TOLERANCE="${BOUNDARY_DRIFT_TOLERANCE:-0.001}"
TWO_CYCLE_RATIO_THRESHOLD="${TWO_CYCLE_RATIO_THRESHOLD:-0.5}"

mkdir -p "$WORK_ROOT/logs" "$OUTPUT_DIR" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "Frozen-environment P-Q fixed-point test"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $ACTUAL_COMMIT"
echo "Checkpoint: $CHECKPOINT"
echo "Frozen batch bank: $FROZEN_BATCH_BANK"
echo "Reference firm data: $REFERENCE_FIRM_DATA"
echo "Reference macro data: $REFERENCE_MACRO_DATA"
echo "Output: $OUTPUT_DIR"

if ! git merge-base --is-ancestor "$DELIVERY_BASE_COMMIT" HEAD; then
  echo "ERROR: GitHub delivery base $DELIVERY_BASE_COMMIT is not an ancestor of $ACTUAL_COMMIT"
  exit 1
fi
# The Python runner verifies AST fingerprints for the audited production P/Q
# methods. This permits the 5195ad0 entry-mechanism commit while preventing a
# silent change to the 4b236ea fitted P/Q mapping used by the GRID run.
for path in "$CHECKPOINT" "$FROZEN_BATCH_BANK" "$REFERENCE_FIRM_DATA" "$REFERENCE_MACRO_DATA"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required input not found: $path"
    exit 1
  fi
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CHECKPOINT FROZEN_BATCH_BANK

python3 - <<'PY'
import json
import os
import torch

from evaluation.pq_fixed_point import (
    checkpoint_provenance,
    validate_checkpoint_bank_provenance,
    validate_frozen_batch_bank,
)

checkpoint = checkpoint_provenance(os.environ["CHECKPOINT"])
bank_payload = torch.load(os.environ["FROZEN_BATCH_BANK"], map_location="cpu")
bank = validate_frozen_batch_bank(bank_payload)
guard = validate_checkpoint_bank_provenance(checkpoint, bank["provenance"])
print("Checkpoint provenance:")
print(json.dumps(checkpoint, indent=2, sort_keys=True))
print("Bank provenance:")
print(json.dumps(bank["provenance"], indent=2, sort_keys=True))
print("Dataset hash:", bank["dataset_sha256"])
print("Provenance guard:", json.dumps(guard, sort_keys=True))
PY

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print("cuda_device_0:", torch.cuda.get_device_name(0))
PY
nvidia-smi

python3 -u experiments/frozen_env_pq_fixed_point.py \
  --checkpoint "$CHECKPOINT" \
  --frozen-batch-bank "$FROZEN_BATCH_BANK" \
  --reference-firm-data "$REFERENCE_FIRM_DATA" \
  --reference-macro-data "$REFERENCE_MACRO_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --cycles "$CYCLES" \
  --fixed-point-seed "$FIXED_POINT_SEED" \
  --b-points 41 \
  --z-points 41 \
  --n-child-shocks "$N_CHILD_SHOCKS" \
  --shock-seed "$SHOCK_SEED" \
  --fit-normalized-tolerance "$FIT_NORMALIZED_TOLERANCE" \
  --distance-tolerance "$DISTANCE_TOLERANCE" \
  --residual-improvement-ratio "$RESIDUAL_IMPROVEMENT_RATIO" \
  --boundary-drift-tolerance "$BOUNDARY_DRIFT_TOLERANCE" \
  --two-cycle-ratio-threshold "$TWO_CYCLE_RATIO_THRESHOLD" \
  --device cuda:0

echo "Completed: $(date)"
