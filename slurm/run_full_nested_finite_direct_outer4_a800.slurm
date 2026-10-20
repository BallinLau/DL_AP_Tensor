#!/bin/bash
#SBATCH -J FINITE_DIRECT_OUTER4_REVIEWED
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

# Prepared for FUTURE explicit approval only. This script does not submit jobs.
set -euo pipefail
source /apps/soft/anaconda3/bin/activate
conda activate DL_HL
cd "${REPO_DIR:?Set immutable reviewed checkout}"
: "${REQUIRED_COMMIT:?Set reviewed commit}"
: "${RUN_ROOT:?Set NEW persistent run directory}"
: "${SAMPLING_DIR:?Set locked reviewed sampling directory}"
: "${INITIAL_CHECKPOINT:?Set verified combined checkpoint}"
: "${APPROVAL_FILE:?Separate explicit user approval required}"
# Schema v2 also binds sampling_signature fingerprint + file SHA and the
# initial combined checkpoint SHA. The Python gate verifies actual contents;
# changing SAMPLING_DIR/INITIAL_CHECKPOINT cannot bypass approval by path alone.
[[ "$(git rev-parse HEAD)" == "$REQUIRED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
python analysis/run_full_nested_finite_direct_outer4.py \
  --sampling-dir "$SAMPLING_DIR" \
  --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --output "$RUN_ROOT" --device cuda:0 \
  --parent-batch-size 128 --fit-parent-batch-size 128 --firm-batch-size 512 \
  --execute-approved --approval "$APPROVAL_FILE" "${extra[@]}"
