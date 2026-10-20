#!/bin/bash
#SBATCH -J FINITE_DIRECT_CONT_OUTER4_11
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
# PREPARED ONLY. Corrected exit v2 + dynamic Train Empirical requires Approval v4.
# Old fixed-bank Approval v3 cannot authorize this code. No self-submission.
set -euo pipefail
source /apps/soft/anaconda3/bin/activate
conda activate DL_HL
cd "${REPO_DIR:?Set immutable reviewed continuation checkout}"
: "${REQUIRED_COMMIT:?Set newly reviewed commit}"
: "${RUN_ROOT:?Set NEW empty continuation run or its own resume root}"
: "${SAMPLING_DIR:?Set reviewed NEW 8192/2048 bank}"
: "${INITIAL_CHECKPOINT:?Set full continuation_initial_checkpoint.pt}"
: "${SOURCE_MANIFEST:?Set certified continuation_source_manifest.json}"
: "${APPROVAL_FILE:?Separate explicit CONTINUATION_APPROVAL_V4_DYNAMIC_EMPIRICAL required}"
[[ "$(git rev-parse HEAD)" == "$REQUIRED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
python analysis/run_finite_direct_large_continuation.py \
  --sampling-dir "$SAMPLING_DIR" --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --source-manifest "$SOURCE_MANIFEST" --output "$RUN_ROOT" --device cuda:0 \
  --parent-batch-size 128 --fit-parent-batch-size 128 --firm-batch-size 512 \
  --execute-approved --approval "$APPROVAL_FILE" "${extra[@]}"
