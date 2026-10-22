#!/bin/bash
# Independent, non-preemptive job; never submits/modifies the H50 job.
#SBATCH --partition=a01
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=h10-h100-outer12-15
#SBATCH --exclude=g53
set -euo pipefail
source /apps/soft/anaconda3/bin/activate
conda activate DL_HL
: "${REPO_DIR:?independent immutable H10 checkout}"
: "${REQUIRED_COMMIT:?new H10 source commit}"
: "${RUN_ROOT:?new persistent output}"
: "${SAMPLING_DIR:?certified sampling input}"
: "${INITIAL_CHECKPOINT:?read-only committed Job702636 Outer11}"
: "${SOURCE_JOURNAL_CHECKPOINT:?certified original transaction}"
: "${SOURCE_LATEST_POINTER:?certified original commit pointer}"
: "${SOURCE_ARTIFACT_DIR:?certified original dynamic artifacts}"
: "${USER_APPROVAL_JSON:?independent V6 content-bound approval}"
cd "$REPO_DIR"
[[ "$(git rev-parse HEAD)" == "$REQUIRED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
python analysis/run_h10_h100_longrun_macro.py \
  --sampling-dir "$SAMPLING_DIR" --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --source-journal-checkpoint "$SOURCE_JOURNAL_CHECKPOINT" \
  --source-latest-pointer "$SOURCE_LATEST_POINTER" --source-artifact-dir "$SOURCE_ARTIFACT_DIR" \
  --output "$RUN_ROOT" --device cuda:0 --parent-batch-size 32 \
  --fit-parent-batch-size 128 --firm-batch-size 2048 --end-outer 15 --skip-bp-head \
  --execute-approved --approval "$USER_APPROVAL_JSON" "${extra[@]}"
