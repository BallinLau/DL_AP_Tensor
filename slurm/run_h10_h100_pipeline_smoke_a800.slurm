#!/bin/bash
#SBATCH --partition=a01
#SBATCH --gres=gpu:a800:1
#SBATCH --exclude=g53
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=h10-h100-pipeline-smoke
set -euo pipefail
source /apps/soft/anaconda3/bin/activate
conda activate DL_HL
: "${REPO_DIR:?independent source}"
: "${REQUIRED_COMMIT:?exact smoke source}"
: "${RUN_ROOT:?output ending _pipeline_smoke}"
: "${SAMPLING_DIR:?certified sampling}"
: "${INITIAL_CHECKPOINT:?committed Outer11}"
: "${SOURCE_JOURNAL_CHECKPOINT:?source transaction}"
: "${SOURCE_LATEST_POINTER:?source pointer}"
: "${SOURCE_ARTIFACT_DIR:?source bank/pool}"
cd "$REPO_DIR"
[[ "$(git rev-parse HEAD)" == "$REQUIRED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python analysis/run_h10_h100_pipeline_smoke.py \
  --sampling-dir "$SAMPLING_DIR" --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --source-journal-checkpoint "$SOURCE_JOURNAL_CHECKPOINT" \
  --source-latest-pointer "$SOURCE_LATEST_POINTER" --source-artifact-dir "$SOURCE_ARTIFACT_DIR" \
  --output "$RUN_ROOT"
