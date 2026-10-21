#!/bin/bash
# PLAN TEMPLATE ONLY. Not submitted by this review.
#SBATCH --partition=a01
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=hstar-h50-review-required
set -euo pipefail
: "${REPO_DIR:?reviewed clean checkout required}"
: "${RUN_ROOT:?new persistent output directory required}"
: "${SAMPLING_DIR:?reviewed input bank directory required}"
: "${INITIAL_CHECKPOINT:?approved committed Outer11 learning state required}"
: "${SOURCE_JOURNAL_CHECKPOINT:?Job702636 final committed transaction required}"
: "${SOURCE_LATEST_POINTER:?Job702636 final pointer required}"
: "${SOURCE_ARTIFACT_DIR:?certified Outer11 pool bank schedule mirror required}"
: "${USER_APPROVAL_JSON:?explicit content-bound V5 approval required}"
cd "$REPO_DIR"
python analysis/run_initial_hstar_h50_broad_macro.py \
  --sampling-dir "$SAMPLING_DIR" --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --source-journal-checkpoint "$SOURCE_JOURNAL_CHECKPOINT" \
  --source-latest-pointer "$SOURCE_LATEST_POINTER" --source-artifact-dir "$SOURCE_ARTIFACT_DIR" \
  --output "$RUN_ROOT" --device cuda:0 --parent-batch-size 128 \
  --fit-parent-batch-size 128 --firm-batch-size 512 \
  --execute-approved --approval "$USER_APPROVAL_JSON"
