#!/bin/bash
# V5 user-approved H50 execution. No self-submission.
#SBATCH --partition=a01
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=h50-hybrid-outer12-20
set -euo pipefail
source /apps/soft/anaconda3/bin/activate
conda activate DL_HL
: "${REPO_DIR:?reviewed clean checkout required}"
: "${RUN_ROOT:?new persistent output directory required}"
: "${SAMPLING_DIR:?reviewed input bank directory required}"
: "${INITIAL_CHECKPOINT:?approved committed Outer11 learning state required}"
: "${SOURCE_JOURNAL_CHECKPOINT:?Job702636 final committed transaction required}"
: "${SOURCE_LATEST_POINTER:?Job702636 final pointer required}"
: "${SOURCE_ARTIFACT_DIR:?certified Outer11 pool bank schedule mirror required}"
: "${USER_APPROVAL_JSON:?explicit content-bound V5 approval required}"
cd "$REPO_DIR"
: "${REQUIRED_COMMIT:?content-bound clean commit required}"
[[ "$(git rev-parse HEAD)" == "$REQUIRED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 30 > "$RUN_ROOT/logs/gpu_telemetry.${SLURM_JOB_ID}.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
coverage_args=()
if [[ -n "${COVERAGE_CONFIG_JSON:-}" ]]; then
  coverage_args=(--coverage-config "$COVERAGE_CONFIG_JSON")
fi
firm_macro_args=()
if [[ -n "${FIRM_MACRO_CONFIG_JSON:-}" ]]; then
  firm_macro_args=(--firm-macro-config "$FIRM_MACRO_CONFIG_JSON")
fi
python analysis/run_initial_hstar_h50_broad_macro.py \
  --sampling-dir "$SAMPLING_DIR" --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --source-journal-checkpoint "$SOURCE_JOURNAL_CHECKPOINT" \
  --source-latest-pointer "$SOURCE_LATEST_POINTER" --source-artifact-dir "$SOURCE_ARTIFACT_DIR" \
  --output "$RUN_ROOT" --device cuda:0 --parent-batch-size 32 \
  --fit-parent-batch-size 128 --firm-batch-size 2048 --end-outer 20 --skip-bp-head \
  "${coverage_args[@]}" \
  "${firm_macro_args[@]}" \
  --execute-approved --approval "$USER_APPROVAL_JSON" "${extra[@]}"
