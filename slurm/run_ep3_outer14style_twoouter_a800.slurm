#!/bin/bash
# Independent EP3->Outer14-style two-Outer weights-only cold start.
# Never submits/modifies the H50 job; never touches Outer12-15 artifacts.
#SBATCH --partition=a01
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=ep3-outer14-2o
#SBATCH --exclude=g53
set -euo pipefail
source /apps/soft/anaconda3/bin/activate
conda activate DL_HL
: "${REPO_DIR:?independent immutable EP3-Outer14-style checkout}"
: "${REQUIRED_COMMIT:?new EP3-Outer14-style source commit}"
: "${RUN_ROOT:?new persistent output}"
: "${SAMPLING_DIR:?certified LEGACY_INITIAL sampling input}"
: "${INITIAL_CHECKPOINT:?read-only EP3 weights-only combined checkpoint}"
: "${USER_APPROVAL_JSON:?independent EP3-Outer14-style content-bound approval}"
cd "$REPO_DIR"
[[ "$(git rev-parse HEAD)" == "$REQUIRED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 30 > "$RUN_ROOT/logs/gpu_telemetry.${SLURM_JOB_ID}.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
python analysis/run_ep3_outer14style_twoouter.py \
  --sampling-dir "$SAMPLING_DIR" --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --output "$RUN_ROOT" --device cuda:0 --parent-batch-size 32 \
  --fit-parent-batch-size 128 --firm-batch-size 2048 --skip-bp-head \
  --execute-approved --approval "$USER_APPROVAL_JSON" "${extra[@]}"
