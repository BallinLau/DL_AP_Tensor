#!/bin/bash
#SBATCH -J mc-full-nested
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/market_cleared_full_nested.%j.out
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/market_cleared_full_nested.%j.err

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:?Set REPO_DIR to the immutable experiment checkout}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/market_cleared_full_nested_equilibrium_20261019}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:?Set REQUIRED_COMMIT to the tested source commit}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:?Set INITIAL_CHECKPOINT to a validated Firm+Macro checkpoint}"
START_AT="${START_AT:-tiny}"
THROUGH="${THROUGH:-formal}"

mkdir -p "$WORK_ROOT/logs" "$RUN_ROOT/logs" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

OBSERVED_COMMIT="$(git rev-parse HEAD)"
if [[ "$OBSERVED_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "Commit mismatch: required=$REQUIRED_COMMIT observed=$OBSERVED_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing training from a dirty checkout" >&2
  git status --short >&2
  exit 2
fi
if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "Initial checkpoint does not exist: $INITIAL_CHECKPOINT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": name, "memory": memory})
if "A800" not in name.upper() or memory < 75 * 1024**3:
    raise SystemExit(f"Expected one A800 80GB, got {name} / {memory}")
PY

echo "${SLURM_JOB_ID:-}" > "$RUN_ROOT/slurm_job_id"
echo "$OBSERVED_COMMIT" > "$RUN_ROOT/source_git_commit"
printf '%s\n' "$INITIAL_CHECKPOINT" > "$RUN_ROOT/initial_checkpoint_path"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 30 > "$RUN_ROOT/logs/gpu_telemetry.csv" &
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT

python3 -u analysis/run_market_cleared_full_nested_equilibrium.py \
  --output-dir "$RUN_ROOT" \
  --device cuda:0 \
  --seed 20261019 \
  --start-at "$START_AT" \
  --through "$THROUGH" \
  --initial-checkpoint "$INITIAL_CHECKPOINT" \
  2>&1 | tee "$RUN_ROOT/logs/chain.log"

echo "Job finished: $(date)"
