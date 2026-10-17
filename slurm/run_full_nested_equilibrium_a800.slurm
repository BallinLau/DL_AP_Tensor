#!/bin/bash
#SBATCH -J full-nested-equilibrium
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/full_nested_equilibrium.%j.out
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/full_nested_equilibrium.%j.err

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/full_nested_equilibrium_${STAMP}}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:?Set REQUIRED_COMMIT to the tested Git commit}"
NESTED_SMOKE="${NESTED_SMOKE:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ALLOW_CODE_MISMATCH="${ALLOW_CODE_MISMATCH:-0}"

mkdir -p "$LOG_DIR" "$RUN_ROOT/logs" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

OBSERVED_COMMIT="$(git rev-parse HEAD)"
if [[ "$OBSERVED_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "Commit mismatch: required=$REQUIRED_COMMIT observed=$OBSERVED_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing formal training from a dirty checkout" >&2
  git status --short >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID:-none}"
echo "Branch: $(git branch --show-current)"
echo "Commit: $OBSERVED_COMMIT"
echo "Run root: $RUN_ROOT"
echo "Smoke: $NESTED_SMOKE"
echo "Resume checkpoint: ${RESUME_CHECKPOINT:-none}"
echo "Python: $(command -v python3)"
echo "MACRO_PRIMARY_TRAINING=MANY_PARENT_ONE_STEP"
echo "LONG_TRAJECTORY_100_PRIMARY_TRAINING=NO"

python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the Slurm allocation")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory
print("gpu_name:", name)
print("gpu_total_memory:", memory)
if "A800" not in name.upper():
    raise SystemExit(f"Expected NVIDIA A800, received {name}")
if memory < 75 * 1024**3:
    raise SystemExit(f"Expected an 80GB-class GPU, received {memory} bytes")
PY

nvidia-smi
echo $$ > "$RUN_ROOT/train.pid"
echo "${SLURM_JOB_ID:-}" > "$RUN_ROOT/slurm_job_id"

EXTRA_ARGS=()
if [[ "$NESTED_SMOKE" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi
MODE_ARGS=(--cold-start)
TEE_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  MODE_ARGS=(--resume "$RESUME_CHECKPOINT")
  TEE_ARGS=(-a)
  if [[ "$ALLOW_CODE_MISMATCH" == "1" ]]; then
    MODE_ARGS+=(--allow-code-mismatch)
  fi
fi

python3 -u analysis/run_full_nested_equilibrium.py \
  "${MODE_ARGS[@]}" \
  --device cuda:0 \
  --output-dir "$RUN_ROOT" \
  --seed 20261018 \
  --equilibrium-outer-max 10 \
  --firm-inner-max 8 \
  --p-batch-size 1024 \
  --p-policy-outer-max 8 \
  --p-adam-steps 100 \
  --p-adam-lr 1e-4 \
  --p-lbfgs-max-iter 100 \
  --p-lbfgs-chunk-iterations 20 \
  --q-solver picard \
  --q-batch-size 2048 \
  --q-picard-outer-max 8 \
  --q-fit-steps 100 \
  --q-lr 1e-4 \
  --macro-inner-max 20 \
  --firms-per-path 200 \
  --branch-num 2 \
  --macro-bank-candidate-paths 1024 \
  --macro-bank-validation-paths 256 \
  --macro-bank-rollout-horizon 8 \
  --macro-bank-capture-times 0,2,4,6,8 \
  --macro-train-parent-count 4096 \
  --macro-validation-parent-count 1024 \
  --macro-bank-candidate-multiplier 1.5 \
  --macro-bank-coverage-fraction 0.20 \
  --macro-parent-batch-size 256 \
  --macro-bank-shard-size 128 \
  --macro-batch-size 4096 \
  --long-diagnostic-paths 50 \
  --long-diagnostic-horizon 100 \
  --no-use-long-rollout-for-macro-training \
  --fc1-production-epochs 10 \
  --fc1-max-epochs 100 \
  --fc1-patience 20 \
  --sdf-chunks 4 \
  --sdf-epochs-per-chunk 100 \
  --sdf-recursive-epochs 0 \
  --disable-fc2 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${TEE_ARGS[@]}" "$RUN_ROOT/logs/run.log"

echo "Job finished: $(date)"
