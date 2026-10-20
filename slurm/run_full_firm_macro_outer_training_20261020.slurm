#!/bin/bash
#SBATCH -J FULL_FIRM_MACRO_OUTER_20261020
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/fit/zhuyingz/WORK/LiuHao/cachedir/full_firm_macro_outer_training_20261020/logs/slurm.%j.out
#SBATCH --error=/home/fit/zhuyingz/WORK/LiuHao/cachedir/full_firm_macro_outer_training_20261020/logs/slurm.%j.err

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:?Set REPO_DIR to the immutable experiment checkout}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/full_firm_macro_outer_training_20261020}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:?Set REQUIRED_COMMIT to the tested source commit}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:?Set INITIAL_CHECKPOINT}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/metadata" "$WORK_ROOT/.mplcache"
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
if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
  echo "Missing initial checkpoint: $INITIAL_CHECKPOINT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"
export REQUIRED_COMMIT RUN_ROOT

if [[ ! -f "$RUN_ROOT/slurm_job_id" ]]; then
  echo "${SLURM_JOB_ID:-}" > "$RUN_ROOT/slurm_job_id"
fi
echo "${SLURM_JOB_ID:-}" > "$RUN_ROOT/slurm_job_id_latest"
printf '%s\n' "${SLURM_JOB_ID:-}" >> "$RUN_ROOT/slurm_job_ids"
if [[ ! -f "$RUN_ROOT/source_git_commit" ]]; then
  printf '%s\n' "$OBSERVED_COMMIT" > "$RUN_ROOT/source_git_commit"
fi
printf '%s\n' "$OBSERVED_COMMIT" > "$RUN_ROOT/source_git_commit_latest"
printf '%s\n' "$OBSERVED_COMMIT" >> "$RUN_ROOT/source_git_commit_history"
printf '%s\n' "$INITIAL_CHECKPOINT" > "$RUN_ROOT/initial_checkpoint_path"

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 30 >> "$RUN_ROOT/logs/gpu_telemetry.csv" &
else
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 30 > "$RUN_ROOT/logs/gpu_telemetry.csv" &
fi
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Mandatory preflight is embedded in this same formal job.  It validates the
# actual A800 and complete fixed candidate schedules, then immediately enters
# optimizer execution; it is not a separate Smoke run.
python3 - <<'PY'
import json
import os
from pathlib import Path
import torch
from analysis.macro_parent_state_bank import (
    RESTRICTED_BETA_FULL_OUTER_V1,
    build_macro_candidate_schedule,
    midpoint_beta_mean,
)

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory
if "A800" not in name.upper() or memory < 75 * 1024**3:
    raise SystemExit(f"Expected one A800 80GB, got {name} / {memory}")

def schedule(split, seed, offset, target, initial, maximum, minimum):
    return build_macro_candidate_schedule(
        split=split,
        namespace=f"{split}:{RESTRICTED_BETA_FULL_OUTER_V1}",
        target_valid=target,
        initial_candidate_budget=initial,
        max_candidate_budget=maximum,
        minimum_partial_valid=minimum,
        firms_per_parent=200,
        seed=seed,
        candidate_id_offset=offset,
        max_initial_weighted_b_mean=0.30,
        distribution_design=RESTRICTED_BETA_FULL_OUTER_V1,
    )

train = schedule("train_outer_000", 20261020 + 30000, 0, 128, 384, 1024, 64)
validation = schedule(
    "global_validation", 20261020 + 40000, 1_000_000_000,
    32, 128, 256, 16,
)
train_ids = {item.candidate_id for item in train.candidate_specs}
validation_ids = {item.candidate_id for item in validation.candidate_specs}
means = [
    midpoint_beta_mean(item.alpha, item.beta, 200)
    for item in (*train.candidate_specs, *validation.candidate_specs)
]
if train_ids & validation_ids or max(means) > 0.30 + 1.0e-12:
    raise SystemExit("formal candidate schedule preflight failed")
payload = {
    "FORMAL_JOB": True,
    "SMOKE": False,
    "gpu": name,
    "gpu_memory": memory,
    "source_commit": os.environ["REQUIRED_COMMIT"],
    "distribution_design": RESTRICTED_BETA_FULL_OUTER_V1,
    "max_initial_weighted_b_mean": 0.30,
    "initial_total_K_bounds": [45.0, 85.0],
    "train_schedule_fingerprint": train.fingerprint,
    "validation_schedule_fingerprint": validation.fingerprint,
    "train_validation_disjoint": True,
    "combined_b_mean_min": min(means),
    "combined_b_mean_max": max(means),
}
root = Path(os.environ["RUN_ROOT"])
target = root / "metadata" / "embedded_preflight.json"
if target.exists():
    target = root / "metadata" / (
        f"embedded_preflight_resume_{os.environ.get('SLURM_JOB_ID', 'unknown')}.json"
    )
target.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(payload, flush=True)
PY

MODE_ARGS=(--cold-start --initial-checkpoint "$INITIAL_CHECKPOINT")
PARENT_BATCH_ARGS=(--macro-parent-batch-size 16)
H_BATCH_ARGS=(--boundary-candidate-h-microbatch-size 1)
TEE_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  MODE_ARGS=(--resume "$RESUME_CHECKPOINT" --allow-code-mismatch)
  PARENT_BATCH_ARGS=(--resume-macro-parent-batch-size 16)
  H_BATCH_ARGS=(--resume-boundary-candidate-h-microbatch-size 1)
  TEE_ARGS=(-a)
  printf '%s\n' "$RESUME_CHECKPOINT" >> "$RUN_ROOT/resume_checkpoint_history"
fi

python3 -u analysis/run_full_nested_equilibrium.py \
  "${MODE_ARGS[@]}" \
  --device cuda:0 \
  --output-dir "$RUN_ROOT" \
  --seed 20261020 \
  --equilibrium-outer-max 4 \
  --overall-gate-min-completed-outers 4 \
  --firm-inner-max 3 \
  --p-batch-size 4096 \
  --p-policy-outer-max 3 \
  --p-adam-steps 200 \
  --p-adam-lr 1e-4 \
  --p-lbfgs-max-iter 100 \
  --p-lbfgs-chunk-iterations 20 \
  --q-solver picard \
  --q-batch-size 8192 \
  --q-picard-outer-max 3 \
  --q-fit-steps 300 \
  --q-lr 1e-4 \
  --policy-distill-max-epochs 1000 \
  --policy-distill-max-steps 1000 \
  --policy-distill-patience 20 \
  --macro-inner-max 2 \
  --firms-per-path 200 \
  --branch-num 2 \
  --macro-train-target-valid 128 \
  --macro-validation-target-valid 32 \
  --macro-train-initial-candidate-budget 384 \
  --macro-train-max-candidate-budget 1024 \
  --macro-validation-initial-candidate-budget 128 \
  --macro-validation-max-candidate-budget 256 \
  --macro-minimum-partial-train 64 \
  --macro-minimum-partial-validation 16 \
  "${PARENT_BATCH_ARGS[@]}" \
  --macro-candidate-distribution-design restricted_beta_full_outer_v1 \
  --max-initial-weighted-b-mean 0.30 \
  --initial-total-k-min 45 \
  --initial-total-k-max 85 \
  --macro-batch-size 4096 \
  --fc1-production-epochs 20 \
  --fc1-max-epochs 60 \
  --fc1-patience 15 \
  --sdf-chunks 2 \
  --sdf-epochs-per-chunk 50 \
  --sdf-target-batches 20 \
  --sdf-min-parent-groups-per-batch 64 \
  --sdf-recursive-epochs 0 \
  --macro-clearing-solver-mode fc1_bounded_improvement_pilot \
  "${H_BATCH_ARGS[@]}" \
  --aggregation-mode continuum_mass_consistent_v1 \
  --bp-actual-use-trace \
  --disable-fc2 \
  --no-run-long-diagnostics \
  --no-use-long-rollout-for-macro-training \
  2>&1 | tee "${TEE_ARGS[@]}" "$RUN_ROOT/logs/run.log"

python3 -u analysis/market_cleared_reporting.py --run-root "$RUN_ROOT" \
  2>&1 | tee "${TEE_ARGS[@]}" "$RUN_ROOT/logs/report.log"

echo "Formal four-Outer job finished: $(date)"
