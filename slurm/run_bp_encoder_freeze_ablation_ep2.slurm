#!/bin/bash
#SBATCH -J bp-enc-freeze-ep2
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/bp_encoder_freeze_ep2_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/bp_encoder_freeze_ep2_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$WORK_ROOT/cachedir/hybridQ_ep1_AL_scaled_eta_bp500_20260925_205639}"
BASELINE_DIAG_DIR="${BASELINE_DIAG_DIR:-$WORK_ROOT/cachedir/bp_teacher_fit_ep2_f8e19ea_formal_20260930_143748}"
EPISODE="${EPISODE:-2}"
STEPS="${STEPS:-500}"
SEED="${SEED:-12345}"
CACHE_MODE="${CACHE_MODE:-auto}"
TRAIN_CACHE="${TRAIN_CACHE:-}"
VAL_CACHE="${VAL_CACHE:-}"
CHECKPOINT="${CHECKPOINT:-$SOURCE_RUN_ROOT/episode_diagnostics/ep_$(printf '%03d' "$EPISODE")/post_bp.pt}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$SOURCE_RUN_ROOT/episode_diagnostics/ep_$(printf '%03d' "$EPISODE")/post_q_final.pt}"
COMBINED_CHECKPOINT="${COMBINED_CHECKPOINT:-}"
FIRM_DATA="${FIRM_DATA:-$SOURCE_RUN_ROOT/data/outputs/ep${EPISODE}_stage_modeb.pkl}"
LEARNING_RATE="${LEARNING_RATE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/cachedir/bp_encoder_freeze_ablation_ep${EPISODE}_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
RUN_PREFLIGHT_ONLY="${RUN_PREFLIGHT_ONLY:-0}"
SKIP_EXTERNAL_EVAL="${SKIP_EXTERNAL_EVAL:-0}"

if [[ -z "$TRAIN_CACHE" && -z "$VAL_CACHE" && "$CACHE_MODE" == "auto" ]]; then
  CACHE_CANDIDATES=(
    "$SOURCE_RUN_ROOT/caches/ep${EPISODE}_bp_train.pt|$SOURCE_RUN_ROOT/caches/ep${EPISODE}_bp_validation.pt"
    "$SOURCE_RUN_ROOT/bp_cache/ep${EPISODE}_train.pt|$SOURCE_RUN_ROOT/bp_cache/ep${EPISODE}_validation.pt"
    "$SOURCE_RUN_ROOT/caches/bp/ep${EPISODE}_train.pt|$SOURCE_RUN_ROOT/caches/bp/ep${EPISODE}_validation.pt"
  )
  FOUND_CACHE_PAIRS=0
  for pair in "${CACHE_CANDIDATES[@]}"; do
    train_candidate="${pair%%|*}"
    val_candidate="${pair#*|}"
    if [[ -f "$train_candidate" && -f "$val_candidate" ]]; then
      TRAIN_CACHE="$train_candidate"
      VAL_CACHE="$val_candidate"
      FOUND_CACHE_PAIRS=$((FOUND_CACHE_PAIRS + 1))
    fi
  done
  if (( FOUND_CACHE_PAIRS > 1 )); then
    echo "ERROR: multiple recognized saved BP cache pairs were found"
    exit 1
  fi
fi

cd "$REPO_DIR"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"

echo "============================================================"
echo "BP encoder freeze/unfreeze paired diagnostic"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $BRANCH"
echo "Commit: $ACTUAL_COMMIT"
echo "Working tree:"
git status --short
echo "Source run: $SOURCE_RUN_ROOT"
echo "Baseline diagnostic: $BASELINE_DIAG_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Cache teacher checkpoint: $TEACHER_CHECKPOINT"
echo "Combined checkpoint: ${COMBINED_CHECKPOINT:-<auto, ambiguity rejected>}"
echo "Firm data: $FIRM_DATA"
echo "Train cache: ${TRAIN_CACHE:-<not provided>}"
echo "Validation cache: ${VAL_CACHE:-<not provided>}"
echo "Cache mode: $CACHE_MODE"
echo "Episode/steps/seed: $EPISODE/$STEPS/$SEED"
echo "Evaluation batch size: ${EVAL_BATCH_SIZE:-<auto: min(training batch, 4096)>}"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

if [[ -n "$REQUIRED_COMMIT" && "$ACTUAL_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "ERROR: REQUIRED_COMMIT mismatch"
  echo "Expected: $REQUIRED_COMMIT"
  echo "Actual:   $ACTUAL_COMMIT"
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: EP${EPISODE} post_bp checkpoint not found: $CHECKPOINT"
  exit 1
fi
if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "ERROR: EP${EPISODE} post_q_final cache-teacher checkpoint not found: $TEACHER_CHECKPOINT"
  exit 1
fi
if [[ ! -d "$BASELINE_DIAG_DIR" ]]; then
  echo "ERROR: baseline diagnostic directory not found: $BASELINE_DIAG_DIR"
  exit 1
fi
if [[ -n "$TRAIN_CACHE" || -n "$VAL_CACHE" ]]; then
  if [[ -z "$TRAIN_CACHE" || -z "$VAL_CACHE" || ! -f "$TRAIN_CACHE" || ! -f "$VAL_CACHE" ]]; then
    echo "ERROR: TRAIN_CACHE and VAL_CACHE must both name existing files"
    exit 1
  fi
elif [[ "$CACHE_MODE" == "saved" ]]; then
  echo "ERROR: CACHE_MODE=saved requires TRAIN_CACHE and VAL_CACHE"
  exit 1
elif [[ ! -f "$FIRM_DATA" ]]; then
  echo "ERROR: cache reconstruction data not found: $FIRM_DATA"
  exit 1
fi
if [[ -e "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: refusing to overwrite non-empty OUTPUT_DIR: $OUTPUT_DIR"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORK_ROOT/.mplcache}"
mkdir -p "$MPLCONFIGDIR"

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this Slurm allocation")
print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

ARGS=(
  --source-run-root "$SOURCE_RUN_ROOT"
  --baseline-diag-dir "$BASELINE_DIAG_DIR"
  --checkpoint "$CHECKPOINT"
  --teacher-checkpoint "$TEACHER_CHECKPOINT"
  --episode "$EPISODE"
  --cache-mode "$CACHE_MODE"
  --firm-data "$FIRM_DATA"
  --steps "$STEPS"
  --record-steps 0 50 100 200 500
  --seed "$SEED"
  --device cuda:0
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$COMBINED_CHECKPOINT" ]]; then
  ARGS+=(--combined-checkpoint "$COMBINED_CHECKPOINT")
fi
if [[ -n "$TRAIN_CACHE" ]]; then
  ARGS+=(--train-cache "$TRAIN_CACHE" --val-cache "$VAL_CACHE")
fi
if [[ -n "$LEARNING_RATE" ]]; then
  ARGS+=(--learning-rate "$LEARNING_RATE")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$EVAL_BATCH_SIZE" ]]; then
  ARGS+=(--eval-batch-size "$EVAL_BATCH_SIZE")
fi
if [[ "$SKIP_EXTERNAL_EVAL" == "1" ]]; then
  ARGS+=(--skip-external-eval)
fi

if [[ "$RUN_PREFLIGHT_ONLY" == "1" ]]; then
  ARGS+=(--dry-run)
fi

python3 -u experiments/run_bp_encoder_freeze_ablation.py "${ARGS[@]}"

echo "Finished: $(date)"
