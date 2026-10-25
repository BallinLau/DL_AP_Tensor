#!/bin/bash
#SBATCH -J conv-eval
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/convergence_eval_%j.out
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/convergence_eval_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00
#SBATCH --exclude=g85,g05
#
# Evaluation-Only runner for the equilibrium convergence evaluator.
# It NEVER trains, NEVER writes to a training RUN_ROOT and NEVER overwrites a
# checkpoint / simulation dataset: all output goes to
#   ${EVAL_RUN_ROOT}/convergence_<RUN_ID>/{manifest.json,bank/,metrics/,figures/,tables/,report/,diagnostics/}
#
# Required environment (missing -> non-zero exit, nothing is guessed):
#   REPO_DIR          worktree with the evaluator code
#   REQUIRED_COMMIT   git commit that must equal `git rev-parse HEAD`
#   EVAL_RUN_ROOT     output root (parent of convergence_<RUN_ID>/)
#   CHECKPOINT_PATHS  comma-separated combined checkpoints (epN_combined.pt)
#   EVALBANK_PATH     evaluation_bank_v1 root
#   EVAL_MODE         all | firm | macro | sim
#   DEVICE            e.g. cuda:0 or cpu
#   SEED              integer seed
#   CHUNK_SIZE        parent-row chunk size of the firm evaluators
#   ROLLOUT_HORIZON   fixed-rollout horizon of Bank E (default 100, echoed)
# Optional:
#   RUN_ID            default slurm_<jobid>; output dir = EVAL_RUN_ROOT/convergence_<RUN_ID>
#   THRESHOLD_CONFIG  JSON gate thresholds (absent -> every gate THRESHOLD_NOT_CONFIGURED)
#   BUILD_BANK_PANELS "firm.pkl,macro.pkl" -> build evaluation_bank_v1 first
#
# CPU-only override example (no GPU requested, device forced to cpu):
#   sbatch --gres= --export=ALL,DEVICE=cpu,REPO_DIR=...,REQUIRED_COMMIT=...,\
# EVAL_RUN_ROOT=...,CHECKPOINT_PATHS=...,EVALBANK_PATH=...,EVAL_MODE=firm,SEED=7,\
# CHUNK_SIZE=256,ROLLOUT_HORIZON=100 slurm/run_convergence_eval.slurm

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

REPO_DIR="${REPO_DIR:?REPO_DIR must point at the evaluation worktree}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:?REQUIRED_COMMIT must pin the evaluator commit}"
EVAL_RUN_ROOT="${EVAL_RUN_ROOT:?EVAL_RUN_ROOT must point at the evaluation output root}"
CHECKPOINT_PATHS="${CHECKPOINT_PATHS:?CHECKPOINT_PATHS must list combined checkpoints (comma separated)}"
EVALBANK_PATH="${EVALBANK_PATH:?EVALBANK_PATH must point at evaluation_bank_v1}"
EVAL_MODE="${EVAL_MODE:?EVAL_MODE must be all|firm|macro|sim}"
DEVICE="${DEVICE:?DEVICE must be cuda:0 or cpu}"
SEED="${SEED:?SEED must be an integer}"
CHUNK_SIZE="${CHUNK_SIZE:?CHUNK_SIZE must be an integer}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-100}"
RUN_ID="${RUN_ID:-slurm_${SLURM_JOB_ID:-local}}"
THRESHOLD_CONFIG="${THRESHOLD_CONFIG:-}"
BUILD_BANK_PANELS="${BUILD_BANK_PANELS:-}"
LOG_DIR="/home/fit/zhuyingz/WORK/LiuHao/logs"
MPL_CACHE="/home/fit/zhuyingz/WORK/LiuHao/.mplcache"

mkdir -p "$LOG_DIR" "$MPL_CACHE" "$EVAL_RUN_ROOT"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$MPL_CACHE"

echo "============================================================"
echo "Equilibrium convergence evaluation (Evaluation-Only)"
echo "============================================================"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Required commit: $REQUIRED_COMMIT"
echo "Eval run root: $EVAL_RUN_ROOT"
echo "Run id: $RUN_ID"
echo "Checkpoints: $CHECKPOINT_PATHS"
echo "EvalBank: $EVALBANK_PATH"
echo "Mode: $EVAL_MODE"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo "Chunk size: $CHUNK_SIZE"
echo "Rollout horizon: $ROLLOUT_HORIZON"
echo "Threshold config: ${THRESHOLD_CONFIG:-<none>}"
echo "Build bank panels: ${BUILD_BANK_PANELS:-<none>}"
echo "============================================================"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "ERROR: HEAD commit $ACTUAL_COMMIT != REQUIRED_COMMIT $REQUIRED_COMMIT"
  exit 1
fi

ARGS=(
  --bank-root "$EVALBANK_PATH"
  --output-root "$EVAL_RUN_ROOT"
  --run-id "$RUN_ID"
  --mode "$EVAL_MODE"
  --device "$DEVICE"
  --seed "$SEED"
  --chunk-size "$CHUNK_SIZE"
  --rollout-horizon "$ROLLOUT_HORIZON"
)

IFS=',' read -r -a CHECKPOINT_ARRAY <<< "$CHECKPOINT_PATHS"
for checkpoint in "${CHECKPOINT_ARRAY[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: checkpoint does not exist: $checkpoint"
    exit 1
  fi
  ARGS+=(--checkpoint "$checkpoint")
done

if [[ -n "$THRESHOLD_CONFIG" ]]; then
  if [[ ! -f "$THRESHOLD_CONFIG" ]]; then
    echo "ERROR: THRESHOLD_CONFIG does not exist: $THRESHOLD_CONFIG"
    exit 1
  fi
  ARGS+=(--threshold-config "$THRESHOLD_CONFIG")
fi

if [[ -n "$BUILD_BANK_PANELS" ]]; then
  ARGS+=(--build-bank "$BUILD_BANK_PANELS")
fi

python3 -u -m evaluation.convergence.run "${ARGS[@]}"
STATUS=$?

echo "============================================================"
echo "Convergence evaluation finished with status $STATUS"
echo "Finished: $(date)"
echo "Output: $EVAL_RUN_ROOT/convergence_$RUN_ID"
echo "============================================================"
exit $STATUS
