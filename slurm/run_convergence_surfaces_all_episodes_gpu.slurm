#!/bin/bash
#SBATCH -J conv-surface-all
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /home/fit/zhuyingz/WORK/LiuHao/logs/convergence_surface_all_output.%j
#SBATCH -e /home/fit/zhuyingz/WORK/LiuHao/logs/convergence_surface_all_error.%j
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

WORK_ROOT="${WORK_ROOT:-/home/fit/zhuyingz/WORK/LiuHao}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/DL_AP_Tensor}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"

RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_scaled_value_20260718_231730}"
RAW_CHECKPOINT_DIR="${RAW_CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints_analysis}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-$RUN_ROOT/data/outputs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_OUTPUT_DIR/convergence_surfaces_all_${STAMP}}"
PACKAGE_ANALYSIS_CHECKPOINTS="${PACKAGE_ANALYSIS_CHECKPOINTS:-1}"
PACKAGING_TRAINING_COMMIT="${PACKAGING_TRAINING_COMMIT:-06b1f0786462bd443bd18ee458bfc4bdecaa67b8}"
VALUE_SCALE_LOG_MAX="${VALUE_SCALE_LOG_MAX:-20.0}"
PV_TRAINING_FLOW="${PV_TRAINING_FLOW:-staged}"
PV_USE_CLIPPED_M="${PV_USE_CLIPPED_M:-1}"
PV_M_CLAMP_MIN="${PV_M_CLAMP_MIN:-0.7}"
PV_M_CLAMP_MAX="${PV_M_CLAMP_MAX:-1.3}"
Q_USE_DETACHED_M="${Q_USE_DETACHED_M:-1}"
BP_GRID_POLICY_LOSS_SPACE="${BP_GRID_POLICY_LOSS_SPACE:-logit}"
BP_GRID_LOGIT_TARGET_EPS="${BP_GRID_LOGIT_TARGET_EPS:-1e-4}"
BP_GRID_LOGIT_HUBER_DELTA="${BP_GRID_LOGIT_HUBER_DELTA:-1.0}"
PV_MIXTURE_ENABLED="${PV_MIXTURE_ENABLED:-1}"
PV_MIXTURE_RATIO="${PV_MIXTURE_RATIO:-0.20}"
PV_MIXTURE_START_EPISODE="${PV_MIXTURE_START_EPISODE:-1}"
PV_MIXTURE_BUDGET_MODE="${PV_MIXTURE_BUDGET_MODE:-fixed_total}"
PV_MIXTURE_SAMPLING_MODE="${PV_MIXTURE_SAMPLING_MODE:-uniform}"
PV_MIXTURE_COVERAGE_GROUP_SIZE="${PV_MIXTURE_COVERAGE_GROUP_SIZE:-2}"
PV_MIXTURE_SEED="${PV_MIXTURE_SEED:-24680}"
PV_MIXTURE_STRATIFIED_VALIDATION="${PV_MIXTURE_STRATIFIED_VALIDATION:-1}"
PV_MIXTURE_PRESERVE_RNG="${PV_MIXTURE_PRESERVE_RNG:-1}"
FIRM_TARGET_UPDATE="${FIRM_TARGET_UPDATE:-stage_hard}"

DEVICE="${DEVICE:-cuda}"
B_MIN="${B_MIN:-0.0}"
B_MAX="${B_MAX:-1.0}"
B_POINTS="${B_POINTS:-41}"
Z_MIN="${Z_MIN:--2.0}"
Z_MAX="${Z_MAX:-2.0}"
Z_POINTS="${Z_POINTS:-41}"
N_CHILD_SHOCKS="${N_CHILD_SHOCKS:-32}"
SEED="${SEED:-2026}"
PARENT_CHUNK_SIZE="${PARENT_CHUNK_SIZE:-512}"
CHILD_CHUNK_SIZE="${CHILD_CHUNK_SIZE:-8192}"
MAX_CHILD_STATE_EVALS="${MAX_CHILD_STATE_EVALS:-2000000}"
ALLOW_LARGE_RUN="${ALLOW_LARGE_RUN:-0}"
INCLUDE_RAW="${INCLUDE_RAW:-0}"
INCLUDE_SIGNED="${INCLUDE_SIGNED:-1}"
HYPERPARAMS_JSON="${HYPERPARAMS_JSON:-}"
CONFIG_JSON="${CONFIG_JSON:-}"
EPISODES="${EPISODES:-}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-06b1f0786462bd443bd18ee458bfc4bdecaa67b8}"
FIXED_ETA="${FIXED_ETA:-}"
FIXED_I="${FIXED_I:-}"
FIXED_X="${FIXED_X:-}"
FIXED_HATCF="${FIXED_HATCF:-}"
FIXED_LNKF="${FIXED_LNKF:-}"
FIXED_HATC_CAL="${FIXED_HATC_CAL:-}"
FIXED_LNK_CAL="${FIXED_LNK_CAL:-}"
LOG_RESIDUAL_SCALE="${LOG_RESIDUAL_SCALE:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT" "$WORK_ROOT/.mplcache"
cd "$REPO_DIR"

if ! git merge-base --is-ancestor "$REQUIRED_COMMIT" HEAD; then
    echo "ERROR: HEAD does not contain required commit $REQUIRED_COMMIT"
    echo "HEAD: $(git rev-parse HEAD)"
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$WORK_ROOT/.mplcache"

echo "============================================================"
echo "Convergence surfaces for all episode checkpoints"
echo "============================================================"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"
echo "Required commit: $REQUIRED_COMMIT"
echo "Raw checkpoint dir: $RAW_CHECKPOINT_DIR"
echo "Analysis checkpoint dir: $CHECKPOINT_DIR"
echo "Run root: $RUN_ROOT"
echo "Data output dir: $DATA_OUTPUT_DIR"
echo "Output root: $OUTPUT_ROOT"
echo "Device: $DEVICE"
echo "B grid: [$B_MIN, $B_MAX] x $B_POINTS"
echo "Z grid: [$Z_MIN, $Z_MAX] x $Z_POINTS"
echo "N_CHILD_SHOCKS=$N_CHILD_SHOCKS"
echo "PARENT_CHUNK_SIZE=$PARENT_CHUNK_SIZE"
echo "CHILD_CHUNK_SIZE=$CHILD_CHUNK_SIZE"
echo "MAX_CHILD_STATE_EVALS=$MAX_CHILD_STATE_EVALS"
echo "ALLOW_LARGE_RUN=$ALLOW_LARGE_RUN"
echo "INCLUDE_RAW=$INCLUDE_RAW"
echo "INCLUDE_SIGNED=$INCLUDE_SIGNED"
echo "PACKAGE_ANALYSIS_CHECKPOINTS=$PACKAGE_ANALYSIS_CHECKPOINTS"
echo "PACKAGING_TRAINING_COMMIT=$PACKAGING_TRAINING_COMMIT"
echo "VALUE_SCALE_LOG_MAX=$VALUE_SCALE_LOG_MAX"
echo "EPISODES=${EPISODES:-auto}"
echo "LOG_RESIDUAL_SCALE=$LOG_RESIDUAL_SCALE"
echo "============================================================"

missing_fixed=()
for name in FIXED_ETA FIXED_I FIXED_X FIXED_HATCF FIXED_LNKF FIXED_HATC_CAL FIXED_LNK_CAL; do
    if [[ -z "${!name}" ]]; then
        missing_fixed+=("$name")
    fi
done
if [[ "${#missing_fixed[@]}" -gt 0 ]]; then
    echo "ERROR: fixed_grid requires these environment variables:"
    printf '  %s\n' "${missing_fixed[@]}"
    exit 1
fi

python3 - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device 0:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available on this Slurm allocation.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python3 -m py_compile \
    analysis/checkpoint_loader.py \
    analysis/convergence_transition.py \
    analysis/convergence_surface.py \
    scripts/plot_convergence_surface.py \
    scripts/package_scaled_episode_checkpoints.py

if [[ "$PACKAGE_ANALYSIS_CHECKPOINTS" == "1" ]]; then
    if [[ ! -d "$RAW_CHECKPOINT_DIR" ]]; then
        echo "ERROR: RAW_CHECKPOINT_DIR does not exist: $RAW_CHECKPOINT_DIR"
        exit 1
    fi
    PACKAGE_ARGS=(
        --run-root "$RUN_ROOT"
        --source-dir "$RAW_CHECKPOINT_DIR"
        --output-dir "$CHECKPOINT_DIR"
        --training-commit "$PACKAGING_TRAINING_COMMIT"
        --value-scale-log-max "$VALUE_SCALE_LOG_MAX"
        --pv-training-flow "$PV_TRAINING_FLOW"
        --pv-m-clamp-min "$PV_M_CLAMP_MIN"
        --pv-m-clamp-max "$PV_M_CLAMP_MAX"
        --bp-grid-policy-loss-space "$BP_GRID_POLICY_LOSS_SPACE"
        --bp-grid-logit-target-eps "$BP_GRID_LOGIT_TARGET_EPS"
        --bp-grid-logit-huber-delta "$BP_GRID_LOGIT_HUBER_DELTA"
        --pv-mixture-ratio "$PV_MIXTURE_RATIO"
        --pv-mixture-start-episode "$PV_MIXTURE_START_EPISODE"
        --pv-mixture-budget-mode "$PV_MIXTURE_BUDGET_MODE"
        --pv-mixture-sampling-mode "$PV_MIXTURE_SAMPLING_MODE"
        --pv-mixture-coverage-group-size "$PV_MIXTURE_COVERAGE_GROUP_SIZE"
        --pv-mixture-seed "$PV_MIXTURE_SEED"
        --firm-target-update "$FIRM_TARGET_UPDATE"
    )
    if [[ -n "$EPISODES" ]]; then
        PACKAGE_ARGS+=(--episodes "$EPISODES")
    fi
    if [[ "$PV_USE_CLIPPED_M" == "1" ]]; then
        PACKAGE_ARGS+=(--pv-use-clipped-m)
    else
        PACKAGE_ARGS+=(--no-pv-use-clipped-m)
    fi
    if [[ "$Q_USE_DETACHED_M" == "1" ]]; then
        PACKAGE_ARGS+=(--q-use-detached-m)
    else
        PACKAGE_ARGS+=(--no-q-use-detached-m)
    fi
    if [[ "$PV_MIXTURE_ENABLED" == "1" ]]; then
        PACKAGE_ARGS+=(--pv-mixture-enabled)
    else
        PACKAGE_ARGS+=(--no-pv-mixture-enabled)
    fi
    if [[ "$PV_MIXTURE_STRATIFIED_VALIDATION" == "1" ]]; then
        PACKAGE_ARGS+=(--pv-mixture-stratified-validation)
    else
        PACKAGE_ARGS+=(--no-pv-mixture-stratified-validation)
    fi
    if [[ "$PV_MIXTURE_PRESERVE_RNG" == "1" ]]; then
        PACKAGE_ARGS+=(--pv-mixture-preserve-rng)
    else
        PACKAGE_ARGS+=(--no-pv-mixture-preserve-rng)
    fi
    echo "------------------------------------------------------------"
    echo "Packaging scaled raw checkpoints into combined analysis checkpoints"
    echo "------------------------------------------------------------"
    python3 -m scripts.package_scaled_episode_checkpoints "${PACKAGE_ARGS[@]}"
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: CHECKPOINT_DIR does not exist after packaging: $CHECKPOINT_DIR"
    exit 1
fi

declare -a ANALYSIS_CKPTS=()
if [[ -n "$EPISODES" ]]; then
    for ep in $EPISODES; do
        ANALYSIS_CKPTS+=("$CHECKPOINT_DIR/ep${ep}_combined.pt")
    done
else
    while IFS= read -r path; do
        ANALYSIS_CKPTS+=("$path")
    done < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'ep*_combined.pt' | sort -V)
fi

if [[ "${#ANALYSIS_CKPTS[@]}" -eq 0 ]]; then
    echo "ERROR: no ep*_combined.pt files found in $CHECKPOINT_DIR"
    exit 1
fi

for CHECKPOINT in "${ANALYSIS_CKPTS[@]}"; do
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "ERROR: missing combined checkpoint: $CHECKPOINT"
        exit 1
    fi

    base="$(basename "$CHECKPOINT")"
    ep="${base#ep}"
    ep="${ep%%_*}"
    EP_OUT="$OUTPUT_ROOT/ep${ep}"
    mkdir -p "$EP_OUT"

    ARGS=(
        --checkpoint "$CHECKPOINT"
        --checkpoint-label "ep${ep}"
        --state-mode fixed_grid
        --eta "$FIXED_ETA"
        --i "$FIXED_I"
        --x "$FIXED_X"
        --hatcf "$FIXED_HATCF"
        --lnkf "$FIXED_LNKF"
        --hatc-cal "$FIXED_HATC_CAL"
        --lnk-cal "$FIXED_LNK_CAL"
        --b-min "$B_MIN"
        --b-max "$B_MAX"
        --b-points "$B_POINTS"
        --z-min "$Z_MIN"
        --z-max "$Z_MAX"
        --z-points "$Z_POINTS"
        --n-child-shocks "$N_CHILD_SHOCKS"
        --seed "$SEED"
        --parent-chunk-size "$PARENT_CHUNK_SIZE"
        --child-chunk-size "$CHILD_CHUNK_SIZE"
        --max-child-state-evals "$MAX_CHILD_STATE_EVALS"
        --device "$DEVICE"
        --output-dir "$EP_OUT"
    )

    if [[ "$ALLOW_LARGE_RUN" == "1" ]]; then
        ARGS+=(--allow-large-run)
    fi
    if [[ "$INCLUDE_RAW" == "1" ]]; then
        ARGS+=(--include-raw)
    fi
    if [[ "$INCLUDE_SIGNED" == "1" ]]; then
        ARGS+=(--include-signed)
    fi

    echo "------------------------------------------------------------"
    echo "Episode $ep"
    echo "Combined checkpoint: $CHECKPOINT"
    echo "Fixed context: eta=$FIXED_ETA i=$FIXED_I x=$FIXED_X hatcf=$FIXED_HATCF lnkf=$FIXED_LNKF hatc_cal=$FIXED_HATC_CAL lnk_cal=$FIXED_LNK_CAL"
    echo "Episode output: $EP_OUT"
    echo "------------------------------------------------------------"

    python3 -m scripts.plot_convergence_surface "${ARGS[@]}" 2>&1 | tee "$EP_OUT/plot_convergence_surface.log"

    for required in manifest.json surface_long.csv raw_surface.pt default_boundary.csv "ep${ep}_p0_conditional_abs.png" "ep${ep}_pi_conditional_abs.png" "ep${ep}_q_conditional_abs.png"; do
        if [[ ! -f "$EP_OUT/$required" ]]; then
            echo "ERROR: missing expected output for episode $ep: $EP_OUT/$required"
            exit 1
        fi
    done

    PNG_COUNT="$(find "$EP_OUT" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
    if [[ "$PNG_COUNT" -lt 3 ]]; then
        echo "ERROR: too few PNG outputs for episode $ep: $PNG_COUNT"
        exit 1
    fi
    echo "Episode $ep done. PNG count: $PNG_COUNT"
done

COMMON_OUT="$OUTPUT_ROOT/common_scale"
mkdir -p "$COMMON_OUT"
echo "------------------------------------------------------------"
echo "Post-processing common-scale fixed-grid plots"
echo "Common output: $COMMON_OUT"
echo "------------------------------------------------------------"

COMMON_ARGS=()
for d in "$OUTPUT_ROOT"/ep*; do
    if [[ -d "$d" ]]; then
        COMMON_ARGS+=("$d")
    fi
done
export COMMON_OUT
export LOG_RESIDUAL_SCALE
python3 - "${COMMON_ARGS[@]}" <<'PY'
import os
import sys
from analysis.convergence_surface import plot_fixed_grid_collection

episode_dirs = sys.argv[1:]
if not episode_dirs:
    raise SystemExit("No episode output directories found for common-scale post-processing.")
manifest = plot_fixed_grid_collection(
    episode_dirs,
    os.environ["COMMON_OUT"],
    log_residual_scale=os.environ.get("LOG_RESIDUAL_SCALE", "0") == "1",
)
print("Common scale manifest:", manifest)
PY

echo "============================================================"
echo "All convergence surface plots completed"
echo "Output root: $OUTPUT_ROOT"
echo "Finished: $(date)"
echo "============================================================"
