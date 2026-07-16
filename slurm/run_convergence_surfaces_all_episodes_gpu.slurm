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

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$WORK_ROOT/cachedir/normalized_sdf_aio_bp_logit_mixture_20260716_003048/checkpoints}"
RUN_ROOT="${RUN_ROOT:-$(dirname "$CHECKPOINT_DIR")}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-$RUN_ROOT/data/outputs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_OUTPUT_DIR/convergence_surfaces_all_${STAMP}}"

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
ALLOW_CURRENT_CONFIG="${ALLOW_CURRENT_CONFIG:-1}"
ALLOW_DEFAULT_HYPERPARAMS="${ALLOW_DEFAULT_HYPERPARAMS:-1}"
HYPERPARAMS_JSON="${HYPERPARAMS_JSON:-}"
CONFIG_JSON="${CONFIG_JSON:-}"
EPISODES="${EPISODES:-}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-be4500d}"
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
echo "Checkpoint dir: $CHECKPOINT_DIR"
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
echo "ALLOW_CURRENT_CONFIG=$ALLOW_CURRENT_CONFIG"
echo "ALLOW_DEFAULT_HYPERPARAMS=$ALLOW_DEFAULT_HYPERPARAMS"
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
    scripts/plot_convergence_surface.py

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: CHECKPOINT_DIR does not exist: $CHECKPOINT_DIR"
    exit 1
fi

declare -a POLICY_CKPTS=()
if [[ -n "$EPISODES" ]]; then
    for ep in $EPISODES; do
        POLICY_CKPTS+=("$CHECKPOINT_DIR/ep${ep}_policy_value.pt")
    done
else
    while IFS= read -r path; do
        POLICY_CKPTS+=("$path")
    done < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'ep*_policy_value.pt' | sort -V)
fi

if [[ "${#POLICY_CKPTS[@]}" -eq 0 ]]; then
    echo "ERROR: no ep*_policy_value.pt files found in $CHECKPOINT_DIR"
    exit 1
fi

for POLICY_CKPT in "${POLICY_CKPTS[@]}"; do
    if [[ ! -f "$POLICY_CKPT" ]]; then
        echo "ERROR: missing policy checkpoint: $POLICY_CKPT"
        exit 1
    fi

    base="$(basename "$POLICY_CKPT")"
    ep="${base#ep}"
    ep="${ep%%_*}"
    SDF_CKPT="$CHECKPOINT_DIR/ep${ep}_sdf_fc1.pt"

    if [[ ! -f "$SDF_CKPT" ]]; then
        echo "ERROR: missing SDF/FC1 checkpoint for episode $ep:"
        echo "  $SDF_CKPT"
        exit 1
    fi
    EP_OUT="$OUTPUT_ROOT/ep${ep}"
    mkdir -p "$EP_OUT"

    ARGS=(
        --policy-checkpoint "$POLICY_CKPT"
        --sdf-checkpoint "$SDF_CKPT"
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
    if [[ -n "$HYPERPARAMS_JSON" ]]; then
        ARGS+=(--hyperparams-json "$HYPERPARAMS_JSON")
    elif [[ "$ALLOW_DEFAULT_HYPERPARAMS" == "1" ]]; then
        ARGS+=(--allow-default-hyperparams)
    fi
    if [[ -n "$CONFIG_JSON" ]]; then
        ARGS+=(--config-json "$CONFIG_JSON")
    elif [[ "$ALLOW_CURRENT_CONFIG" == "1" ]]; then
        ARGS+=(--allow-current-config)
    fi

    echo "------------------------------------------------------------"
    echo "Episode $ep"
    echo "Policy checkpoint: $POLICY_CKPT"
    echo "SDF/FC1 checkpoint: $SDF_CKPT"
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
