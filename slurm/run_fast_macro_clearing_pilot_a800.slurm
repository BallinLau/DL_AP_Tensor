#!/bin/bash
#SBATCH -J fast-macro-clear
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
conda activate DL_HL

REPO_DIR="${REPO_DIR:?Set REPO_DIR to the isolated source directory}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:?Set INITIAL_CHECKPOINT}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
SOURCE_COMMIT="${SOURCE_COMMIT:?Set SOURCE_COMMIT to the archived source commit}"
PARENTS="${PARENTS:-32}"
FIRMS="${FIRMS:-200}"
Z_SCAN_POINTS="${Z_SCAN_POINTS:-65}"
I_SCAN_POINTS="${I_SCAN_POINTS:-33}"
CANDIDATE_MICROBATCH="${CANDIDATE_MICROBATCH:-4}"
SMOKE_TRAIN="${SMOKE_TRAIN:-1}"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs"
cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export SOURCE_COMMIT
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/.mplcache}"
mkdir -p "$MPLCONFIGDIR"

python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory
print({"gpu": name, "memory": memory, "torch": torch.__version__})
if "A800" not in name.upper() or memory < 75 * 1024**3:
    raise SystemExit("pilot requires one 80GB A800")
PY

EXTRA=()
if [[ "$SMOKE_TRAIN" == "1" ]]; then
  EXTRA+=(--smoke-train)
fi

python3 -u analysis/run_fast_macro_clearing_pilot.py \
  --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --output-dir "$RUN_ROOT" \
  --device cuda:0 \
  --parents "$PARENTS" \
  --firms "$FIRMS" \
  --z-scan-points "$Z_SCAN_POINTS" \
  --i-scan-points "$I_SCAN_POINTS" \
  --microbatch "$CANDIDATE_MICROBATCH" \
  "${EXTRA[@]}" \
  2>&1 | tee "$RUN_ROOT/logs/run.log"
