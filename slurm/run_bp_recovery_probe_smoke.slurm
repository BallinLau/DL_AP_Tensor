#!/bin/bash
#SBATCH --job-name=bp_recovery_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=a01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source /apps/soft/anaconda3/bin/activate
CONDA_ENV="${CONDA_ENV:-DL_HL}"
conda activate "$CONDA_ENV"

REPO_ROOT="${REPO_ROOT:-/home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor}"
RUN_ROOT="${RUN_ROOT:-/home/fit/zhuyingz/WORK/LiuHao/cachedir/normalized_sdf_aio_full_20260712_174101}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$RUN_ROOT/data/outputs/target_grid_decomposition}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/data/outputs/bp_recovery_probes}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-}"

EPISODE=2
SEED=12345
N_STATES=20
STEPS=50
RECORD_STEPS="0,1,2,5,10,20,50"
HOLDOUT_FRACTION=0.2
INITIAL_BP_TARGET=0.5
LOGIT_TARGET_EPS="1e-4"
LOGIT_HUBER_DELTA=1.0
LEARNING_RATE=0.001
PROBES="baseline_output_loss,branch_only_output_loss,bias_recenter_output_loss,original_init_logit_loss"

cd "$REPO_ROOT"
mkdir -p logs "$OUTPUT_ROOT"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ -z "$REQUIRED_COMMIT" ]]; then
  REQUIRED_COMMIT="$ACTUAL_COMMIT"
fi
SHORT_SHA="$(git rev-parse --short HEAD)"
SMOKE_OUTPUT_DIR="$OUTPUT_ROOT/smoke/commit_${SHORT_SHA}/ep2_seed12345"
FIRM_PKL="$RUN_ROOT/data/outputs/ep${EPISODE}_stage_modeb.pkl"
SUMMARY_CSV="$SUMMARY_ROOT/ep${EPISODE}/ep${EPISODE}_target_grid_decomposition_summary.csv"
CHECKPOINT="$RUN_ROOT/checkpoints/ep${EPISODE}_policy_value.pt"

echo "============================================================"
echo "BP recovery probe smoke"
echo "============================================================"
date
hostname
pwd
echo "git rev-parse HEAD:"
git rev-parse HEAD
echo "git status --short:"
git status --short
python3 --version
echo "episode=$EPISODE"
echo "seed=$SEED"
echo "n_states=$N_STATES"
echo "steps=$STEPS"
echo "learning_rate=$LEARNING_RATE"
echo "holdout_fraction=$HOLDOUT_FRACTION"
echo "probes=$PROBES"
echo "run_root=$RUN_ROOT"
echo "firm_pickle=$FIRM_PKL"
echo "summary_csv=$SUMMARY_CSV"
echo "output_dir=$SMOKE_OUTPUT_DIR"
echo "required_commit=$REQUIRED_COMMIT"
echo "actual_commit=$ACTUAL_COMMIT"
echo "============================================================"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "ERROR: REPO_ROOT does not exist: $REPO_ROOT"
  exit 1
fi
if [[ ! -d "$RUN_ROOT" ]]; then
  echo "ERROR: RUN_ROOT does not exist: $RUN_ROOT"
  exit 1
fi
if [[ "$ACTUAL_COMMIT" != "$REQUIRED_COMMIT" ]]; then
  echo "Expected commit: $REQUIRED_COMMIT"
  echo "Actual commit:   $ACTUAL_COMMIT"
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean."
  git status --short
  exit 1
fi
for path in "$CHECKPOINT" "$FIRM_PKL" "$SUMMARY_CSV"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required input missing: $path"
    exit 1
  fi
done

python3 - "$SUMMARY_CSV" "$N_STATES" <<'PY'
import sys
import pandas as pd
summary = pd.read_csv(sys.argv[1])
required = int(sys.argv[2])
if "state_pos" not in summary.columns:
    raise SystemExit("summary is missing state_pos")
n_states = summary["state_pos"].nunique()
print(f"summary selectable states: {n_states}")
if n_states < required:
    raise SystemExit(f"summary has {n_states} states, required at least {required}")
PY

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")
print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python3 -m py_compile experiments/run_bp_recovery_probes.py experiments/summarize_bp_recovery_probes.py

python3 -m pytest \
  tests/test_bp_recovery_probes.py \
  tests/test_bp_fixed_teacher_refit.py \
  tests/test_bp_deep_diagnostics.py \
  -q

mkdir -p "$SMOKE_OUTPUT_DIR"
cat > "$SMOKE_OUTPUT_DIR/run_metadata.txt" <<EOF
started_at=$(date)
host=$(hostname)
repo=$REPO_ROOT
commit=$ACTUAL_COMMIT
episode=$EPISODE
seed=$SEED
n_states=$N_STATES
steps=$STEPS
learning_rate=$LEARNING_RATE
holdout_fraction=$HOLDOUT_FRACTION
probes=$PROBES
run_root=$RUN_ROOT
firm_pickle=$FIRM_PKL
summary_csv=$SUMMARY_CSV
output_dir=$SMOKE_OUTPUT_DIR
EOF

python3 experiments/run_bp_recovery_probes.py \
  --run-root "$RUN_ROOT" \
  --episode "$EPISODE" \
  --firm-pkl "$FIRM_PKL" \
  --decomposition-summary "$SUMMARY_CSV" \
  --device cuda \
  --output-dir "$SMOKE_OUTPUT_DIR" \
  --steps "$STEPS" \
  --record-steps "$RECORD_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --holdout-fraction "$HOLDOUT_FRACTION" \
  --seed "$SEED" \
  --n-states "$N_STATES" \
  --initial-bp-target "$INITIAL_BP_TARGET" \
  --logit-target-eps "$LOGIT_TARGET_EPS" \
  --logit-huber-delta "$LOGIT_HUBER_DELTA" \
  --probes "$PROBES"

python3 - "$SMOKE_OUTPUT_DIR" "$EPISODE" "$SEED" "$STEPS" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(sys.argv[1])
episode = int(sys.argv[2])
seed = int(sys.argv[3])
final_step = int(sys.argv[4])
probes = {
    "baseline_output_loss",
    "branch_only_output_loss",
    "bias_recenter_output_loss",
    "original_init_logit_loss",
}
history_path = out / f"ep{episode}_bp_recovery_probe_history.csv"
states_path = out / f"ep{episode}_bp_recovery_probe_states.csv"
summary_path = out / f"ep{episode}_bp_recovery_probe_summary.csv"
for path in [history_path, states_path, summary_path]:
    if not path.exists():
        raise SystemExit(f"missing output CSV: {path}")
history = pd.read_csv(history_path)
states = pd.read_csv(states_path)
summary = pd.read_csv(summary_path)
for label, df in [("history", history), ("states", states), ("summary", summary)]:
    if df.empty:
        raise SystemExit(f"{label} is empty")
    numeric = df.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise SystemExit(f"{label} contains non-finite numeric values")
for df in [history, states, summary]:
    if set(df["episode"].astype(int).unique()) != {episode}:
        raise SystemExit("unexpected episode values")
    if set(df["seed"].astype(int).unique()) != {seed}:
        raise SystemExit("unexpected seed values")
    if set(df["probe"].unique()) != probes:
        raise SystemExit(f"unexpected probes: {sorted(df['probe'].unique())}")
if set(history["split"].unique()) != {"train", "holdout"}:
    raise SystemExit("history must contain train and holdout splits")
steps = set(history["step"].astype(int).unique())
if not {0, final_step}.issubset(steps):
    raise SystemExit(f"missing required steps: {steps}")
if set(history["teacher_semantics"].unique()) != {"checkpoint_online_greedy_proxy"}:
    raise SystemExit("unexpected teacher semantics")
mix_flags = dict(summary.set_index("probe")["mix_training_included"].astype(bool))
if not bool(mix_flags["baseline_output_loss"]) or not bool(mix_flags["bias_recenter_output_loss"]):
    raise SystemExit("full probes must include mix training")
if bool(mix_flags["branch_only_output_loss"]) or bool(mix_flags["original_init_logit_loss"]):
    raise SystemExit("branch probes must exclude mix training")
probe_b = states.loc[states["probe"] == "bias_recenter_output_loss"]
median_bp = float(np.median(np.r_[probe_b["bp0_initial"].to_numpy(), probe_b["bpI_initial"].to_numpy()]))
if not (0.45 <= median_bp <= 0.55):
    raise SystemExit(f"Probe B initial median BP is not near 0.5: {median_bp}")
checkpoint_cols = ["bp0_checkpoint_initial", "bpI_checkpoint_initial", "bp_mix_checkpoint_initial"]
ref = states.loc[states["probe"] == "baseline_output_loss", checkpoint_cols].reset_index(drop=True)
for probe in ["branch_only_output_loss", "bias_recenter_output_loss", "original_init_logit_loss"]:
    cur = states.loc[states["probe"] == probe, checkpoint_cols].reset_index(drop=True)
    if not np.allclose(ref.to_numpy(), cur.to_numpy(), atol=0.0, rtol=0.0):
        raise SystemExit(f"checkpoint initial outputs differ for {probe}")
print("BP recovery smoke validator passed")
PY

echo "BP recovery smoke completed successfully"
