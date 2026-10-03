# BP Encoder Freeze/Unfreeze Ablation

## Question

This diagnostic changes one variable only: whether `policy_encoder` is updated while fitting a fixed EP2 BP teacher cache.

| Arm | Trainable prefixes |
| --- | --- |
| `frozen_encoder` | `bp0_head.`, `bpi_head.` |
| `trainable_encoder` | `policy_encoder.`, `bp0_head.`, `bpi_head.` |

Both arms start from independent deep copies of the same EP2 `post_bp` checkpoint. A reconstructed cache uses the production BP-stage teacher snapshot from EP2 `post_q_final`; the existing external diagnostic keeps its saved EP2 `post_bp` objective semantics. The arms use the same fixed cache, optimizer type/settings, and precomputed batch schedule. This is a diagnostic refit, not full-equilibrium training.

## Inputs

The entry point is `experiments/run_bp_encoder_freeze_ablation.py`.

Required inputs:

- source run root containing an unambiguous `ep2_combined.pt`, EP2 `post_bp.pt`, and EP2 `post_q_final.pt`;
- existing formal BP teacher-fit directory containing `metadata.json` and `tables/bp_fit_state_level_on_distribution.csv`;
- either explicit saved train/validation BP caches, or the saved EP2 firm parent/child dataframe used for one-time reconstruction.

Saved cache mode is preferred:

```bash
--cache-mode saved --train-cache /path/train.pt --val-cache /path/validation.pt
```

If no historical cache was saved, reconstruction uses the production `Episode._build_bp_target_cache()` once and records:

```text
cache_origin = reconstructed
historical_training_cache_exact = false
```

It does not run simulation and does not substitute evaluator Primary states for the training cache.

## Training Semantics

The experiment calls production `Episode._compute_bp_cache_loss()` directly. It therefore preserves checkpoint hyperparameters for:

- logit versus output loss space;
- target-logit clipping and Huber delta;
- BP0/BPI branch weights;
- confidence weights;
- mix loss and survival/sample weights;
- current-parent `eta_t` active-mask reduction.

The default main comparison is `last` versus `last` after 500 successful optimizer updates. `best.pt` is selected only as a supplemental checkpoint using the maximum branch-level confidence-weighted validation MAE. There is no early stopping.

The fixed schedule includes batches without active refinancing supervision. Both arms skip the same such batches and do not count them as successful optimizer updates.

## External Evaluation

Formal execution reconstructs only the existing on-distribution bank from the baseline diagnostic metadata. It requires the reconstructed ordered `source_index` sequence and shock-bank hash to match saved provenance. It evaluates:

- initial checkpoint;
- `frozen_encoder/last.pt` state;
- `trainable_encoder/last.pt` state.

The student supplies BP predictions only. The unchanged baseline teacher supplies objectives, `Phat` masks, value scales, action targets, and regrets. The initial high-error tail IDs are fixed before A/B comparison. Dense-grid and presentation diagnostics are not run.

## Outputs

The output directory is new and non-empty directories are rejected. Main files:

```text
resolved_config.json
cache_manifest.json
parameter_checks.json
training_metrics.csv
comparison.csv
validation_loss.png
summary.md
failure_report.json                  # only on failure
frozen_encoder/best.pt
frozen_encoder/last.pt
trainable_encoder/best.pt
trainable_encoder/last.pt
external_eval/initial_state_level.csv
external_eval/frozen_encoder_last_state_level.csv
external_eval/trainable_encoder_last_state_level.csv
external_eval/comparison.csv
external_eval/fixed_initial_tail_ids.csv
external_eval/fixed_tail_comparison.csv
external_eval/metadata.json
```

`parameter_checks.json` verifies optimizer membership, observed gradients, module changes, non-BP state and buffer invariance, and teacher/cache hash invariance.

## Slurm

The single-job GPU wrapper is:

```text
slurm/run_bp_encoder_freeze_ablation_ep2.slurm
```

Create the log directory before submission because `#SBATCH -o/-e` are resolved before the script runs:

```bash
cd /home/fit/zhuyingz/WORK/LiuHao
mkdir -p logs
```

Preflight-only submission:

```bash
sbatch --export=ALL,RUN_PREFLIGHT_ONLY=1,REQUIRED_COMMIT="$(git -C DL_AP_Tensor rev-parse HEAD)" \
  DL_AP_Tensor/slurm/run_bp_encoder_freeze_ablation_ep2.slurm
```

Formal submission using cache reconstruction from the saved EP2 dataframe:

```bash
sbatch --export=ALL,CACHE_MODE=reconstruct,REQUIRED_COMMIT="$(git -C DL_AP_Tensor rev-parse HEAD)" \
  DL_AP_Tensor/slurm/run_bp_encoder_freeze_ablation_ep2.slurm
```

Formal submission with saved caches:

```bash
sbatch --export=ALL,CACHE_MODE=saved,TRAIN_CACHE=/absolute/train.pt,VAL_CACHE=/absolute/validation.pt,REQUIRED_COMMIT="$(git -C DL_AP_Tensor rev-parse HEAD)" \
  DL_AP_Tensor/slurm/run_bp_encoder_freeze_ablation_ep2.slurm
```

Important environment overrides include `REPO_DIR`, `SOURCE_RUN_ROOT`, `BASELINE_DIAG_DIR`, `OUTPUT_DIR`, `EPISODE`, `STEPS`, `SEED`, `CHECKPOINT`, `TEACHER_CHECKPOINT`, `COMBINED_CHECKPOINT`, `TRAIN_CACHE`, `VAL_CACHE`, `CACHE_MODE`, `LEARNING_RATE`, `BATCH_SIZE`, and `REQUIRED_COMMIT`. Empty optional learning-rate or batch-size values are not passed; the checkpoint hyperparameters are then used. If both `checkpoints/ep2_combined.pt` and `checkpoints_analysis/ep2_combined.pt` exist, set `COMBINED_CHECKPOINT` explicitly instead of allowing an arbitrary choice.

## Local Validation

```bash
python3 -m py_compile experiments/run_bp_encoder_freeze_ablation.py
python3 experiments/run_bp_encoder_freeze_ablation.py --help
bash -n slurm/run_bp_encoder_freeze_ablation_ep2.slurm
python3 -m pytest -q tests/test_bp_encoder_freeze_ablation.py -p no:cacheprovider -p no:randomly
```

The tests do not assert that the trainable encoder outperforms the frozen encoder. That is the unresolved experimental question.
