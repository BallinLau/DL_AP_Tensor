# FC2 Probe vs Episode Trainer Compare

## Purpose

This experiment fixes the simulated FC2 supervised dataset and compares three training protocols on exactly the same path split:

1. `probe_full`
2. `probe_lite_raw`
3. `probe_lite_hatc_y_norm`

The goal is to isolate whether the `hatc` gap comes from:

- dataset / target definition
- or the training protocol itself

This avoids mixing in:

- outer loop effects
- clipping definition changes
- multi-episode drift

## Compared Trainers

### Probe Full

Matches `experiments/run_fc2_supervised_probe.py`:

- path-based train/val/test split
- feature normalization
- target normalization
- `hatc` x-only quadratic baseline fitted outside the network
- validation early stopping
- MLP probe trainer

### Probe-Lite Raw

Matches the current mainline FC2 supervised pretrain more closely:

- same path-based train/val/test split
- raw feature scale
- raw target scale
- `FC2HatcModel` / `FC2LnkModel`
- `hatc` x-baseline fitted via `FC2HatcModel.fit_x_baseline`
- early stopping
- AdamW
- grad clipping
- dropout off by default

### Probe-Lite Hatc Y-Norm

Same as `probe_lite_raw`, except:

- `hatc` still outputs in physical scale
- but the training loss standardizes the `hatc` residual on the train split
- this isolates the marginal value of `y` normalization without changing the FC2 law interface

## Entry Points

- Script:
  - `experiments/run_fc2_trainer_compare.py`
- Slurm:
  - `slurm/run_fc2_trainer_compare_80g.slurm`

## Main Outputs

The comparison run writes:

- `comparison_summary.json`
- `comparison_dataset.pkl`
- `comparison_test_predictions.csv`
- `compare_probe_full_hatc_scatter.png`
- `compare_probe_lite_raw_hatc_scatter.png`
- `compare_probe_lite_hatc_y_norm_scatter.png`
- `compare_probe_full_lnk_scatter.png`
- `compare_probe_lite_raw_lnk_scatter.png`
- `compare_probe_lite_loss_curves.png`

## Interpretation Rule

### If probe-lite hatc y-norm is clearly better than probe-lite raw

Then the remaining `hatc` gap is likely dominated by target-scale optimization geometry.

### If probe-lite raw and probe-lite hatc y-norm are similarly bad

Then the remaining `hatc` gap is not mainly a missing `y` normalization issue; it is more likely:

- target difficulty
- summary sufficiency
- or object definition

## Suggested First Run

Use a small run first:

- `N_PATHS=2048`
- `HORIZON=4`
- `CKPT_PREFIX=ep30`

This is enough to answer the protocol question without paying for a full outer experiment.
