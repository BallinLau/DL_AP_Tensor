# FC2 Probe vs Episode Trainer Compare

## Purpose

This experiment fixes the simulated FC2 supervised dataset and compares two training protocols on exactly the same path split:

1. `probe trainer`
2. `episode-style trainer`

The goal is to isolate whether the `hatc` gap comes from:

- dataset / target definition
- or the training protocol itself

This avoids mixing in:

- outer loop effects
- clipping definition changes
- multi-episode drift

## Compared Trainers

### Probe Trainer

Matches `experiments/run_fc2_supervised_probe.py`:

- path-based train/val/test split
- feature normalization
- target normalization
- `hatc` x-only quadratic baseline fitted outside the network
- validation early stopping
- MLP probe trainer

### Episode-Style Trainer

Matches the mainline FC2 supervised pretrain more closely:

- same path-based train/val/test split
- raw feature scale
- raw target scale
- `FC2HatcModel` / `FC2LnkModel`
- `hatc` x-baseline fitted via `FC2HatcModel.fit_x_baseline`
- fixed epochs
- AdamW
- grad clipping
- optional cosine scheduler
- default dropout kept on unless overridden by CLI

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
- `compare_probe_hatc_scatter.png`
- `compare_episode_hatc_scatter.png`
- `compare_probe_lnk_scatter.png`
- `compare_episode_lnk_scatter.png`
- `compare_episode_style_loss_curves.png`

## Interpretation Rule

### If probe is clearly better than episode-style on the same test split

Then the current `hatc` gap is primarily a training-protocol problem.

### If both are similarly bad

Then the problem is not mainly the trainer; it is more likely:

- target difficulty
- summary sufficiency
- or object definition

## Suggested First Run

Use a small run first:

- `N_PATHS=2048`
- `HORIZON=4`
- `CKPT_PREFIX=ep30`

This is enough to answer the protocol question without paying for a full outer experiment.
