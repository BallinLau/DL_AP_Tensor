# Policy Loss Residual Alignment Audit

## Purpose

This audit tests one core question:

```text
When policy training loss decreases, do the Bellman residual metrics also decrease?
```

It is not a sample-support experiment and it does not run outer episodes or FC2.

## Why This Is Needed

Recent policy datagen comparisons suggest that support sampling is not the main bottleneck.
However, lower training losses do not automatically imply lower final Bellman residuals because the code uses different numerical objectives:

- training P0/PI use AIO-transformed residuals
- convergence P0/PI reports raw absolute residuals
- training P0/PI may use raw / clipped / annealed `M`
- convergence P0/PI currently reports raw-`M` residuals
- training Q total includes boundary, shape, and possible warm-start terms
- convergence Q reports only raw Bellman equation residuals under the configured Q `M` clamp

The audit therefore records both sides on the same fixed batches.

## Entry Points

- Script:
  - `experiments/run_policy_loss_residual_alignment_audit.py`
- Slurm:
  - `slurm/run_policy_loss_residual_alignment_audit_80g.slurm`

## Default Safety

The slurm entry requires a real policy checkpoint by default.
If `CKPT_DIR` does not contain policy checkpoint files, the run fails before training.

Valid checkpoint forms:

```text
epXX_policy_value_q.pt + epXX_policy_value_pvbp.pt
```

or:

```text
epXX_policy_value.pt
```

For smoke tests only, set:

```bash
ALLOW_RANDOM_POLICY_INIT=1
```

## Outputs

The run writes:

- `alignment_summary.json`
- `alignment_history.csv`
- `alignment_loss_vs_residual.png`
- `alignment_conv_means.png`

## Main Metrics

For each recorded epoch, the audit reports:

- `eval_p0_total_loss`, `eval_pi_total_loss`, `eval_q_total_loss`
- `eval_p0_main`, `eval_pi_main`, `eval_q_main`
- `p0_trainM_aio_mean`, `pi_trainM_aio_mean`, `q_trainM_aio_mean`
- `p0_trainM_raw_abs_mean`, `pi_trainM_raw_abs_mean`, `q_trainM_raw_abs_mean`
- `p0_rawM_raw_abs_mean`, `pi_rawM_raw_abs_mean`, `q_rawM_raw_abs_mean`
- `conv_p0_mean`, `conv_pi_mean`, `conv_q_mean`

## Interpretation

If AIO loss falls while raw absolute residual does not fall:

```text
training objective and convergence metric are not aligned
```

If train-`M` residual falls while raw-`M` residual does not fall:

```text
M clipping changes the target being optimized
```

If `pv_m_mode = raw`, then P0/PI training operator is aligned with raw-`M` Bellman residuals by construction.
If `pv_m_mode = anneal`, the audit manifest records both:

- `pv_m_mode`
- `pv_m_clip_anneal_epochs`

so the run can be interpreted against the exact transition schedule.

If Q total loss falls but Q raw Bellman residual does not fall:

```text
Q loss reduction may come from boundary / shape / warm-start terms rather than the Bellman equation
```

If all training objectives and raw residuals fall together but stop at a high floor:

```text
then the next suspect is optimization depth, model capacity, normalization, or equation implementation
```

## Example

```bash
CKPT_DIR=/home/fit/zhuyingz/WORK/LiuHao/cachedir/<run>/checkpoints \
CKPT_PREFIX=ep30 \
sbatch DL_AP_Tensor/slurm/run_policy_loss_residual_alignment_audit_80g.slurm
```
