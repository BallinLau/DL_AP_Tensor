# Full-run checkpoint evaluator

## Purpose

`experiments/evaluate_full_run.py` is a read-only evaluator for a completed
multi-episode run. It does not resume training, build optimizers, update target
models, or change economic parameters.

The evaluator extends the existing firm-side checkpoint evaluator. It does not
replace its frozen-grid surfaces, BPGridTeacher diagnostics, exact future-eta
integration, or objective slices.

## Command

```bash
python3 -u experiments/evaluate_full_run.py \
  --run-root /path/to/completed_run \
  --device cuda:0 \
  --n-child-shocks 64 \
  --robustness-child-shocks 32 64 128
```

Episode selection accepts comma-separated values and inclusive ranges:

```bash
--episodes 0,2:5,9
```

Pass `--training-log` when log metrics are needed. Without it, auto-discovery
is used only when exactly one file exists under `<RUN_ROOT>/logs`; otherwise
log-derived metrics remain unavailable instead of guessing a file.

## Statistical semantics

- P0/PI/Q structural metrics use the common frozen reference grid.
- `ondist_*` P0/PI/Q metrics use each episode's observed parent-state bank.
- Q uses `compute_q_survival_recovery_components()` and reports
  `q_target_total - Q_parent`; it does not reimplement the pricing equation.
- Formal future eta integration remains exact and uses checkpoint `ZETA`.
- SDF held-out metrics use fresh common-random nested shock banks and
  `utils.metrics.conditional_moment_metrics()` for CMSE and the U-statistic.
  They also report the M distribution and the same pooled mean/variance
  inequality constraints used by the SDF implementation.
- FC1 reports one-step RMSE/MAE/R2/correlation/slope/intercept, persistence
  skill, rollout RMSE/MAE/finite ratios, and timing shifts.
- FC2 is optional. When `models.fc2` is absent, the output contains a
  `missing.json` reason and headline values remain `NaN`. When present, parent
  and child nodes are evaluated separately. The evaluator preserves the
  current `FC2LossPipe` macro-input order and absolute-consumption aggregation
  instead of silently correcting those training semantics.
- Adjacent-checkpoint Q/P/bar_z/bp drift is computed by the existing
  `build_convergence_report.compute_function_drift()` implementation. It is
  not reimplemented in this evaluator.
- Equation-error-versus-drift output is diagnostic only; no pass/fail
  threshold is imposed.
- Formal evaluator values have priority over structured artifacts, which have
  priority over parsed log values.

## Output

```text
<RUN_ROOT>/data/outputs/full_run_evaluation/
  run_summary.csv
  run_summary.json
  headline_metrics.csv
  equation_residual_trajectories.csv
  training_log_metrics.csv
  errors.csv
  metadata.json
  missing_artifacts.md
  convergence_dashboard.png
  README.md
  episodes/
    epN/
      summary.csv
      summary.json
      metadata.json
      firm/
        summary.csv
        metadata.json
        on_distribution_metrics.csv
        eta0/
        eta1/
      sdf/
        metrics.csv
        metadata.json
      fc1/
        metrics.csv
        timing_alignment.csv
        rollout.csv
      fc2/
        metrics.csv or missing.json
        node_consistency.csv (when available)
      simulation/
        moments.csv
        metadata.json
      training_log/
        metrics.csv (when an explicit log row is available)
  cross_episode/
    equation_residuals/
    function_drift/
    equation_error_vs_drift/
    macro/
    sdf/
    fc2/
    simulated_moments/
    log_diagnostics/
  figures/
    equilibrium_convergence_dashboard.png
    equation_error_by_episode.png
    equation_tail_error_by_episode.png
    function_drift_dashboard.png
    equation_error_vs_drift.png (when adjacent drift is available)
    sdf_convergence_dashboard.png
    fc1_convergence_dashboard.png
    fc2_convergence_dashboard.png
    simulated_moments_dashboard.png
  tables/
    headline_metrics_by_episode.csv
    equation_metrics_by_episode.csv
    macro_metrics_by_episode.csv
    sdf_metrics_by_episode.csv
    fc2_metrics_by_episode.csv
    simulated_moments_by_episode.csv
    training_log_metrics_by_episode.csv
```

Only early, middle, and final episodes receive full firm-side visual output.
Every discovered/selected episode still receives numerical summaries. Missing
episodes and strict checkpoint reconstruction errors are preserved in
`headline_metrics.csv`, `errors.csv`, and `missing_artifacts.md`. Metadata
records input hashes, checkpoint hashes, model hashes before/after evaluation,
the structural shock-bank hash, evaluator commit, grid, and metric sources.

## Slurm

```bash
sbatch --export=ALL,RUN_ROOT=/path/to/completed_run \
  slurm/run_full_run_evaluator_gpu.slurm
```

Useful overrides:

```bash
EPISODES="0:9"
TRAINING_LOG=/path/to/stdout.log
N_CHILD_SHOCKS=64
ROBUSTNESS_CHILD_SHOCKS="32 64 128"
MAX_SDF_PARENTS=512
RUN_EVALUATOR_TESTS=1
REQUIRED_COMMIT=<minimum-evaluator-commit>
```

The Slurm requires one GPU on partition `a01`, uses the `DL_HL` environment,
performs CUDA and syntax/test preflight checks, and invokes only
`experiments/evaluate_full_run.py`.

## Known unavailable fields

- FC2 metrics are unavailable for checkpoints without `models.fc2`.
- FC1/FC2 metrics are unavailable without an alignable macro artifact.
- Explicit default and investment rates remain unavailable when simulation
  data does not store realized indicators.
- Log metrics remain unavailable when no explicit or unambiguous log exists.
- No pass/fail threshold is imposed on episode drift or equation-error paths.
