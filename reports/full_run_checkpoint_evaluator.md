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

- P0/PI/Q structural metrics use the common frozen reference grid. Each equation
  reports both training-M and raw-M residuals; backward-compatible unqualified
  aliases retain training-M semantics.
  `trainM` means a current-policy fixed-point residual evaluated with the
  checkpoint's `M_used`/clipped-M semantics. It is not a historical optimizer
  residual and does not use a historical target-network RHS. `rawM` evaluates
  the same current-policy fixed point with raw SDF M.
- `ondist_*` P0/PI/Q metrics use each episode's observed parent-state bank and
  likewise retain separate train-M/raw-M outputs.
- Q uses `compute_q_survival_recovery_components()` and reports
  `q_target_total - Q_parent`; it does not reimplement the pricing equation.
- Formal future eta integration remains exact and uses checkpoint `ZETA`.
- `sdf_common_*` uses one deterministic parent bank from the reference artifact
  for every checkpoint. `sdf_ondist_*` uses each episode's visited parent bank.
  Both use fresh common-random nested shock banks and
  `utils.metrics.conditional_moment_metrics()` for CMSE and the U-statistic.
  They also report the M distribution and the same pooled mean/variance
  inequality constraints used by the SDF implementation.
  Model convergence is read primarily from common-bank conditional residuals
  and U-statistics. Distribution-dependent feasibility (`E[M]`, `std(M)`, and
  `g_max`) is read primarily from the on-distribution parent bank.
- FC1 reports one-step RMSE/MAE/R2/correlation/slope/intercept, persistence
  skill, rollout RMSE/MAE/finite ratios, and timing shifts.
- Adjacent-checkpoint Q/P0/PI/P/bar_z/bp drift is computed by the existing
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
  evaluation_timing.json
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
      simulation/
        moments.csv
        metadata.json
      training_log/
        metrics.csv (when an explicit log row is available)
  cross_episode/
    config_comparability.json
    equation_residuals/
    function_drift/
    equation_error_vs_drift/
    macro/
    sdf/
    simulated_moments/
    log_diagnostics/
  figures/
    equilibrium_convergence_dashboard.png
    equation_error_by_episode.png
    equation_tail_error_by_episode.png
    function_drift_dashboard.png
    equation_error_vs_drift_rawM.png (primary, when adjacent drift is available)
    equation_error_vs_drift_trainM.png (training-semantics supplementary)
    sdf_convergence_dashboard.png
    fc1_convergence_dashboard.png
    simulated_moments_dashboard.png
  tables/
    cross_episode_config_invariants.csv
    headline_metrics_by_episode.csv
    equation_metrics_by_episode.csv
    macro_metrics_by_episode.csv
    sdf_metrics_by_episode.csv
    structural_metrics_by_episode_eta.csv
    sdf_metrics_long.csv
    sdf_validation_log_blocks.csv
    simulated_moments_by_episode.csv
    training_log_metrics_by_episode.csv
```

Only early, middle, and final episodes receive full firm-side visual output.
Every discovered/selected episode still receives numerical summaries. Missing
episodes and strict checkpoint reconstruction errors are preserved in
`headline_metrics.csv`, `errors.csv`, and `missing_artifacts.md`. Metadata
records input hashes, checkpoint hashes, model hashes before/after evaluation,
the structural shock-bank hash, evaluator commit, grid, and metric sources.

## Vectorization and comparability

- SDF robustness counts use a nested max-K bank. For child counts 32/64/128,
  the evaluator generates K=128 once, performs one SDF/FC1 forward, and computes
  each smaller result from tensor prefixes. Common and on-distribution scopes
  therefore require two SDF forwards per episode instead of six.
- Frozen transitions are represented as `[N,J,7]` before exact eta expansion
  and `[N,2J,7]` afterwards. Raw/used M use `[N,2J,1]`; branch weights use
  `[N,2J]` and sum to one for every parent.
- The firm robustness matrix loads the checkpoint once, evaluates static
  surfaces and investment once per current eta, builds one Jmax transition per
  eta, and derives smaller-J exact-eta prefixes. Bellman and BP computations
  retain flatten-and-chunk GPU forwards for memory control.
- BP candidate evaluation already uses joint parent x candidate x child tensors
  `[N,B,J,D]`, flattened to `[N*B*J,D]` within each candidate chunk. The
  evaluator now passes transition tensors directly into that path.
- `cross_episode_config_invariants.csv` hashes only stable economic and model
  semantics. Paths, devices, timestamps, and runtime fields are excluded. Any
  audited difference marks `cross_episode_comparable=false` but does not stop
  metric generation.
- `evaluation_timing.json` reports total, firm structural, common/ondist SDF,
  Bellman, BP, investment, FC1, and CUDA peak-memory measurements. CUDA is
  synchronized only at phase boundaries.

FC1 rollout retains its horizon recursion and path/origin loop in this change.
It is timed separately; batching it is deferred until a real run identifies it
as a material bottleneck. FC2 remains explicitly out of scope.

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
OVERWRITE=1  # required to replace a non-empty evaluator output directory
```

The Slurm requires one GPU on partition `a01`, uses the `DL_HL` environment,
performs CUDA and syntax/test preflight checks, and invokes only
`experiments/evaluate_full_run.py`.

## Known unavailable fields

- FC1 metrics are unavailable without an alignable macro artifact.
- Missing episode-specific firm data yields `status=partial`; structural-grid
  and common-reference SDF evaluation still run from the checkpoint and common
  reference artifact.
- Explicit default and investment rates remain unavailable when simulation
  data does not store realized indicators.
- Log metrics remain unavailable when no explicit or unambiguous log exists.
- No pass/fail threshold is imposed on episode drift or equation-error paths.
