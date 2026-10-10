# Investment-Compare Entry Approximation

## Scope

`entry_mode=investment_compare` is an explicit project approximation. It
reuses the trained investment branches to screen entrants; it is not presented
as a complete derivation of the Gomes entry problem.

The version identifier is:

```text
entry_spec_version=investment_compare_v1
```

`legacy` and `value_cost` remain available with their historical behavior.

## Entry decision

Each valid candidate draws productivity from the existing stationary firm
productivity distribution and draws one creation cost from the same support as
ordinary investment cost:

```text
e ~ Uniform(0, effective I_THRESHOLD)
```

The candidate state is fixed to:

```text
b=0, eta=0, i=e
```

The policy/value model's physical-value interface is evaluated once:

```text
P0_entry = V0_physical(0, z, 0, e, x, Hatcf, LnKF)
PI_entry = VI_physical(0, z, 0, e, x, Hatcf, LnKF)
entry_value_gap = PI_entry - P0_entry
accepted = valid_candidate and entry_value_gap >= 0
```

The evaluator does not use aggregate `P`, does not average eta regimes, does
not use `bar_i`, and does not subtract `e` after computing `PI_entry`.
Non-finite candidate inputs or physical P0/PI values raise an error.

## Birth and transition timing

At the child node where the firm is created:

```text
b=0
eta=0
i=entry_cost=e
K=K_birth
bar_i_executed=0
```

`bar_i_model` is retained as a diagnostic, but it is not executed at the birth
node. This prevents the creation investment from also being charged or applied
as ordinary expansion. The child view and promoted-parent view of the same
economic node use the same restriction.

At the next economic period, ordinary eta and i transitions resume. The firm
can then execute the ordinary investment policy, which affects capital in the
following period.

## Resource accounting

Creation spending is recorded once:

```text
I_entry = sum(entry_cost * K_birth)
C_reported = sum(C_operating) - I_entry
```

The birth-node ordinary expansion contribution is zero because
`bar_i_executed=0`. Existing economic-node ledger, independent reconstruction,
resource feasibility and capital decomposition checks remain active.

## Diagnostics and provenance

Firm output adds:

```text
Bar_i_model
Bar_i_executed
entry_P0
entry_PI
entry_value_gap
```

Macro output adds distribution summaries for `entry_P0`, `entry_PI`, and
`entry_value_gap`. The old `entry_cutoff_*` fields remain specific to
`value_cost_v1` and are not relabeled.

Metadata records the criterion, birth state, ordinary-investment cost source,
effective support, and the fact that birth extra expansion is disabled. These
fields enter the existing entry configuration fingerprint and P/Q grid cache
configuration hash.

## Short frozen-checkpoint smoke

The existing smoke command now includes a separate
`D_investment_compare_entry_raw_aggregation` arm:

```bash
python3 -u experiments/run_entry_mechanism_smoke.py \
  --checkpoint /path/to/combined_checkpoint.pt \
  --output-dir /path/to/entry_smoke \
  --device cuda:0 \
  --n-paths 2 \
  --group-size 16 \
  --horizon 3 \
  --branch-num 2
```

## Formal runner

The new mode must be selected explicitly:

```bash
python3 -u experiments/run_multi_episode_job.py \
  --run-root /path/to/run_root \
  --device cuda:0 \
  --entry-mode investment_compare \
  --consumption-aggregation-mode raw \
  --node-accounting-mode economic_node_ledger
```

Do not pass `--entry-cost-max` unless it exactly equals the effective
`I_THRESHOLD`; otherwise the runner rejects the configuration. `entry_dummy_i`
is not used in this mode.
