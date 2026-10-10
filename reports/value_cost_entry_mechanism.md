# Value-cost enterprise entry specification

## Scope

This experiment adds `entry_mode=value_cost` while retaining the historical
`entry_mode=legacy`. It changes simulation entry and node accounting only. It
does not change P0/PI/Q Bellman objectives, BP teachers or losses, SDF/FC1
objectives, network dimensions, or checkpoint tensor shapes.

The formal production path is:

```text
run_multi_episode_job.py
  -> HyperParams
  -> Episode._prepare_simulation_kwargs()
  -> SimulateTS.simulate_tensor()
  -> simulate_tensor_parallel()
  -> data.entry value-cost primitives
```

`Sample` is a bootstrap/coverage generator. `fill_df_to_fullN` is a historical
FC2 full-N helper that can create synthetic positive-mass rows; it is not a
masked value-cost implementation. Both paths reject `entry_mode=value_cost`
instead of silently applying the legacy profit screen. Consequently FC2 must
remain disabled for this first value-cost experiment until its padding/data
contract is redesigned separately.
The old per-path `SimulateTS` compatibility methods also reject value-cost
entry; formal simulation uses the path-parallel tensor implementation.

## Timing and economic definition

For each path, the reference cross-section is the alive parent firms before
the transition. Padding and exited firms are excluded:

```text
K_ref       = sum_j K_j
N_ref       = number of reference firms
mean_K_ref  = K_ref / N_ref
K_birth     = alpha_E * mean_K_ref
K_nominal   = lambda_E * K_ref
n_star      = K_nominal / K_birth
```

`n_star` is stochastically rounded with the independent entry RNG. There is no
minimum candidate count and no replenishment after rejection. If `N_ref=0`,
the path creates no candidate and the formal raw-resource simulation reports
the extinct node instead of inventing capital.

Each candidate draws productivity from the stationary AR(1) distribution and
draws a separate creation cost `e ~ Uniform(0, entry_cost_max)`. Before drawing
the entrant's ordinary refinancing state or investment cost, the frozen policy
value model evaluates

```text
s_eta0 = (b=0, z, eta=0, i=dummy, x, Hatcf, LnKF)
s_eta1 = (b=0, z, eta=1, i=dummy, x, Hatcf, LnKF)
cutoff = (1-ZETA) * P(s_eta0) + ZETA * P(s_eta1)
accept = cutoff - e > 0
```

The accepted entrant is born with `b=0`, `K=K_birth`, and the already sampled
`z`. Only after acceptance are ordinary `eta` and `i` drawn independently.
There is no extra `G` multiplication at birth.

### Why `Bar_i` is not the entry cutoff

`Bar_i` is the model's conditional ordinary-investment object. Entry compares
an independently drawn creation cost with economic-scale integrated equity
value `P`. Replacing `P` with `Bar_i`, or using `PI >= P0`, would answer a
different decision problem and would mix entry cost with ordinary investment
cost.

### Why candidate capital is not actual entry capital

`lambda_E * K_ref` is a nominal candidate-pool scale. Stochastic rounding makes
realized candidate capital differ slightly, and value screening then rejects
some candidates. Therefore the output separately reports nominal potential
capital, integerized candidate capital, gross admitted capital, and surviving
entrant capital.

## Resource accounting

Creation cost and ordinary operation/investment remain distinct:

```text
I_entry = sum_accepted e_j * K_birth_j
C_oper  = sum_j (Y_j - I_oper_j - Phi_j)
C_raw   = C_oper - I_entry
I_total = I_oper + I_entry
```

The value-cost mode requires `consumption_aggregation_mode=raw`. Negative
firm-level operating contributions are not clamped away. If `C_raw <= 0`, the
formal simulation raises an explicit infeasibility error and does not pass a
fabricated positive consumption value to FC1/SDF.

Creation cost is a goods expenditure, while capital addition is a stock-flow
transition. `I_entry` therefore need not equal gross or endpoint entry capital.
The code does not add depreciation or financing charges to the specified
creation-cost account.

## Economic-node ledger

A child node later appears as the next loop's parent. Those are two data views
of one economic node, not two events. The first visit stores authoritative
`K`, `C_raw`, `Hatc`, `LnK`, operating investment, creation spending, and
capital-decomposition fields. Promotion reuses this account after the survivor
mask is applied. It does not redraw entrants, charge creation cost twice, or
recompute a macro state that omits the already-paid cost.

Outputs include `economic_node_id`, `node_view`, `accounting_stage`,
`initial_cohort`, `birth_time`, `entry_cost`, and `K_birth`. Aggregate analysis
must deduplicate `(path, economic_node_id)` before summing across time.

## Capital decomposition

For firm identities shared across start and endpoint (`S`), old exits (`X`),
and endpoint entrants (`E`):

```text
deltaK_incumbent = sum_{j in S} (K_next_j - K_current_j)
K_entry_endpoint = sum_{j in E} K_next_j
K_exit_old       = sum_{j in X} K_current_j

K_next - K_current
  = deltaK_incumbent + K_entry_endpoint - K_exit_old
```

Gross admissions are reported separately. An entrant that defaults in its
birth node still incurred `I_entry` and appears in `K_entry_same_node_exit`,
but it is absent from `K_entry_endpoint`.

## Configuration and compatibility

The runner exposes:

```text
--entry-mode {legacy,value_cost}
--entry-capital-ratio FLOAT
--entry-size-ratio FLOAT
--entry-cost-max FLOAT
--entry-dummy-i FLOAT
--entry-inference-chunk-size INT
--entry-rng-seed INT
--consumption-aggregation-mode {legacy_per_firm_clamp,raw}
--node-accounting-mode {auto,legacy_recompute,economic_node_ledger}
```

The full resolved entry/economic snapshot and its SHA-256 fingerprint are
stored in simulation metadata. Entry fields also participate in the P/Q target
cache configuration hash. Changing the entry mode or its parameters therefore
cannot be represented as the same cache configuration.

An old checkpoint can warm-start this experiment because model tensor shapes
are unchanged. It is not a seamless resume of the same economy: new simulation
data must be generated, and the resulting episode path is a new equilibrium
iteration under a different entry specification.

## Frozen short comparison

The example below runs no training:

```bash
python3 -u experiments/run_entry_mechanism_smoke.py \
  --checkpoint /path/to/epN_combined.pt \
  --output-dir /path/to/entry_smoke \
  --device cuda:0 \
  --seed 12345 \
  --n-paths 2 \
  --group-size 16 \
  --horizon 2 \
  --branch-num 2 \
  --simulation-bp-action-source checkpoint \
  --entry-capital-ratio 0.10 \
  --entry-size-ratio 0.10 \
  --entry-cost-max 1.0 \
  --entry-rng-seed 86420
```

It evaluates:

1. legacy entry plus legacy per-firm clamp;
2. legacy entry plus raw aggregation;
3. value-cost entry plus raw aggregation.

The later `investment_compare_v1` experiment is a separate, explicitly named
entry rule. It is documented in `reports/investment_compare_entry_mechanism.md`
and does not silently replace the `value_cost_v1` semantics described here.

The command resets the same initial seed for each arm, isolates entry draws,
and uses tagged common transition shocks so dynamic capacity cannot shift the
macro/incumbent shock streams. These numerical values are smoke defaults, not
a Gomes calibration. B versus C jointly changes the birth debt, relative size,
and screen, so their difference cannot be attributed to only one submechanism.

The output contains firm/macro pickles, macro CSVs, `summary.json`, accounting
residuals, entry cutoff/cost summaries, provenance, runtime, capacity, and
model-state hashes.
