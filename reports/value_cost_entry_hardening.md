# Value-cost entry hardening audit

## Scope and baselines

This bounded correction is based on implementation commit
`5195ad058077f5fd45865c0ce8ed050be0f4fb6a`. Historical compatibility is
measured against `4b236eaacf47b2ac7cb506508e54b21a199e89b0`.

The correction does not change the entry cutoff, creation-cost distribution,
birth size, `b_birth=0`, P/Q/BP/SDF objectives, debt timing, network shapes, or
checkpoint tensors. It changes provenance, diagnostic random-number matching,
node-accounting selection, and independently checkable accounting outputs.

## Findings and corrections

### 1. Smoke `Namespace` mutation

**Reproduction.** `report["smoke_parameters"] = vars(args)` returned the live
namespace dictionary. Converting `checkpoint` and `output_dir` for JSON then
changed the runtime `Path` objects into strings.

**Correction.** `_jsonable_namespace()` creates a separate dictionary. The
entry-level smoke test calls `main()`, runs all three arms, checks
`summary.json`, and verifies both namespace fields remain `Path` instances.
Failure paths always execute model-state hashing and write the report. Three
economic failures now produce a nonzero program failure rather than a nominal
success.

### 2. BP action source provenance

**Reproduction.** The smoke omitted `bp_action_source` and therefore silently
used the head default, even when the checkpoint experiment used grid actions.

**Correction.** `--simulation-bp-action-source {checkpoint,head,grid}` defaults
to `checkpoint`. Checkpoint mode accepts only an explicitly recorded effective
source. Missing or invalid provenance fails. A checkpoint that selects a
disabled head fails; an explicit diagnostic head override is allowed with a
recorded warning.

Grid mode constructs the production `GridBPSimulationPolicy` with the
checkpoint `firm_target` when present (otherwise `policy_value`), checkpoint
SDF/FC1, economic parameters, hyperparameters, child-shock count, and shock
seed. It does not reimplement the teacher objective and does not fall back to
the head on CPU. A fresh resolver is built per arm, so instrumentation is not
cumulative across A/B/C.

### 3. Identity-keyed common random numbers

**Reproduction.** Resetting a generator by `(t, branch, tag)` and drawing the
current tensor shape ties firm innovations to flattened slots. Capacity growth,
padding, exits, or slot permutation can then reassign shocks among common
firms.

**Correction.** The explicit `stable_firm_identity` mode derives vectorized
uniform/normal innovations from integer keys:

```text
macro: (seed, path, economic_time, branch, shock_type)
firm:  (seed, path, economic_time, branch, firm_id, shock_type)
```

It uses no Python `hash()`, no per-firm generator, and no per-firm network
call. `eps_x`, `eps_z`, the eta uniform, and the ordinary-i uniform are
exported for direct comparison. The historical `legacy_position` mode remains
the default outside the controlled smoke. Matching is claimed only for the
common initial cohort; independently born entrants are explicitly not treated
as the same economic firm merely because numeric IDs happen to coincide.

### 4. Legacy recomputation versus economic-node ledger

**Reproduction.** Reusing the new economic-node cache unconditionally changed
legacy promoted-parent `K/C/Hatc/LnK` after same-node exits.

**Correction.** `node_accounting_mode` resolves as follows:

```text
auto + legacy entry     -> legacy_recompute
auto + value_cost entry -> economic_node_ledger
```

Legacy entry may explicitly opt into the ledger for a controlled comparison.
Value-cost entry may not use `legacy_recompute`, because that would erase
already-paid creation expenditure from the promoted-parent view. The setting
is routed through CLI, `HyperParams`, `Episode`, simulation metadata,
fingerprints, and P/Q grid-cache hashes.

A frozen regression fixture with actual exits reproduces the 4b236ea parent
values (`K`, `C`, `LnK`, `Hatc`, firm count) and verifies the next FC1 call
receives those realized parent macro values.

### 5. Capital-decomposition endpoints

**Reproduction.** The decomposition used post-exit firms while the public
node `K` could describe the complete pre-exit economic node. A zero residual
therefore did not prove that the displayed public `K` was the decomposed
endpoint.

**Correction.** Outputs now distinguish:

```text
K_node_pre_exit
K_endpoint_post_exit
K_decomposition_start
K_decomposition_end
decomposition_start_stage / decomposition_end_stage
decomposition_start_node_id / decomposition_end_node_id
decomposition_transition_valid
```

The checked identity is:

```text
K_decomposition_end - K_decomposition_start
  = deltaK_incumbent + K_entry_endpoint - K_exit_old
```

The first node has no fabricated transition and is marked invalid/NaN. Firm
IDs and capital arrays must be finite, shape-aligned, and unique. With
`enable_exit=False`, the endpoint includes all actual live firms and realized
exit capital is zero. Gross entry is independently checked as surviving entry
capital plus same-node entrant exits.

### 6. Independent resource reconstruction

**Reproduction.** The former residual subtracted `C_oper - I_entry` from the
same expression used to define `C_raw`, making it identically zero even if an
upstream entry expense was missing or duplicated.

**Correction.** `validate_node_resource_account()` rebuilds entry spending
from birth rows (`entry_cost * K_birth`) and operating resources from firm
details (`Y - I_oper - Phi`). It then independently checks:

```text
entry_spend_residual       = I_entry_ledger - I_entry_rebuilt
resource_accounting_residual = C_reported - C_rebuilt
```

Entry and resource residuals use separate scale-aware tolerances. Current
capital, birth capital, costs, and operating details must be finite; raw mode
also requires finite positive consumption. Legacy clamp mode reports the
clamp adjustment and is not described as raw resource clearing.

Fault-injection tests cover a changed ledger amount, duplicated birth,
deleted birth, changed reported consumption, nonfinite values, and correct
positive/negative operating contributions. A separate test reconstructs the
same quantities from the actual exported firm table rather than calling the
production validation helper.

## Controlled A/B/C design

All three smoke arms use one checkpoint, one resolved BP source, matched
initial states, identity-keyed initial-cohort transition shocks, and
`economic_node_ledger`:

| Arm | Entry rule | Consumption aggregation | Interpretation |
|---|---|---|---|
| A | legacy | legacy per-firm clamp | Aggregation-control numerator |
| B | legacy | raw | Aggregation-control denominator |
| C | value-cost | raw | Joint entry-rule package |

A versus B isolates the aggregation convention conditional on the new ledger.
B versus C changes screening, birth debt, and entrant size together; it does
not identify one submechanism. A is not the historical 4b236ea behavior. That
behavior is covered separately by `legacy_recompute + legacy_position`.

If one arm is economically infeasible, its node/error is recorded and the
remaining arms continue. Program/configuration/accounting errors are recorded
and terminate the smoke. If every arm is economically infeasible, the command
fails after writing `summary.json` and checking model immutability.

## Validation status

The CPU test suite covers the six failure mechanisms, including nonzero macro
and firm shocks, two paths, capacity growth, padding, slot permutation, and
actual exits. It also verifies the smoke entry point and grid-source routing
with a resolver spy.

Executed in this worktree:

```text
pytest tests/test_entry_value_cost.py tests/test_simulate_ts.py
  39 passed

pytest tests/
  774 passed, 1 skipped

python -m compileall -q analysis config data experiments training tests
  passed

git diff --check
  passed
```

Not yet executed in this worktree:

- a real checkpoint smoke;
- CUDA `GridBPSimulationPolicy` execution;
- Slurm or long training.

Those omissions are intentional: this task has no supplied real checkpoint
and forbids launching training. A CPU routing mock is not evidence that the
GPU grid teacher has been numerically exercised.

## Real frozen-smoke command

For a checkpoint that explicitly records its simulation BP source:

```bash
python3 -u experiments/run_entry_mechanism_smoke.py \
  --checkpoint /path/to/epN_combined.pt \
  --output-dir /path/to/entry_smoke \
  --device cuda:0 \
  --simulation-bp-action-source checkpoint \
  --seed 12345 \
  --n-paths 2 \
  --group-size 16 \
  --horizon 2 \
  --branch-num 2 \
  --entry-capital-ratio 0.10 \
  --entry-size-ratio 0.10 \
  --entry-cost-max 1.0 \
  --entry-rng-seed 86420
```

Use an explicit `head` or `grid` only when intentionally overriding missing or
different checkpoint provenance. Grid mode requires CUDA and a checkpoint
containing SDF/FC1 plus the recorded grid hyperparameters.
