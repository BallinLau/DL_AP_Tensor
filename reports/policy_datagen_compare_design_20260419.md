# Policy Datagen Compare Design

## Purpose

This experiment tests one specific mechanism hypothesis:

- current `modeb` only promotes `main_branch` children into future parents
- side branches are used as one-step next-state targets but are not repeatedly revisited as Bellman parents
- this may weaken recursive Bellman closure on future states

The goal is not to build the final neural solver yet.
The goal is to verify whether **children replay into future parent pool** materially improves micro Bellman convergence.

## Compared Data Generation Schemes

### A. `main_branch`

Matches the current `SimulateTS` rollout logic:

1. current parent expands to all branch children
2. all branch children appear in current Bellman tuples
3. only `main_branch` becomes the next parent

### B. `children_promote`

Keeps the same one-step Bellman tuple structure, but changes how the next parent pool is updated:

1. current parent expands to all branch children
2. all branch children appear in current Bellman tuples
3. the next parent pool is sampled from **all children across the current pool**

This keeps the parent pool size fixed, so the tree does not explode.

## What Stays Fixed

To isolate the data-generation mechanism, the following stay fixed across A/B:

- initial checkpoint
- model architecture
- loss functions
- optimizer family
- policy staged training configuration
- parent pool size
- rollout depth
- training batch size
- total training epochs

## What Changes

Only one thing changes:

- how future parents are selected from current children

## Data Construction

The harness uses `SimulateTS` internals directly:

- `_initialize_path_tensor`
- `_process_node_tensor`
- `_expand_branches_tensor`
- `_apply_entry_tensor`
- `_apply_exit`

For each current parent:

1. emit one parent row block (`branch=-1`)
2. emit all child row blocks (`branch=0,1,...`)
3. update the next parent pool according to either:
   - `main_branch`
   - `children_promote`

The resulting tensor table is then fed into:

- `Episode._create_firm_batches_from_tensor(...)`
- `Episode._run_batches(..., train_modules=['policy_value'])`

## Primary Acceptance Metrics

Only micro-level metrics matter here:

- `conv_p0_mean / conv_p0_p90`
- `conv_pi_mean / conv_pi_p90`
- `conv_q_mean / conv_q_p90`

These are evaluated:

1. before training
2. after training

The updated harness reports metrics at two levels:

1. `train_support_*`
   - residuals on each arm's own training support
2. common evaluation supports
   - `eval_main_*`
   - `eval_children_*`
   - `eval_mixed_*`

The common evaluation metrics are the main decision metrics.
The train-support metrics are diagnostic only, because each arm trains on a different state distribution.

## Common-Support Evaluation

The first harness version evaluated each arm on its own support:

```text
train main_branch      -> eval main_branch support
train children_promote -> eval children_promote support
```

That is not a clean A/B because `children_promote` may generate harder states.

The updated harness evaluates both trained models on the same fixed supports:

```text
                 eval_main    eval_children    eval_mixed
train_main
train_children
```

This directly tests whether children promotion improves Bellman residuals on shared validation supports.
For `eval_mixed`, the child-promoted validation table is assigned a disjoint path-id range before concatenation.
This avoids accidental duplicate `(path,t,branch)` keys when the batching code reconstructs parent-child tuples.

## Clean A/B Initialization Guard

The harness now constructs the initial model once, snapshots every model `state_dict`, and loads that same snapshot into both arms.
The arms no longer rebuild independently from checkpoint paths.

Before any training starts, the harness evaluates both arms on the same common eval batches and asserts:

```text
max_abs_diff(common_eval_before_residuals) <= before_eval_tol
```

Default:

- `before_eval_tol = 1e-8`

If this check fails, the run aborts before training.
This prevents comparing:

```text
main_branch      = data A + init A
children_promote = data B + init B
```

The summary also records:

- `initialization.shared_initial_model_state`
- `initialization.policy_checkpoint_status`
- `initialization.state_stats`
- `initialization.before_eval_check`

## Stage-Level Evaluation

The harness also evaluates common-support convergence at:

- `q_stage_end`
- `pvbp_stage_end`
- `q_refresh_end`

This is necessary because the final stage may be `q_refresh`; final loss summaries alone can omit fresh `p0/pi` training terms.

## Decision Rule

### If `children_promote` beats `main_branch`

Meaning:

- lower post-train `conv_p0/pi/q`
- especially better in `conv_q` or `conv_pi`

Then the current rollout support is likely a real bottleneck.

### If they are similar

Then this mechanism is not the main blocker, and effort should move away from support replay and back toward:

- objective structure
- staged optimization
- regime-specific hardness

## Main Outputs

The run writes:

- `comparison_summary.json`
- `compare_conv_means.png`
- `compare_promoted_branch_share.png`
- `compare_eval_main_conv_means.png`
- `compare_eval_children_conv_means.png`
- `compare_eval_mixed_conv_means.png`
- `compare_eval_main_stage_conv_means.png`
- `compare_eval_children_stage_conv_means.png`
- `compare_eval_mixed_stage_conv_means.png`

The `eval_*` outputs are the primary outputs for interpretation.

## Entry Points

- Script:
  - `experiments/run_policy_datagen_compare.py`
- Slurm:
  - `slurm/run_policy_datagen_compare_80g.slurm`
