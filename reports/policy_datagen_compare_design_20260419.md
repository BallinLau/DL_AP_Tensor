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

## Entry Points

- Script:
  - `experiments/run_policy_datagen_compare.py`
- Slurm:
  - `slurm/run_policy_datagen_compare_80g.slurm`
