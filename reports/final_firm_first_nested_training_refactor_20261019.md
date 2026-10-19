# Unified Firm-First Nested Equilibrium training refactor

Date: 2026-10-19  
Phase: A — code refactor and lightweight verification only  
Repository: `BallinLau/DL_AP_Tensor`

## Executive conclusion

The canonical runner now has one production Macro-data route:

```text
Firm Inner
  → predeclared path-specific B-only candidate schedules
  → boundary-aware Parent + Child 0 + Child 1 clearing
  → FC1 fit
  → same-candidate post-FC1 one-step refresh
  → SDF/Wealth fit
  → cross-block and outer evaluation
  → next Firm Inner
```

The formal entry no longer calls the short-rollout Parent harvest or the old
shared-distribution Macro initializer. Candidate failures are isolated and
audited; only complete Parent/two-Child groups enter FC1/SDF batches. The train
and validation namespaces are disjoint, and the validation candidate identity
set is locked after its first successful construction.

No model training, economic smoke, long simulation, Slurm submission, or push
was performed in this phase.

## A. Git and scope

```text
SOURCE_COMMIT = a067bfb4d79649dcd35f5775fb43b5644aa5df39
BRANCH = exp/final-firm-first-nested-training
PARTIAL-ACTIVE PREREQUISITE COMMIT = 3171092
IMPLEMENTATION COMMITS:
  ecf8e9f22f1d9527f4cc5f694751a937ec4fcead
  19a70b8  Align FC1 inputs with cleared current state
  601fceb  Harden stage resume against candidate re-evaluation
PUSHED = FALSE
```

The prerequisite commit is a selective port of the already-audited
partial-active market-clearing state merge. It preserves terminal-path status
and full-state merge semantics; it does not change calibration.

Implementation files:

- `analysis/macro_parent_state_bank.py`
- `analysis/run_full_nested_equilibrium.py`
- `data/market_clearing.py`
- `data/boundary_aware_market_clearing.py`
- `tests/test_final_firm_first_nested_training.py`
- `tests/test_full_nested_equilibrium.py`
- `tests/test_macro_parent_state_bank.py`
- `tests/test_boundary_aware_market_clearing.py`

Documentation files:

- `reports/final_firm_first_nested_training_refactor_20261019.md`
- `reports/final_training_commands_draft.md`

The original checkout's unrelated dirty/untracked files were not modified. All
work was performed in the isolated worktree
`/private/tmp/DL_AP_Tensor_final_firm_first_20261019`.

## B. Project structure and production call chain

### Entry file

The only formal orchestrator remains:

```text
analysis/run_full_nested_equilibrium.py::main
```

No second training runner or scheduler was added.

### Core modules

| Module | Responsibility |
|---|---|
| `analysis/run_full_nested_equilibrium.py` | Outer/inner ordering, freeze checks, checkpoints, resume, metrics |
| `analysis/macro_parent_state_bank.py` | Immutable candidate schedules, B-only initialization, failure isolation, one-step pair construction |
| `data/boundary_aware_market_clearing.py` | Continuous-mass decision integration, resource ledger, selected transport record |
| `data/market_clearing.py` | Existing vectorized coarse-to-fine root search and root classification |
| `training/episode.py` | Existing FC1/SDF pair and batch construction, optimizer execution |
| `analysis/run_macro_block_isolation.py` | Existing FC1 and SDF fitting routines and freeze invariants |

### Dependency relationships

```text
run_full_nested_equilibrium.main
  ├─ Firm P/Q solve and BP distillation
  ├─ _build_or_refresh_macro_candidate_context
  │    └─ build_macro_candidate_schedule
  ├─ build_one_step_macro_dataset
  │    ├─ make_boundary_candidate_initial_state
  │    ├─ _evaluate_boundary_candidates_with_isolation
  │    │    └─ BoundaryAwareMarketClearingController
  │    │         └─ existing MarketClearingController root search
  │    ├─ boundary_child_state (two keyed conditional macro shocks)
  │    └─ Episode._build_sdf_pairs_from_macro_tensor
  ├─ run_fc1_fit
  ├─ build_one_step_macro_dataset (same candidate IDs, post-FC1)
  ├─ run_sdf_chunks
  └─ cross-block Firm residual + outer metrics
```

Important function locations at the implementation commit:

- candidate schedule: `analysis/macro_parent_state_bank.py:374`
- B-only initializer: `analysis/macro_parent_state_bank.py:643`
- conditional child transition: `analysis/macro_parent_state_bank.py:881`
- failure isolation: `analysis/macro_parent_state_bank.py:1699`
- canonical one-step builder: `analysis/macro_parent_state_bank.py:1789`
- operator config lock: `analysis/run_full_nested_equilibrium.py:286`
- candidate context: `analysis/run_full_nested_equilibrium.py:1370`
- production stage recorder: `analysis/run_full_nested_equilibrium.py:1475`
- canonical `main`: `analysis/run_full_nested_equilibrium.py:1735`

Line numbers may move in later documentation-only commits; function names are
the stable references.

## C. Removed and retained legacy logic

### Removed from the formal entry

The body of `main()` contains no calls to:

- `prepare_macro_parent_schedule`;
- `harvest_available_macro_parent_schedule_with_candidate`;
- `_build_or_refresh_macro_parent_context`;
- the old short-rollout capture-time schedule;
- `SimulateTS._sample_initial_b` for primary Macro candidate creation.

The obsolete harvest scheduler was removed from the canonical runner rather
than left behind a runtime flag. A static production-entry test enforces this.

### Retained outside the formal route

Generic `SimulateTS` initializers and historical short-rollout helpers remain
available to Firm bootstrap, legacy replay, and independent diagnostics. They
are not selectable as an alternative formal Macro-training source. This avoids
breaking unrelated diagnostics while maintaining one formal training semantic.

Long-horizon simulation is controlled by `--run-long-diagnostics`, defaults to
false, and its artifacts are explicitly labelled diagnostic-only. The parser
still rejects `--use-long-rollout-for-macro-training`.

## D. New Macro sampling contract

### Candidate schedule

Candidate identity, lineage, split, Beta parameters, and shock keys are fixed
before model evaluation. The parameter schedule uses scrambled two-dimensional
Sobol points within seven balanced support strata:

| Family | alpha range | beta range |
|---|---:|---:|
| low leverage concentration | 0.65–2.00 | 4.00–10.00 |
| medium-low leverage concentration | 1.50–3.50 | 3.50–7.00 |
| middle concentration | 2.00–8.00 | 2.00–8.00 |
| medium-high leverage concentration | 3.50–7.00 | 1.50–3.50 |
| high leverage concentration | 4.00–10.00 | 0.65–2.00 |
| wide dispersion | 0.80–2.25 | 0.80–2.25 |
| two-tail concentration | 0.25–0.90 | 0.25–0.90 |

These are coverage-design parameters, not economic calibration values.

### Initial distribution

For each Parent (p):

\[
b_{pn}=b_{\min}+(b_{\max}-b_{\min})
F^{-1}_{\mathrm{Beta}(\alpha_p,\beta_p)}(q_n).
\]

Midpoint quantiles avoid endpoint singularities. The inverse-CDF operation is
vectorized across Parent/component coordinates with SciPy's verified beta
inverse, then transferred as one tensor to the target device. All subsequent
distribution construction, market evaluation, reductions, and child
transitions are batched tensors. There is no per-Parent/per-firm Python network
call. A native GPU inverse-CDF lookup/interpolation benchmark remains a Phase-B
performance item; no inaccurate approximation was introduced merely to claim
GPU residency.

Only `b` changes under a fixed keyed baseline. Tests compare two different
Beta schedules and establish bit-exact equality for current `x`, `hatcf`,
`lnkf`, capital allocations, component mass/mask, conditional z moments, and
eta states. Mass and total capital identities are also checked.

### Candidate budgets

```text
train target valid = 2048
train initial budget = 4096
train maximum budget = 8192
train minimum partial = 1024

validation target valid = 512
validation initial budget = 1024
validation maximum budget = 2048
validation minimum partial = 256

children per Parent = 2
```

The schedule contains the maximum candidate sequence from the start. Failure
does not cause resampling: evaluation advances through this predeclared order.
Records distinguish initial-budget attempts from bounded replenishment.

### Accepted dataset semantics

A valid training unit is exactly:

```text
one cleared Parent
  + cleared Child 0
  + cleared Child 1
```

Child shocks are keyed by candidate identity, time, and branch. Invalid or
terminal Child states remain in the audit but never enter FC1/SDF batches.
Candidate outcomes include family, parameters, all identities, Parent/Child
status, failure reason, acceptance flag, distribution hash, and root residuals.
The report also records outcome counts by distribution family, making selection
bias observable.

### Failure isolation

The normal evaluator is vectorized by candidate batch. If a typed economic
market-clearing failure occurs, the batch is recursively bisected until the
failed Candidate is isolated. Only typed economic failures are converted into
candidate records. Schema, interface, and other engineering failures propagate
immediately.

The mandatory 32-candidate mock produces exactly:

```text
23 valid
4 Parent clearing failures
3 Child-0 clearing failures
2 Child-1 clearing failures
```

It verifies complete pairing, no duplicate identities, bounded replenishment,
partial return on budget exhaustion, and engineering fail-fast behavior.

## E. FC1 and SDF data semantics

The Macro table keeps beliefs as diagnostic columns but feeds FC1 the true
current cleared state:

```text
FC1 input:
  x_t
  x_{t+1}
  h_t_cleared
  LnK_t_calculated

FC1 labels:
  h_{t+1}_cleared
  LnK_{t+1}_calculated
```

`h_belief` and `lnk_belief` are separate audit fields. The pair table carries
candidate lineage, time, and branch. An invariant requires exactly branches
0 and 1 at child time 1 for every accepted Parent.

After FC1 fit, one-step data are refreshed using precisely the accepted input
Candidate IDs. Training can no longer switch to a new candidate family simply
because FC1 changed. Fixed validation IDs never receive replacement children.
The refreshed complete pairs are the SDF/AiO source.

## F. Freeze contract

| Stage | Trainable | Required frozen/invariant |
|---|---|---|
| Firm P/Q/BP | Firm block selected by existing solver | SDF/FC1 Macro block hash |
| Macro candidate generation | none | Firm and Macro parameters |
| FC1 | FC1-C/FC1-K | Firm, SDF/Wealth/value submodules |
| post-FC1 refresh | none | all model parameters |
| SDF | SDF/Wealth | Firm and FC1 hashes |
| outer evaluation | none | all model parameters |

The runner already checks Firm hashes across MacroSolve and FC2 call/hash
invariance. `run_fc1_fit` and `run_sdf_chunks` retain their module-scope hash
checks. FC2 remains structurally guarded; recursive SDF epochs remain zero.

Because Phase A intentionally ran no real optimizer step, runtime freeze checks
are wired and unit-covered but a full GPU training freeze parity result is not
claimed.

## G. Checkpoint and resume contract

In addition to existing Firm P/Q and policy-distillation checkpoints, the
runner now publishes:

- `candidate_generation_complete.pt`;
- `macro_inner_*/fc1_complete.pt`;
- `macro_inner_*/post_fc1_refresh_complete.pt`;
- `macro_inner_*/sdf_complete.pt`;
- the existing macro-inner and outer-complete checkpoints.

Candidate batches are atomic `.pt` shards with schedule-parameter and dataset-
provenance fingerprints. Provenance includes model hashes at data generation,
integration config, root config, split, and component count. Resume from a
committed later stage is allowed to load those historical shards; it is not
allowed to silently recompute a missing committed shard under a newer model.

FC1 resume loads the committed summary/model checkpoint instead of repeating
FC1 optimization. SDF resume does the same after `sdf_complete.pt`. Partial
candidate generation continues at deterministic batch boundaries. The stage
trace is idempotent so a committed stage is not double-counted after resume.

## H. Tests

### Focused verification

```text
39 passed
```

Covered files:

- `tests/test_final_firm_first_nested_training.py`
- `tests/test_full_nested_equilibrium.py`
- `tests/test_macro_parent_state_bank.py`
- `tests/test_boundary_aware_market_clearing.py`

Coverage includes:

- deterministic/disjoint candidate schedules;
- all seven B-distribution support families;
- B-only preservation, mass and capital checks;
- production-entry exclusion of old Parent harvest;
- 23-valid/9-failure isolation with required 4/3/2 split;
- chunk-size invariant accepted IDs;
- bounded replenishment and partial dataset status;
- fixed validation identities with no replacement;
- engineering fail-fast;
- committed-shard resume without evaluator re-entry;
- exact Parent/two-Child pairing;
- two-outer Firm-first stage trace and idempotent resume;
- FC1 input equality with cleared `Hatc/LnK` rather than belief columns;
- partial-active terminal state merge and market-clearing regressions.

### Repository-wide verification

After the final resume hardening and documentation generation, the complete
suite reported:

```text
1083 passed, 1 skipped, 62 warnings
runtime = 264.37 seconds
```

The skip and warnings are pre-existing test-suite behavior (mostly plotting
axis warnings plus one explicit BP-head warning and one tensor-construction
warning); there were no failures.

### Not run

- no A800 GPU interface/parity benchmark;
- no real 2048/512 data generation;
- no optimizer training;
- no economic smoke;
- no Slurm job.

## I. Architecture decisions and limitations

1. The old generic initializers remain for non-formal tools, but cannot be
   reached from canonical `main()` for Macro training.
2. Exact Beta inverse quantiles are currently generated in one vectorized CPU
   operation and transferred in bulk. This is numerically preferable to an
   unvalidated GPU approximation, but A800 profiling is still required.
3. Economic failure isolation may recursively split a failed candidate batch;
   successful normal batches remain vectorized. Engineering failures are not
   swallowed.
4. Validation identities are fixed after the first valid selection. If a later
   model makes one invalid, it is reported and not replaced; the minimum-valid
   gate decides whether training may continue.
5. This refactor does not establish economic convergence, survival behavior,
   or forecast quality because training was explicitly out of scope.

## J. Next-stage readiness

The only next action after external code review should be the reviewed smoke in
`reports/final_training_commands_draft.md`. The smoke/pilot/full templates use
only real argparse options and were not executed.

No expansion to 8192/2048 valid parents is authorized by this change.

## Machine-readable status

```text
SOURCE_COMMIT = a067bfb4d79649dcd35f5775fb43b5644aa5df39
NEW_COMMIT = 601fceb

FIRM_FIRST_ORDER = PASS
CANONICAL_TRAINING_RUNNER = PASS
OLD_MACRO_INITIALIZATION_REMOVED = TRUE
PATH_SPECIFIC_B_DISTRIBUTION = PASS
B_DISTRIBUTION_COVERAGE = PASS
Z_LAW_UNCHANGED = TRUE
ETA_LAW_UNCHANGED = TRUE
I_LAW_UNCHANGED = TRUE
K_INITIALIZATION_UNCHANGED = TRUE

ROLLOUT_DEPENDENT_MACRO_PARENT_SOURCE_REMOVED = TRUE
BOUNDARY_AWARE_PRODUCTION_INTEGRATED = TRUE
PARENT_TWO_CHILD_CONTRACT = PASS
PARTIAL_FAILURE_TOLERANCE = PASS
BOUNDED_CANDIDATE_REPLENISHMENT = PASS
CANDIDATE_ID_REPRODUCIBILITY = PASS
FAILED_CHILD_REPLACEMENT = FALSE
FAILED_LINEAGE_AUDIT_PRESERVED = TRUE
TRAIN_VALIDATION_ISOLATION = PASS
FIXED_VALIDATION_SCHEDULE = PASS
FC1_TRUE_STATE_INPUT = PASS
FC1_POST_REFRESH = PASS
SDF_AIO_PAIRING = PASS

FIRM_MACRO_FREEZE_PARITY = NOT_VERIFIED
OUTER_STAGE_SEQUENCE = PASS
CHECKPOINT_RESUME = PASS
MOCK_END_TO_END = PASS
GPU_CPU_INTERFACE_PARITY = NOT_VERIFIED

ECONOMIC_EQUATIONS_CHANGED = FALSE
ECONOMIC_CALIBRATION_CHANGED = FALSE
MODEL_ARCHITECTURE_CHANGED = FALSE
CONSUMPTION_FLOOR_ADDED = FALSE
ARTIFICIAL_SURVIVAL_ADDED = FALSE

TRAINING_PERFORMED = FALSE
SLURM_JOB_SUBMITTED = FALSE
FORMAL_RUN_STARTED = FALSE
READY_FOR_EXTERNAL_CODE_REVIEW = TRUE
```
