# Firm-First Nested Training — Phase B Code Review Fixes

Date: 2026-10-19  
Repository: `BallinLau/DL_AP_Tensor`  
Branch: `exp/final-firm-first-review-fixes-20261019`  
Base commit: `ac1780a8ff1819b81a8edaec23484af26c513b8a`  
Implementation commit: `9e84a929ed8bc4cac2c11bfa9a3de21f893ef941`

## 1. Scope and outcome

This phase repaired the four confirmed P0 engineering defects and the P1
post-FC1 replenishment contract without changing the Firm-first architecture:

```text
Firm Inner
  -> Macro Candidate Generation
  -> FC1 Training
  -> Post-FC1 One-Step Refresh
  -> SDF / Wealth Training
  -> Outer Evaluation
  -> Next Firm Inner
```

No real Firm, FC1, or SDF optimization was run. No economic training dataset
was generated at production scale. No A800/Slurm job was submitted. No branch
was pushed.

Final code-review status: `CODE_REVIEW_FIXES_COMPLETED`.

## 2. Changed files

The implementation commit changes exactly these six files:

- `analysis/macro_parent_state_bank.py`
- `analysis/nested_equilibrium_utils.py`
- `analysis/run_full_nested_equilibrium.py`
- `data/tensor_data.py`
- `tests/test_final_firm_first_nested_training.py`
- `training/episode.py`

No Firm Bellman loss file, SDF loss file, boundary integration equation,
market-clearing equation, B-distribution parameter range, model architecture,
or economic configuration file changed.

## 3. P0-1 — FP32 Candidate ID collision

### Original failing call chain

```text
MacroCandidateSpec.candidate_id
  -> boundary_macro_rows(path=lineage_id=candidate_id)
  -> _macro_rows_to_table(all fields cast to float32)
  -> Episode._build_sdf_pairs_from_macro_tensor()
  -> float32 path converted back to int64
  -> Episode._create_sdf_batches_from_macro_tensor()
```

Large canonical identities such as `1000000000` and `1000000001` cannot both
be represented exactly as distinct FP32 values. Once the canonical ID entered
the float table, later conversion to `long` could not recover the lost bits.
Changing only an audit comparator would not have repaired production pairing.

### Repair

`TensorTable` now carries optional metadata while retaining the same numeric
matrix and column interface. For boundary-aware one-step Macro datasets:

- canonical candidate identity remains `int64` in metadata;
- the numeric FP32 table stores a dataset-local path ID `0..N-1`;
- parent and both children use the same exact local path ID;
- `lineage_id` in the numeric table uses the same exact local identity;
- `canonical_to_local` and `local_to_canonical` are stored as a bijection;
- `Episode._build_sdf_pairs_from_macro_tensor()` propagates the mapping;
- pairing audit translates local identities back to canonical identities;
- candidate target movement aligns on canonical candidate ID, not on a local
  ID that may be compacted after candidate failure/replenishment.

The float table schema did not gain or lose a numeric column. The container
schema gained metadata:

```text
identity_schema = canonical_candidate_to_local_path_v1
canonical_candidate_ids: int64 tensor
local_path_ids: int64 tensor
canonical_to_local: lossless map
local_to_canonical: lossless map
```

### Production-method tests

The tests call the real methods on `training.episode.Episode`, not only the
legacy `_FakeEpisode`:

- `Episode._build_sdf_pairs_from_macro_tensor()`;
- `Episode._create_sdf_batches_from_macro_tensor()`.

Covered canonical IDs:

```text
1000000000
1000000001
1000000002
1000000063
1000000064
1000000065

20000000
20000001
20000002

512 consecutive IDs beginning at 1000000000
```

The tests verify no collision, two correctly grouped children per parent,
branch `0/1`, exact canonical/local recovery, exact lineage grouping, and
canonical alignment when two datasets assign different local IDs.

Result: `PASS`.

## 4. P0-2 — Unconditional long rollout before Firm

### Original failing call chain

Three cold-start paths could run long simulations before the first Firm solve:

```text
main()
  -> _build_or_load_fixed_banks()
     -> run_simulation(horizon=args.bootstrap_horizon; default 100)

main()
  -> _build_or_load_macro_probe()
     -> _simulation_pair()
     -> run_simulation(horizon=args.long_diagnostic_horizon)

main()
  -> initial historical market comparison
     -> run_simulation(horizon=args.long_diagnostic_horizon)
```

`--no-run-long-diagnostics` did not suppress all three paths.

### Repair

- The Firm conditional transition bootstrap uses the production simulator for
  exactly one step. It still constructs matched incumbent parents and two
  independently cleared children with the existing economic operator.
- The fixed Macro function probe uses one-step production pairs.
- The initial historical comparison is now guarded by
  `args.run_long_diagnostics`.
- `_simulation_pair()` remains the long-horizon diagnostic implementation and
  is called only inside explicit long-diagnostic branches.
- `bootstrap_horizon` and `long_diagnostic_horizon` remain recorded arguments;
  they are not silently reused as pre-Firm training prerequisites.

No synthetic macro state incompatible with the Bellman equations was added.
The bounded bootstrap still uses `make_initial_state`, production
`run_simulation`, root-cleared matching, and the original child-shock fields.

### Control-flow tests

- A production fixed-bank test asserts both bootstrap calls use `horizon=1`.
- A real `main()` dependency-injection test makes every unauthorized
  `run_simulation(horizon > 1)` raise `AssertionError("Long rollout forbidden")`.
- That test reaches the production Firm problem construction call before any
  Macro candidate generation or optimizer invocation.

Result: unconditional pre-Firm long rollout removed; first actual optimization
stage remains Firm.

## 5. P0-3 — Resume configuration and optimizer state

### Original failing call chain

```text
parse current CLI defaults
  -> _args_fingerprint(current defaults)
  -> load checkpoint produced with non-default original args
  -> fingerprint mismatch
```

The documented short resume command did not reconstruct immutable arguments.
The base runner also referenced `_resolve_resume()` without defining it, so the
real resume entry path was not covered by prior production-flow tests.

In addition, checkpoints explicitly recorded
`optimizer_state_restored=False`; model-only resume could restart optimizer
moments and violate interrupted/uninterrupted equivalence.

### Repair

Resume now performs this sequence:

```text
resolve run directory / latest checkpoint
  -> read metadata/run.json
  -> restore every immutable training argument
  -> permit only runtime-only overrides
  -> recompute and verify original fingerprint
  -> construct matching models and optimizers
  -> load model + optimizer + RNG state
  -> validate bank/data/stage provenance
```

Runtime-only fields are limited to:

- resume target;
- device;
- output path;
- code-mismatch switch;
- cold-start/resume mode marker.

Device is excluded from the immutable fingerprint. Economic calibration,
sampling schedules, candidate budgets, root tolerances, architecture, optimizer
budgets, initial checkpoint provenance, and outer semantics remain immutable.
An explicitly supplied conflicting immutable argument fails fast.

The original warm-start path is restored as provenance but is not loaded a
second time during resume; the complete stage checkpoint is authoritative.

Checkpoint payloads now contain:

- model states and hashes;
- optimizer states and hashes;
- RNG state;
- immutable argument fingerprint;
- committed stage and data provenance.

An optimizer set mismatch or missing expected optimizer state fails fast.

### Tests

- non-default outer count, macro-inner count, train/validation target counts,
  seed, and custom initial checkpoint are recovered using only `--resume`;
- a conflicting explicit seed is rejected;
- Adam moments are populated, saved, loaded into a newly constructed optimizer,
  and hash-compared with the original state;
- RNG restoration remains part of the existing checkpoint path.

Result: `PASS`.

## 6. P0-4 — Macro-inner committed dataset version

### Original failing call chain

Uninterrupted execution did this:

```text
Macro Inner k
  -> FC1 fit on current dataset
  -> post-FC1 refresh
  -> current_dataset = post_dataset
  -> Macro Inner k+1 consumes post_dataset
```

Resume reconstructed this instead:

```text
resume Macro Inner k+1
  -> rebuild outer pre_fc1 dataset
  -> Macro Inner k+1 consumes pre_fc1 dataset
```

Thus the same checkpoint could continue from a different fixed-point input.

### Repair

Every post-FC1 train and validation dataset is atomically committed as a
tensor payload with:

- Macro table and Pair table;
- lossless canonical/local identity metadata;
- pairing audit and accepted canonical IDs;
- semantic dataset fingerprint;
- producer schedule fingerprint;
- Firm/FC1 model hashes;
- input and output dataset fingerprints;
- committed stage.

At `macro_inner=k>0`, both uninterrupted semantics and resumed semantics now
use `macro_inner=k-1` committed output. Missing payloads, missing manifests,
fingerprint mismatch, or a stage other than `macro_inner_complete` fail fast.
There is no silent resampling or candidate re-evaluation.

Committed dataset loading does not build randomized batches by default and
therefore does not advance RNG. FC1/SDF batches are constructed once at the
same production point in both uninterrupted and resumed execution.

The runner also persists:

- Macro-outer start function snapshot;
- per-inner input function snapshot;
- Macro probe history;
- FC1, post-refresh, SDF, macro-inner and outer checkpoints.

This prevents a resume at `fc1_complete`, `post_fc1_refresh_complete`, or
`sdf_complete` from replacing the pre-stage function baseline with the current
post-stage function when movement metrics are computed.

### Interrupted/uninterrupted tests

- committed Macro dataset save/load preserves tensors, identity maps and hash;
- loading does not change the torch RNG state;
- uninterrupted in-memory output and resumed previous-inner committed output
  have identical train/validation fingerprints;
- optimizer state round-trip is exact;
- repeated Firm-first stage recording remains idempotent;
- missing or inconsistent committed provenance is fail-fast by construction.

Result: `PASS`.

## 7. P1 — Post-FC1 Train replenishment

### Original behavior

Formal post-FC1 train refresh used:

```python
candidate_ids=accepted_train_ids
allow_replenishment=False
```

If four previously valid train candidates failed under updated FC1, the formal
train bank shrank even when the predeclared schedule still contained unused
candidates.

### Repair

Train refresh now:

- evaluates previously accepted candidates first with their original states
  and child shocks;
- retains their failures in the audit;
- continues only through the already fixed unused schedule;
- does not reuse an ID;
- stops at target valid count or the fixed maximum budget.

Validation refresh remains:

- fixed identity;
- no replenishment;
- no failed-child redraw;
- allowed to report a smaller currently valid subset without silently replacing
  it.

Each macro inner writes `post_fc1_replenishment_audit.json` with failed original
train IDs, replacement IDs, duplicate detection, pre/post validation IDs, and
the no-validation-replacement invariant.

The synthetic `16 -> 12 + 4 replacements` test verifies that train returns to
16 valid candidates using IDs `20000016..20000019`, while validation retains
only the 12 still-valid original IDs.

Result: `PASS`.

## 8. Why prior tests missed the defects

The existing orchestration mock validated only a predefined stage-name list.
It did not execute production `main()`, did not pass large canonical IDs through
the real Episode pairing path, and did not compare a committed Macro dataset or
optimizer state across interruption.

The new tests add production-path coverage for:

1. large canonical ID -> local FP32 ID -> real Episode Pairing;
2. 512 consecutive validation identities;
3. canonical movement alignment under different local numbering;
4. bounded one-step Firm bootstrap;
5. real `main()` reaching Firm while long rollout is forbidden;
6. immutable resume configuration reconstruction and conflict rejection;
7. model/optimizer/RNG checkpoint semantics;
8. committed dataset round-trip and prior-inner loading;
9. bounded post-FC1 train replenishment and fixed validation;
10. existing Firm-first stage-order/idempotence behavior.

## 9. Test evidence

### Targeted Phase B and related tests

```text
PYTHONPATH=. pytest -q \
  tests/test_final_firm_first_nested_training.py \
  tests/test_full_nested_equilibrium.py \
  tests/test_macro_parent_state_bank.py

43 passed
```

### Final full repository regression on implementation commit

```text
PYTHONPATH=. pytest -q

1094 passed, 1 skipped, 62 warnings in 256.06s
```

The warnings are pre-existing plotting, explicit BP-head override, and tensor
construction warnings. There was no test failure.

The pre-Phase-B report stated `1083 passed, 1 skipped`; this phase adds 11
tests, yielding `1094 passed, 1 skipped`.

## 10. Schema, equations, architecture and calibration

### Data schema

`TensorTable` gained optional metadata. Numeric table columns and dtypes used by
existing Firm/SDF consumers are unchanged. Boundary-aware Macro tables now use
exact local IDs in FP32 plus lossless canonical `int64` provenance.

### Economic equations

Unchanged. No resource equation, market-clearing residual, entry rule,
investment/default rule, Bellman target, AiO constraint, AL loss, root
tolerance, survival mass, consumption floor, or child filtering changed.

### Model structure

Unchanged. No layer, head, input dimension, output dimension, or parameterization
changed.

### Sampling design and calibration

Unchanged. B-only Beta family ranges, Sobol/stratified design, child shocks,
boundary quadrature, candidate maximum budgets, and fixed validation schedule
remain the same.

## 11. Items still requiring A800 smoke after external review

This Phase B intentionally did not run GPU smoke. After external diff review,
an A800 smoke should verify:

- production checkpoint size and optimizer serialization overhead;
- CUDA device transfer after loading optimizer states;
- one-step Firm bootstrap memory and runtime at smoke scale;
- candidate dataset commit/load on the persistent filesystem;
- resume independently from `fc1_complete`,
  `post_fc1_refresh_complete`, `sdf_complete`, and
  `macro_inner_complete` artifacts;
- identical semantic dataset/model/function-history hashes for interrupted and
  uninterrupted smoke routes.

These are runtime validations, not unresolved code-review failures.

## 12. Final status matrix

```text
BASE_COMMIT = ac1780a8ff1819b81a8edaec23484af26c513b8a

IMPLEMENTATION_COMMIT = 9e84a929ed8bc4cac2c11bfa9a3de21f893ef941

FP32_CANDIDATE_ID_COLLISION_FIXED = PASS

PRODUCTION_PAIRING_IDENTITY_PARITY = PASS

UNCONDITIONAL_BOOTSTRAP_LONG_ROLLOUT_REMOVED = TRUE

FIRST_ACTUAL_OPTIMIZER_STAGE_IS_FIRM = TRUE

RESUME_CONFIG_PARITY = PASS

MACRO_INNER_RESUME_DATA_PARITY = PASS

CHECKPOINT_STAGE_IDEMPOTENCE = PASS

POST_FC1_TRAIN_REPLENISHMENT_CONTRACT = PASS

FIXED_VALIDATION_IDENTITIES = PASS

FIRM_FIRST_ORDER_PRESERVED = TRUE

BOUNDARY_AWARE_ECONOMIC_OPERATOR_UNCHANGED = TRUE

B_DISTRIBUTION_DESIGN_UNCHANGED = TRUE

FIRM_BELLMAN_EQUATIONS_UNCHANGED = TRUE

SDF_AIO_OBJECTIVE_UNCHANGED = TRUE

ECONOMIC_CALIBRATION_CHANGED = FALSE

ARTIFICIAL_SURVIVAL_ADDED = FALSE

FAILED_CHILDREN_DROPPED = FALSE

REAL_TRAINING_PERFORMED = FALSE

SLURM_SUBMITTED = FALSE

READY_FOR_EXTERNAL_REVIEW = TRUE
```

`READY_FOR_EXTERNAL_REVIEW = TRUE` does not mean training convergence or A800
runtime validation. It means the requested Phase B fixes are implemented,
covered by targeted production-path tests, and pass the full CPU repository
test suite.
