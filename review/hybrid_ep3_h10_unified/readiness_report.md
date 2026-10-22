# EP3 + H10 Unified Two-Outer Experiment — Implementation Readiness

## Status

```ini
CODE_IMPLEMENTED = TRUE
TARGETED_TESTS = PASS
FORMAL_TRAINING_STARTED = FALSE
SLURM_JOB_SUBMITTED = FALSE
APPROVAL_CREATED = FALSE
OPTIMIZER_STEPS_EXECUTED_IN_THIS_TASK = 0
CURRENT_PHASE = WAITING_FOR_USER_REVIEW
```

## Experiment Contract

This experiment starts from the selected EP3 checkpoint weights only:

```text
normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_investment_compare_fresh100_20261014_144528
checkpoints/ep3_combined.pt
```

No EP3 optimizer, scheduler, RNG, simulation state, or training data state is restored. The converted initial checkpoint must use the reviewed `finite_direct_outer4_initial_weights_v1` weights-only format.

The intended run has exactly two new Outers. Each Outer follows:

```text
unified Fresh100 Firm data
H10 explicit P solver
EP3 direct-Q regime training (zero / default / survival / polish)
BP head skipped
H10 Macro Inner
80% on-policy Fresh100 replay + 20% H10-style coverage
FC1
SDF / AiO
cross-block evaluation
Outer checkpoint
```

## Unified Firm Data

P and Q consume the same committed Fresh100 parent/child pool inside an Outer.

The Fresh100 contract is:

```ini
MODE = modeb
PATHS = 200
FIRMS = 200
HORIZON = 100
BRANCHES = 2
ENTRY_MODE = investment_compare
STATE_BANK_ENABLED = false
PV_MIXTURE_ENABLED = true
PV_MIXTURE_COVERAGE = 0.20
PV_TRAINING_FLOW = staged
BP_HEAD_TRAINING = false
SIMULATION_BP_ACTION_SOURCE = grid
EPISODE_ID = 1
```

The unified pool has a content fingerprint. Both the P and Q helpers verify the same fingerprint before execution.

The P helper delegates to the existing H10 `_p_policy_outer`. It additionally verifies that Q and BP parameters are unchanged.

The Q helper delegates to the existing EP3 `Episode._run_q_regime_training` phases:

```text
zero
default
survival
polish
```

It verifies direct-Q semantics and that P and BP parameters are unchanged.

The BP helper never invokes a BP optimizer. It checks that:

```ini
BP_HEAD_TRAINING_SKIPPED = true
BP_HEAD_OPTIMIZER_STEPS = 0
SIMULATION_BP_ACTION_SOURCE = grid
```

and records the BP parameter hash.

## Macro Data

Training macro rows use:

```ini
ON_POLICY_FRESH100 = 80%
H10_STYLE_COVERAGE = 20%
TARGET_PARENT_TIME_GROUPS = 4096
```

The existing 50/50 H10 replay selector was generalized with an explicit `on_policy_fraction`. Its default remains 0.5 for backward compatibility; the hybrid experiment passes 0.80.

The Fresh100 macro adapter:

- uses the primary Child branch;
- applies the H10 20-period burn-in;
- preserves actual `C`, `K`, `Hatc`, and `LnK` ledgers;
- marks undefined rows invalid;
- never fabricates a second Child or fills an undefined account.

Validation does not use the 80/20 mixture. It remains the fixed H10 Validation bank and CRN.

## What Is Not Implemented Yet

The production two-Outer execution backend and Slurm submission are intentionally not connected in this review commit. The runner is plan-only and requires a future explicit, content-bound approval after this review.

This is a safety boundary, not a claim that training cannot be executed. The next implementation step after user approval is to wire the reviewed helpers into the existing H10 StageJournal/Outer execution backend.

## Tests

Targeted non-optimization tests passed:

```text
68 passed
```

Covered:

- exact Fresh100 contract;
- two-Outer plan and unified data stages;
- BP grid source and disabled BP head;
- P/Q same-pool fingerprint;
- P-only parameter guard;
- Q-only parameter guard;
  - Q zero/default/survival/polish phase contract;
- 80/20 macro mixture;
- fixed Validation rejection of 80/20 mixture;
- Fresh100 macro replay adapter and burn-in;
- no duplicate parent-time rows;
- default runner plan-only behavior and no approval artifact.

No real GPU simulation, optimizer step, Slurm submission, or formal training was executed.
