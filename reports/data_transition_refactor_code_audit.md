# Data Transition Refactor — Code Audit

Commit audited before the patch: `ab47fd98f0b0072a72cf6d2fdb8eb2eff80c8fa6`

Server parity: THU checkout `/home/fit/zhuyingz/WORK/LiuHao/DL_AP_Tensor` was on the same `exp/full-nested-equilibrium` commit.

## Production simulation path

```text
analysis/run_full_nested_equilibrium.py
  -> run_simulation()
  -> analysis/run_macro_outer_inner_pq.py::_simulation_kwargs()
  -> data/simulate_ts.py::SimulateTS.simulate_tensor()
  -> data/simulate_ts_parallel.py::simulate_tensor_parallel()
  -> _process_node_batched(parent)
       computes Hatc_cal_t and LnK_cal_t from realized resources
  -> _expand_branches_batched()
       calls _predict_macro_fc1_batched(
         x_t, x_t1, Hatc_cal_t, LnK_cal_t
       )
  -> _process_node_batched(child)
       records child forecast, M, realized resources and node IDs
```

Production `SimulateTS` already used the intended `cal_t -> f_{t+1}` timing.

## Old Firm fixed-point transition path

```text
analysis/run_full_nested_equilibrium.py::_build_or_load_fixed_banks()
  -> training/inner_fixed_point.py::build_collocation_bank()
       independently sampled macro forecast support and micro support
       discarded node-matched calculated macro context
  -> training/inner_fixed_point.py::build_child_bank()
       OLD MacroTransitionContext(
         hatc_cal=parent[:, 5:6],
         lnk_cal=parent[:, 6:7],
       )

analysis/run_full_nested_equilibrium.py::_refresh_children()
  -> analysis/run_macro_outer_inner_pq.py::refresh_transition_bank()
       OLD forward_step(...,
         hatcf_prev=parents[:, 5:6],
         lnkf_prev=parents[:, 6:7],
       )
```

The Firm parent columns 5/6 are `Hatcf/LnKF`, so both calls implemented `f_t -> f_{t+1}` while naming the values as calculated macro state in one path.

## Patched path

```text
build_collocation_bank()
  -> many-to-one node match on path/t/branch[/economic_node_id]
  -> parent_states[N,7] = b,z,eta,i,x,Hatcf,LnKF
  -> parent_macro_cal[N,2] = Hatc_cal,LnK_cal
  -> unified_firm_macro_bank_v1

build_child_bank()
  -> parent_bank.require_macro_context()
  -> MacroTransitionContext(Hatc_cal_t,LnK_cal_t)
  -> build_child_exogenous_bundle()
  -> one production SDF/FC1 forward

refresh_transition_bank()
  -> requires parent_macro_context[N,2]
  -> forward_step(x_t,x_t1,Hatc_cal_t,LnK_cal_t)
  -> writes child Hatcf/LnKF/raw M
  -> applies the existing common M clamp exactly once
  -> audit_m_parity() verifies P/Q receive the same used M
```

Legacy seven-column bank payloads remain readable as `legacy_firm_only_v0`, but any operation that needs a Macro transition calls `require_macro_context()` and fails. There is no fallback to `f_t -> f_{t+1}` and no silent column rename.

## Short-horizon data path

```text
analysis/run_diverse_macro_unified_bank.py
  -> data/diverse_short_horizon.py::make_diverse_initial_state()
       eight configurable cross-section families
       correlated b/z and nonnegative normalized K weights
       production eta/i initialization
  -> production SimulateTS (entry/exit/resources/BP/FC1/SDF unchanged)
  -> build_short_horizon_pairs()
       parent branch=-1 at t
       child branch=0/1 at t+1
       explicit burn-in validity
       no future calculated-state input
  -> build_collocation_bank()
       on-support matched Firm/Macro rows (default 75%)
       synthetic micro coverage conditional on the same macro node (25%)
  -> transition_parity(), old_vs_new_transition(), support and sufficiency audits
```

The 100-period rollout remains a long-horizon validation/recursive diagnostic. It is no longer the intended primary Macro-inner data source once this generator is connected to the equilibrium runner.

## Entry, resource accounting and BP

- `SimulateTS` remains the sole owner of entry/exit and resource accounting.
- The runner records an explicit economic configuration and defaults the new pilot to `investment_compare`, `raw`, and `economic_node_ledger`.
- `_simulation_kwargs()` now resolves leverage source through `Episode._prepare_simulation_kwargs()`; grid mode therefore reuses `GridBPSimulationPolicy`/`BPGridTeacher` and is applied to the actual transition.
- No Bellman equation, Q equation, wealth/SDF objective, FC1 loss, network input dimension, or calibrated economic parameter was edited.
