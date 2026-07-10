# training

Training orchestration for episodes and modules.

## episode.py
`Episode` encapsulates a single training episode:
- `generate_data` (legacy) and `run_episode` (current flow)
- SDF, Policy/Value, FC2 losses and batch builders

### SDF/FC1: fresh wealth shock pairs
- Formal Mode B defaults to `sdf_wealth_loss_mode='signed_aio'` and `sdf_fresh_pair_enabled=True`; use `legacy_abs_log1p` only when explicitly running the old wealth-loss ablation.
- `HyperParams.sdf_fresh_pair_enabled` enables refreshable aggregate shock pairs for the SDF wealth Euler loss.
- `sdf_wealth_loss_mode='signed_aio'` now requires `sdf_fresh_pair_enabled=True`; `Episode` and the multi-episode CLI entry points raise immediately if signed AiO is requested without fresh double sampling.
- `sdf_fresh_pair_enabled=True` with legacy wealth loss is still allowed for diagnostics/ablations, but it emits a warning because the bank was designed for signed AiO.
- The shock tape stores only AR(1) innovations with shape `[n_parent, sdf_child_bank_size, 1]`.
- At each optimizer step, `_compute_sdf_loss(...)` samples two different children per parent (`j1 != j2`) and converts them into fresh `x_{t+1}` values.
- Fresh children are used only for:
  - wealth Euler residuals
  - `sdf_wealth_loss_mode` (`signed_aio` by default, `legacy_abs_log1p` for explicit ablations)
  - SDF moment and mean-anchor penalties
- Fixed Treatment B children remain the source of:
  - FC1 true-state reconstruction targets
  - FC1 forecast-state reconstruction targets
  - delta and Jacobian penalties
- Teacher-forcing phases mark fresh pairs as requested but not used, and do not advance the pair generator, because FC1 teacher targets must stay tied to the fixed Treatment B children.
- Parent indices are compacted after resampling/capping before they reach the shock bank. The original row id is kept as `parent_source_index`, while `parent_index` is the dense bank index, avoiding huge sparse-bank allocations.
- The bank lifecycle is keyed by `(episode_id, epoch, capacity, bank_size, device, dtype)` and is reset at each `run_episode(...)` call, so reused `Episode` instances cannot carry stale shock tapes across episodes.
- Relevant controls:
  - `sdf_child_bank_size` (default `16`)
  - `sdf_child_bank_refresh_epochs` (default `1`)
  - `sdf_child_bank_seed` (default `12345`)
  - CLI flags: `--sdf-fresh-pair-enabled`, `--sdf-child-bank-size`, `--sdf-child-bank-refresh-epochs`, `--sdf-child-bank-seed`
- Diagnostics include `sdf_pair_collision_rate`, `sdf_pair_unique_ratio` (index-pair uniqueness), `sdf_index_pair_coverage_ratio`, `sdf_parent_pair_unique_ratio`, `sdf_j1_hist_entropy`, `sdf_j2_hist_entropy`, `sdf_eps_cross_corr`, `sdf_eps1_std`, `sdf_eps2_std`, `sdf_bank_size`, and `sdf_bank_refresh_id`.
- Signed AiO also logs finite-sample diagnostics such as `sdf_signed_aio_se` and `sdf_signed_aio_negative_share`. These do not change the objective; they only make noisy negative sample estimates visible.

### SDF/FC1: forecast Jacobian penalty schedule
- `fc1_jacobian_penalty_weight` enables the forecast-state local Jacobian penalty for FC1.
- The penalty uses higher-order autograd, so it is not evaluated on every batch by default.
- `fc1_jacobian_penalty_interval` controls the schedule:
  - `10` by default: compute the Jacobian penalty every 10 SDF/FC1 optimizer steps.
  - `1`: compute it on every step, matching the original expensive behavior.
  - `<=0`: skip the Jacobian penalty even if its weight is positive.
- Non-Jacobian batches still compute the ordinary true-state reconstruction, forecast-state reconstruction, and delta penalties; they skip only the four `torch.autograd.grad(..., create_graph=True)` calls.
- The schedule uses `sdf_fc1_step_count`, not the global training step, so Policy/Value and FC2 steps cannot starve the Jacobian penalty.
- Diagnostics include `sdf_jacobian_penalty_interval`, `sdf_jacobian_penalty_active`, and `sdf_fc1_step_count`.
- Slurm jobs expose the same control as `FC1_JACOBIAN_PENALTY_INTERVAL`.

### SDF/FC1: explicit training phases
- `sdf_training_schedule_enabled=True` runs simulated macro SDF/FC1 training as explicit phases:
  - `episode0_bootstrap`: Episode 0 only; use `Hatcf_t/LnKF_t` as bootstrap macro state, freeze FC1, train SDF/value with Stage1 moment/anchor weights, and do not require true `Hatc_t/LnK_t`.
  - `fc1_only`: train only `fc1_model`; Euler, moment, and SDF mean-anchor weights are zero.
  - `sdf_true_only`: freeze FC1 and train only `sdf_model`/`value_model` using true macro parent state.
  - `sdf_recursive_only`: freeze FC1 and train only `sdf_model`/`value_model` using recursive forecast parent state, plus a true-state Euler baseline.
- The phase is recorded as `sdf_training_phase`. Effective loss weights are logged as `*_weight_effective` fields.
- `fc1_only` never samples fresh wealth pairs and never computes Euler/moment/anchor terms.
- Episode 0 does not run the Stage2 FC1/SDF schedule. Stage2 is valid only for `episode_id > 0`.
- In Mode B, the ordering is now: simulate with previous policies, build macro transitions, run `fc1_only -> sdf_true_only -> sdf_recursive_only` with gates, replay the same simulation seed to refresh firm data with the accepted FC1/SDF, then train Q/P/bp. Gate failure raises `NumericalStageFailure` and skips downstream Policy/Value training.
- After the SDF refresh replay, Mode B runs a validation-only post-refresh gate before Policy/Value. This checks the distribution that Policy/Value will actually use, not only the pre-refresh macro distribution.
- SDF/FC1 gates are evaluated on a path-level holdout split controlled by `sdf_fc1_val_fraction` and `sdf_fc1_val_seed`; train and validation paths are disjoint when at least two paths are available.
- If `sdf_fc1_val_fraction > 0` but fewer than two paths are available, the run fails fast unless `allow_in_sample_sdf_gate_for_debug=True` is set explicitly.
- Mode A is disabled by default for Episode>0 when both `sdf_fc1` and `policy_value` are active, because it trains Policy/Value before the FC1/SDF gate. Set `allow_modea_sdf_after_pv=True` only for legacy comparison runs.
- `fc1_rollout_weight` requires same-path sequence tensors: `fc1_rollout_initial_x`, `fc1_rollout_initial_state`, `fc1_rollout_future_x`, and `fc1_rollout_target_states`. Child branches are not treated as rollout time steps.
- FC1 rollout tensors are constructed from consecutive macro rows in the tensor pipeline. If rollout is enabled but these tensors are missing, FC1 training fails fast instead of silently using a zero rollout loss.
- `sdf_true_only` and `sdf_recursive_only` require true `Hatc_t`/`LnK_t` in the parent batch; they no longer fall back to forecast state while logging a true-state phase.
- SDF gates include finite-ratio diagnostics for `M`; non-finite `M` values are no longer silently ignored by the pass/fail decision.
- SDF tail checks report `tail_gate_active`. The default p99/max thresholds are infinite, so tail diagnostics are logged but not binding until finite thresholds are configured from a healthy short run.
- `stage_gate_required_consecutive_passes=1` matches the current fixed-epoch schedule. Multi-pass adaptive gates require a separate min/max-epoch training loop and are not advertised by default.
- Episode 0 is explicitly recorded as `episode0_bootstrap_policy_training`; it remains a bootstrap exception rather than a formally gated SDF/FC1 stage.

### Policy/Value: Q-first training hooks
- `train_step(..., policy_loss_terms=...)` now supports selective optimization among `['q', 'p0', 'pi']`.
- `_run_batches(...)` supports Q pretraining via `HyperParams.q_pretrain_epochs` and `HyperParams.q_warmstart_epochs`:
  - `epoch < max(q_pretrain_epochs, q_warmstart_epochs)`: optimize `q` only
  - otherwise: optimize `p0 + pi + q` jointly
- `_compute_q_loss(...)` now includes:
  - AIO residual aggregation (`compute_aio_residual`) instead of pure branch-product aggregation
  - Optional `M` detach + clamp (`q_use_detached_m`, `q_m_clamp_min`, `q_m_clamp_max`)
  - Q-only parameter freezing (`q_freeze_non_q_in_pretrain`):
    - keep original economic equation (`bar_i`, `bp`, `bar_z` stay model outputs)
    - freeze non-Q parameters during Q-only stage
    - trainable scope controlled by `q_pretrain_trainable_scope`:
      - `q_head_only` (default, strict freeze)
      - `q_path` (`share_layer + q_head`)
  - Warm-start supervision (`q_warmstart_*`):
    - add `MSE(Q, Q_warm_target)` in early epochs
    - target is a structured prior over `(b,z,x)` with Gaussian shape in `b`
  - Q-shape regularization:
    - `dQ/dz > 0`
    - `dQ/db > 0` on low-`b` region
    - `dQ/db < 0` on high-`b` region
  - Additional diagnostics in loss dict:
    - `q_physics`, `q_warmstart`, `q_warm_weight`
    - `q_pretrain_mode`, `q_freeze_mode`

### Policy/Value: target-grid bp teacher
- `HyperParams.pv_bp_training_mode='target_grid'` enables the new Grid-B policy/value training path.
- During PV training, `_compute_p0_loss(...)` and `_compute_pi_loss(...)` call `BPGridTeacher`:
  - build a coarse bp-grid, optionally refine locally;
  - construct continuation child states with `b_child = eta_child * bp_candidate + (1 - eta_child) * b_parent`;
  - construct issuance-Q states with debt set directly to `bp_candidate`;
  - evaluate issuance `Q` with the frozen target network and child continuation `(P, bar_z)` through `target_model.forward_equity(...)`;
  - choose the candidate maximizing economic RHS `cashflow + continuation value`;
  - train `P0/PI` with Huber loss toward the selected target-grid Bellman backup;
  - train `bp0/bpI` with Huber distillation loss toward the selected grid argmax.
- The simulation leverage action is treated as an ex-ante mixed action:
  - `output.bp = survival_prob * (bar_i_cond * bpI + (1 - bar_i_cond) * bp0) + (1 - survival_prob) * bp0`
  - PI target-grid training constructs `V_mix = (1 - bar_i_cond_target) * V0_grid + bar_i_cond_target * VI_grid`, where `bar_i_cond_target` comes from the frozen firm target network.
  - The mixed distillation loss supervises conditional leverage `bp_cond = bar_i_cond_stopgrad * bpI + (1 - bar_i_cond_stopgrad) * bp0`, not the survival-fallback `output.bp`.
  - Mixed loss is weighted by target survival probability, so near-default states do not force economically irrelevant leverage targets.
  - Diagnostics include `mix_grid_bp_mae`, `mix_grid_regret_p90`, and `mix_grid_target_survival_weight_mean`.
- FOC/KKT is no longer the default bp training signal in `target_grid` mode. The old path is still available with `pv_bp_training_mode='legacy_foc_kkt'`.
- Simulation calls `PolicyValueModel.forward_simulation(...)` through `data.simulation_forward.forward_policy_value_for_simulation(...)`; it uses network outputs directly and never calls `BPGridTeacher` or runs a bp-grid.
- `PolicyValueModel.forward_policy(...)` is available for policy-only diagnostics and avoids `Q/P/Phat/bar_z` and the internal `i`-grid.
- `PolicyValueModel.forward_equity(...)` is available for target-grid child continuation and avoids Q/policy-head work.
- Target-moving controls:
  - `firm_target_update='epoch_hard'` by default: the target network is fixed inside each epoch and hard-copied from online at epoch end.
  - `soft`, `hard`, `epoch_soft`, and `none` remain available for ablations.
- Main target-grid controls:
  - `bp_grid_coarse_size`, `bp_grid_refine_enabled`, `bp_grid_fine_size`
  - `bp_grid_value_huber_delta`, `bp_grid_policy_huber_delta`, `bp_grid_policy_weight`
  - `bp_grid_mix_policy_weight`
  - `bp_grid_margin_scale`, `bp_grid_confidence_relative`, `bp_grid_confidence_min`
  - `bp_grid_parent_chunk_size`, `bp_grid_candidate_chunk_size`, `bp_grid_max_expanded_states`
  - `bp_grid_conv_mae_thresh`, `bp_grid_conv_regret_p90_thresh`, `bp_grid_conv_max_batches`, `bp_grid_conv_survival_eps`
  - `pv_target_grid_val_fraction`
  - CLI flags: `--pv-bp-training-mode`, `--bp-grid-coarse-size`, `--bp-grid-fine-size`, `--bp-grid-refine-enabled`, `--bp-grid-policy-weight`, `--bp-grid-mix-policy-weight`, `--bp-grid-margin-scale`, `--bp-grid-parent-chunk-size`, `--bp-grid-candidate-chunk-size`, `--bp-grid-max-expanded-states`, `--bp-grid-confidence-relative`, `--no-bp-grid-confidence-relative`, `--pv-target-grid-val-fraction`, `--firm-target-update`
- Confidence is computed from coarse/global top-two margins by default. In relative mode it uses `(coarse_top2_margin / |V_star|) / bp_grid_margin_scale`; fine-grid margins are logged separately as `*_grid_fine_top2_margin_mean`.
- `bp_grid_candidate_chunk_size=0` means evaluate the full candidate grid for each parent chunk when `bp_grid_max_expanded_states` permits it. With the default `bp_grid_parent_chunk_size=2048`, `bp_grid_coarse_size=21`, and `bp_grid_max_expanded_states=65536`, the coarse grid is evaluated one-shot because `floor(65536 / 2048) = 32 >= 21`.
- Global low/high bp diagnostics come from the coarse grid endpoints. Local fine interval endpoints are logged only as `*_grid_local_value_left_mean` and `*_grid_local_value_right_mean`.
- `bp_grid_quadratic_refine=True` re-evaluates `value_star`, `q_issue_at_star`, `p_child_at_star`, and `default_at_star` at the refined continuous `bp_star`.
- `bp_grid_use_survival_gate` has been removed; continuation values use the already clipped/default-adjusted `P` from the target equity evaluator.
- Target-grid policy convergence is checked in addition to Bellman residual convergence. In `target_grid` mode, `_run_batches(...)` reserves tail batches according to `pv_target_grid_val_fraction`; for joint stages, only PV training uses the reduced train split while SDF/FC1 continues to see all batches.
- Mixed-policy convergence is evaluated on survival-active states using `target survival_prob > bp_grid_conv_survival_eps`; all-state metrics are also reported as `*_all` fields. This prevents near-default fallback states from dominating the convergence gate.
- `evaluate_bellman_convergence(...)` passes only if Bellman residuals and validation teacher policy MAE/regret checks pass.
- Diagnostics include:
  - `p0_grid_bp_mae`, `pi_grid_bp_mae`
  - `mix_grid_bp_mae`, `mix_grid_regret_p90`
  - `p0_grid_regret_mean`, `pi_grid_regret_mean`
  - `p0_grid_boundary_low_share`, `p0_grid_boundary_high_share`
  - `p0_grid_top2_margin_mean`, `pi_grid_top2_margin_mean`
  - `p0_grid_default_at_star_mean`, `pi_grid_default_at_star_mean`
  - low/high-bp curve endpoints such as `p0_grid_value_low_bp_mean`, `p0_grid_value_high_bp_mean`, and matching default/P/Q endpoint means

### Episode flow
- Episode 0:
  1) Sample-based SDF/FC1 training (`build_sdf_fc1_df`)
  2) Sample-based Policy/Value training (`build_policy_value_df`)
  3) FC2 training using firm-level aggregation and Policy/Value outputs
- Episode >= 1:
  - Use `SimulateTS` to generate firm-time series, then train SDF/FC1, Policy/Value, FC2

### Batch builders
- `_create_sdf_batches_from_macro_df`: macro cross-section for SDF/FC1
- `_create_firm_batches_from_df`: firm triplets for Policy/Value losses
- `_create_fc2_batches`: parent/children firm groups + K for FC2

## trainer.py
`Trainer` orchestrates multi-episode training, checkpoints, and callbacks.

## scheduler.py
Learning rate and loss-weight schedulers used by `Episode`.

## gradient_utils.py
Gradient clipping and NaN protection.
