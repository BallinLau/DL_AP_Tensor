# training

Training orchestration for episodes and modules.

## episode.py
`Episode` encapsulates a single training episode:
- `generate_data` (legacy) and `run_episode` (current flow)
- SDF, Policy/Value, FC2 losses and batch builders

### SDF/FC1: fresh wealth shock pairs
- `HyperParams.sdf_fresh_pair_enabled` enables refreshable aggregate shock pairs for the SDF wealth Euler loss.
- The shock tape stores only AR(1) innovations with shape `[n_parent, sdf_child_bank_size, 1]`.
- At each optimizer step, `_compute_sdf_loss(...)` samples two different children per parent (`j1 != j2`) and converts them into fresh `x_{t+1}` values.
- Fresh children are used only for:
  - wealth Euler residuals
  - `sdf_wealth_loss_mode` (`legacy_abs_log1p` or `signed_aio`)
  - SDF moment and mean-anchor penalties
- Fixed Treatment B children remain the source of:
  - FC1 true-state reconstruction targets
  - FC1 forecast-state reconstruction targets
  - delta and Jacobian penalties
- Relevant controls:
  - `sdf_child_bank_size` (default `16`)
  - `sdf_child_bank_refresh_epochs` (default `1`)
  - `sdf_child_bank_seed` (default `12345`)
  - CLI flags: `--sdf-fresh-pair-enabled`, `--sdf-child-bank-size`, `--sdf-child-bank-refresh-epochs`, `--sdf-child-bank-seed`
- Diagnostics include `sdf_pair_collision_rate`, `sdf_eps_cross_corr`, `sdf_eps1_std`, `sdf_eps2_std`, `sdf_bank_size`, and `sdf_bank_refresh_id`.

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
  - PI target-grid training also constructs `V_mix = (1 - bar_i_cond) * V0_grid + bar_i_cond * VI_grid`
  - `output.bp` is directly supervised against `argmax_b V_mix`, with diagnostics such as `mix_grid_bp_mae` and `mix_grid_regret_p90`.
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
  - `bp_grid_conv_mae_thresh`, `bp_grid_conv_regret_p90_thresh`, `bp_grid_conv_max_batches`
  - CLI flags: `--pv-bp-training-mode`, `--bp-grid-coarse-size`, `--bp-grid-fine-size`, `--bp-grid-refine-enabled`, `--bp-grid-policy-weight`, `--bp-grid-mix-policy-weight`, `--bp-grid-margin-scale`, `--bp-grid-parent-chunk-size`, `--bp-grid-candidate-chunk-size`, `--bp-grid-max-expanded-states`, `--bp-grid-confidence-relative`, `--no-bp-grid-confidence-relative`, `--firm-target-update`
- Confidence is computed from coarse/global top-two margins by default; fine-grid margins are logged separately as `*_grid_fine_top2_margin_mean`.
- Global low/high bp diagnostics come from the coarse grid endpoints. Local fine interval endpoints are logged only as `*_grid_local_value_left_mean` and `*_grid_local_value_right_mean`.
- `bp_grid_quadratic_refine=True` re-evaluates `value_star`, `q_issue_at_star`, `p_child_at_star`, and `default_at_star` at the refined continuous `bp_star`.
- `bp_grid_use_survival_gate` has been removed; continuation values use the already clipped/default-adjusted `P` from the target equity evaluator.
- Target-grid policy convergence is checked in addition to Bellman residual convergence. In `target_grid` mode, `evaluate_bellman_convergence(...)` passes only if both Bellman residuals and held-out teacher policy MAE/regret checks pass.
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
