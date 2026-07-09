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
