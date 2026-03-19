# losses

Loss functions for SDF, Policy/Value, and FC2. All losses support multi-branch
inputs (lists per branch).

## sdf_loss.py
`SDFLoss` enforces Euler equation residuals and moment penalties for SDF:
- Inputs: `w_parent`, `w_children`, `M_list`, `k_parent`, `k_children`, `c_parent`, `c_children`
- Computes per-branch residuals, combines them, and adds moment constraints on `M`.

## p0_loss.py
`P0Loss` for no-invest equity value:
- Inputs include `P0`, state inputs, `Q`, `Qp_children`, `M_list`, `P_children`, `bar_z_children`
- Core APIs:
  - `compute_bellman_residual(...)`
  - `compute_foc_residual_from_bp(...)` (autograd-based FOC from `CF0p`/`P_children` w.r.t. `bp`)
- In `training/episode.py`, `_compute_p0_loss` aggregates:
  - Bellman main loss (`prod across children`)
  - `z` penalty on Bellman residual
  - FOC main loss (`prod across children`)
  - `z` penalty on FOC residual

## pi_loss.py
`PILoss` for invest equity value:
- Similar structure to P0 but with investment cashflow and invest-specific terms
- Core APIs:
  - `compute_bellman_residual(...)`
  - `compute_foc_residual_from_bp(...)` (autograd-based FOC from `CFip`/`P_children` w.r.t. `bp`)
- In `training/episode.py`, `_compute_pi_loss` aggregates:
  - Bellman main loss (`prod across children`)
  - `z` penalty on Bellman residual
  - `b > 1` penalty
  - FOC main loss (`prod across children`)
  - `z` penalty on FOC residual

## q_loss.py
`QLoss` for bond pricing:
- Inputs: `Q`, `bar_i`, `M_list`, `Qsp_children`, `bar_z_children`, `x_children`, `z_children`
- Enforces pricing residual and boundary constraints (low/high b)

## fc2_loss.py
`FC2Loss` for macro consistency:
- `forward`: compares FC2 outputs to aggregated macro targets
- `forward_with_aggregation`: runs resource accounting + aggregation, then applies loss

## Notes
- Losses are module-agnostic and expect correctly shaped tensors from training.
- All losses are called from `training/episode.py`.
