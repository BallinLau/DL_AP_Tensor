# models

Neural network modules for the asset pricing system.

## policy_value.py
`PolicyValueModel` outputs firm-level decision/value objects.

### Input
Firm-state (7D):
```
(b, z, ETA, i, x, Hatcf, LnKF)
```

### Output (PolicyValueOutput)
- `Q`, `bp0`, `bpI`, `P0`, `PI`, `bar_i`, `bar_z`, `P`, `Phat`, `bp`

### Update rule
`update_leverage(b_old, bp, eta)`:
```
b_new = eta * bp + (1 - eta) * b_old
```

## sdf_fc1.py
`SDFFC1Combined` combines SDF, FC1, and value model `W`.

Key API:
- `forward_fc1(x)`: FC1 now predicts increments `(ΔHatcf, ΔLnKF)` internally, then outputs
  levels via:
  - `Hatcf_{t+1} = Hatcf_t + ΔHatcf`
  - `LnKF_{t+1} = LnKF_t + ΔLnKF`
  Input remains `(x_prev, x_curr, Hatcf_prev, LnKF_prev)`.
- `forward_step(...)`: returns `(w_prev, w_curr, M, hatcf_curr, lnkf_curr)`

SDF stability update:
- `compute_sdf` now clamps the exponent term before `exp` (config: `Config.SDF_EXPONENT_CLAMP`)
  to reduce early-training numerical explosions in `M`.
- `ValueFunctionW` now uses a structural parameterization:
  - `w = exp(c) + surplus`
  - `surplus = softplus(raw) + Config.W_SURPLUS_FLOOR`
  This enforces `w - exp(c) > 0` by construction and avoids near-zero denominators.

This is used by SDF loss and macro proxy prediction.

## fc2.py
`FC2Model` maps cross-sectional distribution features to macro proxies.

Input:
- `phi = [b_quantiles(100), z_quantiles(100), x]` (default 201 dims)

Output:
- `(hatc, lnk)`

## share_layer.py
Shared backbone + heads used by policy/value to enforce structure
(e.g., no-invest heads do not depend on investment cost `i`).
