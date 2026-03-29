# q_unit Shape Constraint Update

## Why The Constraint Target Changes

The previous `q_shape` heuristic constrained `Q(b)` directly, effectively assuming where the total-debt-value peak should appear. That is too strong a prior and is not implied by theory.

The more theory-aligned object is the unit debt price:

```math
Q(b) = b \cdot q_{unit}(b)
```

The economically meaningful shape restrictions are:

```math
\frac{\partial q_{unit}}{\partial b} \le 0,
\qquad
\frac{\partial q_{unit}}{\partial z} \ge 0
```

These say:

- more leverage should not raise the price of one unit of debt;
- better firm productivity should not lower debt price.

The peak of total debt value `Q(b)` is then allowed to emerge endogenously from the interaction between the quantity effect `b` and the pricing effect `q_unit(b)`.

## Code Changes

1. Added direct `q_unit` accessors:
   - `SharedModel.get_q_unit(...)`
   - `PolicyValueModel.get_q_unit(...)`

2. Replaced the old `Q`-gradient shape penalty with a `q_unit`-gradient shape penalty:

```math
L_{shape}
=
\lambda_z \mathbb E[\max(0,-\partial q_{unit}/\partial z)]
+
\lambda_b \mathbb E[\max(0,\partial q_{unit}/\partial b)]
```

3. Added diagnostics:
   - `q_unit_mean`
   - `dq_unit_db_mean`
   - `dq_unit_dz_mean`

## Intended Effect

This change should stop the model from learning the economically implausible pattern where `q_unit(b)` rises with leverage, while still avoiding a hard-coded peak location for `Q(b)`.
