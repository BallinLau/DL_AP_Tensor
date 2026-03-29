# Q Relative Residual Update

## Why Change The Main Q Residual

The recent diagnostics show that the model can satisfy the `Q` Bellman equation with an economically meaningless near-zero solution:

```math
q_{unit}(b,z) \approx 10^{-8}, \qquad Q(b,z) = b \cdot q_{unit}(b,z) \approx 0.
```

In that situation, the debt-adjustment channel disappears numerically, but the absolute residual

```math
R_Q = \text{RHS} - Q
```

can still look deceptively small because both sides of the equation are tiny.

The problem is therefore not only shape; it is the scaling of the loss.

## This Update

The Bellman equation itself is unchanged. Only the numerical loss scale is changed.

Instead of using the raw residual directly, the main `Q` residual is normalized as

```math
R_Q^{rel}
=
\frac{
\text{RHS} - Q
}{
\varepsilon + |Q| + |\text{RHS}|
},
\qquad \varepsilon = 10^{-4}.
```

Then the existing AiO aggregation is applied to this relative residual.

## Why This Is Less Prior-Driven

This change does **not** tell the model what level `Q` should have.
It only tells the optimizer that:

- a pricing error of `0.01` is large when the object itself is `0.02`;
- the same absolute error is small when the object is `2.0`.

So the training objective now emphasizes **relative pricing accuracy** rather than absolute residual magnitude.

## Intended Effect

The purpose is to make the near-zero collapse less attractive numerically:

- if both `Q` and the RHS collapse to tiny values, the residual is no longer automatically “cheap”;
- the model must satisfy the Bellman equation in relative terms, not just absolute terms.

This is a numerical stabilization change, not a new economic prior.
