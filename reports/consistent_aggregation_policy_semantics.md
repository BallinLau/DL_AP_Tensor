# Consistent aggregation policy semantics

This document records the execution contract implemented by
`consistent_aggregation_market_clearing_20261019`. It is a code-level contract,
not a claim that the model is identical to any external Gomes implementation.

## Existing model objects

- `P0` / `PI` are the physical no-investment and investment branch values.
- The legacy `bar_i_cond` is
  `sigmoid((PI-P0)/tau_i)`. `bar_i` additionally multiplies this by the smooth
  survival gate. It is therefore a trainable-value-derived smooth gate, not the
  CDF `H(i_star)` and not a realized project indicator.
- `Phat` is the model's uniform-grid average of `max(P0,PI)` over the original
  investment-cost support. Limited-liability equity is `max(Phat,0)`.
- The legacy `bar_z` is `1-sigmoid(Phat/tau_z)`. It is a smooth gate, not an
  independently calibrated default probability and not a realized default.
- Conditional BP heads are controls only when current `eta_t=1`. The realized
  leverage transition remains
  `b_{t+1}=eta_t*bp_t+(1-eta_t)*b_t`.

## Realized decisions in the new explicit modes

`hard_realized_reference` and `continuum_mass_consistent_v1` do not use the
legacy soft gates as realized actions:

- survival is `Phat>0` and default is its complement;
- investment is `PI>=P0` for the realized/integrated cost state, conditional on
  survival;
- the same hard investment/default objects enter project expenditure,
  liquidation, capital growth, and exit;
- newborn firms retain the existing rule that their entry capital is created at
  the node and they do not receive an additional same-node project expansion or
  incumbent maintenance charge.

The resource definitions remain the production definitions. For each unit,
output is `exp(x+z)K`; maintenance is `delta*K` when eligible; default contributes
the existing `-K` capital-transfer term and the existing
`(1-PHI)*(1+exp(x+z))*K` bankruptcy-loss term. No consumption floor or new cash
flow shock is added.

## Entry

The entry comparison remains `PI-P0>=0` at `b=0`, `eta=0`, and
`i=entry_cost`. The cost is already an input to `PI` and is not subtracted a
second time. Entry capital, entry resource cost, accepted entry mass, birth
provenance, and the next distribution all use the same accepted set.

The continuum mode removes only finite-firm stochastic rounding of potential
entrant counts. It integrates the unchanged stationary-z and uniform-cost
distributions into entrant mass. A genuine common macro shock is never averaged
across the two SDF child branches.

## Timing

The selected root's immutable `NodeDecisionRecord` is attached before node
commit. The record contains the snapshot hash, selected `h`, hard actions,
entry acceptance, mass, resource ledger, BP provenance, and transition payload.
The current-node resource account is recorded before hard-default mass is
removed. Exit is then applied before that child is promoted to the next parent.

A child observation may evaluate a Grid BP for diagnostics, but it is marked
`BP_EXECUTION_APPLICABLE=FALSE`. Only a parent-to-child transition records an
executed BP and written child leverage. Under Grid configuration there is no
branch-based fallback to the BP head.

## Representation changes

In `continuum_mass_consistent_v1`, `n_slots`, `firm_mass`, capital per unit, and
aggregate capital are distinct. Idiosyncratic AR(1) z innovations, Bernoulli eta,
and uniform investment costs are integrated with deterministic probability
weights. Projection groups nearby numerical states while preserving total mass,
capital, debt, and the current output proxy; its error is gated and reported.

These are explicit aggregation and numerical-state-representation changes.
The calibrated economic parameters, value comparisons, transition laws, entry
criterion, SDF macro branching, and root residual tolerance are unchanged.

