# Timing and information set

1. A parent node carries `(x_t,h_t^belief,lnK_t^belief)` and the full firm-mass
   distribution.
2. For every root candidate `h`, Firm values and hard decisions are recomputed
   at that candidate while `x`, `lnK`, capital, mass, and candidate entry draws
   remain fixed.
3. The coarse-to-fine controller selects only a verified root and freezes its
   exact `NodeDecisionRecord`.
4. The node ledger and FC1 current label use that record. The commit cannot
   redraw entry or reinterpret the investment/default actions.
5. Parent BP is resolved using the configured Grid policy and current `eta_t`;
   the resulting leverage is written to each child. Child-view BP is diagnostic
   until that child becomes a parent.
6. Each macro child receives its own common `x` shock. Conditional on that
   common shock, idiosyncratic z, eta, and investment cost are integrated and
   conservatively projected.
7. Each child is cleared independently. Aggregation precedes `log(C/K)` and the
   two macro branches remain separate inputs to SDF training.

Firm Bellman equations continue to use FC1 as a perceived aggregate law of
motion. A price-taking firm's candidate `b'` does not trigger a full market
clearing solve. Simulation separately reports the error between perceived and
realized cleared transitions.

