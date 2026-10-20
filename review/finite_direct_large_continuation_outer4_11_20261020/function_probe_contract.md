# Fixed function probes

Formal new-run baseline creates immutable, hash-bound Firm 7D inputs and complete
Validation macro parent/two-child records. No new baseline simulation has been
executed during review. It will run only behind approval v3.
Firm b,z,eta,i,x,hatcf,lnkf are unchanged across Outers; outputs P0/PI/Q and
bp0/bpI plus hard default/investment comparisons are saved. BP functions here are
HEAD functions (explicit label); actual Simulation still executes production Grid.
FC1 fixed (x_t,x_t1,Hatc_cal_t,LnK_cal_t) outputs are physical scale.
Wealth/M uses fixed true Parent and fixed common child x, current FC1 perceived
child, then existing value_model and sdf_model.get_M formula. Raw/used M are
separate and used M follows original Firm clamp. Additional TRUE-state wealth/M
on fixed realized children is separately named; it is not confused with Firm M.

Movement: RMS, MAE, abs P90/P99; binary flips; two-Outer distance; update-direction
cosine (undefined zero update is NOT_AVAILABLE). Distributions include mean,
variance, P90/P99 and clamp fraction. Probe/output hashes fail on changed inputs.
No Jacobian/spectral-radius experiment is introduced. CPU test loads actual
checkpoint and observed old probes, verifies production forward APIs and exact
repeatability/no model or RNG mutation with optimizer.step forbidden. Those
test fixtures are not represented as new-bank baseline or continuation results.
