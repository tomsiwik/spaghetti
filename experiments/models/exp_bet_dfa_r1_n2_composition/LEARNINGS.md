# LEARNINGS — exp_bet_dfa_r1_n2_composition (KILLED, K2314)

**Core finding.** Mathematically exact output-side orthogonality (disjoint B-frames, per-layer
overlap 2e-16) recovered only 6.5pp (17.6%) of a 37pp N=2 behavioral interference gap on GSM8K —
far below the pre-registered 50% gate. B-side param-space disjointness does not buy behavioral
non-interference, the same way A-orthogonality didn't.

**Why.** Interference is not a single-layer linear-algebraic collision: each adapter's delta
perturbs the residual stream, so every later layer's A-matrix sees off-distribution activations.
No output-side frame at layer L can remove a mechanism that propagates through the nonlinear
stack. The +6.5pp D−C crumb says output deflection is directionally right but an order of
magnitude too weak.

**Implication for the next experiment.** Frozen-vector geometry (A-side or B-side) is a dead
class for N≥2 composition; per the ladder, R2 (retrain with the frame) is skipped. The surviving
rung is R3: the JEPA shared-frozen-predictor objective — align adapters in function space at
train time, gated on composition additivity (N=3 composed ≥ best-solo −3pp). Reviewer caveat to
carry: QR ordering was python-first (math's delta partially deflated by design); a math-first
variant is a follow-up, not a rescue, and re-scoring with numeric answer normalization left the
kill intact (19.4% recovery).

**PIERRE-IMPACT:** shelved — killed rung; no code change to bet/dfa-init. R1's projection path
must NOT be merged; the bet survives only via R3 (train-time function-space alignment).
