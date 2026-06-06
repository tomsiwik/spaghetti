# LEARNINGS — exp_rdt_lti_injection_mlx

## What we proved

- OpenMythos LTI-injection primitive (`main.py:643-686`) ports to MLX with
  5-"9"s cosine parity on 100 random float32 inputs and 4.77e-7 max absolute
  diff — within expected numerical noise for identical-init float32 ops.
- ρ(A_d) < 1 holds empirically across 1000 Adam steps (max_rho = 0.366)
  with MSE on synthetic targets. Theorem 1 (exact-arithmetic stability)
  verified for central-regime training dynamics.
- NaN-freedom holds at all 4 extreme corners of (log_A, log_dt) ∈ {-20, 20}².
  Both forward and backward paths finite.

## What we learned (non-obvious)

- **Float32 boundary degradation at s = -20** (Lemma 1): in IEEE 754 float32,
  `exp(-exp(-20))` rounds to exactly 1.0 because float32 unit-round-off
  (5.96e-8) exceeds the delta (2.06e-9). The construction is exact-arithmetic
  stable but float32-stable only if the argument stays away from -20. Reusable
  insight for any MLX primitive that relies on `exp(-exp(small_positive))`.
- The clamp at ±20 is necessary to prevent overflow at the upper end:
  `exp(20) ≈ 4.85e8`, `exp(-4.85e8) = 0` (underflow, safe). But at the lower
  end, unclamped `log_dt+log_A = -∞` would give `exp(-exp(-∞)) = exp(0) = 1`
  which is the failure mode the clamp is supposed to prevent; the clamp
  bounds the boundary, doesn't eliminate it.
- Under MSE-on-Gaussian training from zeros init, log_A and log_dt stay
  tightly centered (|log_A| < 1 after 1000 steps). The stability-at-extremes
  property is not stressed under typical training; adversarial training
  signals could stress it.

## Reusable antipattern / pattern

- **Pattern — float-boundary documentation**: when a primitive has
  exact-arithmetic guarantee but float32-rounded boundary, document both
  Theorem (exact) and Lemma (float boundary). Split KC into "functional
  trajectory" (use Theorem) and "extreme corners" (test a weaker property
  like NaN-freedom). K1737/K1739 demonstrate the split.

## Impact on downstream / related experiments

- `exp_rdt_loop_lora_gemma4` (P=0, depends_on this) is **unblocked**. The
  LTI primitive is safe to use as-is.
- `exp_rdt_act_halting_throughput` (P=1) shares the primitive requirement
  via the broader RDT chain.
- Downstream RDT loops should monitor `log_A + log_dt` during training and
  optionally add L2 regularization on these params to avoid drifting toward
  -20 under adversarial gradients.

## Not covered / deferred

- MLX float16 / bfloat16 variants (M3+ would benefit from bfloat16 BW).
  Deferred because downstream uses float32 primitives today.
- Compiled (`mx.compile`) variant. Trivial JIT target; the element-wise
  primitive is already 0.77s for 1000 steps uncompiled, so compilation
  is an optimization not a correctness concern.
- Composition with `TransformerBlock`. Tested in parent RDT experiments.
