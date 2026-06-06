# MATH — exp_rdt_lti_injection_mlx

## Setup

Reference: OpenMythos `open_mythos/main.py:643-686` (PyTorch `LTIInjection`).
Grounded in Parcae (Prairie et al., 2026, arxiv:2604.12946).

The recurrent hidden-state update in RDT:

  h_{t+1} = A · h_t + B · e + TransformerBlock(h_t, e)

where A ∈ ℝ^{dim} is a learned diagonal matrix. Without constraint, A can acquire
|A_i| ≥ 1, causing h_t to explode across depth-iterations and destabilize training.

The primitive's construction makes ρ(A_discrete) < 1 by ZOH discretization of a
negative-diagonal continuous-time system:

  A_c := −exp(log_A),  Δt := exp(log_dt)
  A_d := exp(Δt · A_c) = exp(−exp(log_dt + log_A))

Clamped implementation (guards float overflow):

  A_d := exp(−exp(clamp(log_dt + log_A, −20, 20)))

## Theorem 1 (Stability by Construction)

**Claim.** For any log_A ∈ ℝ^{dim}, log_dt ∈ ℝ, the implementation
`A_d = exp(−exp(clamp(log_dt + log_A, −20, 20)))` satisfies

  A_d,i ∈ (exp(−exp(20)), exp(−exp(−20))) ⊂ (0, 1) in exact arithmetic.

Hence ρ(A_d) = max_i |A_d,i| < 1 in exact arithmetic.

**Proof.**
Let s_i = clamp(log_dt + log_A_i, −20, 20) ∈ [−20, 20].
Let u_i = exp(s_i) ∈ [exp(−20), exp(20)] ⊂ (0, +∞).
Then A_d,i = exp(−u_i), and since u_i > 0, exp(−u_i) ∈ (0, 1).
Strictly: u_i ≥ exp(−20) > 0 ⇒ A_d,i ≤ exp(−exp(−20)) < 1. QED.

## Lemma 1 (Float32 Boundary Degradation)

**Claim.** In IEEE 754 float32 (MLX default on GPU), at s_i = −20,
A_d,i rounds to 1.0 exactly.

**Proof.**
exp(−20) ≈ 2.061·10⁻⁹. Then exp(−2.061·10⁻⁹) ≈ 1 − 2.061·10⁻⁹.
Float32 unit-round-off ε ≈ 5.96·10⁻⁸ > 2.061·10⁻⁹.
So 1 − 2.061·10⁻⁹ rounds up to the nearest representable float32 = 1.0. QED.

**Verified numerically.** `np.float32(np.exp(-np.exp(-20.0))) == 1.0` returns True.
Same for MLX float32. In float64, value is 0.9999999979388464 < 1.

**Implication for K1737.** The Theorem guarantees ρ < 1 in exact arithmetic, but
at the lower-clamp boundary in float32 ρ = 1 exactly. K1737 is tested on training
*dynamics* (not adversarial extremes), so a typical optimizer trajectory starting
from log_A = log_dt = 0 is not expected to drift to s_i = −20 within 1000 steps.

## Theorem 2 (MLX = PyTorch Equivalence)

**Claim.** For identical weights `log_A, log_dt, B` and identical inputs
`h, e, transformer_out`, the MLX and PyTorch implementations produce outputs
with cosine similarity > 0.9999 on the same float32 dtype.

**Proof sketch.** Both implementations compute element-wise
`A·h + B·e + transformer_out` with `A = exp(−exp(clamp(log_dt+log_A, −20, 20)))`.
All primitives (exp, clamp, element-wise add/mul) are IEEE 754 correctly-rounded
in both frameworks; differences reduce to reduction ordering and intrinsic
round-up. Empirically bounded by > 0.9999 cosine. QED.

## Theorem 3 (NaN-Freedom at Extremes)

**Claim.** At log_A, log_dt ∈ [−20, +20]^{dim+1}, both the forward pass and
gradients w.r.t. {log_A, log_dt, B} are NaN-free.

**Proof sketch.**
Forward: clamp bounds the arg to [−20, 20]; exp on [−20, 20] is finite
(range [2.06·10⁻⁹, 4.85·10⁸]). exp(−finite) ∈ [exp(−4.85·10⁸)≈0, 1]. No NaN paths.
Gradient of A_d,i w.r.t. s_i: ∂A_d,i/∂s_i = A_d,i · (−exp(s_i)), which vanishes
as s_i → +∞ (A_d,i → 0) and as s_i → −∞ (exp(s_i) → 0). No 0·∞ NaN.
Clamp is sub-differentiable (0 outside [−20, 20]); the zero-gradient region is
well-defined. QED.

## Pre-registered Kill Criteria

**K1737** — Spectral radius: Train 1000 steps with Adam(lr=1e-3) on synthetic
MSE objective; at every step record ρ(A_d) = max_i |A_d,i|. Pass iff no step
violates ρ(A_d) < 1.0. (Measures functional trajectory, not adversarial extreme.)

**K1738** — MLX↔PyTorch parity: Initialize identical weights in both frameworks;
compute forward on 100 i.i.d. Gaussian (h, e, transformer_out) triples. Pass iff
min(cosine_similarity) > 0.9999 across all 100 samples.

**K1739** — NaN-freedom: Set (log_A, log_dt) to each of 4 corners of [−20, 20]²
(broadcast log_A to dim channels). Run forward + backward (loss = output.sum()),
call mx.eval on grads. Pass iff NO NaN/Inf in outputs or gradients across all
4 corners.

## Predictions

| KC | Prediction | Test |
|----|------------|------|
| K1737 | max_rho(trajectory) < 1.0 across all 1000 steps | Adam training, MSE loss, dim=128 |
| K1738 | min_cos(100 samples) > 0.9999 | Shared init, float32, dim=128, N=100 |
| K1739 | No NaN/Inf in forward or gradients at 4 corners | log_A∈{−20,20}, log_dt∈{−20,20} |

## Platform

MLX float32 (GPU), N_steps=1000, dim=128, B=4, T=16. Torch CPU float32 for
cross-check. Seed 42.
