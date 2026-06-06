# LEARNINGS.md — T1.3: Givens Rotation Orthogonality at d=2816

**Finding #413 | Status: Supported | Date: 2026-04-09**

## Core Finding

Givens rotations on q_proj NoPE dims [128:512] are isometrically exact in float32 (error 2.384e-07, 420× margin over threshold), structurally parallel in a single Metal kernel, and require only 1408 params/layer — 32× fewer than LoRA r=8.

## Why

Block-diagonal structure from disjoint index pairs {2k, 2k+1} guarantees O^T O = I_d algebraically. Float32 accumulation error is O(ε_mach) per vector (independent of d), not O(d^{3/2} × ε_mach) as in the naive explicit-matrix test. Composition across L=1..8 layers holds the constant at the float32 precision floor (2.384e-07) — depth does not degrade isometry.

## Methodological Insight (carry forward to all experiments)

The explicit O^T O test (build d×d matrix, compute Frobenius norm) FAILS at large d due to dense float32 matmul accumulation: ‖error‖_F ~ d^{3/2} × ε_mach ≈ 0.018 at d=2816. Always use the isometry test (apply to random unit vectors, measure ‖Ox‖²) for large-d orthogonality verification.

## Implications for Next Experiment

Combined with T0.3 (NoPE isolation) and T0.4 (Q-only KV sharing), the P1 adapter is:
- Orthogonal via Givens on NoPE dims [128:512] (384 dims → 192 params/layer)
- Position-invariant: NoPE dims are algebraically insensitive to sequence position
- KV-cache-safe: Q-only adapters leave K/V unchanged by construction

Next: T1.6 algorithm bake-off (Givens vs HRA vs Cayley vs PoLAR) on real Gemma 4 dimensions once mlx_lm adds gemma4 support. Or: T1.1 (Householder) on same synthetic setup.
