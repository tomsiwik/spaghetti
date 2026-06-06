# LEARNINGS: T0.1 Grassmannian QR on Gemma 4 Weight Shapes

## Core Finding
Grassmannian QR construction achieves algebraic zero interference (1.7e-16 ≈ float64 floor)
at d=512/1024, rank=4. This verifies Theorem 1 at 410-547× below the predicted bound — the
theorem is d-independent, so smoke dims suffice to guarantee the same at d=2816/d=5376.

## Why It Works
QR orthogonality is algebraic: Q^T Q = I by construction, so pairwise A_i^T A_j = 0 exactly.
The Grassmannian partition (row-block slices of a single QR-decomposed matrix) inherits this
guarantee for any dimension. Theorem 2 bound O(sqrt(N*r)*eps_mach) is loose by ~400-500×.
Reference: standard numerical linear algebra (Golub & Van Loan, QR decomposition).

## Analytical Corollaries for P1 (not measured, but algebraically certain)
- Gemma 4 full q_proj (d=2816, r=16): N_max = 176 >> 25-domain target (7× headroom)
- Gemma 4 NoPE subspace (d=384, r=16): N_max = 24 per layer — sufficient for Gemma 4 local layers
- Construction: ~1ms for 20 adapters — never a bottleneck

## Implications for Next Experiment
T0.1 completes the T0 foundation. T1.5 (PoLAR landing field on Gemma 4) is now unblocked
by the DB dependency graph. However, Gemma 4 is still not loadable by mlx_lm 0.29.1
(gemma4 missing from MODEL_REMAPPING) — T1.5 must use synthetic layers at correct dimensions
(as T0.3/T0.4/T0.1 did) or wait for mlx_lm update.

## Reviewer Note (REVISE lesson)
Round 1 PAPER.md reported fabricated d=2816 values not in results.json. Fixed in REVISE
pass: PAPER.md now shows measured values (d=512/1024) and labels Gemma 4 capacity as
analytical corollaries. RULE: for algebraic theorems, smoke dimensions fully verify —
label extrapolated claims as "analytical corollary", never as measurements.

## PROCEED — Finding #417 (supported). T1.5 unblocked.
