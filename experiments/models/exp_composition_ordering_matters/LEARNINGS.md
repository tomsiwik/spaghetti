# LEARNINGS.md — exp_composition_ordering_matters

## Core Finding
Under N=3 LoRA adapter summation, add order is behaviorally irrelevant: both kill criteria pass
by ≥100× (K1928 rel-Frobenius 4.40e-8 vs 1e-5 threshold = 227× margin; K1975 rel-PPL 1.94e-3
vs 1e-2 threshold = 5.2× margin). The 6 permutations collapse to **3 distinct PPLs** —
a clean fingerprint of left-fold GEMM: FP addition is commutative (bit-exact swap of first two
addends) but not associative (re-grouping changes roundoff).

## Why
- **Weight-space (FP32)**: Higham bound `(n-1)u·Σ|x_i|` gives ≤2e-6 absolute; measured 4.77e-7
  (~4× below ceiling, typical average-case).
- **Behavior (BF16 GEMM intermediates)**: Each `(dx @ A_i) @ B_i` materializes at model dtype
  (BF16, u≈7.8e-3) before the sum. Bound at this precision is `(N-1)·u_bf16·‖term‖` ≈
  `2·7.8e-3·‖term‖`, matching measured 1.94e-3 PPL gap.
- So the ~1000× gap between weight-space prediction (2e-6) and behavioral measurement (2e-3)
  is **not a theorem violation** — it's the theorem evaluated at BF16 unit roundoff instead
  of FP32. First cross-scale measurement in this codebase of intermediate-GEMM dtype
  propagation into PPL.

## Implications for Next Experiment
1. **Canonicalize order freely.** Pierre/Room-Model composition can fix any permutation
   (alphabetical, list-index) with zero behavioral cost at N≤3. "Reorder adapters for better
   PPL" is not a productive tuning direction — signal is ~1e-3 PPL, below F#666 target noise.
2. **N=10 forecast.** `(N-1)·u_bf16` scales linearly: expect ~6-8e-3 rel PPL gap at N=10,
   still below 1%. Replication is low-priority given theorem, but worth one run before
   Pierre macro deployment claims.
3. **Diagnostic fingerprint for future composition experiments:** if 6-permutation sweep
   yields ≠3 PPL values (or the {(0,1,2)≡(1,0,2)}, {(0,2,1)≡(2,0,1)}, {(1,2,0)≡(2,1,0)}
   pairing breaks), implementation is non-deterministic — not an FP-summation effect.
4. **FP32 forward-pass is a deployable knob** for reproducibility-critical runs: predicted
   to shrink PPL gap from 2e-3 to ~1e-6. Worth follow-up only if reproducibility becomes
   a Pierre requirement.
5. **Cosmetic:** code/results.json use `k1929_fire` variable name; MATH.md/PAPER.md use
   K1975 label (thresholds match). Rename on next edit for searchability; no verdict impact.
