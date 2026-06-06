# LEARNINGS: T1.5 PoLAR Landing Field — KILLED (Finding #419)

## Core Finding
Constraining U alone (Stiefel retraction on the right factor) does NOT prevent rank collapse in ΔW = V@U — the V matrix also collapses to rank ≈ 2 because the task gradient ∂L/∂V = (∂L/∂ΔW)@U^T is rank-1 (single-domain SFT has a rank-1 gradient subspace), regardless of U's orthogonality.

## Why It Failed
U^T is an isometry — it preserves gradient magnitudes per column but does NOT rotate different gradient directions to different V columns. Every V column receives the same approximately rank-1 gradient from GSM8K SFT (all chain-of-thought steps improve the same dominant direction), so Adam drives all columns to co-adapt: sr(V) → 1–2. PoLAR quality gap: 3.3% vs LoRA 13.3% (4×) in 200 steps.

## What Was Verified
Theorem 1 holds: periodic polar retraction maintains ‖UU^T−I‖_F = 2.46e-08 (400× under threshold). Retraction is exact, cheap (< 1ms per layer), and numerically stable. Theorem 2 bound sr(ΔW) ≥ sr(V)·(1−ε) is also correct — the wrong claim was the corollary sr(V) ≈ r/2.

## Resurrection Path
1. **Joint Stiefel on U×V (product manifold):** retract BOTH U and V to orthonormal rows every 10 steps. Each rank-1 term U_i ⊗ V_i contributes independently → sr(ΔW) = r by construction. Requires alternating retraction or Riemannian Adam on St(r,d_in)×St(r,d_out).
2. **Diverse gradient:** 3+ domains (code + math + QA) raises gradient subspace rank to ≥ 3, enabling sr(V) ≥ 3 even without Stiefel constraint on V.

## Implications for T1.6 Bake-off
- All single-U-retraction methods (PoLAR, Cayley, Givens on U only) will suffer V-collapse on single-domain SFT.
- T1.6 must compare with **joint retraction** on both U and V, OR use multi-domain data.
- Equal-params comparison (T1.2 lesson) still applies: use matching parameter budgets across methods.
- Givens (T1.3, proven isometry) is the most efficient building block for joint retraction.
