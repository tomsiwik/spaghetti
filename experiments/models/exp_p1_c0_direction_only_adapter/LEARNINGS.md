# LEARNINGS: C0.2 Direction-Only Adapter (KILLED)

## Core Finding
Direction-only LoRA (unit-norm B) achieves 83.3% of standard LoRA accuracy (60% vs 72% GSM8K), falling short of the 90% threshold — but this is a **training algorithm gap, not a representational gap**.

## Why
RMSNorm scale-invariance (Theorem 1) is confirmed: B-norms remain constant at √6 throughout 1000 steps. However, the failure has three compounding causes:
1. Post-hoc projection disrupts B initialization (zero → random unit vectors at step 1)
2. AdamW momentum accumulates on wrong scale after each retraction
3. Post-hoc retraction ≠ native Riemannian gradient step (first-order approximation only)

Loss curves converge to similar final values (0.84 vs 0.81), confirming the representations are learnable — the path to them is broken, not the destination.

**Theorem 1 scope caveat (from adversarial review):** Scale invariance holds when `delta_W >> W_q`. In standard LoRA regime (`delta_W << W_q`), re-scaling the adapter changes the combined output direction — so the claim "scale hyperparameter is irrelevant" is too strong. Post-normalization, only combined direction matters, but scale affects that direction.

## Implications for C1.1 (PoLAR Re-test)
- Use **native Riemannian GD** on the Stiefel manifold for BOTH U (lora_a) and V (lora_b rows):
  `G_tangent = G - B @ G^T @ B; B_new = retract(B + lr * G_tangent)`
- Do NOT use post-hoc projection (this is what failed here and in T1.5)
- Reference: Wen & Yin 2013 arxiv:2309.03737 (Cayley retraction / exponential map)
- Include scale sweep {1,5,10,20}: if T1 holds, direction-only accuracy should be identical across scales. If it differs, confirms the T1 scope gap.

## Secondary Discoveries
- **Stable rank benefit confirmed:** Direction-only sr=2.62 > standard sr=1.78 — unit-norm constraint forces rank diversity, prevents collapse.
- **Higher stable rank ≠ higher accuracy** under broken training algorithm.
