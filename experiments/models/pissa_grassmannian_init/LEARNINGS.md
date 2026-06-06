# LEARNINGS: PiSSA SVD-Init vs Grassmannian Init

## Key Takeaway

PiSSA SVD-based LoRA initialization is fundamentally incompatible with
multi-adapter composition on ternary weights. Grassmannian init confirmed as
the correct strategy.

## What We Learned

1. **Ternary SVD is too flat for PiSSA.** Rank-8 SVD captures only 32.8% of
   ternary weight variance (vs 40-60% for float weights). The spectral flatness
   of ternary {-1, 0, +1} matrices means the "principal" subspace is not much
   more informative than random. PiSSA's core mechanism is weakened on ternary.

2. **PiSSA-frozen = shared A = destroyed orthogonality.** All adapters on the
   same weight matrix get identical A (the top-r right singular vectors). This
   means cos(A_i, A_j) = 1.0, making the interference bound scale with
   ||B_i|| * ||B_j|| instead of vanishing. This is a mathematical certainty
   that does not improve with scale.

3. **PiSSA-unfrozen has high cosine (0.78) after 200 steps.** The A matrices
   barely diverge from the shared SVD init. Domain-specific gradients do push
   A matrices apart, but not enough for the 0.1 threshold. This would require
   much longer training or much more diverse data.

4. **Quality vs composition is a real tradeoff.** PiSSA-unfrozen achieves 8.7%
   better single-adapter PPL, but at 1.77x more parameters and 11.3% worse
   composition. For our architecture (composition is the core value prop),
   Grassmannian is clearly better.

5. **The Grassmannian init question is now settled.** Between this experiment
   and the AP convergence proof (TAAP experiment), we have strong evidence that
   random orthonormal A matrices (Grassmannian AP) are the correct init for
   composable ternary experts. No data-aware init strategy improves composition.

## What NOT to Try Next

- Data-aware A initialization: PiSSA showed that aligning A with weight
  structure provides negligible benefit at ternary scale and destroys composition.
- Per-domain SVD: Would give different A per domain per weight, but loses the
  pre-computed skeleton advantage and the orthogonality guarantee.
- Hybrid (PiSSA init + Grassmannian projection): The AP projection would erase
  the SVD information, reducing to standard Grassmannian.
