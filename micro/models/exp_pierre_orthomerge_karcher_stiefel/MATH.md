# MATH.md — OrthoMerge: Karcher mean on Stiefel for shared-A B-only

## Hypothesis

OrthoMerge (arxiv 2602.05943, "Orthogonal Model Merging") averages model
weights on the **Riemannian manifold of orthogonal matrices** with magnitude
correction, which is the most direct geometric match to PoLAR's Grassmannian
sibling design. Pierre's Fisher-Rao norm-rescaled mean is essentially trying
to do this — average on a manifold and correct for naive-averaging shrinkage —
but in a B-only euclidean approximation. OrthoMerge does it properly:

1. Extract a rotation $R_t$ per adapter via Orthogonal Procrustes
2. Map to Lie algebra (skew-symmetric tangent space) via **inverse Cayley** (paper's explicit choice — not matrix log)
3. Average in the tangent space with **magnitude correction** $c = \sum_t \|Q_t\|_F / \|\sum_t Q_t\|_F$
4. Map back to $O(n)$ via Cayley
5. Merge the linear residuals separately in Euclidean space

The magnitude-correction step is what Fisher-Rao norm-rescaling tries to do.
OrthoMerge's version is mathematically grounded.

> **Does Karcher-mean-on-Stiefel + magnitude correction outperform
> Fisher-Rao for Pierre's shared-A B-only composition?**

## Adaptation to Pierre

OrthoMerge operates on full $W$. Like ACE, we adapt by materializing
$\Delta W_t = \text{scale} \cdot A \cdot B_t$ at composition time and
running OrthoMerge on the materialized deltas. The output is a fused delta
installed via `_FusedDeltaLinear`.

**One implementation choice the paper does NOT specify**: how to apply
OrthoMerge to LoRA-factored adapters when the base $W_0$ is the **frozen
base model weight** (not part of the adapter). Two paths:

- **(A) With-base path**: pass the base linear's weight $W_0$ to the merger.
  Procrustes finds $R_t$ that rotates $W_0$ toward $W_0 + \Delta W_t$; the
  merger returns $\Delta W_{\text{final}} = R_{\text{merged}} W_0 - W_0 + \rho_{\text{merged}}$.
  Geometrically faithful but requires reading the base linear weight per
  layer — non-trivial in MLX with quantized weights.

- **(B) No-base path** (fallback): treat $W_0 = I$, run Procrustes against
  the delta directly. Mathematically a different operation but architectural
  honesty (uses only the adapter state). Documented as a degenerate form.

This experiment **starts with path (B)** for simplicity and adds path (A)
as a follow-up if (B) shows promise. Both are pre-registered.

## Algorithm (paper-faithful pseudocode)

For each (layer, module) key:

1. Materialize $\Delta W_t = s_t \cdot A \cdot B_t$ for each of K adapters.
2. Procrustes per adapter:
   - $M_t = (W_0 + \Delta W_t) \cdot W_0^\top$ (path A) or fallback (path B)
   - $U \Sigma V^\top = \text{SVD}(M_t)$, $R_t = U V^\top$
3. Inverse Cayley: $Q_t = (R_t - I)(R_t + I)^{-1}$
4. Magnitude-corrected tangent-mean:
   - $Q_{\text{sum}} = \sum_t Q_t$, $Q_{\text{mean}} = Q_{\text{sum}}/T$
   - $c = (\sum_t \|Q_t\|_F) / \|Q_{\text{sum}}\|_F$
   - $Q_{\text{merged}} = c \cdot Q_{\text{mean}}$
5. Cayley forward: $R_{\text{merged}} = (I + Q_{\text{merged}})(I - Q_{\text{merged}})^{-1}$
6. Residual: $\rho_t = (W_0 + \Delta W_t) - R_t W_0$, $\rho_{\text{merged}} = \text{mean}_t(\rho_t)$
7. Final: $\Delta W_{\text{final}} = R_{\text{merged}} W_0 - W_0 + \rho_{\text{merged}}$

## Pre-registered Kill Criteria

- **K1 (DECISION)** OrthoMerge avg ≥ Fisher-Rao avg + 3pp.
- **K2 (ARCH GAP CLOSURE)** Full-delta DARE avg − OrthoMerge avg ≤ 4pp.
- **K3 (PREPROCESS BUDGET)** OrthoMerge full pipeline ≤ 60s total
  (heaviest of the three: Procrustes SVD + 2 matrix inversions per adapter per layer).
- **K4 (SANITY)** $K=1$ adapter case reproduces single-adapter eval (within 1pp on each of the 3 benchmarks).

## Verdict logic

Same as ACE-Merging:

| K1 | K2 | K3 | Outcome |
|----|----|----|---------|
| ✓ | ✓ | ✓ | **SUPPORTED** — Pierre default becomes OrthoMerge. |
| ✓ | ✓ | ✗ | **SUPPORTED** with caveat — accept slowdown for offline merge. |
| ✓ | ✗ | * | **SUPPORTED** — adopt; gap not fully closed. |
| ✗ | * | * | **KILLED** — geometrical principles don't transfer in this form. |

## Eval protocol

Same as `exp_pierre_dare_b_vs_fisher_rao` and other Pierre composition experiments.

## Honest gaps & implementation choices

- **No public code** for OrthoMerge; implementation is from paper equations only. Higher porting risk than Pico (which has paper-only spec) or ACE (which has working repo).
- **Cayley singularity unguarded** in paper — happens if any $R_t$ has eigenvalue $-1$ (180° rotation). We add `eps · I` to denominator inverses; flagged as a port choice.
- **No-base path (B) is mathematically distinct** from the paper's intended algorithm. Results from path (B) are an approximation; if successful, run path (A) as a follow-up.
- **d_out × d_out matrix operations** at d=2048 cost ~150ms per layer for SVD + 2 inversions × 7 adapters = ~1.0s × 42 layers = ~42s. K3 budget set to 60s with margin.
- **Linear residual merger** uses simple mean; paper recommends "any standard merger" including TIES. Mean is the closest match to Pierre's existing Fisher-Rao baseline for fair comparison.

## References

- OrthoMerge paper (arxiv 2602.05943): https://arxiv.org/abs/2602.05943
- Implementation spec: research agent `acbe00274a1a6eb9c` (paper HTML extraction)
- Prior measurement: `exp_pierre_dare_b_vs_fisher_rao` (Fisher-Rao 64.7%, full-delta DARE 71.3%)
- Finding #831 canonical `_FusedDeltaLinear` pattern (used here for fused-delta install)
