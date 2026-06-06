# SHINE S3: Meta LoRA Encoding + Multi-Projection Generation

## Type: Guided Exploration
**Proven framework**: M2P generates functional LoRA through quantized Gemma 4 (Finding #484).
**Unknown**: Can meta LoRA + multi-projection + diversity loss break the centroid trap?

## Grounding
- arXiv:2602.06358 (SHINE) S3.1: Meta LoRA during encoding enriches memory states
- Finding #484: M2P generates LoRA with 86.6% CE reduction, centroid trap (cos=0.998)
- Finding #345: Algebraic proof of centroid trap — B-matrix homogenization starves capacity by 1/k
- Finding #480: v_proj+o_proj LoRA unlocks behavioral format priors (+70pp SOAP, +90pp Legal)

## Problem Analysis

### The Centroid Trap (Finding #345, #484)
S2 demonstrated M2P → LoRA → reconstruction works, but converges to a single adapter:
- Mean pairwise LoRA cosine = 0.998 across 40 contexts
- Finding #345 formalizes: homogenized B-matrix scales optimal weights by 1/k

### The Disease
Three independent structural causes, each requiring its own fix:
1. **No diversity signal**: NTP loss alone cannot penalize identical LoRA for different contexts
2. **Insufficient output capacity**: q_proj-only LoRA has limited degrees of freedom
3. **Generic memory states**: Base Gemma 4 extraction is not specialized for the M2P task

### Three Structural Interventions

**1. Meta LoRA (rank 128, q_proj, all layers)**
Trainable LoRA applied during memory extraction. The model learns to read contexts
in a way that maximizes M2P's ability to differentiate them. Zero-initialized B
(no effect at init), trained end-to-end through NTP loss.

Gradient path: NTP loss → generated LoRA → M2P → memory states → extraction pass → meta LoRA

**2. Multi-projection (q + v + o)**
Expand M2P output from q_proj-only to q + v + o projections. Finding #480 shows
v_proj+o_proj encode behavioral format priors. This 3× the output dimensionality,
making the centroid less attractive in the higher-dimensional LoRA space.

**3. Diversity regularizer**
$$\mathcal{L}_{total} = \mathcal{L}_{ntp} + \lambda \cdot \mathcal{L}_{div}$$
$$\mathcal{L}_{div} = \frac{1}{|C|} \sum_{i < j} \cos(\ell_i, \ell_j)^2$$

where $\ell_i$ is the flattened LoRA vector for context $c_i$, computed against
a running cache of recent detached LoRA vectors. Warmup $\lambda$ from 0 to target
over first 100 steps.

## Theorem 1 (Diversity Loss Makes Centroid a Saddle Point)

**Setup.** Let $\ell^*$ be the centroid LoRA minimizing average NTP:
$\ell^* = \arg\min_\ell \sum_i \mathcal{L}_{ntp}(\ell, c_i)$.

**Claim.** For any $\lambda > 0$, the centroid solution $\ell_i = \ell^*$ for all $i$
is NOT a local minimum of $\mathcal{L}_{total}$.

**Proof.** At the centroid, $\mathcal{L}_{div} = 1$ (all cosines = 1).
Consider perturbation $\ell_i = \ell^* + \varepsilon_i$ with orthogonal $\varepsilon_i$.

NTP change: Since $\ell^*$ approximately minimizes each $\mathcal{L}_{ntp}(\cdot, c_i)$,
the first-order change is small: $\Delta \mathcal{L}_{ntp} = O(\varepsilon^2)$.

Diversity change: $\cos(\ell^* + \varepsilon_i, \ell^* + \varepsilon_j) =
1 - \frac{\|\varepsilon_i - \varepsilon_j\|^2}{2\|\ell^*\|^2} + O(\varepsilon^2)$.
For orthogonal $\varepsilon_i$: $\|\varepsilon_i - \varepsilon_j\|^2 = 2\varepsilon^2$.
So $\Delta \mathcal{L}_{div} = -\varepsilon^2 / \|\ell^*\|^2 + O(\varepsilon^2) < 0$.

Total: $\Delta \mathcal{L}_{total} = O(\varepsilon^2) - \lambda \varepsilon^2/\|\ell^*\|^2 < 0$
for small enough $\varepsilon$.

Therefore the centroid is a saddle point of $\mathcal{L}_{total}$. QED.

**Caveat**: Proves centroid is not a local min. Does not guarantee convergence to
context-specific LoRA — depends on optimization landscape.

## Theorem 2 (Multi-Projection Increases Escape Dimensionality)

**Argument.** With $K$ projections, the LoRA vector $\ell_i \in \mathbb{R}^{K \cdot D_{lora}}$.
The number of orthogonal escape directions from the centroid saddle scales as $O(K \cdot D_{lora})$.
With K=1 (q only): D ≈ 42 × (2560 × 2 + 2 × 2048) ≈ 387K.
With K=3 (q+v+o): D ≈ 3 × 387K ≈ 1.16M.
More escape directions → gradient descent more likely finds them.

This is a heuristic argument. The key insight: multi-projection provides INDEPENDENT
dimensions for expressing context differences.

## Predictions

| ID | Prediction | Threshold | Reasoning |
|----|-----------|-----------|-----------|
| P1 | S3 test CE ratio < S2 | ratio < 0.134 | Richer memory states + multi-projection capacity |
| P2 | q+v+o CE < q-only CE | improvement > 0 | Multi-projection adds capacity (Finding #480) |
| P3 | Diversity loss → mean LoRA cos < 0.9 | cos < 0.9 | Theorem 1: centroid is saddle point |
| P4 | Meta LoRA near-orthogonal to generated LoRA | cos < 0.3 | Random subspaces in R^2560 |

Note on P4: Random subspaces of rank 128 and rank 2 in R^2560 have expected max
principal cosine ≈ sqrt(128 × 2 / 2560) ≈ 0.32. K1260 threshold of 1e-4 is
unreachable without explicit orthogonalization. We predict cos < 0.3 (natural regime).

## Kill Criteria

- **K1258**: Meta LoRA improves memory state quality (lower reconstruction CE vs S2)
  → S3 test CE ratio < 0.134 (S2 baseline)
- **K1259**: Multi-projection (q+v+o) LoRA improves reconstruction vs q-only
  → Full CE < q-only CE in ablation
- **K1260**: Meta LoRA + generated LoRA use orthogonal Grassmannian slots (cos < 1e-4)
  → Note: threshold likely unreachable for rank 128 vs rank 2 in R^2560.
  We measure and report the actual value.

## Failure Modes

1. **Memory explosion**: Two Gemma 4 forward passes (extraction + NTP) hold simultaneous
   computation graphs. Estimated: ~7 GB (vs 4.68 GB in S2). Budget: 48 GB. Low risk.

2. **Meta LoRA dominance**: Meta LoRA becomes so powerful that memory states trivially
   encode context, but only because meta LoRA is doing the adaptation. Generated LoRA
   norms → 0. Detection: monitor generated LoRA norms. Fix: reduce meta LoRA rank.

3. **Diversity loss prevents learning**: Too strong λ prevents NTP convergence.
   Detection: NTP loss stays high after warmup. Fix: reduce λ or extend warmup.

4. **Still centroid despite diversity loss**: Optimization doesn't find escape directions.
   Detection: mean cos > 0.9. Fix: increase λ or use hard negative mining.
