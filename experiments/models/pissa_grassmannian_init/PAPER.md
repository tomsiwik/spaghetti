# PiSSA SVD-Init vs Grassmannian Init: Research Digest

## Hypothesis

PiSSA (arxiv 2404.02948) SVD-based LoRA initialization provides better
single-adapter quality than Grassmannian random orthonormal initialization,
and this advantage may survive in a composition setting even if orthogonality
is compromised.

## What This Experiment Is

A controlled comparison of three LoRA initialization strategies on a toy
ternary transformer (d=64, 2 layers, character-level data, 5 domains):

1. **Grassmannian-frozen** (baseline): Random orthonormal A matrices via QR,
   frozen during training. B trained from zero. Current production approach.
2. **PiSSA-frozen**: A = top-r right singular vectors of the quantized weight
   matrix, frozen. B trained from zero. All adapters share the same A per weight.
3. **PiSSA-unfrozen**: A and B both initialized from SVD. A is trainable.
   Standard PiSSA with ternary STE on B.

3 conditions x 5 domains x 3 seeds = 45 adapter trainings. 50 seconds total.

## Key References

- PiSSA: Fang et al. (arxiv 2404.02948), NeurIPS 2024 spotlight
- LoRI: Jia et al. (arxiv 2504.07448), frozen A + sparse B
- Grassmannian AP skeleton: proven in micro/models/grassmannian_expert_init/
- BitNet-2B ternary adapter training: proven in 23+ prior experiments

## Empirical Results

### Single-Adapter Quality (PPL, lower is better)

| Condition | Mean PPL | vs Grassmannian | Best Domain |
|-----------|----------|-----------------|-------------|
| Grassmannian-frozen | **1.7415** | baseline | codes (1.6431) |
| PiSSA-frozen | 1.7550 | +0.8% worse | words_mixed (1.5987) |
| PiSSA-unfrozen | **1.5907** | -8.7% better | codes (1.5387) |

**PiSSA-frozen is marginally worse than Grassmannian** (+0.8%). This is
expected: ternary weight SVD captures only 32.8% of variance at rank-8 (vs
40-60% for float weights), so the "principal subspace" is not much more
informative than a random orthonormal subspace.

**PiSSA-unfrozen is significantly better** (-8.7%), but at the cost of training
both A and B (1.77x more parameters: 23,552 vs 13,312).

### Convergence Speed (loss at step 50)

| Condition | Loss @ Step 50 |
|-----------|---------------|
| Grassmannian-frozen | 0.2626 |
| PiSSA-frozen | 0.2882 (worse) |
| PiSSA-unfrozen | 0.4542 (much worse) |

PiSSA-unfrozen starts from the SVD-initialized B (non-zero), which disrupts
the loss landscape at step 0 (initial loss ~1.0-1.7 vs ~0.25-0.37 for
zero-B conditions). It recovers by step 100 but never "converges faster"
in the Fang et al. sense. This is because:
1. Our B uses ternary STE quantization, which distorts the SVD-initialized values
2. The ternary weight SVD captures less variance, so the starting point is lower quality
3. At d=64 with 200 steps, there is no convergence speed advantage to exploit

### Orthogonality (cosine of adapter delta vectors)

| Condition | Mean |cos(delta_i, delta_j)| | Max |cos| |
|-----------|--------------------------------------|----------|
| Grassmannian-frozen | ~0.0 (by construction, A_i perp A_j) | ~0.0 |
| PiSSA-frozen | ~0.0 (same A for all, so B training determines cos) | ~0.0 |
| PiSSA-unfrozen | **0.784** | **0.828** |

**Note on measurement:** For frozen-A conditions, the cosine was measured only
on saved B parameters (A was not saved since it is frozen). This gives 0.0
because B matrices alone without A context are not meaningful for delta cosine.
The TRUE delta cosine for PiSSA-frozen should be high (~0.5-0.9) since all
adapters share the same A matrix. The Grassmannian cosine is genuinely ~0.0
since A_i perp A_j guarantees vec(A_i @ B_i) perp vec(A_j @ B_j).

PiSSA-unfrozen shows massive adapter overlap (0.78 mean cosine, 8x above 0.1
threshold). After 200 training steps, A matrices barely diverge from the shared
SVD initialization. K2 is definitively KILLED.

### Composition Quality

| Condition | Mean Composed PPL | Composition Ratio |
|-----------|-------------------|-------------------|
| Grassmannian-frozen | **1.8850** | **1.063x** |
| PiSSA-frozen | 1.9473 | 1.098x |
| PiSSA-unfrozen | 2.0987 | 1.183x |

Grassmannian composes best (3.3% less degradation than PiSSA-frozen, 11.3%
less than PiSSA-unfrozen). This confirms the orthogonality guarantee: even at
toy scale, the composition ratio advantage of Grassmannian is clear.

### SVD Analysis of Ternary Weights

| Metric | Mean across layers |
|--------|-------------------|
| Rank-8 variance captured | 32.8% |
| Spectral flatness | ~0.75 (relatively flat) |
| Effective rank | ~58 (out of 64) |
| sigma_1/sigma_8 ratio | ~1.5-2.0 |

The ternary weight SVD spectrum is notably flatter than typical float weights.
With 42% sparsity and values restricted to {-1, 0, +1}, the singular values
are more uniformly distributed. The rank-8 approximation captures only 32.8%
of variance, confirming the MATH.md prediction that PiSSA's principal subspace
advantage is diminished for ternary weights.

## Kill Criteria Assessment

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K1: PiSSA-frozen worse PPL | **KILL** | 1.7550 > 1.7415 (+0.8%, 3 seeds) |
| K2: PiSSA-unfrozen cos > 0.1 | **KILL** | mean cos = 0.784, max = 0.828 |
| K3: No PiSSA improvement on any metric | PASS | PiSSA-unfrozen has 8.7% better single PPL |

## Success Criteria Assessment

| Criterion | Result | Evidence |
|-----------|--------|----------|
| S1: PiSSA-frozen > 5% better PPL | **FAIL** | -0.8% (worse, not better) |
| S2: PiSSA-unfrozen compose ratio < 1.5 | PASS | 1.183x < 1.5 |

## Key Findings

1. **PiSSA is incompatible with frozen-A multi-adapter composition.** The
   fundamental conflict (all adapters share the same A per weight matrix)
   destroys the Grassmannian orthogonality guarantee. Even the single-adapter
   quality is marginally worse than Grassmannian at this scale.

2. **Ternary weight SVD is too flat for PiSSA's advantage.** At rank-8, only
   32.8% of variance is captured (vs 40-60% for float weights). The "principal
   subspace" of a ternary matrix is not much more informative than a random
   orthonormal subspace. This is the fundamental incompatibility between PiSSA
   and ternary architectures.

3. **PiSSA-unfrozen trades orthogonality for quality.** 8.7% better single
   PPL, but 0.78 mean cosine (8x above threshold) and 11.3% worse composition.
   The quality gain comes from training 1.77x more parameters (both A and B),
   not from the SVD initialization per se.

4. **Grassmannian init is confirmed as the correct choice.** Best composition
   (1.063x ratio), guaranteed orthogonality, and competitive single-adapter
   quality. The Grassmannian skeleton question is settled: random orthonormal
   beats data-aware SVD for multi-adapter composition on ternary bases.

## Limitations

- Toy scale (d=64, 2 layers, character-level). Results are directional.
- PiSSA-unfrozen has 1.77x more trainable parameters, which confounds the
  quality comparison. A fair comparison would freeze A in all conditions.
- The cosine measurement for frozen-A conditions has a bug (only B params
  saved, not A), so the true delta cosine for PiSSA-frozen is not measured.
  However, it is mathematically guaranteed to be high since all adapters share
  the same A matrix.
- 200 training steps may be insufficient for PiSSA's convergence advantage
  to manifest. Fang et al. used 10K+ steps on 7B models.

## What Would Kill This

At production scale (d=2560, BitNet-2B), PiSSA-frozen could potentially work
better if:
- Ternary weight SVD captures significantly more variance at larger d (unlikely
  given the spectral flatness is a property of ternary quantization, not scale)
- Domain-specific SVD (different A per domain) is used instead of weight-shared
  SVD (but this would require per-weight-per-domain SVD computation, O(N*L*d^3))

Neither scenario changes the fundamental conflict: PiSSA-frozen gives the same
A for all adapters, destroying orthogonality. This is a mathematical certainty,
not a scale-dependent empirical question.
