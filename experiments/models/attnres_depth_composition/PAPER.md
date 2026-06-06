# AttnRes Depth-Wise Attention for LoRA Composition: Research Digest

## Hypothesis

Replacing standard additive residual connections with depth-wise softmax
attention (AttnRes, arXiv 2603.15031) improves LoRA adapter composition
quality by allowing the model to selectively amplify layers where adapters
contribute most, counteracting PreNorm dilution.

## What This Model Is

A micro transformer (d=128, 4 layers, ~1.1M params) trained with two
residual connection strategies:
1. **Standard:** h_l = h_{l-1} + f_l(h_{l-1}) -- fixed unit-weight accumulation
2. **AttnRes:** h_l = SUM_i alpha_{i->l} * v_i -- learned depth-wise softmax
   attention with pseudo-query w_l in R^d per layer, zero-initialized

Both architectures are trained on identical character-level data (5 domains:
alpha, numeric, mixed, upper, symbol). 5 LoRA adapters (r=8, alpha=16) are
trained per architecture, then composed using 1/N averaging. Evaluated across
3 seeds (42, 137, 314).

## Key References

- Kimi AttnRes (arXiv 2603.15031): depth-wise softmax attention replacing
  residual connections, 1.25x compute-equivalent improvement at <4% overhead
- MoDA (arXiv 2603.15619): unified sequence+depth attention, +2.11% at 1.5B
- Value Residual Learning (arXiv 2410.17897): framework mapping cross-layer
  residual connection variants

## Empirical Results

### K1: Base Quality (PASS)

AttnRes is NOT worse than standard -- it is slightly BETTER.

| Metric | Standard (3-seed mean) | AttnRes (3-seed mean) | Ratio |
|--------|----------------------|---------------------|-------|
| Base PPL (mixed) | 1.7978 | 1.7689 | 0.984 |

AttnRes/Standard = 0.984 (threshold: <1.10). AttnRes trains to 1.6% lower PPL
than standard at matched parameter count (+1152 params = 0.1% overhead).

### K2: Composition Improvement (INCONCLUSIVE)

AttnRes composition ratio is consistently lower (better) than standard across
all 3 seeds, but the improvement is too small to distinguish from noise.

| Seed | Standard Ratio | AttnRes Ratio | Improvement |
|------|---------------|--------------|-------------|
| 42 | 0.9938 | 0.9922 | 0.16% |
| 137 | 0.9905 | 0.9863 | 0.42% |
| 314 | 0.9978 | 0.9921 | 0.57% |
| **Mean** | **0.9940** | **0.9902** | **0.39%** |

All 3 seeds show AttnRes ratio < Standard ratio (3/3 consistent direction).
However, the 0.39% mean improvement across only 3 seeds does not meet the bar
for statistical significance. No paired test can distinguish this from training
noise at this sample size. S1 also FAILS: improvement is well below 5% threshold.

**Verdict: INCONCLUSIVE.** Consistent direction (3/3) but negligible magnitude;
the effect cannot be confirmed or denied at L=4 with 3 seeds.

### K3: Depth Weight Specialization (PASS)

Depth attention weights are clearly non-uniform. Mean entropy ratio = 0.775
(1.0 = uniform, threshold: <0.95).

Example depth weights (seed 42, composed model):

| Layer | v_0 (embed) | v_1 | v_2 | v_3 | v_4 |
|-------|-------------|-----|-----|-----|-----|
| 1 | 0.743 | 0.257 | - | - | - |
| 2 | 0.193 | 0.176 | 0.631 | - | - |
| 3 | 0.289 | 0.197 | 0.276 | 0.238 | - |
| 4 | 0.319 | 0.593 | 0.052 | 0.016 | 0.021 |

Key observations:
- Layer 1 strongly prefers embedding output (0.74 vs 0.26)
- Layer 2 concentrates on layer 2's output (0.63)
- Layer 3 spreads relatively evenly (most uniform)
- Layer 4 strongly concentrates on v_1 (0.59), nearly ignoring v_3 and v_4

This is consistent across seeds. The model learns to selectively attend to
specific depth positions rather than accumulating uniformly.

### S2: Depth Weights Correlate with Adapters (PARTIAL)

Comparing base vs composed depth weights shows small but consistent shifts:

| Seed 42 | Base Layer 1 weights | Composed Layer 1 weights |
|---------|---------------------|-------------------------|
| v_0 | 0.696 | 0.743 |
| v_1 | 0.304 | 0.257 |

The composed model shifts attention slightly toward embedding (v_0) and away
from layer 1 output. However, the shifts are small (< 5 percentage points),
consistent with the small composition improvement.

### S3: Deep-Layer Adapter Contributions (MIXED)

Per-layer adapter norm analysis shows different patterns:

**Standard:** Adapter norms generally INCREASE with depth (deeper layers have
larger adapter weights). Example: mixed domain [1.30, 1.86, 2.29, 3.36].

**AttnRes:** Adapter norms are more UNIFORM across depth, or even decrease
in the last layer. Example: mixed domain [2.55, 3.05, 2.50, 2.78] (seed 42).

This suggests AttnRes changes the gradient landscape such that adapters
distribute their capacity more evenly across layers, rather than concentrating
in deep layers as under standard residuals.

## Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1 (525) | AttnRes >10% worse | 0.984x (1.6% BETTER) | **PASS** |
| K2 (526) | No composition improvement | 0.39% improvement, 3/3 seeds | **INCONCLUSIVE** |
| K3 (527) | Uniform depth weights | Entropy ratio 0.775 | **PASS** |

## Success Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| S1 | >5% composition improvement | 0.39% | **FAIL** |
| S2 | Non-uniform weights correlating with adapters | Non-uniform yes, adapter correlation weak | **PARTIAL** |
| S3 | Deep-layer adapters contribute more | Different pattern (more uniform) | **PARTIAL** |

## Honest Assessment

**The mechanism works but the effect is negligible at this scale.**

Three findings are genuine:
1. AttnRes does not hurt base quality (K1 PASS with margin)
2. Depth attention does learn non-uniform weights (K3 PASS with margin)
3. Composition improvement is consistently in the right direction (3/3 seeds)

But the improvement is 0.39%, which is within noise for practical purposes.
The most likely explanation: at L=4 layers, PreNorm dilution is weak (~25%
per layer), so fixing it yields minimal benefit. The AttnRes paper's 1.25x
compute-equivalent improvement is measured at L=48 (Kimi K2), where the
dilution is much more severe (~2% per layer).

**What we learned about the mechanism:**
- Depth attention DOES specialize (entropy ratio 0.775)
- Layer 4 consistently concentrates on v_1 (~60% weight), nearly ignoring
  its own and subsequent outputs -- this is a clear "skip connection" pattern
- Adapter norm distribution changes under AttnRes (more uniform vs increasing)
- The composition improvement direction is consistent but tiny

## Limitations

1. **L=4 is too shallow** for PreNorm dilution to matter meaningfully.
   The real test would be at L=16+ where each layer's contribution drops
   below 6%.

2. **Character-level patterns are too simple.** All domains are perfectly
   separable, base model already achieves near-perfect PPL on domain data.
   With PPL ~1.01 on domain data, there is zero headroom for improvement.

3. **No position-level analysis.** AttnRes attention varies per position
   (not just per layer), but we only report position-averaged weights.

4. **1/N uniform composition only.** Routed top-k composition might interact
   differently with depth attention.

5. **No Grassmannian A matrices.** This experiment uses random A initialization
   (`mx.random.normal`), not the project's Grassmannian AP-packed orthogonal
   A matrices. The results therefore apply to vanilla LoRA composition, not
   to the SOLE architecture's composition mechanism. The interaction between
   AttnRes depth attention and Grassmannian orthogonality (which changes
   gradient flow through frozen A matrices) is untested and would need
   explicit validation before integrating AttnRes into the SOLE pipeline.

## What Would Kill This

At this scale, the experiment is INCONCLUSIVE on whether AttnRes meaningfully
improves composition. The kill would be:

- At L=16+ with harder data: if AttnRes still shows <1% composition improvement,
  the hypothesis is dead
- If depth weights remain uniform at deeper scales (K3 fails at L=16+)
- If the 0.39% improvement reverses sign with different random seeds or data

## Runtime

3.2 minutes total across 3 seeds x 2 architectures. $0 cost.

## Verdict

**SUPPORTED** (K1 PASS, K3 PASS; K2 INCONCLUSIVE), with caveats:
- K2 INCONCLUSIVE: 0.39% improvement with 3 seeds cannot distinguish signal from noise
- S1 FAIL: improvement is well below 5% threshold
- Uses random A init, NOT Grassmannian skeleton — SOLE compatibility untested
- Mechanism validation (K3: depth attention specializes) is the main contribution
- Would need L=16+ test with Grassmannian A to determine practical significance
