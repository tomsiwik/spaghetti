# Quality Degradation Detection: Research Digest

## Hypothesis

Canary queries (small fixed test sets per expert) can detect when adding
a new expert degrades an existing expert, with less than 20% false negative
rate and less than 10 min overhead at N=500.

## What This Model Is

This experiment addresses a critical reviewer attack on the SOLE architecture:
"At N=500, there are 124,750 pairwise interactions. How do you verify expert
#347 hasn't subtly degraded expert #12?"

We test four degradation detection strategies:
(a) **Full eval:** Evaluate all N existing experts before/after adding (gold standard)
(b) **Random sampling:** Evaluate a random subset of experts
(c) **Cosine-gated:** Only evaluate experts with high cosine to the new expert
(d) **Canary queries:** Fixed small test set (20 examples) per expert, evaluate all

The experiment trains 8 synthetic domain experts (arithmetic, reverse, repeat,
sort, parity, subtract, multiply, length) using a full autograd transformer
(d=64, rank 8, 2 layers), composes them additively, and measures per-expert
degradation when each expert is added to the composition.

## Lineage in the Arena

```
exp_gram_schmidt_composition (proven)
  |
  +-- exp_domain_similarity (proven, within-cluster 7.84x)
  |
  +-- exp_collision_scaling (supported, beta=-0.575)
  |
  +-- exp_quality_degradation_detection (THIS EXPERIMENT)
```

## Key References

- Lopez-Paz & Ranzato, "Gradient Episodic Memory", NeurIPS 2017 (backward transfer metric)
- Chaudhry et al., "Riemannian Walk for Incremental Learning", ECCV 2018 (forgetting measures)
- Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023
- Yadav et al., "TIES-Merging: Resolving Interference", NeurIPS 2023

The canary query approach is essentially a lightweight backward transfer
test suite applied to the SOLE additive composition setting. The contribution
is validation engineering (proving this simple approach works and cosine
shortcutting fails), not a novel detection algorithm.

## Empirical Results

All results from the full configuration: d=64, rank=8, N=8 experts, 3 seeds,
15-20 epochs. Runtime: 414 seconds on Apple Silicon (CPU, autograd).
Results saved in `results.json` with config_name="full".

### Phase 1: Degradation is Anti-Correlated with Cosine

| Metric | Value | 95% Bootstrap CI |
|--------|-------|-----------------|
| Total pairs tested | 168 (56 per seed x 3 seeds) | -- |
| Degradation rate | 82.7% (139/168 pairs) | -- |
| Occupancy rho | 1.0 (N*r/d = 8*8/64) | -- |
| Pearson r(|cos|, degradation) | -0.41 +/- 0.13 | [-0.50, -0.31] |
| Spearman rho | -0.52 +/- 0.14 | [-0.60, -0.35] |

**Key finding:** Degradation is NEGATIVELY correlated with cosine similarity.
Experts with LOW cosine to the new expert are MORE degraded, not less. This
inverts the intuitive hypothesis that cosine-gated checking of HIGH-cosine
experts would catch degradation.

Per-seed Pearson correlations: -0.44 (seed 42), -0.25 (seed 142), -0.55 (seed 242).
All negative, with seed 142 showing weaker anti-correlation. The pooled
bootstrap CI excludes zero, confirming the anti-correlation is robust.

### Phase 2: Detection Method Comparison (d=64, 3 seeds)

| Method | FNR (mean) | FNR (std) | FPR | Coverage | Time at N=500 |
|--------|------------|-----------|-----|----------|---------------|
| Full eval | 0.0% | 0.0% | 0.0% | 100% | 36s |
| Cosine > 0.005 | 0.0% | 0.0% | 0.0% | 100% | 36s |
| Cosine > 0.05 | 1.2% | 1.7% | 0.0% | 99% | 35s |
| **Cosine > 0.10** | **33.8%** | **7.0%** | 0.0% | 74% | 27s |
| Cosine > 0.15 | 60.6% | 8.5% | 0.0% | 46% | 17s |
| Cosine > 0.20 | 78.4% | 7.7% | 0.0% | 31% | 11s |
| Random 50% | 56.1% | 2.5% | 0.0% | 43% | 15s |
| **Canary (n=20)** | **2.0%** | **0.1%** | **6.6%** | **100%** | **7s** |

**Important negative result:** Cosine gating at tau=0.10 has FNR=33.8%, far
above the 20% kill threshold. This is because the anti-correlation means
cosine gating systematically misses the most degraded experts (those with
low cosine). Cosine-based shortcutting is counterproductive for degradation
detection.

### Best Method: Canary Queries

Canary queries achieve the best cost-quality tradeoff:
- FNR = 2.0% +/- 0.1%, 95% CI [1.9%, 2.1%] (well below 20% kill threshold)
- Per-seed FNR: 1.8%, 2.1%, 2.1% (remarkably stable across seeds)
- FPR = 6.6% +/- 4.2% (acceptable false positive rate)
- Projected time at N=500: 7.3 seconds (well below 10 min)
- No cosine computation needed (avoids the anti-correlation trap)

### Kill Criteria

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| K1: Best FNR > 20% | 20% | 0.0% (full eval), 2.0% (canary) | **PASS** |
| K2: Time > 10 min at N=50 | 10 min | 0.57s max | **PASS** |

**Verdict: SUPPORTED**

Status is "supported" rather than "proven" because: (1) the canary approach
is validated only at N=3-8 experts; the N=500 projection is linear
extrapolation of wall-clock time that does not test whether 20-example canary
sets remain sensitive at larger N; (2) the anti-correlation finding is
directionally robust but its production-scale relevance depends on occupancy
rho (see MATH.md for full reconciliation).

## Key Insights

### 1. The Anti-Correlation Surprise

The original hypothesis proposed checking HIGH-cosine experts. But
degradation is anti-correlated with cosine because at high occupancy
(rho ~ 1.0), similar experts reinforce each other while dissimilar
experts add noise. This means cosine-gated detection at aggressive
thresholds (tau > 0.1) systematically MISSES the most degraded experts.

### 2. Degradation Regime Depends on Occupancy rho

At micro scale (d=64, r=8, N=8), rho=1.0 and degradation is universal
(82.7% of pairs degrade). At production scale, rho controls the absolute
magnitude of degradation while cosine controls the relative ranking.

The reconciliation of "low cosine implies more degradation" with "production
experts have low cosine but near-zero degradation" is that degradation
magnitude ~ f(rho) * g(cos), where f(rho) -> 0 as rho -> 0. See MATH.md
section "Reconciling the Anti-Correlation Paradox" for the full argument.

### 3. Canary Queries are the Practical Solution

The simplest approach wins: maintain a fixed set of 20 domain-specific test
queries per expert. On every expert addition, run all canaries. This:
- Scales as O(N * 20) per addition (linear, small constant)
- Projects to 7.3 seconds at N=500
- Achieves 2.0% FNR without needing cosine computation
- Works regardless of the cosine-degradation correlation direction
- Is trivially parallelizable

### 4. Low Cosine Threshold = Full Eval in Disguise

At d=64, nearly all expert pairs have |cos| > 0.01, so cosine gating
at any threshold below 0.05 checks ALL experts -- providing no
computational savings. At production dimensions where cos ~ 0.0002,
this problem is worse: no threshold can usefully partition experts.

## Micro-Scale Limitations

1. **Extreme occupancy:** At d=64, rank 8, N=8, rho=1.0, causing universal
   degradation. Production rho is much lower but exact value depends on N.

2. **Autograd model:** The model is a 2-layer transformer trained with autograd
   for 15-20 epochs. Not a production-scale model.

3. **Additive composition:** We test raw additive composition (W + sum delta_i)
   without Gram-Schmidt orthogonalization. GS composition would further reduce
   interference.

4. **Fixed epsilon:** The 2% relative loss increase threshold is arbitrary.
   Production systems would need calibration.

5. **Small test sets:** 100 test examples per domain (20 for canary). Production
   canary sets should be curated for sensitivity to subtle degradation.

6. **Flattened cosine measure:** Cosine of concatenated parameter vectors is
   a coarse metric. Layer-wise cosine might show different (possibly positive)
   correlation in individual layers. The anti-correlation may be partly a
   property of the coarse measure.

## What Would Kill This

### At micro scale
- FNR consistently > 20% for canary detection across multiple configurations
- Canary query evaluation time scales super-linearly with N

### At macro scale
- Structural orthogonality fails to prevent degradation at production d (d=896)
- Canary queries miss systematic degradation patterns not captured by 20 examples
- False positive rate becomes unacceptably high (> 10%), causing unnecessary
  expensive re-evaluations

## Implications for SOLE Architecture

1. **Degradation detection should use canary queries, not cosine gating.**
   The anti-correlation between cosine and degradation makes cosine-based
   filtering counterproductive at aggressive thresholds.

2. **At production scale, detection may be unnecessary.** When rho << 1 and
   structural orthogonality holds, adding experts cannot significantly degrade
   others. Detection is insurance against edge cases.

3. **The merge pipeline remains fast.** Even full eval at N=500 takes only 36s.
   Canary queries take 7s. The 10-minute budget is never approached.

4. **Canary sets are cheap operational overhead.** Storing 20 examples per expert
   is ~0.1KB per expert. At N=10,000, this is 1MB total.
