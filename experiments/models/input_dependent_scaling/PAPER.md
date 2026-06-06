# Input-Dependent Adapter Scaling: Proof Verification Report

## Theorem (Restated)

**Proposition 1 (Monotone Scale-Similarity Relationship):** If TF-IDF embedding
similarity to domain centroids preserves distributional similarity to adapter
training data, then behavioral quality should be monotonically related to
embedding similarity, and per-query scale modulation should outperform fixed
per-domain scales.

## Predictions vs Measurements

| # | Prediction (from framework) | Measured | Match? |
|---|----------------------------|----------|--------|
| P1 | Embedding-similarity correlates with score (r > 0.3) | r = -0.079 (overall) | NO |
| P2 | Per-query scaling improves on >= 2/3 domains | 1/3 improved (math only, +10%) | NO |
| P3 | Math shows smallest improvement | Math showed LARGEST improvement (+10%) | NO |
| P4 | Code shows largest improvement | Code: +2.3% (below 2% significance threshold) | NO |
| P5 | Coherence maintained (< 20% incoherent) | 3.3% incoherent | YES |

## Hypothesis

Per-query embedding-based scale selection (using cosine similarity between query
TF-IDF embedding and domain centroid) improves behavioral quality over fixed
per-domain scales on at least 2/3 tested domains.

**Verdict: NOT SUPPORTED.** K2 (correlation) definitively fails. The mechanism
(TF-IDF similarity -> scale) has no predictive power.

## What This Experiment Is

A guided exploration testing whether TF-IDF embedding space geometry predicts
per-query adapter effectiveness. The approach:
1. Compute TF-IDF centroids for each domain's training data
2. For each test query, compute cosine similarity to routed domain centroid
3. Map similarity to scale: s(q,d) = s_d * max(alpha, normalized_sim)
4. Compare behavioral quality against fixed per-domain scales

The scale modulation maps [p25, p75] of in-domain similarity to [alpha, 1.0],
where alpha=0.3 is the floor. This means queries near the 25th percentile of
domain similarity get 30% of optimal scale, while queries above the 75th
percentile get full scale.

## Key References

- Finding #249: Two behavioral regimes (FORMAT vs CAPABILITY), per-domain scaling required
- Finding #252: Math phase transition is evaluation artifact, not universal
- LoRAuter (arXiv:2602.21222): Input-dependent adapter scaling via vector retrieval
- MoLoRA (arXiv:2603.15965): Per-token routing with learnable scale coefficients

## Empirical Results

### Domain Comparison (Fixed vs Input-Dependent)

| Domain | Fixed Score | Dynamic Score | Delta | % Change | Improved? |
|--------|-----------|--------------|-------|----------|-----------|
| Math | 0.800 | 0.900 | +0.100 | +12.5% | YES |
| Code | 0.624 | 0.638 | +0.015 | +2.3% | NO (< 2% threshold) |
| Medical | 0.291 | 0.277 | -0.014 | -4.7% | NO |

### Correlation Analysis (K2)

| Domain | r (sim vs score) | Sim Mean | Score Mean |
|--------|-----------------|----------|------------|
| Math | -0.009 | 0.132 | 0.900 |
| Code | 0.032 | 0.259 | 0.638 |
| Medical | -0.093 | 0.165 | 0.277 |
| **Overall** | **-0.079** | 0.185 | 0.605 |

**The correlation is essentially zero.** TF-IDF similarity to domain centroids
has no predictive power for per-query behavioral quality.

### Scale Distribution

The dynamic scaling produced a wide range of scales:
- Math: 6.0 to 20.0 (4/10 queries got reduced scale)
- Code: 6.0 to 20.0 (5/10 queries got reduced scale)
- Medical: 6.0 to 20.0 (3/10 queries got reduced scale)

### Per-Prompt Analysis: Where Dynamic Helped/Hurt

**Math improvements (+2 correct, -1 correct = net +1):**
- Prompt 1 (sim=0.082, scale=6.0): Fixed WRONG, dynamic CORRECT. Lower scale
  produced cleaner format with step-by-step solution.
- Prompt 9 (sim=0.065, scale=6.0): Fixed WRONG, dynamic CORRECT. Same pattern --
  lower scale yielded better structured output.
- Prompt 4 (sim=0.133, scale=11.7): Fixed CORRECT, dynamic WRONG. Intermediate
  scale caused arithmetic error.

**Code: Large variance, no pattern:**
- Prompt 0 (sim=0.205, scale=6.0): Fixed 0.771, dynamic 0.079. Reduced scale
  broke syntax validity.
- Prompt 9 (sim=0.156, scale=6.0): Fixed 0.030, dynamic 0.790. Reduced scale
  FIXED degenerate repetition (fixed output was "......." repetition).

**Medical: Mixed, small effects:**
- Prompt 1 (sim=0.124, scale=7.8): +0.104 improvement at reduced scale.
- Prompt 4 (sim=0.133, scale=10.1): -0.296 degradation at reduced scale.

### Kill Criteria Assessment

| Kill Criterion | ID | Value | Threshold | Result |
|---------------|-----|-------|-----------|--------|
| K1: Domains improved | #663 | 1/3 | 0/3 to kill | **PASS** (not killed) |
| K2: Embedding-scale correlation | #664 | r=-0.079 | \|r\| < 0.3 | **FAIL** |
| K3: Incoherent output | #665 | 3.3% | > 20% | **PASS** |

**Overall: K2 FAIL.** The experiment is not killed (K1 passes, K3 passes), but
the core hypothesis that embedding similarity predicts optimal scale is refuted.

## Interpretation

### Why K2 Fails: TF-IDF Distance != Adapter Effectiveness

TF-IDF embedding similarity measures lexical overlap with training data.
Adapter effectiveness depends on:
1. Whether the adapter's learned transformations (B@A) are useful for the query
2. How the perturbation interacts with the base model's existing capability
3. Format activation (for math: GSM8K chain-of-thought format)

These are representation-space properties that TF-IDF cannot capture. A math
query with unusual wording (low TF-IDF similarity) may still benefit fully from
the math adapter because the model's internal representations align. Conversely,
a query with high lexical overlap may not benefit if the adapter's B@A
perturbation is miscalibrated for that specific computation.

### Why Math Improved Despite Zero Correlation

The math improvement (+1 correct, from 8/10 to 9/10) is a stochastic effect:
- 2 queries that were wrong at scale=20 became correct at scale=6 (different
  generation path, happened to hit correct answer)
- 1 query that was correct at scale=20 became wrong at scale=11.7

This is NOT evidence that lower scale is better for math. It is noise from
binary evaluation at n=10. The correlation r=-0.009 confirms: similarity
does not predict which queries benefit from scale reduction.

### What Would Work Instead

The literature (LoRAuter, MoLoRA) uses LEARNED routing/scaling:
- LoRAuter: learns a retrieval function mapping queries to scale vectors
- MoLoRA: per-token learnable scale coefficients via gradient descent

The key insight: the mapping from query to optimal scale must be LEARNED, not
derived from a fixed embedding space. TF-IDF captures topic, not adapter
effectiveness.

## Limitations

1. **n=10 per domain:** Too small for reliable correlation estimates. r with n=10
   has 95% CI of approximately +/-0.63, so even r=-0.079 is consistent with true
   correlations anywhere in [-0.70, +0.55].
2. **Single alpha value (0.3):** Different floors might produce different results,
   but this is unlikely to change the zero-correlation finding.
3. **TF-IDF only:** Other embedding spaces (e.g., sentence transformers, model
   hidden states) might have predictive power. This experiment only tests TF-IDF.
4. **3 domains only:** Legal and finance untested.

## What Would Kill This Direction Entirely

The direction (input-dependent scaling) is NOT killed. What is killed is the
specific mechanism (TF-IDF similarity -> scale). The broader question -- can
per-query scale selection improve over fixed per-domain? -- remains open.

Evidence that per-query scaling CAN help: the code prompt 9 case (degenerate
repetition at scale=20, clean output at scale=6) shows that SOME queries genuinely
benefit from reduced scale. The problem is IDENTIFYING those queries.

Fully killing input-dependent scaling would require showing that even with
perfect oracle scale selection per-query, there is no improvement over fixed
per-domain scales. This was NOT tested.

## Status: Provisional

The finding is provisional: TF-IDF similarity does not predict adapter
effectiveness, but the broader question of input-dependent scaling remains
open for exploration with learned routing functions.
