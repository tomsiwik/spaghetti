# Hierarchical Expert Composition: Research Digest

## Hypothesis

A two-level LoRA hierarchy (foundation rank-4 for broad cluster + specialist rank-4 per domain) handles clustered cross-domain queries better than flat rank-8 experts of equivalent total rank budget.

## Status: KILLED (K1)

Hierarchical composition does not outperform flat composition for within-cluster cross-domain queries. The PPL-probe weighted flat approach is strictly better.

## What This Experiment Tests

Whether explicitly separating shared knowledge (foundation) from domain-specific knowledge (specialist) via SVD decomposition improves composition quality over flat experts that implicitly contain both.

**Architecture comparison (equalized rank budget = 8 per domain):**

| Component | Flat | Hierarchical |
|-----------|------|-------------|
| Per-domain adapter | rank-8 | foundation rank-4 + specialist rank-4 |
| Shared knowledge | Implicit in each expert | Explicit foundation per cluster |
| Cluster structure | None | 2 clusters (symbolic: arith+parity, string: reverse+repeat+sort) |
| Total rank budget | 8 per domain | 4+4 = 8 per domain |

**Composition approaches tested:**
1. `flat_equal` -- equal-weight flat composition
2. `flat_ppl` -- PPL-probe weighted flat composition
3. `hier_equal` -- foundation + equal-weight specialists
4. `hier_ppl` -- foundation + PPL-probe weighted specialists

## Key References

- exp_orthogonality_by_domain_type (proven): within-cluster |cos| 7.84x higher than cross-cluster. This motivated the hierarchy hypothesis.
- exp_cross_domain_dilution_vs_k (proven): PPL-probe weighting resolves dilution (+9.34pp over equal-weight). The parent result that this experiment builds on.
- MoE-Adapters4CL (arXiv 2404.09855): MoE adapters on frozen base for continual learning. Closest architecturally to our approach.
- AdapterFusion (arXiv 2005.00247): Two-stage adapter composition (extract, then fuse). Similar philosophy to hierarchical approach.
- X-LoRA (arXiv 2402.07148): Dynamic layer-wise LoRA mixing, biologically-inspired hierarchical reuse.

## Empirical Results

**Configuration:** d=64, H=4, L=2, 5 domains in 2 clusters, rank_flat=8, rank_foundation=4, rank_specialist=4, 5 seeds, 200 train/domain.

### Within-Cluster Cross-Domain Queries (4 pair types, 20 measurements)

| Strategy | Mean Gap vs Base | Std | Max |
|----------|-----------------|-----|-----|
| flat_equal | -7.25% | 10.99% | +8.35% |
| **flat_ppl** | **-16.57%** | 11.70% | +4.82% |
| hier_equal | -7.29% | 10.31% | +12.70% |
| hier_ppl | -13.58% | 8.90% | +1.98% |

Within-cluster: flat_ppl beats hier_ppl by 2.99pp (p=0.381, not significant, but consistently worse across seeds).

### Across-Cluster Cross-Domain Queries (6 pair types, 30 measurements)

| Strategy | Mean Gap vs Base | Std | Max |
|----------|-----------------|-----|-----|
| flat_equal | +4.14% | 17.53% | +36.48% |
| flat_ppl | -13.53% | 11.97% | +14.14% |
| hier_equal | -4.24% | 17.77% | +34.16% |
| **hier_ppl** | **-14.17%** | 11.65% | +13.52% |

Across-cluster: hier_ppl slightly better by 0.64pp. Foundation averaging helps when experts come from different clusters.

### Per-Type Breakdown (flat_ppl vs hier_ppl)

| Cross-Type | Scope | Flat PPL | Hier PPL | Delta |
|-----------|-------|----------|----------|-------|
| arith_reverse | across | -12.88% | -8.69% | -4.19pp |
| arith_sort | across | -12.76% | -6.94% | -5.82pp |
| arith_repeat | across | -23.92% | -24.58% | +0.66pp |
| arith_parity | **within** | -4.40% | -6.50% | **+2.10pp** |
| reverse_repeat | **within** | -32.16% | -24.16% | -8.00pp |
| reverse_sort | **within** | -12.74% | -13.73% | **+0.99pp** |
| reverse_parity | across | -13.94% | -20.20% | +6.26pp |
| repeat_sort | **within** | -16.99% | -9.94% | -7.05pp |
| repeat_parity | across | +4.30% | -2.35% | +6.66pp |
| sort_parity | across | -21.99% | -22.26% | +0.27pp |

The hierarchy helps in some cases (arith_parity, reverse_sort, repeat_parity) but hurts significantly in others (reverse_repeat -8.00pp, repeat_sort -7.05pp). The large regressions dominate.

### Subspace Sharing (averaged across seeds)

| Pair | Cluster | cos |
|------|---------|-----|
| arithmetic vs parity | symbolic | 0.05-0.13 |
| reverse vs sort | string | 0.44-0.49 |
| reverse vs repeat | string | 0.18-0.28 |
| repeat vs sort | string | 0.22-0.25 |
| cross-cluster | mixed | 0.02-0.15 |

The "string" cluster has meaningful shared subspace (cos 0.18-0.49), while "symbolic" has very low sharing (cos 0.05-0.13). This explains why hierarchy helps for some across-cluster pairs (reverse_parity +6.26pp: foundation captures string manipulation) but not for the symbolic cluster.

### Complexity

| Metric | Flat | Hierarchical | Overhead |
|--------|------|-------------|----------|
| Processing time | 89.81s | 90.05s | +0.3% |

SVD extraction is negligible. The overhead comes from having to score foundation+specialist combos vs flat experts during PPL-probe. K2 PASSES easily.

## Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: hier better than flat (within-cluster) | hier_mean < flat_mean | flat_ppl=-16.57% vs hier_ppl=-13.58%, delta=-2.99pp | **KILL** |
| K2: complexity <30% OR improvement >=5% | overhead OR quality | overhead=0.3%, improvement=-2.99pp | PASS |

**Overall: KILLED.** Hierarchy does not improve within-cluster composition over flat+PPL-probe.

## Why It Failed

1. **Foundation averages away discriminative information.** SVD of stacked deltas finds the shared subspace, but this shared component is already present in each flat expert. When PPL-probe can weight experts correctly, the foundation adds no new information -- it just constrains the rank budget.

2. **Rank allocation is rigid.** The rank-4 foundation + rank-4 specialist split is fixed at training time. For pairs where the shared subspace is large (reverse+sort, cos=0.48), this might be appropriate. For pairs where sharing is minimal (arith+parity, cos=0.10), the foundation wastes rank capacity on noise.

3. **PPL-probe already solves the problem.** The flat_ppl strategy achieves -16.57% improvement over base by simply weighting the right expert higher. This is a strictly simpler mechanism that adapts at inference time rather than requiring a priori cluster structure.

4. **Hierarchy helps the WRONG cases.** It marginally helps across-cluster pairs (+0.64pp) where the foundation acts as a denoiser, but hurts within-cluster pairs (-2.99pp) where it was supposed to help most. The foundation for a cluster constrains what the specialists can represent, reducing adaptivity.

## Implications for SOLE

1. **Flat composition + PPL-probe weighting is the correct architecture.** No need for explicit cluster structure or foundation layers. The 7.84x within-cluster cosine similarity is a real phenomenon but does not translate into a useful architectural feature.

2. **Adaptive weighting beats structural assumptions.** Rather than building cluster knowledge into the architecture (which requires knowing clusters a priori), let the PPL-probe discover relevance at runtime. This is more flexible and more effective.

3. **Domain hierarchy is NOT a routing/composition feature.** It is a training data organization feature. Clustering helps for data generation and expert naming, but the composition layer should remain flat.

4. **The shared subspace exists but is better utilized via weight selection, not extraction.** When two string experts (cos=0.48) both contribute to a query, PPL-probe naturally upweights the more relevant one while keeping the shared signal from both. Foundation extraction forces this shared signal through a single fixed representation.

## Limitations

1. **Micro scale only.** At d=4096, within-cluster cosine may be more meaningful, and foundation extraction might capture richer shared representations. However, the fundamental argument (PPL-probe adapts better than rigid structure) likely holds.

2. **Synthetic data with known cluster structure.** Real domains may have more complex hierarchical relationships. A three-level hierarchy (meta-category > category > domain) might behave differently.

3. **Only two clusters tested.** The hypothesis might hold with more clusters or different cluster compositions.

4. **Foundation extraction via SVD.** Training a foundation adapter on cluster-mixed data (rather than extracting via SVD) might work better, but adds significant training cost.

## What Would Kill This (if it had passed)

- Foundation extraction failing to converge at d=4096
- PPL-probe cost becoming prohibitive at large K (K+1 forward passes)
- Within-cluster cosine too low for meaningful foundation (<0.05)
- Foundation training cost exceeding flat expert training by >50%
