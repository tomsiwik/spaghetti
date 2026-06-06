# Union-Find Expert Merging: Research Digest

## Hypothesis

Union-find with path compression enables dynamic expert library compression
by transitively merging similar capsules, achieving >=20% compression at
<=3% quality loss.

**Falsifiable**: Kill if merged experts >3% worse than pre-merge, or if
merging reduces library size <20%.

**Result: KILLED.** Both kill criteria exceeded simultaneously. Union-find's
transitive closure creates catastrophically large clusters (400+ capsules
merged into one) in Layer 0, destroying quality (+11.55% mean degradation
across 3 seeds at the most favorable threshold). The mechanism is
mathematically sound but the core assumption -- that similarity is
transitive -- is false for co-activation Jaccard.

---

## What This Experiment Tests

Whether the union-find (disjoint-set) data structure with path compression
(Tarjan 1975) can compress composed expert libraries by transitively merging
behaviorally similar capsules. The key innovation over behavioral_dedup's
greedy pairing is transitive closure: if A~B and B~C, all three merge into
one canonical expert.

Protocol:
1. Train shared base on all data (300 steps)
2. Fine-tune MLP-only per domain (200 steps, attention frozen)
3. Compose by concatenating A/B weight matrices (P=512 per layer)
4. Profile behavioral similarity (Jaccard + output correlation)
5. Build union-find: union all pairs above threshold
6. Extract connected components -> merge clusters
7. Merge weights: a-average, b-sum (same as capsule_dedup)
8. Evaluate quality degradation vs compression ratio

---

## Lineage in the Arena

```
gpt -> capsule_moe -> relu_router -> capsule_dedup -> behavioral_dedup -> union_find_merge
                       (composition    (weight-cosine    (co-activation     (transitive
                        by concat)      dedup, KILLED)    Jaccard dedup)     closure merging)
```

---

## Key References

**Union-Find / Disjoint-Set** (Tarjan 1975): Efficient data structure for
maintaining disjoint sets with near-constant-time find and union operations
via path compression and union by rank.

**Behavioral Deduplication** (this project, Exp behavioral_dedup): Found
19.3% of capsules in behaviorally-redundant pairs at tau_rho=0.3, all
concentrated in Layer 0 (J=0.527 mean). Greedy pairing preserves quality.

**TIES-Merging** (Yadav et al., NeurIPS 2023): Trim-elect-sign-merge
resolves sign conflicts in parameter merging. Addresses interference.

**DARE** (Yu et al., 2023): Random drop + rescale of delta params before
merging. Exploits parameter redundancy.

---

## Empirical Results

### Threshold Sweep (3-seed mean: seeds 42, 123, 7)

| Threshold | Compression | Quality Delta | Verdict |
|-----------|-------------|---------------|---------|
| J>0.15, c>0.15 | 23.7% | +17.75% | KILLED |
| J>0.20, c>0.20 | 22.5% | +15.85% | KILLED |
| J>0.30, c>0.30 | 20.3% | +11.55% | KILLED |
| J>0.40, c>0.30 | 19.5% | +11.25% | KILLED |
| J>0.50, c>0.40 | 17.0% | +13.51% | KILLED |

No threshold achieves both kill criteria simultaneously.
Higher thresholds reduce compression below 20% without sufficiently
improving quality.

### Union-Find vs Greedy Pairing (J>0.3, c>0.3, 3-seed mean)

| Method | Compression | Quality Delta |
|--------|-------------|---------------|
| Union-Find (transitive) | 20.3% | +11.55% |
| Greedy (behavioral_dedup) | 9.8% | -0.65% |

Greedy pairing achieves lower compression but dramatically better quality.

### Root Cause: Layer 0 Cluster Explosion

The failure is concentrated in Layer 0, where capsules have high
mean Jaccard similarity (~0.5). At J>0.3 threshold:

| Seed | Layer 0 Before | Layer 0 After | Max Cluster |
|------|----------------|---------------|-------------|
| 42   | 512 | 33 | 476 |
| 123  | 512 | 255 | 239 |
| 7    | 512 | 75 | 433 |

Transitive closure chains loosely-similar capsules into mega-clusters.
Merging 400+ capsules into one destroys Layer 0's representational
capacity.

---

## Key Insight: Similarity Is Not Transitive

The fundamental assumption of union-find -- that the union relation is
transitive -- fails for approximate similarity:

```
sim(A,B) > tau  AND  sim(B,C) > tau  =/=>  sim(A,C) > tau
```

This is a well-known problem in clustering theory. Union-find implements
single-linkage clustering, which is known to produce "chaining" artifacts
where long chains of marginally-similar elements get merged into one
cluster. The effect is catastrophic in Layer 0 where the similarity
graph is dense (mean J~0.5).

## What Was Learned

1. **Greedy pairing is correct for expert merging.** Each capsule should
   merge at most once, with its most similar partner. Transitive closure
   is the wrong abstraction.

2. **Layer 0 is qualitatively different.** Its high co-activation Jaccard
   (~0.5 mean) makes any threshold-based transitive merging scheme
   unstable. Layer 0 capsules share low-level features but are NOT
   functionally interchangeable.

3. **Union-find is useful for the WRONG problem.** It excels at finding
   connected components in graphs with binary edges (connected or not).
   Expert similarity is continuous and non-transitive -- it needs
   average-linkage or complete-linkage clustering, not single-linkage.

4. **The UF vs greedy comparison validates behavioral_dedup.** Greedy
   pairing at J>0.3 achieves -0.65% quality change -- essentially zero
   degradation. This confirms that per-pair merging works; the problem
   is only with transitive closure.

---

## Micro-Scale Limitations

- Only 2 domains tested (binary split a-m vs n-z)
- P=256 per domain pool (512 composed) -- small enough that transitive
  closure is especially destructive
- Character-level tokenization limits domain differentiation
- Only tested a-average/b-sum merging rule (TIES or DARE might help
  with large cluster merging)

## What Would Kill This

Already killed. Both kill criteria exceeded:
- Quality: +11.55% mean (>3% threshold)
- Compression: only achievable with unacceptable quality loss

## Potential Recovery Directions

If the core idea of iterative transitive merging is to be salvaged:

1. **Cluster size cap**: Limit union-find clusters to max size k (e.g., k=3).
   This prevents mega-clusters while allowing small transitive chains.

2. **Average-linkage clustering**: Replace single-linkage (union-find) with
   average-linkage, where a capsule joins a cluster only if its mean
   similarity to ALL existing members exceeds the threshold.

3. **Layer-specific thresholds**: Layer 0 needs tau_J > 0.8 (or skip it
   entirely), while deeper layers could use tau_J > 0.3.

4. **Hierarchical merging**: Merge in rounds with quality gates. After each
   round, re-evaluate quality; stop if degradation exceeds budget.
