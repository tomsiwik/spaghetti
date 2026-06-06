# MATH.md — T4.2: LSH Spatial Routing at N=25 and N=100

## Problem Statement

TF-IDF routing (Finding #431) achieves 96.6% at N=5 and 86.1% at N=25, but requires O(N)
centroid similarity computation per query. As N grows to 100+, O(N) becomes a bottleneck.
We ask: can Locality-Sensitive Hashing (LSH) achieve comparable routing accuracy with
O(log N) lookup time?

---

## Theorem 1 (Indyk-Motwani 1998, STOC): Random Hyperplane LSH

**Setup**: Let x, y ∈ R^d be TF-IDF feature vectors. For a random unit vector v ~ N(0, I_d)/||N(0,I_d)||, define the hash:

    h_v(x) = sign(v^T x)  ∈ {-1, +1}

**Claim**: P(h_v(x) = h_v(y)) = 1 - arccos(cos(x,y)) / π

**Proof**: P(h_v(x) = h_v(y)) = P(v^T x and v^T y have same sign).
Since v is random, the probability of a sign disagreement equals the probability that v
falls in the halfspace separating x from y, which subtends angle arccos(cos(x,y))/π
of the unit sphere. QED.

**Corollary (k-bit code)**: Define g(x) = [h_{v_1}(x), ..., h_{v_k}(x)] using k independent
random hyperplanes. Then:

    P(g(x) = g(y)) = (1 - arccos(cos(x,y))/π)^k

This is the probability of EXACT bucket match in one LSH table.

---

## Theorem 2: LSH Recall with L Tables

**Setup**: Use L independent LSH tables, each with k-bit codes. Item y is in the candidate set
for query x if g_l(x) = g_l(y) for at least one table l ∈ {1,...,L}.

**Recall (true positive rate)**:
    P(y ∈ candidates | cos(x,y) = c) = 1 - (1 - (1 - arccos(c)/π)^k)^L

**Precision factor** (false positive rate):
    P(z ∈ candidates | cos(x,z) = c') = 1 - (1 - (1 - arccos(c')/π)^k)^L

**Routing accuracy** ≈ P(correct domain in candidate set AND has highest similarity score).

---

## Parameter Design

We need k, L such that:
1. Recall ≥ 90% for same-domain pairs (c_true ≥ 0.8 typical for TF-IDF)
2. Low false positive rate (few wrong domains in candidate set)
3. Latency < 0.5ms for N=100 queries

**Calculation for k=8, L=16**:

Cosine sim = 0.8: arccos(0.8) ≈ 0.6435 rad = 36.87°
- Per-table recall: (1 - 36.87/180)^8 = (0.7951)^8 ≈ 0.1665
- L=16 tables: 1 - (1 - 0.1665)^16 = 1 - (0.8335)^16 ≈ 1 - 0.0613 = **0.9387**

Cosine sim = 0.5 (false pair): arccos(0.5) = 60°
- Per-table: (1 - 60/180)^8 = (0.6667)^8 ≈ 0.0390
- L=16 tables: 1 - (1 - 0.0390)^16 = 1 - (0.9610)^16 ≈ 1 - 0.524 = 0.476

So ~47.6% false positive rate for c=0.5 pairs. However, routing uses score-based tie-breaking
(pick max cosine sim in candidate set), so false positives don't cause errors unless
a wrong domain has HIGHER cosine sim than the correct domain.

**Expected routing accuracy** (N=25): depends on the true/false cosine sim distributions
of TF-IDF domain centroids vs. queries. From Finding #431, TF-IDF achieves 86.1% at N=25.
LSH recall at c=0.8 is 93.9%, so LSH routing accuracy ≤ 93.9% × 86.1% ≈ 81% if errors compound.
But errors don't compound: LSH miss-rate (6.1%) replaces exact centroid lookup with a fallback
(search all centroids = TF-IDF fallback). With fallback: LSH accuracy ≈ TF-IDF accuracy = 86.1%.

**Key prediction**: LSH routing ≈ TF-IDF routing (same accuracy) with lower median latency
because most queries hit the correct bucket in O(Lk) time, avoiding O(N*d) centroid search.

---

## Theorem 3: Latency Analysis

**Per-query cost**:
- Hash computation: O(L × k × d) for L tables with k bits each. At L=16, k=8, d≈10000: 1.28M ops
- Hash table lookup: O(L) table lookups = O(L) ops
- Candidate scoring: O(|C| × d) where |C| = expected candidate set size

**Expected candidate set size at N=100**:
    E[|C|] = N × P(false pair in same bucket) + 1 (true pair)
    = 99 × 0.476 + 1 ≈ 48 per table, but with L=16 tables and deduplication:
    
Hmm, this seems high. Let me recalculate:

For k=8, L=16, c_false = 0.5:
- P(false pair in AT LEAST ONE table) = 0.476 (computed above)
- Expected false positives: 99 × 0.476 ≈ 47 domains in candidate set

That's 47 false positives out of 100 domains — too many. We need fewer false positives.

**Revised parameters: k=16, L=8**:
- c_true = 0.8: (1 - 36.87/180)^16 × L=8: recall = 1 - (1-0.7951^16)^8 = 1 - (1-0.0277)^8 = 1 - 0.798 = 0.202

That's only 20.2% recall — too low.

**Optimal parameters: k=4, L=24**:
- c_true = 0.8: (0.7951)^4 = 0.4005 per table; recall = 1 - (1-0.4005)^24 = 1 - (0.5995)^24 ≈ 1 - 0.00020 = **0.9998**
- c_false = 0.5: (0.6667)^4 = 0.1975 per table; FP rate = 1 - (1-0.1975)^24 = 1 - (0.8025)^24 ≈ 1 - 0.0085 = **0.9915**

Wait, 99% false positive rate means almost all 100 domains are in the candidate set — no speedup.

The problem: at N=100 with many similar domains (MMLU subjects), c_false can be 0.5+.
For truly distinct domains (c_false < 0.2), k=4, L=8 gives:
- c_true = 0.8: recall = 1 - (1-0.4005)^8 = 1 - (0.5995)^8 ≈ 1 - 0.0168 = 0.983
- c_false = 0.2: FP rate = 1 - (1-(0.5))^8... 

Let me recalculate:
  arccos(0.2) = 78.46°; per-table: (1 - 78.46/180)^4 = (0.5641)^4 = 0.1011
  FP rate with L=8: 1 - (1-0.1011)^8 = 1 - (0.8989)^8 ≈ 1 - 0.434 = 0.566

Still 56% FP rate for c=0.2 pairs. The fundamental challenge: at high N with similar domains,
LSH's FP rate is high because the hash buckets are not fine-grained enough to separate
semantically adjacent domains.

**Resolution**: Use two-stage routing:
1. Stage 1: LSH hash → candidate set (O(L) time)
2. Stage 2: Exact centroid similarity for candidate set (O(|C|×d))
If |C| << N, we save time. If |C| ≈ N, fall back to linear scan.

For truly distinct domains (math, code, medical — c_false < 0.1), LSH is highly effective.
For adjacent domains (MMLU subjects — c_false = 0.4-0.6), LSH has high FP rate, so the
routing accuracy falls back to TF-IDF centroid selection (same accuracy, same speed).

**Latency prediction** (two-stage):
- Hash: L × k × d/64 = 8 × 8 × 10000/64 ops ≈ 10K ops → ~0.01ms
- Candidate scoring: O(|C| × d) where E[|C|] ≈ 5 (for highly distinct domains)
  = 5 × 10000 = 50K ops → ~0.05ms
- Total: ~0.06ms << 0.5ms threshold

**Prediction**: 
- K1077 (N=25 ≥ 90%): We expect ~86-90% (same as TF-IDF ≈ 86.1%, possibly slightly lower
  due to LSH miss-rate on adjacent MMLU subjects)
- K1078 (N=100 ≥ 80%): We expect ~78-84% (TF-IDF floor is ~80% at this scale)
- K1079 (N=100 < 0.5ms): Expected median ~0.1-0.2ms for distinct domains, ~0.3ms for similar
- K1080: Zero params (trivially true — only random hyperplanes, no learned parameters)

---

## Kill Criteria Predictions

| Criterion | Predicted Value | Pass? |
|-----------|----------------|-------|
| K1077: N=25 accuracy ≥ 90% | ~86-91% | UNCERTAIN |
| K1078: N=100 accuracy ≥ 80% | ~78-85% | UNCERTAIN |
| K1079: N=100 latency < 0.5ms | ~0.1-0.3ms | EXPECTED PASS |
| K1080: Zero trainable params | 0 | CERTAIN PASS |

**Note**: K1077/K1078 are uncertain because they depend on whether TF-IDF similarity
structure allows LSH to avoid false negatives. If true-pair cosine sim < 0.8 for some
MMLU domains, recall drops and accuracy suffers.

---

## References

- Indyk & Motwani (1998) "Approximate nearest neighbors: Towards removing the curse of dimensionality"
  ACM STOC 1998. — Random hyperplane LSH collision probability theorem.
- Charikar (2002) "Similarity Estimation Techniques from Rounding Algorithms"  
  ACM STOC 2002. — Cosine similarity-preserving LSH variant used here.
- Finding #431: TF-IDF routing 96.6% at N=5, 86.1% at N=25 (baseline for comparison)
- exp_lsh_capsule_routing: Prior micro experiment showing LSH routing viable (-1.34% vs softmax)
