# LEARNINGS.md — T4.2: LSH Spatial Routing

**Status: KILLED** | Finding #448

## Core Finding

LSH random hyperplane routing on TF-IDF features fails because the query-centroid cosine
similarity (c=0.23) is far below the assumed threshold (c=0.8). At c=0.23, LSH recall
per query is only 45%, causing routing accuracy to collapse to 44.6% at N=25 vs 86.1%
for TF-IDF brute-force. 

**The positive finding**: LSH IS 1.6× faster than TF-IDF at N=60 (0.90ms vs 1.46ms),
confirming the O(log N) latency property in principle.

## Why LSH Fails on TF-IDF

TF-IDF features are sparse (50/20,000 non-zero entries), producing low cosine similarity
even between a query and its correct domain centroid (c=0.23). The gap between true-pair
(c=0.23) and false-pair (c≈0.20) cosine similarities is only Δc=0.03.

Random hyperplane LSH cannot discriminate pairs with Δc=0.03:
- Differential collision probability ≈ 0.003 per hash table
- Indistinguishable from statistical noise at any reasonable number of tables

Any LSH configuration achieving 90%+ recall also accepts 90%+ false positives,
reducing to brute-force with LSH overhead.

## Implications for Architecture

**TF-IDF routing (T4.1) is the right choice for N≤100.**
At N=100, TF-IDF brute-force costs ~1.5ms (N × 20,000 sparse-dense multiply).
This is fast enough. LSH provides no benefit.

**LSH would help at N≥1,000** where TF-IDF becomes ≥15ms. But P1 targets N≤100.

**Dense embeddings fix LSH.** Gemma 4 hidden states achieve c_true≈0.7-0.9 for
same-domain queries. At c=0.8, LSH with k=6, L=16 achieves 93.9% recall — effective.
A future experiment could test: Gemma 4 hidden state clustering + LSH routing.

## Key Numbers

| Metric | N=25 | N=60 |
|--------|------|------|
| LSH accuracy | 44.6% | 38.2% |
| TF-IDF accuracy | 86.1% | 74.5% |
| Gap (LSH - TF-IDF) | -41.5pp | -36.3pp |
| Query-centroid cos sim (mean) | 0.236 | (same) |
| LSH recall (theoretical) | 44.9% | 44.9% |
| Candidate set mean | 7.96 domains | 17.9 domains |
| LSH latency (median) | 1.08ms | 0.90ms |
| TF-IDF latency (median) | 1.08ms | 1.46ms |
| LSH speedup | 1.0× | 1.6× |

## References

- Charikar (2002) — Random hyperplane LSH for cosine similarity
- Finding #431 — TF-IDF routing baseline (the right tool for N≤100)
- Finding #448 — This experiment's finding
