# Routing Strategy Comparison: Mathematical Foundations

## 1. Problem Setup

Given N experts with quality matrix Q in R^{D x N} where Q[d,e] is the quality
of expert e on domain d, and a query embedding q in R^E, select expert(s) to
maximize expected quality while minimizing routing latency.

### Variables

| Symbol | Shape | Definition |
|--------|-------|------------|
| N | scalar | Total number of experts |
| D | scalar | Number of distinct domains |
| E | scalar | Embedding dimension |
| C | scalar | Number of domain clusters |
| k | scalar | Number of experts selected per query |
| Q | (D, N) | Quality matrix: Q[d,e] = quality of expert e on domain d |
| q | (E,) | Query embedding |
| c_e | (E,) | Expert centroid embedding for expert e |
| mu_c | (E,) | Cluster centroid for cluster c |
| s | scalar | Specialization strength |

### Quality Model

Each expert e has a home domain h(e). Quality decomposes as:

    Q[d, e] = q_base + s * I(d = h(e)) + s * alpha * I(cluster(d) = cluster(h(e)))

where:
- q_base ~ Uniform(0, 0.2): background quality
- s: specialization strength (how much better an expert is at its home domain)
- alpha = 0.3: within-cluster knowledge sharing factor
- I(.) is the indicator function

## 2. Routing Strategies: Complexity Analysis

### 2.1 Pre-merge (Strategy A)

**Operation:** Merge all N experts into base weights offline. No routing at query time.

    W' = W + (1/N) * sum_{i=1}^{N} B_i @ A_i

**Routing complexity:** O(1) -- no routing step.
**Quality:** Q_premerge(d) = (1/N) * sum_{e=1}^{N} Q[d, e] = mean quality across all experts.
**Offline cost:** O(N * r * d) for merge (one-time, amortized).

For large N with specialization:
    Q_premerge(d) ~ q_base + s/N + s*alpha*(D_c - 1)/N

where D_c = D/C is domains per cluster. As N grows, specialization bonus dilutes as O(1/N).

### 2.2 Hash Ring (Strategy B)

**Operation:** Hash query to select expert via consistent hashing.

    h(q) = MD5(query_key) mod ring_size
    expert = ring[bisect(hashes, h(q))]

**Routing complexity:** O(log(N * V)) where V = virtual nodes per expert (150).
With V=150: O(log(150N)).
At N=100: log2(15000) = 13.9 comparisons.
At N=1000: log2(150000) = 17.2 comparisons.

**Quality:** Expected quality = E[Q[d, h(d)]] = (1/N) * sum_e Q[d,e] = same as random.
Hash ring is content-unaware: quality = random baseline.

**Measured:** 0.8 us/query at N=30, scaling O(log N).

### 2.3 Embedding Similarity (Strategy C)

**Operation:** Cosine similarity of query embedding against all expert centroids.

    selected = argmax_e (q / |q|) . (c_e / |c_e|)

**Routing complexity:** O(N * E) -- one dot product per expert.
At E=64, N=100: 6400 multiply-adds = ~1us on modern CPU.
At E=4096, N=500: 2M multiply-adds = ~10us.

**With FAISS index:** O(log N * E) via approximate nearest neighbor.

**Quality:** Depends on embedding quality. With oracle centroids (domain centroids),
achieves quality_lift of 0.023-0.078 over random (varying with N and ratio).

**Measured:** 3.1 us/query at N=30, scaling linearly with N.

### 2.4 Tiny Classifier (Strategy D)

**Operation:** Two-layer MLP: E -> H -> N with ReLU activation.

    h = ReLU(q @ W1 + b1)     # W1: (E, H), b1: (H,)
    logits = h @ W2 + b2       # W2: (H, N), b2: (N,)
    selected = argmax(logits)

**Routing complexity:** O(E*H + H*N). With E=64, H=32, N=100: 5248 multiply-adds.
N-dependent through the output layer, but fast for small H.

**Quality:** After training on (embedding, best_expert) pairs. Competitive with
embedding similarity at small N, but requires labeled training data.

**Measured:** 3.5 us/query at N=30.

### 2.5 Hierarchical (Strategy E)

**Operation:** Two-stage routing:
1. Cluster classification: cosine similarity against C cluster centroids, O(C*E)
2. Hash ring within cluster: O(log(N/C * V))

    cluster = argmax_c (q / |q|) . (mu_c / |mu_c|)
    expert = cluster_ring[cluster].lookup(hash(query_key))

**Total complexity:** O(C*E + log(N*V/C))
At C=3, E=64, N=100: 192 + log2(5000) = 192 + 12.3 ~ O(200) operations.
Dominated by cluster cosine at small C; by hash at large N/C.

**Quality:** Gets cluster-level accuracy (~95-97% from prior experiments) but
within-cluster selection is random (hash ring). Quality lift depends on
inter-cluster quality differences.

**Measured:** 3.9 us/query at N=30.

## 3. Quality Capture Analysis

### 3.1 Oracle Quality

The oracle selects the best expert for each domain:

    Q_oracle(d) = max_e Q[d, e]

For the quality model with N experts and D domains:
    E[Q_oracle(d)] ~ q_base_max + s = 0.2 + 0.8 = 1.0

where q_base_max = max of N Uniform(0, 0.2) draws ~ 0.2 * (1 - 1/(N+1)).

### 3.2 Quality Capture

    quality_capture = E[Q_router(d)] / E[Q_oracle(d)]

For pre-merge:
    quality_capture_premerge = (q_base_mean + s/N + ...) / (q_base_max + s)
    ~ (0.1 + 0.8/N) / (0.2 + 0.8)
    = (0.1 + 0.8/N) / 1.0

At N=15: ~ 0.153. At N=30: ~ 0.127. At N=100: ~ 0.108.

This explains why quality_capture is fundamentally low: the oracle benefits from
the full specialization bonus s, while pre-merge dilutes it by 1/N.

### 3.3 Quality Lift Over Random

A fairer comparison:

    quality_lift = (Q_router - Q_random) / (Q_oracle - Q_random)

This measures what fraction of the ACHIEVABLE gap is captured.

For embedding similarity at N=30: lift = 0.034 (3.4% of achievable gap).
For embedding similarity at N=5 (1:1): lift = 0.078 (7.8%).

### 3.4 Fundamental Limitation: Embedding Insufficiency

Even with perfect domain embeddings, a routing strategy cannot achieve >90%
quality capture because:

1. **Per-query quality variation:** Q[d,e] has per-query noise. The oracle has
   per-query information; the router only has embedding information.

2. **Multi-expert-per-domain:** With N/D > 1, the router must distinguish between
   experts in the same domain. Embeddings do not encode this.

3. **Quality != similarity:** The best expert for a query is not necessarily the
   most similar expert in embedding space.

The achievable quality capture for any embedding-based router is bounded by:

    quality_capture <= (s * domain_accuracy + q_base_mean) / (s + q_base_max)

where domain_accuracy is the probability of correct domain classification.
At domain_accuracy = 1.0, s = 0.8, q_base = 0.1, q_base_max = 0.2:
quality_capture <= (0.8 * 1.0 + 0.1) / (0.8 + 0.2) = 0.90

So 90% capture is THEORETICALLY possible with perfect domain routing, but only
when there is exactly 1 expert per domain and domain accuracy is 100%.
With N/D > 1, the bound tightens to quality_capture <= 0.90 * D/N.

## 4. Latency Scaling Summary

| Strategy | N=10 | N=30 | N=100 | N=500 | N=1000 |
|----------|------|------|-------|-------|--------|
| Pre-merge | 0 | 0 | 0 | 0 | 0 |
| Hash ring | ~0.8us | ~0.8us | ~0.9us | ~1.1us | ~1.3us |
| Embed sim | ~3us | ~3us | ~3.5us | ~17us | ~33us |
| Classifier | ~3.5us | ~3.5us | ~3.7us | ~15us | ~30us |
| Hierarchical | ~3.9us | ~3.9us | ~4.0us | ~4.3us | ~4.5us |

Reference: micro forward pass ~1ms. Macro forward pass ~30ms.

All strategies are <0.5% of inference time at N<=100.
Embedding similarity and classifier become notable at N>500.

## 5. Worked Example

**Setup:** N=30, D=15, C=3, E=64, s=0.8

**Quality matrix (illustrative row for domain d=0):**
- Expert 0 (home domain): Q[0,0] = 0.15 + 0.8 = 0.95
- Expert 15 (same cluster): Q[0,15] = 0.12 + 0.24 = 0.36
- Expert 5 (different cluster): Q[0,5] = 0.08

**Oracle selects expert 0:** Q_oracle(0) = 0.95

**Pre-merge:** Q_premerge(0) = mean(Q[0,:]) ~ 0.1 + 0.8/30 + 0.24*4/30 = 0.159
quality_capture = 0.159 / 0.95 = 16.7%

**Embedding similarity:** If embedding correctly identifies domain 0, selects
expert 0 or 15. Expected quality ~ (0.95 + 0.36) / 2 = 0.655.
quality_capture = 0.655 / 0.95 = 68.9%. But domain accuracy is ~27%, so
expected capture is much lower.

**Hash ring:** Random expert. Q_random(0) ~ 0.155.
quality_capture = 0.155 / 0.95 = 16.3%.

This shows the fundamental tradeoff: pre-merge and hash ring achieve similar
quality (~16%), embedding similarity CAN achieve higher quality (~69% if domain
routing is perfect) but WITH THESE EMBEDDINGS achieves only ~27% due to
imperfect domain classification.

## 6. Assumptions

1. Quality is additive across experts (linear composition)
2. Expert quality is a function of domain, not individual query content
3. Embedding similarity correlates with domain membership
4. Routing latency is independent of query content (data-independent)
5. Specialization strength is uniform across experts
6. No expert-expert interaction effects (quality of combining 2 experts is mean)
