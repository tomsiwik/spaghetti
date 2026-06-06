# Semantic Router: Research Digest

## Hypothesis

A learned semantic router can classify queries into the correct expert domain with >70% accuracy and <5ms latency per query, outperforming content-agnostic hash ring routing.

## What This Model Is

A controlled comparison of 6 routing strategies for selecting among 15 domain experts, using richer character n-gram embeddings (unigram + bigram + trigram) instead of the bag-of-words embedding used in the prior content_aware_routing experiment:

1. **Hash ring** (baseline): content-agnostic consistent hashing, O(log N)
2. **Keyword frequency**: character frequency profile matching via L2 distance
3. **Cosine similarity**: n-gram embedding vs pre-computed expert centroids
4. **LSH partitioning**: SimHash-style binary codes for spatial bucketing
5. **Utterance matching (1-NN)**: nearest exemplar lookup (semantic-router pattern)
6. **Utterance matching (aggregated)**: mean similarity to per-domain exemplars

Architecture: Pure numpy. 15 domains in 3 clusters (code/reasoning/knowledge), generated via Markov chain transition matrices. Character n-gram features (dim=224) projected to D=64 via random Gaussian projection.

**Key difference from content_aware_routing:** This experiment focuses purely on the routing mechanism (accuracy + latency), not downstream NTP quality. We already know that downstream quality is vacuous at micro scale because LoRA experts do not specialize at d=64.

## Lineage in the Arena

```
content_aware_routing (KILLED: K1, K3)
    |
    +-- semantic_router  <-- this experiment
```

## Key References

- Karger et al. 1997 -- Consistent hashing
- aurelio-labs/semantic-router -- utterance-based routing library
- Ostapenko et al. 2024 -- Arrow: zero-shot LoRA routing via embedding similarity
- Switch Transformers (Fedus et al. 2022) -- k=1 learned router at scale
- Soft MoE (Puigcerver et al. 2024) -- soft token-to-expert assignment

## Empirical Results

### Routing Accuracy (3 seeds, mean +/- std)

| Strategy | Domain Acc | Cluster Acc | Latency/query |
|----------|-----------|-------------|---------------|
| Hash ring (baseline) | 0.069 +/- 0.002 | 0.330 | 2.08 us |
| Keyword frequency | **0.273** +/- 0.009 | 0.935 | 5.74 us |
| Cosine similarity | 0.254 +/- 0.010 | 0.938 | **0.19 us** |
| LSH partitioning | 0.196 +/- 0.014 | 0.821 | 0.33 us |
| Utterance 1-NN | 0.222 +/- 0.001 | **0.971** | 1.75 us |
| Utterance aggregated | 0.226 +/- 0.015 | 0.935 | 1.96 us |
| Oracle | 1.000 | 1.000 | 0 |

### Kill Criteria Assessment

| Criterion | Threshold | Observed | Verdict |
|-----------|-----------|----------|---------|
| K1: Best domain accuracy | >= 70% | 27.3% (keyword) | **KILL** |
| K2: Router latency | < 5ms | 5.74 us (keyword) | PASS |
| K3: Router overhead | < 2% of inference | 5.74 us < 10 us | PASS |

**Overall Verdict: KILL (K1)**

## What We Actually Learned

### 1. Cluster routing is trivially solved (91-97%)

All semantic strategies achieve 82-97% cluster accuracy (3 clusters). The best is utterance 1-NN at 97.1%. This confirms the prior finding from content_aware_routing: inter-cluster discrimination is easy because the Markov chain prototypes create distinct character frequency distributions across clusters.

### 2. Within-cluster domain discrimination is fundamentally information-limited

All strategies achieve only 20-27% domain accuracy, which is barely above the 20% within-cluster random baseline (5 domains per cluster). The information bottleneck analysis (MATH.md Section 4.2) explains why: within-cluster domains differ by only O(noise_scale * 0.5) = O(0.075) in transition probabilities, which compresses to ~1 bit of distinguishing information in the stationary distribution. Domain-level routing requires 2.32 bits within each cluster.

### 3. N-gram features did NOT meaningfully improve over bag-of-words

Despite adding bigram and trigram features (224-dim raw features vs 32-dim unigrams), domain accuracy improved only marginally (27.3% vs 26.5% in prior experiment). The within-cluster Markov chain variation is too small for ANY frequency-based feature to discriminate reliably.

### 4. New routers (LSH, utterance) confirmed the same ceiling

- **LSH** was the worst semantic router (19.6% domain, 82.1% cluster) because binary hash codes lose fine-grained similarity information
- **Utterance 1-NN** achieved the best cluster accuracy (97.1%) but mediocre domain accuracy (22.2%), confirming that nearest-exemplar matching is great for coarse classification but insufficient for fine-grained domain discrimination
- **Utterance aggregated** performed similarly to cosine (22.6% domain), as expected since mean exemplar similarity converges to centroid similarity

### 5. All routers are comfortably within production latency budget

Every strategy runs in <6 us/query. Cosine similarity is the fastest at 0.19 us (pure matrix multiply). Even at production scale (N=500, D=4096), cosine routing would require ~10 us (2M multiply-adds, vectorized). The latency concern is definitively resolved: semantic routing is NOT a latency bottleneck.

### 6. Production recommendation: hierarchical routing

The results point to the same conclusion as content_aware_routing:

1. **Route to cluster** (trivially solved, 97% with utterance 1-NN)
2. **Hash ring within cluster** (avoids the unsolvable within-cluster discrimination problem)

This hierarchical approach would give cluster-correct routing for 97% of queries while maintaining hash ring's plug-and-play simplicity for within-cluster expert selection.

## Micro-Scale Limitations

1. **Synthetic Markov chain data.** The within-cluster domain variation (noise_scale=0.15) is artificially uniform and small. Real domains (python vs javascript vs rust) have much richer distinguishing features (keywords, syntax patterns, semantic content).

2. **Character-level tokenization (V=32).** Real tokenizers (V=32K+) encode word-level and subword-level semantics that would dramatically increase discriminative power.

3. **No expert specialization tested.** We measured routing accuracy only, not whether correct routing matters for downstream quality. At micro scale, expert specialization is absent, so this is appropriate.

4. **Random projection embedding.** Production systems would use pretrained sentence encoders (BERT, sentence-transformers) that capture deep semantic features. Our random projection of character n-grams is a minimal baseline.

5. **15 domains is small.** At N=500, the routing problem becomes harder (more classes) but the embedding space may be more structured (real domain embeddings form tighter clusters in semantic space).

## What Would Kill This at Macro Scale

1. **Domain accuracy <80% with pretrained embeddings at N=50.** If sentence-transformer embeddings from a pretrained encoder cannot discriminate 50 real domains (e.g., math, coding, medical, law, cooking) with >80% accuracy, then semantic routing fails even with real features. This would mean the inter-domain signal is insufficient even for non-parametric methods.

2. **Routing latency >1ms at N=1000 with FAISS.** If approximate nearest neighbor search cannot maintain sub-millisecond routing at scale, hash ring's O(log N) property wins.

3. **Cluster-level routing insufficient.** If within-cluster expert selection significantly affects quality (i.e., math expert is much better than physics expert for math queries), then the hierarchical routing recommendation fails and we need fine-grained domain routing.

4. **Expert quality is so robust that routing doesn't matter.** If every expert helps every domain (due to shared base model knowledge), then content-aware routing adds complexity without benefit. Hash ring wins by being simpler.
