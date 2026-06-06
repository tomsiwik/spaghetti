# Content-Aware Routing: Research Digest

## Hypothesis

Content-aware routing strategies (MLP classifier, cosine similarity, keyword matching) outperform hash ring routing on domain-specific queries, as measured by expert selection accuracy and downstream NTP quality.

## What This Model Is

A controlled comparison of 5 routing strategies for selecting among 15 LoRA domain experts:

1. **Hash ring** (baseline): content-agnostic consistent hashing, O(log N)
2. **MLP classifier**: embed query -> Linear(d, N_experts) -> softmax, trained on (embedding, domain) pairs
3. **Cosine similarity**: query embedding vs precomputed expert centroids
4. **Keyword matching**: character frequency histogram vs domain frequency profiles
5. **Oracle**: perfect domain labels (upper bound)

Plus two composition baselines: base model only, and pre-merge (average all 15 experts).

Architecture: 4-layer MLP, d=64, d_ff=256, rank-8 LoRA on all MLP layers. 15 domains in 3 clusters (code/reasoning/knowledge), generated via Markov chain transition matrices. Pure numpy, no MLX.

## Lineage in the Arena

```
orthogonality_by_domain_type
    |
    +-- content_aware_routing  <-- this experiment
    |
    +-- (future) exp_routing_at_scale
```

## Key References

- Karger et al. 1997 — Consistent hashing: 1/N displacement property
- Shazeer et al. 2017 / Switch Transformers — top-k expert routing
- Soft MoE (2023) — soft token-to-expert assignment
- Union of Experts (2024) — expert weights contain routing signal
- MoRAM (2025) — LoRA as key-value associative memory

## Empirical Results

### Routing Accuracy (3 seeds, mean +/- std)

| Strategy | Domain Acc | Cluster Acc | Latency/query |
|----------|-----------|-------------|---------------|
| Hash ring | 0.066 (random) | 0.33 (random) | 98.3 us |
| MLP classifier | 0.085 | ~0.62 | 1.4 us |
| Cosine similarity | 0.255 | **0.95** | 1.5 us |
| Keyword matching | **0.265** | **0.96** | 5.8 us |
| Oracle | 1.000 | 1.000 | 0 |

### NTP Quality (3 seeds)

| Strategy | Mean NTP Loss |
|----------|--------------|
| Base model (no expert) | 3.4657 |
| Hash ring | 3.4657 |
| MLP classifier | 3.4657 |
| Cosine similarity | 3.4657 |
| Keyword matching | 3.4657 |
| Oracle (perfect routing) | 3.4657 |
| Pre-merge all 15 | 3.4657 |

**All losses are identical to 4 decimal places.** Routing has zero effect on NTP quality because the LoRA experts did not meaningfully specialize (loss ~3.466 throughout training, barely moving from initialization). This is the known micro-scale limitation: d=64 with 300 steps and V=32 character-level data does not provide enough gradient signal for LoRA specialization.

### Kill Criteria Assessment

| Criterion | Threshold | Observed | Verdict |
|-----------|-----------|----------|---------|
| K1: Best CA accuracy | >= 60% | 26.5% (keyword) | **KILL** |
| K2: CA latency | < 10ms | 5.8 us | PASS |
| K3: Quality gap | > 5% vs hash | 0.00% | **KILL** |

**Overall Verdict: KILL (K1, K3)**

## What We Actually Learned

Despite the kill, the experiment reveals three important directional findings:

### 1. Cluster-level routing is solved (~96% accuracy)
Both cosine similarity and keyword frequency matching achieve near-perfect CLUSTER-level routing (95-96%), far above the 33% random baseline. The data's cluster structure (different character frequency profiles) is trivially detectable by even simple methods.

### 2. Within-cluster discrimination is the hard problem
All routers struggle to distinguish individual domains within a cluster (26% domain accuracy vs 20% within-cluster random baseline). This is consistent with the orthogonality_by_domain_type finding: within-cluster experts are 7.84x more similar. The routing difficulty mirrors the interference difficulty -- the domains that are hardest to route to correctly are exactly the ones where picking the wrong expert matters most.

### 3. The MLP classifier catastrophically fails at micro scale
The MLP router achieves only 8.5% accuracy (barely above 6.7% random). It learns to predict only one cluster (all queries -> code, or all queries -> reasoning) and never recovers. This is because:
- The embedding space has too little variance at d=64 with bag-of-words
- 500 SGD steps on a 15-class problem with minimal feature separation is insufficient
- The MLP lacks hidden layers to learn nonlinear decision boundaries

In contrast, cosine similarity and keyword matching -- which use NO training -- are 3x better. This suggests that at micro scale, non-parametric methods dominate.

### 4. Routing is irrelevant when experts don't specialize
The identical NTP losses across all strategies (including oracle) prove that at this scale, expert selection does not affect prediction quality. The LoRA deltas are too small to meaningfully change the model's output distribution. This means K3 is trivially killed -- but it would likely PASS at macro scale where experts genuinely specialize.

## Implications for the SOLE Architecture

1. **Hash ring is sufficient at micro scale** because routing doesn't matter when experts don't specialize. The real test is at macro scale.

2. **At macro scale, consider hierarchical routing:** First route to cluster (trivially solvable with cosine/keyword), then hash ring within cluster. This avoids the hard within-cluster discrimination problem while capturing the easy inter-cluster signal.

3. **Cosine routing is the sweet spot:** 1.5 us latency, 96% cluster accuracy, no training required, no parameters to maintain. It uses the base model's existing embedding layer.

4. **Skip MLP routers at small N:** For N < 50, the MLP has too little signal and too many failure modes. Cosine similarity or keyword matching are more robust.

5. **Routing matters only when experts specialize.** Before investing in routing, ensure distillation quality (priority 1 in the roadmap).

## Micro-Scale Limitations

1. **Experts did not specialize.** Loss ~3.466 throughout training for all 15 experts. The LoRA deltas reflect gradient noise direction, not learned features. This makes the quality comparison (K3) vacuous.

2. **Bag-of-words embedding.** Real systems use transformer hidden states which carry much richer contextual information. Routing from CLS tokens or mean pooling over transformer layers would be substantially more discriminative.

3. **Character-level tokenization.** V=32 character-level tokens provide minimal semantic signal. Real tokenizers (V=32K+) encode word and subword semantics that would dramatically improve both expert specialization and routing accuracy.

4. **15 domains is small.** At N=500 or N=1000, the routing problem becomes harder (more classes) but also more structured (deeper cluster hierarchies).

5. **No gradient through router.** The routers are trained independently from the experts. Joint training (as in Switch Transformer / Mixtral) could improve routing accuracy.

## What Would Kill This at Macro Scale

1. **Cosine routing accuracy < 80% at cluster level with real embeddings.** If transformer embeddings from a pretrained base don't carry enough domain signal to route at cluster level, content-aware routing is dead.

2. **Routing latency > 1ms at N=1000.** If approximate nearest neighbor search (FAISS) cannot maintain sub-millisecond routing at scale, hash ring's O(log N) property wins.

3. **Expert specialization is so strong that ANY expert helps.** If distilled experts are individually strong enough that even a wrong-domain expert improves over base, then routing accuracy doesn't matter and hash ring wins by being simpler.

4. **Within-cluster routing matters.** If math vs physics expert selection significantly affects quality (not just math-cluster vs knowledge-cluster), then we need routers that can discriminate at the domain level, which this experiment suggests is fundamentally harder.
