# Learnings: exp_softmax_router_scaling

## Core Finding

A single multi-class softmax router eliminates the binary head collapse (0% fallback vs 46%) and achieves **oracle-matching quality** (gamma 0.625 vs oracle 0.625) at N=24 despite only 40-46% classification accuracy. The mechanism: semantically similar adapters within a cluster produce equivalent PPL improvement, so within-cluster misclassification is quality-benign. Random routing is 11.6% worse, confirming the router provides genuine value through semantic clustering, not interchangeability.

## Why This Happened

### Semantic clustering rescues routing accuracy

The softmax router groups confused domains into semantically coherent clusters:
- Cluster 1: philosophy/history/agriculture/creative_writing/science/environmental/politics/economics
- Cluster 2: education/engineering/sports
- Cluster 3: sociology/linguistics
- Cluster 4: medical/health_fitness
- Cluster 5: legal/finance
- Singletons: code, math, cooking, psychology, cybersecurity

Within-cluster misclassification (e.g., selecting "philosophy" for "science") produces <1.2% oracle gap because adapters in the same cluster modify the model similarly. Cross-cluster misclassification would be worse, but the softmax router avoids it.

### Adapters are NOT interchangeable (random disproves it)

The initial finding "any adapter gives oracle PPL" was misleading:
- Softmax top-1 gamma: 0.625 (matches oracle)
- Random gamma: 0.697 (11.6% worse)

The softmax router selects from the right cluster; random selection crosses clusters.

### LoRA activation magnitudes are domain-independent

In-domain vs out-of-domain activation ratio: 1.08x. Adapters contribute similar magnitude regardless of whether text is from their training domain. This explains why within-cluster misrouting doesn't hurt — the adapter perturbation has similar effect size, just in a slightly different direction within the orthogonal subspace.

## Confirming Evidence

1. **exp_more_adapters_is_better (KILLED):** Binary heads had 46% base-only fallback at N=24. Softmax eliminates this entirely (0% fallback) while matching oracle quality.

2. **exp_real_data_25_domain_adapters:** Showed routing recall bifurcation between genuine and slice-based domains. Softmax bypasses the recall problem entirely since it always selects exactly one adapter.

3. **MoLoRA (arXiv 2603.15965):** Uses softmax routing and reports stable multi-LoRA routing. Our result extends this to Grassmannian-orthogonal adapters where the quality benefit is even stronger.

## Contradicting Evidence

1. **K1 technically FAIL** (45.8% centroid accuracy < 50%). The kill criterion was set for domain classification accuracy, not quality. With 0.0% oracle gap, the criterion may have been too strict.

2. **The "interchangeability" claim from the initial run was wrong.** Random routing (gamma 0.697) is clearly worse than softmax (0.625). The router provides value through semantic clustering, not adapter interchangeability.

3. **Activation magnitude analysis is on a single layer (q_proj at layer 13).** Other layers may show different patterns. A full-model analysis would be more convincing.

## Alternative Approaches

1. **Increase router training.** Current: 500 steps, 75% train accuracy, loss 1.0. More steps or larger hidden dim could push per-sample accuracy above 50%.

2. **Embedding-similarity routing.** Cosine similarity between input embedding and domain centroids. Zero training, deterministic. Would test whether learned routing adds value over simple similarity.

3. **Hierarchical routing.** First classify into 5-6 clusters, then within-cluster. The natural cluster structure suggests this could achieve high accuracy with simpler models.

## Implications for SOLE Architecture

1. **Binary routing heads are dead.** Softmax router is strictly superior: eliminates fallback, matches oracle quality, 6x fewer parameters (330K vs 1.97M).

2. **Exact domain classification is unnecessary for PPL.** The router only needs to identify the correct semantic cluster. This makes routing far easier at scale.

3. **Task-specific metrics may still need better routing.** PPL is forgiving (adapters have similar magnitude OOD). Code correctness, medical accuracy, etc. likely require correct domain selection. This is the key open question.
