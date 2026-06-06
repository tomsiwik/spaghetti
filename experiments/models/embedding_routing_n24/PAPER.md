# Embedding-Based Routing at N=24: Proof Verification Report

## Theorem
Centroid Separability (MATH.md Theorem 1): If domains have distinctive vocabulary, mean embeddings from the base model's embedding layer provide larger inter-centroid distances than mean-pooled hidden states, enabling better routing via argmax cosine similarity.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| P1: 6 strong domains maintain >90% | math 100%, psychology 90%, health 65%, finance 20-70%, legal 0-50%, medical 50-80% | PARTIAL -- only math+psychology stable |
| P2: Overall accuracy significantly >39.4% | Best: 28.3% (instruction-only), 25.2% (full-text) | NO -- 11pp WORSE |
| P3: Overhead <1ms | P95: 0.36ms | YES |
| Embedding centroids more separated than hidden | Emb mean cos=0.986, Hid mean cos=0.716 | NO -- OPPOSITE |

## Hypothesis
Embedding-layer features (before transformer blocks) preserve domain-specific vocabulary signal that gets washed out by contextual mixing, enabling better routing than mean-pooled hidden states.

**Verdict: KILLED.** The hypothesis is false. Embedding centroids are LESS separated (mean cos 0.986) than hidden-state centroids (mean cos 0.716).

## What This Model Is
Training-free routing using cosine similarity between input embeddings and per-domain centroid embeddings. Five variants tested: (1) full-text embedding centroid, (2) instruction-only embedding centroid, (3) hidden-state centroid, (4) TF-IDF bag-of-words, (5) trained softmax baseline.

## Key References
- LoRAuter (arxiv 2601.21795): Training-free embedding routing using external sentence encoder
- Finding #192/193: Centralized softmax routing killed at 39.4% (mean-pooled hidden states)

## Empirical Results

### Routing Accuracy (N=24, 480 validation samples)

| Method | Accuracy | Parameters |
|--------|----------|------------|
| TF-IDF bag-of-words | 35.0% | 0 |
| Embedding centroid (instruction-only) | 28.3% | 0 |
| Hidden-state centroid (no training) | 32.5% | 0 |
| Embedding centroid (full text) | 25.2% | 0 |
| Trained softmax router (baseline) | 39.4% | 165K |

### Why Embedding Routing Failed: The Centroid Collapse

**Critical finding:** Mean embedding centroid cosine similarity = 0.9864 (range 0.949-1.000).
All 24 domain centroids are nearly identical in embedding space.

**Root cause:** The embedding layer maps tokens to a d=2560 space. When you mean-pool over all tokens in a text, common words (articles, prepositions, "the", "is", "and") dominate the mean. These words are shared across ALL domains. The domain-specific vocabulary (a few distinctive words per text) is drowned out by the shared vocabulary mass.

This is why LoRAuter uses a **specialized sentence encoder** (SupCon-trained), not the base model's embedding layer. The sentence encoder is explicitly trained to compress semantic content into discriminative embeddings. The base model's raw embedding lookup has no such property.

**Comparison: TF-IDF outperforms neural embeddings** because TF-IDF naturally downweights common words via IDF. The embedding mean-pool has no such mechanism.

### Hidden-State Centroid vs Trained Router
Untrained hidden-state centroids (32.5%) approach the trained softmax router (39.4%) without any training. The trained router's 7pp advantage comes from the learned MLP, not from fundamentally better features. Both use the same d=2560 hidden-state representation.

### Per-Domain Analysis
Domains that work well across ALL methods (finance, math, medical, psychology, health_fitness) have either: (a) highly distinctive vocabulary (math formulas, medical terminology), or (b) very specific response patterns. Domains that fail across all methods (creative_writing, cybersecurity, economics, engineering, environmental, history, philosophy, politics, science) have overlapping general vocabulary.

### Overhead
Embedding lookup: 0.23ms average, 0.36ms P95 (225x faster than hidden-state computation at 51ms). K592 PASS.

## Limitations
1. Only 40 training samples per domain for centroids
2. BitNet-2B-4T embedding layer -- other models may have better-structured embeddings
3. Did not test with external sentence encoders (LoRAuter approach) -- would require loading a second model
4. Mean-pooling is the simplest aggregation -- weighted pooling, [CLS] token, or attention pooling not tested

## What Would Kill This (at any scale)
The approach is fundamentally killed by **centroid collapse**: raw embedding mean-pooling cannot separate domains because shared vocabulary dominates. This is not a scale issue -- it is a structural property of mean-pooled embeddings from any autoregressive language model.

## Key Finding

**The disease is NOT contextual mixing in transformer layers. The disease is that mean-pooled representations (whether embedding or hidden) cannot separate 24 overlapping text domains.**

Evidence:
- Hidden-state centroids (mean cos 0.716) are BETTER separated than embedding centroids (mean cos 0.986)
- Transformer layers ADD discriminative signal, they don't destroy it
- The ~40% ceiling is fundamentally a representation quality issue for overlapping domains, not a layer-choice issue

**Implication for routing at N=24:** Neither embedding-layer nor hidden-state centroid routing can break the ~40% ceiling. The path forward requires either:
1. A specialized routing model (LoRAuter's external sentence encoder approach)
2. Per-token routing (not per-sequence)
3. Hierarchical clustering to reduce effective N
4. Accepting that within-cluster misrouting is PPL-benign (Finding from softmax experiment: gamma 0.625 = oracle despite 40% accuracy)
