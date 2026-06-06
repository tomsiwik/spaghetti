# Partial RoPE Semantic Routing: Research Digest

## Hypothesis

In Partial RoPE (only 25% of head dimensions use positional encoding), the position-free dimensions (75%) learn pure semantic similarity patterns that naturally cluster by domain, enabling zero-parameter routing without a learned router.

## What This Experiment Tests

We extract pre-RoPE query projections from BitNet-2B-4T (which uses full RoPE) and simulate partial RoPE by splitting each head's dimensions into "RoPE dims" (first 25%) and "free dims" (last 75%). We test whether the position-free Q features cluster by domain well enough to serve as a zero-parameter routing signal. Five feature types are compared: position-free Q (last layer), position-included Q (last layer), full Q (last layer), position-free Q (middle layer), and full hidden states (the softmax router's feature space).

## Key References

- RoFormer (arXiv 2104.09864): RoPE original paper
- Parameter Golf / T2L scaling (arXiv 2506.06105): partial RoPE analysis showing position-free dims learn semantic similarity
- exp_softmax_router_scaling: 40% classification accuracy, oracle PPL quality, 330K params

## Empirical Results

**Model:** BitNet-2B-4T (n_heads=20, head_dim=128, n_kv_heads=5, 24 layers)
**Data:** 24 domains, 50 validation samples each (1200 total)
**RoPE split:** 32/128 dims (25%) simulated as position-encoding, 96/128 (75%) as position-free

### Clustering and Routing Comparison

| Feature Type    | Dims  | Silhouette | Centroid Acc | K-means Acc | Var Ratio |
|----------------|-------|------------|-------------|-------------|-----------|
| q_free_last    | 1920  | -0.007     | 52.4%       | 30.3%       | 0.200     |
| q_rope_last    | 640   | -0.014     | 49.1%       | 28.2%       | 0.221     |
| q_full_last    | 2560  | -0.009     | 51.7%       | 28.9%       | 0.205     |
| q_free_mid     | 1920  | -0.010     | 50.7%       | 30.2%       | 0.183     |
| hidden (baseline) | 2560 | -0.028  | **65.0%**   | **36.3%**   | 0.158     |

### Kill Criteria Assessment

| Criterion | Metric | Threshold | Result | Verdict |
|-----------|--------|-----------|--------|---------|
| K1 | Silhouette score | >= 0.3 | -0.007 | **FAIL** |
| K2 | Routing accuracy | > 4.2% (1/24) | 52.4% | PASS |
| S1 | Routing accuracy | > 60% | 52.4% | FAIL |
| Bonus | Silhouette | > 0.5 | -0.007 | FAIL |

**VERDICT: KILLED (K1)**

## Key Findings

### 1. ALL feature spaces have negative silhouette scores

Every feature type tested -- Q-free, Q-rope, Q-full, hidden -- shows negative silhouette scores on true domain labels. This means domain boundaries are NOT the natural clustering structure of ANY of these representations. The model's internal representations organize around something other than our 24 domain labels.

### 2. Centroid routing works despite no clustering

The paradox: negative silhouette (samples closer to other-domain centroids than own-domain centroid ON AVERAGE) yet 52-65% centroid routing accuracy (12-15x random). This occurs because centroid routing uses the NEAREST centroid, and the domain signal is embedded in a small number of discriminative dimensions that centroids capture, while the silhouette score is dominated by the majority of non-discriminative dimensions that create inter-domain overlap.

### 3. Full hidden states dominate Q projections for routing

Full hidden states (65.0%) substantially outperform all Q-based features (49-52%). The domain routing signal lives primarily in the residual stream (MLP contributions + accumulated skip connections), not in the attention Q projections. This makes sense: the Q projection is optimized for "what to attend to" (a syntactic/positional function), not "what domain is this" (a semantic function).

### 4. Position-free vs position-included Q: negligible difference

q_free_last (52.4%) vs q_rope_last (49.1%): only 3.3pp difference. Splitting Q into "position-free" and "position-included" dimensions produces nearly identical routing signal. In a full-RoPE model, all Q dimensions carry similar domain signal because the Q projection weights already mix all hidden dimensions. The split is meaningless when all dims go through the same linear projection.

### 5. Variance ratio is similar across all Q features (~0.20)

Between-domain variance accounts for ~20% of total variance in Q features vs ~16% in hidden states. Counter-intuitively, Q features have HIGHER between/total variance ratio than hidden states, yet lower routing accuracy. This means the hidden state domain signal is more CONCENTRATED (fewer, more discriminative dimensions), while Q features spread domain signal diffusely across many dimensions.

## Why the Hypothesis Failed

1. **Pre-RoPE Q is not position-free in a full-RoPE model.** The hidden state h arriving at the last layer has already been processed by 23 layers of RoPE-based attention. Position information is deeply encoded in h. The Q projection W_Q applied to h produces a vector that is semantically meaningful but NOT position-free -- it inherits position from the residual stream.

2. **The "position-free dims" concept requires actual partial RoPE training.** The Parameter Golf finding (arXiv 2506.06105) is about models TRAINED with partial RoPE, where the optimizer learns to use the free dims for semantic features. In our simulation, the model was trained with full RoPE, so there's no incentive for any Q dimension to specialize as "semantic-only."

3. **Domain routing signal lives in the residual stream, not attention.** The 65% vs 52% gap shows that MLP layers and skip connections accumulate domain-specific features that the Q projection discards. The attention mechanism focuses on syntactic/structural patterns, while domain identity is a higher-level semantic property.

## Limitations

- We SIMULATE partial RoPE on a full-RoPE model. A model actually trained with partial RoPE might produce different Q features. This experiment can only test whether pre-RoPE Q features from the existing model are useful, not whether partial RoPE training would produce domain-discriminative features.

- 24 domains with 50 samples each. Some domains (many "slice-based" domains per real_data_25_domain_adapters LEARNINGS) share common text distributions, making them genuinely hard to separate.

- Centroid-based routing only (no learned routing on Q features). A learned router on Q features might achieve higher accuracy, but that defeats the "zero-parameter" motivation.

- Numerical warnings (overflow in matmul) suggest some features have extreme values; z-score normalization may not fully address this.

## What Would Kill This (Already Killed)

K1: Silhouette < 0.3 -- **KILLED at -0.007** (negative = worse than random assignment).

## Implications for the Project

1. **Zero-parameter routing from attention features is dead.** The 330K softmax router (0.46% overhead) is needed and cannot be replaced by centroid matching on Q features.

2. **Partial RoPE would NOT automatically produce routing-ready features.** Even if we moved to partial RoPE for other reasons, the position-free dims would not replace a learned router. Domain routing is a residual-stream phenomenon, not an attention-projection phenomenon.

3. **The softmax router's 65% centroid accuracy (from hidden states) vs its 40% learned accuracy is interesting.** The centroid approach on hidden states actually OUTPERFORMS the trained softmax router on classification (65% vs 40-46%), suggesting the softmax router's 500-step training was insufficient. But both match oracle PPL, confirming that within-cluster misrouting is quality-benign.

## Runtime

65.6 seconds total (62.7s feature extraction, 2.8s clustering analysis). Single seed (42).
