# Learnings: exp_molora_per_token_mlx

## Core Finding
Per-token routing provides no benefit over per-sequence routing on cleanly separated domains (-0.46%, 7.63 vs 7.60 PPL), but the Gumbel-sigmoid routing mechanism itself works correctly on MLX with negligible overhead (0.58%) and non-degenerate routing patterns (diversity 2.42).

## Why This Happened (Literature-Grounded)

The null result is predicted by the routing granularity literature. When sequences are domain-homogeneous (all tokens share a single domain label), per-token routing reduces to a noisy version of per-sequence routing -- every token in a "python" sequence should activate the same experts, so token-level decisions add variance without signal.

**Mod-Squad** (Chen et al., NeurIPS 2023) explicitly moved away from token-level routing, finding that task-level "squads" of experts outperform per-token gating when domain boundaries are clear. Their mutual information loss encourages expert specialization at the task granularity, not the token granularity.

**L2R** (Learning to Route) routes at the sequence level using the `[CLS]` token through a Gumbel-sigmoid router. Their architecture choice implicitly assumes domain identity is a sequence-level property, not a token-level one -- and this is validated by strong results on continual learning benchmarks (MTL5, WOS, AfriSenti).

The key insight is that **routing granularity should match supervision granularity**. Our router was trained with per-domain labels (all tokens in a sequence get the same label), so it learned per-sequence patterns. Token-level routing without token-level supervision is an architecture-data mismatch.

## Confirming Evidence

1. **Mod-Squad** -- Task-level routing outperforms token-level routing when domains are clearly separated. Mutual information loss drives expert specialization at the task level.

2. **L2R** (Ponti et al., 2023) -- Sequence-level Gumbel-sigmoid routing achieves strong results without per-token granularity. On near-domain (homogeneous) benchmarks, simple sequence-level routers are competitive because domain identities are easily distinguishable without token-level analysis.

3. **MoLoRA** (Feng et al., arXiv:2403.03432) -- The original MoLoRA paper focuses on multi-task settings with clearly separated tasks, operating at the task/sequence level rather than demonstrating per-token benefits on homogeneous data.

4. **Memory wall on edge devices** -- Per-token routing causes memory thrashing from weight shuffling between experts. Sequence-level routing maintains a consistent expert set, which is especially important on edge devices (M5 Pro). Our 0.58% overhead at N=5 experts is acceptable, but this would not scale to N=50+.

## Contradicting Evidence

No papers in the research corpus demonstrate per-token routing significantly outperforming per-sequence routing on homogeneous/single-domain data. This absence is itself informative -- the literature appears to assume that per-token routing's value proposition requires intra-sequence heterogeneity.

However, the **MoLoRA paper** (arXiv:2603.15965) claims "Qwen3-1.7B + 4 adapters > 8B" using per-token routing. The critical difference is that their evaluation likely includes mixed-domain or instruction-following data where tokens within a single sequence genuinely benefit from different experts. Our experiment used cleanly separated domains, which is the wrong test for per-token routing's value proposition.

## Gumbel-Sigmoid vs Softmax (Controlled Evidence)

L2R provides the only controlled comparison in our source library. Using the same L2R-wavg architecture:

| Benchmark | Gumbel-sigmoid | Softmax | Delta |
|-----------|---------------|---------|-------|
| MTL5 | 78.0 | 77.8 | +0.2 |
| WOS | 79.90 | 60.74 | +19.2 |
| AfriSenti | 60.04 | 59.90 | +0.1 |

**Mechanism**: Softmax forces destructive competition -- it often incorrectly concentrates all activation on a single adapter. Gumbel-sigmoid uses independent Bernoulli distributions, allowing simultaneous multi-adapter blending that respects task hierarchies. Our experiment's diversity metric of 2.42 confirms this non-degenerate blending is working on MLX.

## Alternative Approaches (What We Could Try Instead)

### 1. Hypernetwork-Generated Adapters (Skip Routing Entirely)
- **SHINE** (arXiv:2602.06358): Generates LoRA weights from context in a single forward pass via Memory-to-Parameter Transformer. Eliminates the routing problem entirely by dynamically creating task-specific adapters on the fly.
- **Text-to-LoRA** (Charakorn et al.): MLP maps task descriptions directly to LoRA weights. Relevant to our project's text-conditioned composition goal.

### 2. Entropy-Based Confidence Gating (Our exp_entropy_gated_experts)
Our own entropy gating experiment showed 63% of tokens can be skipped entirely when the base model is confident. Combining this with per-token routing would mean: skip routing for confident tokens, only route the uncertain 37%. This is the natural next step. **Caution**: models exhibit overconfidence on OOD data, so entropy thresholds need calibration.

### 3. Token Clustering for Batch Routing
**Routing Transformers** (TACL 2020) and **ClusterFormer** use learnable sparsity to cluster tokens before routing. Applied to LoRA composition: cluster tokens by hidden-state similarity, then route clusters (not individual tokens) to experts. This gives sub-sequence granularity without per-token overhead.

### 4. Retrieval-Based Fusion
**LoraHub** (arXiv:2307.13269): Embeds queries to retrieve top-k similar adapters from a library, fuses via linear interpolation. No router training needed -- purely similarity-based. Could complement our Grassmannian framework by using subspace distance as the similarity metric.

### 5. Quality-Diversity Search for Adapter Combinations
**LLMatic** (arXiv:2506.00823): Uses QD algorithms to search for optimal network architectures. Could be adapted to search the combinatorial space of adapter compositions, finding Pareto-optimal blends for different domain mixtures.

## Implications for Next Experiments

### What changes:
1. **Mixed-domain evaluation is mandatory** for per-token routing claims. Clean domain separation is the wrong test. Future experiments need sequences with genuine intra-sequence heterogeneity (code+comments, legal+math formulas, medical+drug dosages).
2. **Token-level supervision is needed** to train per-token routing. Per-domain labels teach per-sequence patterns. Consider: per-token adapter loss (train on which adapter produces lowest loss at each position), or unsupervised clustering of hidden states.
3. **Evaluation architecture needs fixing**. The per-group forward pass approximation makes small PPL differences uninterpretable. A single-pass architecture with per-layer adapter switching is needed for rigorous per-token evaluation.

### What stays:
1. **Gumbel-sigmoid routing is validated** for MLX. Use it as the default gate type going forward, not softmax.
2. **0.58% overhead** confirms per-token routing is feasible on Apple Silicon at N=5 experts.
3. **164K router params** are sufficient. No need for larger routers.

### Strategic recommendation:
The highest-value next experiment is **entropy-gated per-token routing on mixed-domain data**: combine exp_entropy_gated_experts' confidence skipping with the Gumbel-sigmoid router, evaluated on sequences that contain genuine domain mixing. This tests per-token routing where it should actually help, with the entropy gate reducing overhead on the ~63% of tokens that don't need expert composition.
