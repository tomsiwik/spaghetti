# MoLoRA Per-Token Routing: Research Digest

## Hypothesis
Per-token adapter routing with independent Gumbel-sigmoid gates enables finer-grained expert composition than per-sequence routing, improving PPL on mixed-domain text while maintaining low overhead on Apple Silicon.

## What This Model Is
A per-token routing mechanism for composing domain-specific LoRA adapters. Each token independently selects its top-2 experts via Gumbel-sigmoid non-competing gates (L2R-style), as opposed to traditional softmax top-k which forces experts to compete. Tokens are grouped by expert set for efficient pre-merge, avoiding the memory wall of per-token adapter loading.

The router is a 164K-parameter 2-layer MLP (d=2560 -> 64 -> 5) trained with binary cross-entropy on independent expert gates. At inference, sigmoid gates are used without Gumbel noise, and top-2 experts are selected per token with score-proportional weighting.

## Key References
- MoLoRA (arxiv 2603.15965): per-token LoRA routing, 1.7B+4 adapters beats 8B
- L2R: Gumbel-sigmoid non-competing routing, +0.1-19.2pts over softmax
- exp_tiny_routing_heads: per-adapter binary routing heads (baseline comparison)
- exp_bitnet_per_token_routing: per-sequence centralized router (different architecture, 15 domains, softmax -- NOT directly comparable to this experiment)

## Empirical Results

### PPL Comparison (5 domains, BitNet-2B-4T, rank-16 LoRA)

| Method | Python | Math | Medical | Legal | Creative | **Avg** |
|--------|--------|------|---------|-------|----------|---------|
| Uniform 1/N | 2.52 | 4.94 | 6.20 | 20.44 | 5.96 | **8.01** |
| Per-seq top-2 (Gumbel-sigmoid) | 2.22 | 3.60 | 6.53 | 20.71 | 4.93 | **7.60** |
| Per-token top-2 (Gumbel-sigmoid) | 2.22 | 3.61 | 6.38 | 20.85 | 5.11 | **7.63** |
| Oracle (individual) | 2.22 | 3.60 | 4.75 | 16.56 | 4.92 | **6.41** |

### Primary Finding: Informative Null Result

**Per-token routing does not improve over per-sequence routing on cleanly separated domains.** The average PPL difference is -0.46% (per-token 7.63 vs per-sequence 7.60, with per-token slightly worse). This is the honest, fair comparison -- both use the same Gumbel-sigmoid gates, same 5 domains, same router architecture, same training procedure.

This null result is expected and informative:
1. **Domain homogeneity**: Python code is uniformly "python", legal text is uniformly "legal". There are no mixed-domain sequences in the evaluation data. Per-token routing adds value only when tokens within a single sequence benefit from different experts.
2. **Router training signal**: The router is trained with per-domain labels (all tokens in a python text labeled "python"). This teaches per-sequence patterns, not per-token patterns. True per-token supervision would require token-level domain annotations.
3. **Medical improvement**: The one domain where per-token routing helps (+2.35%) is medical, which plausibly contains mixed content (disease names, drug dosages, clinical procedures spanning multiple knowledge domains).

### Revised Kill Criteria

| Criterion | Definition | Metric | Threshold | Result |
|-----------|------------|--------|-----------|--------|
| K1 (REVISED) | Per-token PPL <= per-sequence PPL on same 5 domains | 7.63 vs 7.60 | per-token <= per-seq | **NULL RESULT** (7.63 > 7.60, -0.46%) |
| K2 | Router overhead | 0.58% | < 10% | **PASS** |
| K3 | Expert diversity | 2.42 | > 1.0 | **PASS** |

**K1 note**: The original K1 compared against PPL 13.65 from exp_bitnet_per_token_routing, which is an invalid comparator. That experiment used 15 domains (including hard domains like physics at PPL 73.7), a different router architecture (256-dim hidden), different training (2000 steps), and softmax gating. The comparison is confounded by at least 4 variables and cannot isolate the effect of per-token routing. The revised K1 compares per-token vs per-sequence routing within this experiment, which is the only fair comparison available.

### Success Criteria

| Criterion | Result | Status |
|-----------|--------|--------|
| S1: Per-token PPL < per-sequence PPL | 7.63 > 7.60 | **FAIL** (null result) |
| S2: Diversity > 2.0 | 2.42 | **PASS** |
| S3: Medical/math improve most | Medical +2.35% | **PASS** |

### Routing Diversity Per Domain

| Domain | Avg distinct expert sets per sequence |
|--------|--------------------------------------|
| Python | 1.68 |
| Math | 2.72 |
| Medical | 2.96 |
| Legal | 2.80 |
| Creative | 1.92 |

Medical (2.96) and legal (2.80) domains have the highest token-level routing diversity, suggesting these domains benefit most from varying expert composition within a sequence. Python (1.68) has the lowest, consistent with its highly homogeneous content.

### Timing

| Operation | Time | Overhead |
|-----------|------|----------|
| Base forward | 36.65ms | -- |
| Router inference | 0.21ms | 0.58% |

## Evaluation Methodology Limitation

The per-token routing evaluation has a known approximation flaw. For each token group (tokens sharing the same expert pair), a **separate full forward pass** is run through the model with that group's merged adapter, and loss is extracted only at that group's token positions. However, tokens at positions 0..t-1 may belong to a DIFFERENT group with different adapters. This means:

- When computing loss for group B (e.g., tokens 51-100), the hidden states at those positions are conditioned on group B's adapter applied at positions 0-50, even though positions 0-50 actually belong to group A with different adapter weights.
- The per-sequence routing baseline does not have this problem because one adapter is used for the entire sequence.

**Implication**: The per-token PPL of 7.63 is an approximation. It can be interpreted as an upper bound on the quality achievable with per-token routing under this grouping scheme, but it is not directly comparable to per-sequence PPL at the precision needed to detect small differences. The -0.46% gap is within the range that could be explained by this evaluation artifact.

A correct evaluation would require either (a) a single forward pass with per-layer adapter switching, or (b) a causal-consistent evaluation where each token position uses the adapter that was actually assigned to it throughout the sequence prefix.

## What the Gumbel-Sigmoid Router Proved

The Gumbel-sigmoid non-competing gate mechanism works on MLX with negligible overhead:
1. **Independent binary gates** allow natural multi-adapter blending without mode collapse
2. **164K router params** suffice for effective domain routing
3. **0.58% overhead** makes per-token routing feasible on Apple Silicon

**Regarding Gumbel-sigmoid vs softmax**: exp_bitnet_per_token_routing (softmax, 15 domains) achieved PPL 13.65, while this experiment (Gumbel-sigmoid, 5 domains) achieved PPL 7.60-7.63. However, this comparison is **confounded** by differences in domain count, domain difficulty, router architecture, and training procedure. We cannot attribute the gap to gate type alone. A controlled ablation (Gumbel-sigmoid vs softmax on the same 5 domains with the same router) was not performed and would be needed to make any causal claim.

## Limitations

1. **Toy domains**: 5 trivially separable domains do not stress-test per-token routing. Mixed-domain text is needed to show clear benefit.
2. **Per-domain supervision**: Training with domain-level labels teaches per-sequence patterns. Token-level training signals (e.g., per-token adapter loss) would be better.
3. **Evaluation methodology approximation**: Per-token loss is computed by running the full sequence through each token group's merged adapter. Tokens at earlier positions may belong to a different group, creating a mismatch between the adapter used for context and the adapter assigned to those positions. This makes per-token PPL an approximation (likely an upper bound on true per-token routing quality). See "Evaluation Methodology Limitation" section above.
4. **No entropy pre-filter**: This experiment does not integrate the entropy gating from exp_entropy_gated_experts, which could skip 63% of tokens entirely.
5. **Pre-merge weight approximation**: The implementation uses first-token weights as the representative for each token group rather than averaging all token weights within the group. See MATH.md for details.

## What Would Kill This

- **At scale**: Per-token routing fails to beat per-sequence routing even on mixed-domain evaluation data (e.g., code+comments, legal+math)
- **Memory wall**: Loading N adapters per sequence causes memory thrashing on edge devices (not observed at N=5, would appear at N=50+)
- **Router collapse at scale**: With many more experts, the router may fail to learn meaningful per-token routing patterns
- **Production overhead**: The current implementation requires multiple forward passes per sequence (one per token group). A single-pass architecture with per-layer routing would be needed for production.
