# Dynamic Weight Composition at Macro Scale: Research Digest

## Hypothesis

Per-query expert weighting beats equal-weight pre-merge (the current SOLE default)
at macro scale, achieving >= 2% quality improvement with < 50ms latency overhead.

## What This Experiment Is

Equal-weight pre-merge of N=50 LoRA experts dilutes individual expert quality by
127% at N=5 (pilot50 composition quality, SUPPORTED). The dilution is expected
to worsen dramatically at N=50 (SNR = 1/49 from MATH.md Section 3.1).

This experiment tests whether smarter weighting -- using domain embeddings, cached
PPL scores, or hybrid approaches -- can restore individual expert quality while
keeping latency within interactive bounds.

Five strategies are compared:

1. **equal_premerge** (A): Current SOLE default. Weight = 1/N. Zero overhead.
2. **embed_topk** (B2): Select top-5 experts by embedding cosine. ~0.1ms overhead.
3. **embed_weighted** (C2): Softmax cosine scores for all N experts. ~0.1ms overhead.
4. **ppl_precomputed** (D): Cache expert-domain quality matrix. Use domain-specific
   weights. Zero per-query overhead (domain classification only).
5. **hybrid_k3** (C3): Embed top-3 filter + PPL rerank. ~90ms overhead.

## Key References

- **LoRA-Flow** (Wang et al., 2024): Per-token per-layer gates. 0.26M gate params
  for k=2. Outperforms static merging on generative tasks. Gate scales O(k*d*L),
  infeasible at k>=100.
- **LoRA Soups / CAT** (Prabhakar et al., COLING 2025): Per-layer learned scalar
  weights. +12% over data mixing on binary composition. Limited to k=2.
- **X-LoRA** (Buehler & Buehler, 2024): Hidden-state gating for dynamic mixing.
  Requires training the gating network.
- **LoRAHub** (Huang et al., 2023): Gradient-free few-shot composition. No
  additional parameters or gradients needed.
- **TIES-Merging** (Yadav et al., NeurIPS 2023): Sign conflict resolution before
  merging. Reduces interference in weight averaging.
- **DARE** (Yu et al., 2023): Random drop + rescale of delta parameters.
  Exploits parameter redundancy.

## Prior Results (micro, from exp_cross_domain_dilution_vs_k)

| Strategy | Mean Gap | Oracle r | Production? |
|----------|----------|----------|-------------|
| equal_weight | -0.6% | -- | Yes |
| PPL-probe weighted | -9.94% | 0.990 | Yes |
| loss_weighted (oracle) | -10.06% | 1.000 | No |
| activation_weighted | -0.71% | 0.023 | Yes |
| top-1 oracle | -5.85% | -- | No |

PPL-probe weighting achieves r=0.990 correlation with oracle and +9.34pp
improvement over equal weight at micro scale (d=32, r=4, K=2).

## Empirical Results

TO BE FILLED from `results/dynamic_weight_composition/results.json`

### Base Model and Single-Expert PPL

| Domain | Base PPL | Single-Expert PPL |
|--------|----------|-------------------|
| TBD | TBD | TBD |

### Strategy Comparison at N=50

| Strategy | Mean PPL | Mean Deg. vs Single | Domains > Base | Latency (ms) |
|----------|----------|---------------------|----------------|--------------|
| equal_premerge | TBD | TBD | TBD | 0.0 |
| embed_topk_k5 | TBD | TBD | TBD | TBD |
| embed_weighted | TBD | TBD | TBD | TBD |
| ppl_precomputed | TBD | TBD | TBD | TBD |
| hybrid_k3 | TBD | TBD | TBD | TBD |

### Kill Criteria

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| K1: best dynamic >= 2% improvement over equal | >= 2% | TBD | TBD |
| K2: winning strategy < 50ms latency | < 50ms | TBD | TBD |
| K3: equal-weight NOT Pareto optimal | dominated | TBD | TBD |

## Limitations

1. **4-bit quantization** may mask or amplify composition effects. Both base and
   composed models use the same quantization, so the comparison is fair within
   this regime, but absolute PPL values may differ from fp16.

2. **Embedding-based routing was not validated at micro.** Micro experiments
   tested activation magnitude and logit difference (both failed with r<0.25).
   Embedding cosine was not tested. It may or may not work better at d=3584.

3. **PEFT per-adapter weighting is scalar, not per-layer.** LoRA-Flow and CAT
   both use per-layer weights. PEFT's `add_weighted_adapter` applies one scalar
   per adapter across all layers. This may underperform per-layer weighting
   for experts that are useful in some layers but not others.

4. **Domain classification as proxy for query relevance.** The ppl_precomputed
   strategy assumes queries can be classified into domains. Ambiguous or
   cross-domain queries may not match any single domain well.

5. **N=50 only.** Does not test scaling to N=500 or N=5000. At larger N,
   embedding-based selection may degrade (more similar centroids).

## What Would Kill This

**K1 KILL (improvement < 2%):** If the best dynamic strategy improves mean PPL
by less than 2% over equal-weight, the complexity is not justified. This would
imply that either (a) equal-weight dilution is not severe at macro scale
(contradicting the N=5 result), or (b) the routing signals tested here are
insufficient. In either case, equal-weight pre-merge remains the SOLE default.

**K2 KILL (latency > 50ms):** If the only strategy that achieves >= 2%
improvement has > 50ms latency, dynamic weighting is too slow for interactive
use. This would redirect toward offline composition (query-class-aware
pre-merge) rather than per-query routing.

**K3 KILL (Pareto optimal equal-weight):** If equal-weight pre-merge lies on
the Pareto frontier (no strategy is simultaneously better in quality AND
acceptable in latency), then the current SOLE default is confirmed optimal.
