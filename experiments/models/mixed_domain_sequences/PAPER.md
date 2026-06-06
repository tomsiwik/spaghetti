# Mixed-Domain Sequences: Research Digest

## Hypothesis

Per-token routing will outperform per-sequence routing by >5% on mixed-domain
sequences where different tokens genuinely belong to different domains, because
per-token routing can assign the appropriate expert to each segment while
per-sequence routing must compromise.

**Result: KILLED. Both kill criteria fail.**

## What This Experiment Does

Creates synthetic mixed-domain sequences by concatenating segments from different
domains (e.g., 128 Python tokens + 128 math tokens in one sequence). Compares
four routing strategies on these sequences:

1. **Uniform 1/N**: Equal-weighted blend of all 5 adapters
2. **Per-sequence**: Mean-pool hidden states, router selects top-2 for entire sequence
3. **Per-token**: Router selects top-2 experts independently at each position
4. **Oracle**: Uses perfect domain knowledge (domain-A adapter for first half, domain-B for second)

Setup: BitNet-2B-4T base + 5 LoRA adapters (python, math, medical, legal, creative),
Gumbel-sigmoid router (164K params), 10 domain pairs x 20 sequences each.

## Key References

- MoLoRA (arXiv:2603.15965): per-token routing, Qwen3-1.7B+4 adapters > 8B
- L2R: Gumbel-sigmoid validated
- Mod-Squad (Chen et al., NeurIPS 2023): task-level routing on homogeneous domains
- exp_molora_per_token_mlx: null result on homogeneous domains (-0.46%)

## Empirical Results

### Overall (averaged across 10 domain pairs, 200 sequences)

| Condition | Avg PPL | vs Per-Seq |
|-----------|---------|------------|
| Uniform 1/N | 9.08 | -- |
| Per-sequence top-2 | 8.61 | baseline |
| **Per-token top-2** | **8.59** | **+0.28%** |
| Oracle | 7.03 | +18.4% |

**K1 FAIL**: Per-token improvement is only +0.28% (threshold: >=5%).

### Per-Pair Breakdown

| Pair | Uniform | PerSeq | PerToken | Oracle | PT vs PS |
|------|---------|--------|----------|--------|----------|
| python+math | 4.47 | 3.95 | 4.21 | 3.36 | **-6.4%** |
| python+medical | 4.79 | 4.54 | 4.55 | 4.00 | -0.0% |
| python+legal | 11.52 | 10.90 | 10.83 | 9.11 | +0.6% |
| python+creative | 5.50 | 5.16 | 5.13 | 4.58 | +0.6% |
| math+medical | 7.01 | 5.62 | 5.64 | 5.24 | -0.2% |
| math+legal | 13.59 | 11.60 | 11.64 | 9.71 | -0.3% |
| math+creative | 7.02 | 6.17 | 6.17 | 5.25 | -0.0% |
| medical+legal | 16.90 | 17.48 | 17.37 | 13.00 | +0.6% |
| medical+creative | 7.91 | 8.27 | 8.10 | 6.28 | +2.0% |
| legal+creative | 12.09 | 12.45 | 12.27 | 9.73 | +1.4% |

Notable: python+math is the ONE pair where per-token routing succeeds at boundary
detection (97.17%) -- yet it is actually **worse** than per-sequence (-6.4%).
This is because per-token routing introduces noise: it must run separate forward
passes per expert group, and cross-attention from the wrong-adapter segment
contaminates all positions.

### Boundary Detection (K2)

| Pair | Accuracy | Domain A | Domain B |
|------|----------|----------|----------|
| python+math | **97.17%** | 95.98% | 98.36% |
| python+medical | 50.00% | 100.00% | 0.00% |
| python+legal | 50.00% | 100.00% | 0.00% |
| python+creative | 49.96% | 99.92% | 0.00% |
| math+medical | 50.00% | 100.00% | 0.00% |
| math+legal | 49.94% | 99.88% | 0.00% |
| math+creative | 50.00% | 100.00% | 0.00% |
| medical+legal | 0.07% | 0.15% | 0.00% |
| medical+creative | 0.00% | 0.00% | 0.00% |
| legal+creative | 0.00% | 0.00% | 0.00% |

Average: 39.71%. **K2 FAIL** (threshold: >40%).

**Critical finding**: The router has collapsed into a hierarchical detector:
- Python and math are well-separated in hidden space (near-perfect detection)
- Medical, legal, creative are indistinguishable to the router (0% detection)
- When python or math appears as domain A, the router "detects" it (100% accuracy
  on A) but defaults to the same expert for domain B (0% accuracy on B)

## Why Per-Token Routing Fails Here

Three compounding factors:

1. **Evaluation methodology bias**: Per-token routing requires separate forward passes
   per expert group. Each forward pass processes the FULL sequence with the same adapter,
   but only scores the tokens belonging to that group. Tokens in the "wrong" segment
   still influence the hidden states of the "right" segment via self-attention.
   This contaminates the per-token advantage.

2. **Router capacity insufficient for fine-grained domain discrimination**: The 2-layer
   MLP (2560->64->5) learns to separate python and math (structurally distinct hidden
   representations: code tokens vs mathematical notation) but cannot distinguish
   medical, legal, and creative text (all "natural prose" with similar hidden distributions).

3. **Oracle gap is real but unreachable**: The oracle (22.6% better than uniform) proves
   that correct per-segment expert assignment has substantial value. But capturing this
   value requires EITHER a much stronger router OR a fundamentally different architecture
   that avoids the cross-attention contamination problem.

## Limitations

- Synthetic mixed-domain sequences (concatenation, not natural mixing) create
  artificially sharp boundaries. Natural mixed-domain text (e.g., "explain this
  medical procedure in Python pseudocode") would have gradual transitions.
- Only 5 domains, 2 of which (python, math) are structurally very distinctive.
  The 3 "prose" domains are hard to separate at the hidden-state level.
- The evaluation methodology (full forward pass per group, score only group tokens)
  is an approximation that penalizes per-token routing via cross-attention contamination.
- Router trained for only 800 steps. More training or a different architecture
  (e.g., contrastive pre-training on domain embeddings) might help with the
  collapse problem.

## What Would Kill This

This hypothesis IS killed. Both K1 and K2 fail.

For per-token routing to be revived, it would need:
- A router architecture that can distinguish all domains (not just code vs math)
- An evaluation method that avoids cross-attention contamination (e.g., segment-level
  composition rather than full-sequence forward passes)
- Alternatively: a fundamentally different approach like "segment-level routing"
  that detects domain boundaries first, then applies per-segment composition

## Implications for the Project

1. **Per-token routing does NOT provide meaningful advantage over per-sequence routing**,
   even on the data regime specifically designed to show its value. This is now a
   two-experiment kill: null on homogeneous domains (-0.46%), null on mixed domains (+0.28%).

2. **The oracle gap (22.6%) proves composition value exists**, but it requires correct
   segment-level domain knowledge, not token-level routing from hidden states.

3. **Router collapse is the real problem**: 3/5 domains are indistinguishable in hidden
   space. Better routing requires either (a) domain-specific training signals
   (contrastive learning on domain embeddings), (b) external domain classifiers,
   or (c) a fundamentally different approach like entropy gating (which sidesteps
   domain detection entirely by measuring base model confidence).

4. **Entropy gating remains the best routing approach**: It achieved 63% skip rate
   at 1.13% PPL cost without needing to identify domains at all. For the remaining
   37% of tokens, per-adapter routing heads (100% accuracy) are the proven approach.
