# Mixture-of-Depths: Dynamically Allocating Compute in Transformers

**Source:** https://arxiv.org/abs/2404.02258 (Raposo et al., 2024)

**Key Insight:** Instead of processing every token through every layer, MoD uses
a learned router to decide WHICH tokens skip a layer entirely. This is adaptive
compute at the layer level (depth), not within-layer expert selection.

**Relevance to our work:**
- Closest prior art to entropy-adaptive routing, but operates at different granularity
- MoD: skip entire layers for easy tokens (depth-wise sparsity)
- Our approach: reduce number of experts for easy tokens (width-wise sparsity)
- MoD uses top-k routing to select which tokens participate, with a fixed budget
- We use entropy threshold to decide per-token k (variable budget)
- Complementary: could combine MoD (skip layers) with entropy-adaptive (fewer experts)

**What to use:**
- The token-level confidence concept (some tokens need less compute)
- Their routing mechanism design
- Their compute budget enforcement strategy

**Difference from our approach:**
- MoD: binary skip/process per layer, fixed total compute budget
- Ours: variable k per token within a layer, no fixed budget constraint
