# Soft MoE (Puigcerver et al., 2024)

**Source:** https://arxiv.org/abs/2308.00951 (Google, 2024)

**Key Insight:** Replaces discrete token-to-expert assignment with soft
combinations — each expert receives a weighted mixture of all tokens,
eliminating routing collapse, load balancing problems, and token dropping.
No top-k selection needed.

**Relevance to our work:**
- Alternative to our softmax+top-k routing paradigm
- Soft assignment could be better for composition of independently-trained
  experts since it avoids the hard selection that causes k=1 catastrophe
- Trade-off: no sparsity savings (all experts process all tokens), so
  inference cost is higher but composition quality may be better
- Worth testing for `exp5_macro_match` as an alternative routing strategy

**What to use:**
- The soft assignment mechanism
- Their comparison against standard top-k routing
- Their analysis of load balancing properties
