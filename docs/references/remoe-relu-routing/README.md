# ReMoE: ReLU-Based Self-Routing for MoE

**Source:** https://arxiv.org/abs/2408.14711 (ICLR 2025)

**Key Insight:** Replaces softmax+top-k routing with ReLU activation on routing
logits. This enables dynamic per-token expert allocation — each token activates
a variable number of experts based on ReLU output magnitude, not a fixed k.
Outperforms standard top-k MoE at equal compute.

**Relevance to our work:**
- Directly related to `exp5_macro_match` — ReMoE routing could replace our
  softmax router for macro-scale composition
- Our `relu_router` micro experiment was inspired by this (see IDEA-RELU-ROUTER.md)
- Key difference: ReMoE trains routing end-to-end; we compose independently-trained
  experts. The question is whether ReLU routing works for post-hoc composition.

**What to use:**
- The routing mechanism itself (ReLU on routing logits)
- Their load balancing approach (capacity factor without auxiliary loss)
- Their ablation on routing collapse prevention
