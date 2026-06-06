# Mixtral of Experts

**Source:** https://arxiv.org/abs/2401.04088 (Mistral AI, January 2024)
Full analysis in: `macro/mixtral-8x7b/PAPER.md`

**Key Insight:** First open-weight production MoE. 8 experts per layer, top-2
routing. 12.9B active / 46.7B total params. Matches Llama 2 70B quality at
~6x less active compute.

**Relevance to our work:**
- Validates top-2 routing at production scale (consistent with our micro
  finding that k=2 is the minimum viable sparsity)
- Their expert offloading approach is what VISION.md describes for inference:
  base model active, experts loaded on demand
- Architecture is the closest production analog to our composition protocol

**What to use:**
- Their top-2 routing implementation details
- Expert offloading and caching strategy
- Benchmark comparison methodology
