# Arcee AI Trinity Large: 400B MoE

**Source:** https://arxiv.org/abs/2602.17004
**GitHub:** https://github.com/arcee-ai/trinity-large-tech-report

**Key Insight:** 400B MoE (13B active) combining MoE + GQA + Sliding Window
Attention. Uses alternating local:global attention at 3:1 ratio with 4096-token
sliding windows. Depth-scaled RMSNorm initialization for training stability.

## Novel Techniques
- **Sliding Window Attention (SWA)**: local attention with 4096-token window,
  cheaper than full attention for long sequences
- **Depth-scaled RMSNorm**: initialization varies by layer depth for stability
- **QK-Norm + Gated Attention**: similar to Qwen3.5's approach

## Relevance to our work
- The 3:1 local:global attention ratio mirrors Qwen3.5/Qwen3-Coder-Next
- Sliding window attention could interact with capsule composition differently
  than full attention (our micro finding: shared attention is the composition
  bottleneck, Exp 4)
- Relevant to `exp5_macro_match` architectural decisions
