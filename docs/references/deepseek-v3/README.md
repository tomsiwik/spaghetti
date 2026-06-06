# DeepSeek-V3 Technical Report

**Source:** https://arxiv.org/abs/2412.19437 (December 2024)
Full analysis in: `macro/deepseek-v3/PAPER.md`

**Key Insight:** Fine-grained MoE with 256 experts per layer, top-8 routing.
Auxiliary-loss-free load balancing via per-expert bias terms. 671B total / 37B
active params. FP8 mixed-precision training on 2048 H800 GPUs.

**Relevance to our work:**
- Their auxiliary-loss-free balancing is directly relevant — our router
  calibration uses balance loss, but their approach avoids it entirely
- Fine-grained experts (256 small vs 8 large) aligns with our capsule approach
  (many small capsule groups vs few large experts)
- Their MLA (Multi-head Latent Attention) reduces KV cache — relevant for
  inference efficiency at macro scale
- Most relevant to `exp5_macro_match` architecture decisions

**What to use:**
- Auxiliary-loss-free load balancing (per-expert bias)
- Fine-grained expert sizing analysis
- Their training infrastructure choices for reference
