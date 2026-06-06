# Qwen3-Coder-Next: Hybrid Attention MoE for Code

**Source:** https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
**Tech Report:** https://github.com/QwenLM/Qwen3-Coder/blob/main/qwen3_coder_next_tech_report.pdf
**Blog:** https://qwen.ai/blog?id=qwen3-coder-next

**Key Insight:** 80B params / 3B active — extreme sparsity MoE with 512 routed
experts + 1 shared expert, top-10 routing. Same hybrid attention as Qwen3.5
(GatedDeltaNet + Gated Attention at 3:1 ratio) but with code-specific agentic
training. Achieves SWE-Bench-Pro comparable to models with 10-20x more active params.

## Architecture

| Component | Detail |
|-----------|--------|
| Total params | 80B |
| Active params | 3B (3.75% activation ratio) |
| Layers | 48 (36 DeltaNet + 12 Gated Attention) |
| Hidden dim | 2048 |
| MoE experts | 512 routed + 1 shared |
| Top-k | 10 |
| Expert intermediate dim | 512 |
| GQA heads | 16 Q / 2 KV (Gated Attention layers) |
| Head dim | 256 (attention) / 128 (DeltaNet) |
| Context | 262K native, 1M with YaRN |

## Novel Techniques
1. **Multi-Token Prediction (MTP)**: predicts multiple next tokens simultaneously,
   boosts training quality AND inference speed (speculative decoding)
2. **Agentic training**: trained on large-scale executable task synthesis with
   environment interaction and RL — not just code completion but code EXECUTION
3. **Zero-centered layernorm + weight-decayed layernorm**: stability techniques
   for robust pre-training at extreme sparsity
4. **512 experts with top-10**: much finer granularity than Mixtral (8 experts, top-2)
   or even DeepSeek-V3 (256 experts, top-8)

## Relevance to our work

**Directly relevant to `exp5_macro_match`:**
- This IS the state of the art for sparse MoE coding models
- Our composition protocol aims to match 1.5B monolithic at 1/3 active params;
  Qwen3-Coder-Next achieves comparable goals at massive scale (80B/3B)
- Their 512 experts with top-10 routing validates fine-grained expert composition
- The agentic training approach is relevant for functional evaluation

**Architecture lessons:**
- 512 experts × 512 intermediate dim = very small per-expert MLPs (our capsule
  groups are conceptually similar: many small adapters vs few large experts)
- The shared expert (always active) maps to our frozen base model concept
- Top-10 out of 512 (~2% activation) vs our top-2 out of N (~25-50% at small N)
- GatedDeltaNet for linear attention layers — O(n) complexity, relevant for
  long-context composition scenarios

**What to study:**
- The expert granularity choice: 512 × 512dim vs fewer × larger
- Their routing stability at 512 experts (does it collapse?)
- MTP interaction with MoE routing (do experts specialize differently under MTP?)
- How the shared expert interacts with routed experts (parallels to our base model)
