# Spring 2026 Architecture Trends (Rasbt Overview)

**Source:** https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight
**Tweet:** https://x.com/rasbt/status/2026659971467706603

**Key Insight:** 10 open-weight LLM releases in Jan-Feb 2026 show a clear trend:
architectures moving from classic GQA → MLA and linear attention hybrids.
Efficiency (lower latency, longer context) is the dominant driver.

## The 10 Models

| Model | Params | Active | Key Innovation |
|-------|--------|--------|----------------|
| Arcee Trinity | 400B | 13B | MoE + GQA + SWA |
| Kimi K2.5 | 1T | ? | DeepSeek-like MoE, multimodal |
| Step 3.5 Flash | 196B | 11B | MTP-3 (multi-token prediction) |
| **Qwen3-Coder-Next** | **80B** | **3B** | **512 experts, GatedDeltaNet hybrid, agentic code training** |
| GLM-5 | 744B | 40B | MLA + DeepSeek Sparse Attention |
| MiniMax M2.5 | 230B | 230B | Classic GQA (deliberately simple) |
| Nanbeige 4.1 | 3B | 3B | Llama-style, no weight tying |
| Qwen3.5 | 397B | 17B | GatedDeltaNet hybrid in main line |
| Ling 2.5 | 1T | ? | Lightning Attention + MLA |
| Tiny Aya | 3.35B | 3.35B | Parallel transformer blocks |

## Architectural Trends Relevant to Our Work

1. **Hybrid attention is mainstream**: GatedDeltaNet + standard attention at 3:1
   ratio appears in Qwen3.5, Qwen3-Coder-Next, and influenced others
2. **Fine-grained MoE**: 256-512 small experts (DeepSeek, Qwen) vs 8 large
   experts (Mixtral) — aligns with our capsule approach
3. **MLA replacing GQA**: Multi-head Latent Attention (DeepSeek origin) reduces
   KV cache at inference — relevant for expert composition inference
4. **MTP for free speedup**: Multi-token prediction as a training+inference technique
5. **Shared expert pattern**: Qwen3-Next uses 1 always-on shared expert alongside
   512 routed ones — directly parallels our frozen base model concept

## Implication for exp5_macro_match
The state of the art has moved to extremely sparse MoE (3-5% activation ratio)
with hybrid attention. Our composition protocol should target these architectures,
not just vanilla transformers. The base model (Qwen2.5-Coder-0.5B) is already
in this family.
