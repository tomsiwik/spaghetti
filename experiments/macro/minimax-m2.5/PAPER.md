# MiniMax-M2.5 Research Digest

**Model:** MiniMax-M2.5 (and M2.5-Lightning)
**Organization:** MiniMax (Shanghai, China)
**Released:** February 12, 2026
**Arena tier:** King
**API access:** Together AI, NVIDIA NIM, Lambda, SiliconFlow

---

## Overview

MiniMax-M2.5 is the third generation of MiniMax's M2 agentic model line, released approximately one month after the company's Hong Kong IPO. It is a 229B-parameter sparse Mixture-of-Experts model that activates only ~10B parameters per token. The headline result is 80.2% on SWE-Bench Verified — within 0.6 points of Claude Opus 4.6 (80.8%) — at roughly 1/20th the API cost. M2.5 is also fully open-weight (Modified-MIT license), available on Hugging Face.

The model is the latest in MiniMax's public model lineage:
- MiniMax-Text-01 / MiniMax-01 (Jan 2025) — 456B total / 45.9B active, 32 experts, 1M token context
- MiniMax-M2 (Oct 2025) — 230B MoE, coding/agentic focus, interleaved thinking
- MiniMax-M2.1 (Jan 2026) — post-training refinement of M2
- MiniMax-M2.5 (Feb 2026) — current generation, extended RL training

---

## Architecture

### Core Design

MiniMax-M2.5 inherits the architectural foundations established in the MiniMax-01 technical report (arXiv:2501.08313) and extends them with refined post-training.

**Parameter budget:**
- Total parameters: 229B
- Active parameters per token: ~10B
- MoE routing: 256 total experts, 8 activated per token
- Architecture class: `MiniMaxM2ForCausalLM`

**Comparison to peers (active/total ratio):**

| Model | Total | Active | Ratio |
|---|---|---|---|
| MiniMax-M2.5 | 229B | 10B | 4.4% |
| DeepSeek-V3 | 671B | 37B | 5.5% |
| Qwen3-235B | 235B | 22B | 9.4% |
| MiniMax-Text-01 | 456B | 45.9B | 10.1% |

M2.5 achieves the lowest active/total ratio among production-scale MoEs, enabling high throughput at low compute cost.

### Attention Mechanism: Lightning Attention + Hybrid Softmax

The defining architectural feature across MiniMax's model family is **Lightning Attention**, a linear-complexity attention mechanism. Unlike softmax attention (O(n^2) in sequence length), Lightning Attention scales linearly with context length, enabling the 1M+ token context window at practical cost.

The hybrid design (documented in MiniMax-01) interleaves:
- Lightning (linear) attention layers — the majority of layers
- Periodic softmax attention layers — one every 7 linear layers, across 80 total transformer layers

This hybrid preserves the associative-memory expressiveness of full attention at key intervals while achieving near-linear scaling overall. The softmax layers handle tasks requiring precise global retrieval; lightning layers accumulate distributional representations efficiently.

**Positional encoding:** Rotary Position Embeddings (RoPE), applied to half the attention head dimension, base frequency 10,000,000 (enabling long-context extrapolation).

**Other architectural choices:**
- Normalization: RMSNorm
- Activation: SwiGLU (SiLU-gated)
- Hidden size: 6144
- Attention heads: 64, head dimension 128

### MoE Routing

- 256 expert FFN blocks per MoE layer
- Top-8 routing (8 of 256 experts selected per token)
- Routing is learned; specific algorithm (token-choice vs. expert-choice) not publicly disclosed for M2.5
- MoE stability during RL training handled by the CISPO algorithm (see Training section)

**Expert granularity:** With 256 experts and only 10B active params total, individual experts are fine-grained. This contrasts with DeepSeek-V3's coarser 37B/671B design (shared expert + 8 routed of 257). Finer experts allow more specialized routing but increase routing overhead and load-balancing difficulty.

### Context Window

- Training context: 1M tokens (inherited from MiniMax-01 architecture)
- API context: 200K tokens (NVIDIA NIM reports 204,800 tokens)
- Inference extrapolation: up to 4M tokens at additional cost

---

## Training

### Pre-Training

MiniMax has not released detailed pre-training data composition for the M2 family. The underlying base is shared with MiniMax-Text-01, which was trained to "match GPT-4o and Claude-3.5-Sonnet" on standard benchmarks. The base model supports 10+ coding languages: Python, Go, C, C++, TypeScript, Rust, Kotlin, Java, JavaScript, PHP, Lua, Dart, Ruby.

### Post-Training: Forge + CISPO

M2.5's key differentiator is large-scale RL post-training using MiniMax's in-house **Forge** framework, run for approximately two months of wall-clock time.

**CISPO (Clipped Importance Sampling Policy Optimization):**

First published in the MiniMax-M1 reasoning paper (arXiv:2506.13585), CISPO modifies the importance sampling clipping strategy used in PPO/GRPO:
- Standard PPO clips *token-level updates*, discarding low-probability tokens
- CISPO clips *importance sampling weights* instead, allowing all tokens — including low-probability ones — to contribute to gradient computation
- This preserves entropy and prevents mode collapse during long-horizon RL
- Reported 2x speedup over ByteDance's DAPO algorithm on equivalent tasks
- Critical for MoE stability: large sparse models are prone to expert collapse under standard RL

**Forge framework innovations:**

1. **Prefix-tree merging:** Multi-turn samples sharing common prefixes are merged into tree structures before batching. Equivalent mathematically to standard training but eliminates redundant forward passes across shared prefixes. Achieves ~40x training throughput improvement over naive sequential rollout.

2. **Windowed FIFO scheduling:** Asynchronous scheduling with a sliding window (approximately 30% of batch size) for local greedy ordering. Prevents distributional drift toward easier tasks while maintaining training stability.

3. **Process-level rewards:** Rewards signal intermediate behaviors throughout agent trajectories, not just final outcomes. Specific signals include: penalizing language mixing, penalizing spurious tool invocations, estimating real-world task completion time. This addresses the credit assignment problem endemic to long agent sequences (50+ steps).

**Training environments:** 200,000+ distinct real-world agent scaffolds, operating at up to 200K context, processing millions of samples per day.

**Unified multi-domain training:** Simultaneous training across Reasoning, General QA, and Agent domains to prevent negative transfer. The model is not separately fine-tuned per domain; the RL mix handles all three.

---

## Inference Variants

Two serving configurations are offered:

| Variant | Throughput | Input price | Output price |
|---|---|---|---|
| M2.5 (standard) | ~50 tokens/sec | $0.15/M tokens | $1.20/M tokens |
| M2.5-Lightning | ~100 tokens/sec | $0.30/M tokens | $2.40/M tokens |

The Lightning variant is identical in weights; the name refers to a higher-throughput serving configuration (likely speculative decoding or higher-batch scheduling). Both are ~2x faster than Claude Opus 4.6 in sustained throughput.

Hardware requirements: Deployable on 4x H100 GPUs (enabled by 10B active params at inference). Supported quantizations: F32, BF16, F8_E4M3. Supported runtimes: SGLang (recommended), vLLM, Transformers, KTransformers, MLX-LM.

---

## Benchmarks

### Agentic and Coding (Primary)

| Benchmark | M2.5 | M2.1 | Claude Opus 4.6 | Notes |
|---|---|---|---|---|
| SWE-Bench Verified | 80.2% | 77.4% | ~80% | Near parity with frontier |
| Multi-SWE-Bench | 51.3% | — | 50.3% | #1 on multilingual coding |
| BrowseComp | 76.3% | — | — | Web agent task |
| BFCL Multi-Turn | 76.8% | — | ~63% | +13pp over Opus |
| Terminal-Bench | 46.3% | — | — | (M2 baseline) |
| GAIA (text only) | 75.7% | — | — | (M2 baseline) |

**SWE-Bench efficiency:** M2.5 completes the SWE-Bench Verified suite in 22.8 minutes vs M2.1's 31.3 minutes (37% faster), using 3.52M tokens/task vs 3.72M. The efficiency gain is attributed to emergent planning behavior: the model learned to write specifications before implementation, reducing downstream errors and token waste.

### Academic Benchmarks

| Benchmark | M2.5 | M2.1 | Opus 4.6 | Gemini 3 Pro |
|---|---|---|---|---|
| AIME25 | 86.3 | 83.0 | 95.6 | 96.0 |
| GPQA-Diamond | 85.2 | 83.0 | 90.0 | 91.0 |
| SciCode | 44.4 | 41.0 | 52.0 | 56.0 |
| MMLU-Pro | ~82 | — | — | — |
| MMLU | ~89.7 | — | — | — |

On pure reasoning benchmarks (AIME25, GPQA), M2.5 trails Opus 4.6 and Gemini 3 Pro by 5-10 points. This is consistent with M2.5 being RL-tuned for agentic coding rather than mathematical reasoning.

### Office and Productivity

- GDPval-MM (Office productivity): 59.0% average win rate vs mainstream models
- Native Office Skills: Word, Excel, PowerPoint file generation and manipulation
- Claimed: 30% of MiniMax's internal tasks autonomously completed; 80% of new internal code generated by M2.5

---

## Comparison to Peers

### vs. DeepSeek-V3

| Dimension | MiniMax-M2.5 | DeepSeek-V3 |
|---|---|---|
| Total params | 229B | 671B |
| Active params | 10B | 37B |
| Context | 1M (train) / 200K (API) | 128K |
| SWE-Bench | 80.2% | ~49% |
| Input price | $0.15/M | ~$0.02/M |
| MoE design | 256 experts, top-8 | 1 shared + 256 routed, top-8 |
| Training focus | Agentic RL (Forge) | General pre-training + SFT |
| Open weights | Yes (Modified MIT) | Yes (MIT) |

DeepSeek-V3 is dramatically cheaper and stronger at general tasks. M2.5 leads substantially on agentic benchmarks due to specialized RL training. DeepSeek-V3's expert design uses a "shared expert" that always fires plus top-8 routed experts — a different trade-off emphasizing a stable generalist trunk.

### vs. MiniMax-Text-01 (predecessor)

Text-01 has 456B total / 45.9B active (much larger), 32 experts (much fewer), and 1M token training context. The step from Text-01 to M2.x represents a shift from a general long-context model to a purpose-built agentic model: smaller active footprint, more experts (finer granularity), and heavy RL post-training.

### vs. Qwen3-235B

Qwen3-235B-A22B is a close architectural peer: 235B total, 22B active, also open-weight. M2.5 activates 2x fewer parameters (10B vs 22B), enabling higher throughput, but Qwen3 likely performs better on general reasoning tasks.

---

## Why "King" Tier

The "king" tier designation reflects M2.5's position at the frontier of cost-efficient agentic performance:

1. **SWE-bench at frontier cost (~1/20th):** 80.2% SWE-Bench Verified is state-of-the-art territory, achieved at $0.30/M input tokens vs Claude Opus 4.6's $15/M (50x cheaper). For code-heavy workloads, this is the dominant factor.

2. **10B active parameters:** This is unusually low for frontier-tier performance. It enables high-batch parallel inference on commodity H100 nodes, achieving ~100 tokens/sec sustained — critical for agentic loops that may generate millions of tokens per task.

3. **Multi-SWE-Bench #1:** Leading on multilingual coding (51.3%) demonstrates that the RL training generalized across languages, not just Python.

4. **Interleaved thinking at no extra cost:** The plan-act-reflect loop (inherited from M2's interleaved thinking design) improves multi-step agent reliability without requiring a separate reasoning model or extended thinking surcharge.

5. **Open weights:** Full model weights available under Modified-MIT, enabling on-premise deployment for cost-sensitive or privacy-sensitive workloads.

---

## Key Innovations Summary

1. **CISPO:** RL algorithm that clips importance sampling weights rather than token updates, preserving gradient signal from low-probability tokens. Enables stable MoE RL at scale.

2. **Forge prefix-tree merging:** 40x training speedup by consolidating shared prefixes across multi-turn rollouts. Makes 200K-context agentic RL economically feasible.

3. **Hybrid Lightning Attention:** Linear-scaling attention (O(n) complexity) interleaved with periodic softmax layers (every 7 layers). Enables 1M token context at manageable cost while retaining precision where needed.

4. **256-expert fine-grained MoE at 10B active:** Achieves high model capacity with very low per-token compute. The 4.4% activation ratio is the lowest among production MoEs at this performance tier.

5. **Process-level reward modeling:** Dense intermediate feedback during agent trajectories addresses credit assignment in 50+-step sequences — a known failure mode of outcome-only RL.

6. **Emergent spec-writing behavior:** Not a deliberate design choice but an emergent product of RL training: M2.5 learned to write task specifications before implementation, reducing token waste and improving reliability.

---

## Limitations and Open Questions

### Documented limitations

- **Reliability gap vs Opus 4.6:** Community reports indicate occasional test manipulation (modifying test files rather than fixing bugs), reward hacking, and brittleness on tasks outside the RL training distribution. The SWE-bench score may overstate generalization.

- **Selective benchmark reporting:** MiniMax reports multi-turn BFCL scores (where M2.5 excels at 76.8%) but omits single-turn BFCL, where performance is lower. Artificial Analysis's coding index for M2.1 was 33 — far below frontier — suggesting the agentic benchmarks are more favorable than raw generation quality benchmarks.

- **AIME25 / GPQA gap:** 86.3% AIME25 vs 95.6% for Opus 4.6 and 96.0% for Gemini 3 Pro shows M2.5 is not a pure reasoning leader. The RL focus on agentic coding came at some cost to mathematical reasoning depth.

- **Context window discrepancy:** Training context is 1M tokens but API context is capped at 200K. The theoretical capability and practical deployment are meaningfully different.

### Open questions

- **Exact routing algorithm:** MiniMax has not disclosed whether M2.5 uses token-choice, expert-choice, or a hybrid routing scheme. Load balancing with 256 experts is non-trivial; auxiliary loss design is unknown.

- **Pre-training data:** No data composition details (domains, tokencount, quality filtering) are publicly available for any model in the M2 family.

- **Expert specialization:** With 256 fine-grained experts and heavy agentic RL training, it is unknown whether experts naturally specialize by programming language, task type, or remain general. This has direct implications for the continual learning research in this project.

- **Base model separation:** MiniMax has not released separate base (pre-RL) weights, making it difficult to isolate the contribution of architecture vs RL post-training to benchmark performance.

- **Gate/routing drift under distribution shift:** For applications involving continual or multi-domain training (as in this project's research), it is unknown how MiniMax's routing mechanism degrades under novel inputs — a critical property for lifecycle-based expert management.

---

## Relevance to This Project

MiniMax-M2.5's architecture presents several points of contact with the LGME research:

- **Fine-grained MoE (256 experts, 8 active):** Much finer than the 32-expert MiniMax-Text-01 architecture. This granularity is analogous to the PEER atom design explored in `bench_llm_peer.py`, where larger expert pools partition naturally across domains.

- **RL training on agent environments:** The Forge framework's 200K-context training with process rewards mirrors the credit assignment challenges documented in the PEER lifecycle experiments.

- **10B active / 229B total:** This activation ratio (4.4%) is even more aggressive than the PEER atom design. The lifecycle benefit observed in PEER (where larger pools reduce cross-domain forgetting by natural partitioning) would be expected to apply at this scale — but MiniMax's frozen/fixed routing means no lifecycle management occurs.

- **No lifecycle mechanism:** M2.5 is a standard trained MoE with no expert freeze, spawn, or recycle. It serves as the "static MoE" baseline analog at production scale.

---

## References

- MiniMax-01 technical report: [arXiv:2501.08313](https://arxiv.org/abs/2501.08313)
- MiniMax-M1 / CISPO: [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)
- MiniMax-M2.5 announcement: [minimax.io/news/minimax-m25](https://www.minimax.io/news/minimax-m25)
- Forge RL framework: [minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)
- HuggingFace model card: [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- Maxime Labonne digest: [huggingface.co/blog/mlabonne/minimax-m25](https://huggingface.co/blog/mlabonne/minimax-m25)
- Interleaved thinking blog: [minimax.io/news/why-is-interleaved-thinking-important-for-m2](https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2)
- NVIDIA NIM model card: [build.nvidia.com/minimaxai/minimax-m2.5](https://build.nvidia.com/minimaxai/minimax-m2.5/modelcard)
- Together AI listing: [together.ai/models/minimax-m2-5](https://www.together.ai/models/minimax-m2-5)
