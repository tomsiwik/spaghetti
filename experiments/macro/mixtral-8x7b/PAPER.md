# Mixtral 8x7B: Mixture of Experts at Scale

**Paper:** "Mixtral of Experts" — Jiang et al., Mistral AI, January 2024
**arXiv:** https://arxiv.org/abs/2401.04088
**Released:** December 11, 2023
**License:** Apache 2.0

---

## Model Overview

Mixtral 8x7B is a sparse Mixture of Experts (SMoE) language model developed by Mistral AI. It was the first widely available open-weight MoE model at this scale and demonstrated that sparse expert routing could match or exceed much larger dense models while using a fraction of the active compute. At release it was the strongest open-weight model with a permissive license, and the first open model to achieve GPT-3.5-level performance on standard benchmarks.

**Catalog entry:** `mistralai/Mixtral-8x7B-Instruct-v0.1`
**Tier:** Large — excellent quality/compute ratio at time of release, now surpassed by newer MoE models (DeepSeek-V3, Qwen2.5-MoE, Llama 4) but remains a solid large-tier reference.

---

## Architecture

### Sparse Mixture of Experts

Mixtral is a decoder-only transformer that extends the Mistral 7B architecture by replacing every FFN (feed-forward network) sublayer with a sparse MoE layer. Non-FFN components (attention, RMSNorm, embeddings) are unchanged and shared — only the FFN blocks are replicated eight times.

**Key parameters:**

| Parameter | Value |
|---|---|
| Total parameters | 46.7B |
| Active parameters per token | 12.9B |
| Number of layers | 32 |
| Hidden size (d_model) | 4096 |
| FFN intermediate size (per expert) | 14,336 |
| Number of attention heads | 32 |
| Head dimension | 128 |
| Number of experts per layer | 8 |
| Experts activated per token | 2 (top-2) |
| Context length | 32,768 tokens |
| Vocabulary size | 32,000 |

The total parameter count is ~46.7B, not 56B (8 x 7B) — because only the FFN is replicated, not the full model.

### Expert Structure

Each expert is a full FFN identical in structure to the Mistral 7B FFN: a two-layer MLP with SwiGLU activation and intermediate dimension 14,336. There is no weight sharing between experts. The eight experts per layer are therefore eight independent SwiGLU FFNs sitting behind a single learned router.

### Top-2 Gating / Router

For each token at each layer, a learned linear router scores all 8 experts and selects the top 2. The output is a weighted sum of the two selected expert outputs, with weights given by the softmax over the top-2 scores:

```
y = sum_i  Softmax(Top2(x @ W_gate))_i * Expert_i(x)
```

where `Top2` sets all but the two largest logits to `-inf` before softmax, producing a convex combination of exactly two expert outputs. The router is a simple linear projection (`d_model -> n_experts`), one per layer, with no nonlinearity. There is no auxiliary load-balancing loss in Mixtral; the authors report that expert assignment is naturally balanced close to the theoretical uniform rate (1/8 per expert) after training.

### Inherited from Mistral 7B

- **Grouped-Query Attention (GQA):** 8 KV heads shared across 32 query heads, reducing KV cache memory.
- **Sliding Window Attention (SWA):** Each token attends to a fixed window of prior tokens at each layer. The window size is 4,096 tokens per layer; with 32 layers, the effective receptive field grows to ~128K tokens in theory, though the model was trained on 32K context.
- **RMSNorm:** Pre-norm architecture with RMSNorm instead of LayerNorm.
- **SentencePiece tokenizer:** 32K vocabulary, byte-fallback.
- **Rotary position embeddings (RoPE).**

---

## Training

Mistral AI has not publicly disclosed the full training recipe. Known or inferred details:

- **Pretraining data:** Large-scale web corpora. Multilingual emphasis (English, French, German, Spanish, Italian) relative to Mistral 7B. No disclosed dataset size or composition.
- **Context length:** 32,768 tokens during pretraining.
- **Optimizer:** Not disclosed. Standard practice (AdamW) assumed.
- **No auxiliary load-balancing loss** — the paper reports naturally balanced routing emerged without it, in contrast to Switch Transformer and other MoE predecessors that required explicit capacity penalties.
- **Expert parallelism** is the natural deployment strategy: with 8 experts per layer across 32 layers, experts can be sharded across GPUs. At inference, only 2 of 8 experts activate per token, giving ~5x active-compute reduction vs. a 70B dense model.

**Instruct tuning (Mixtral-8x7B-Instruct-v0.1):**
- Supervised fine-tuning (SFT) on curated instruction datasets (undisclosed).
- Direct Preference Optimization (DPO) using paired human preference data (undisclosed).
- No RLHF / PPO. DPO was chosen for stability and simplicity.

---

## Benchmark Performance

Mixtral 8x7B outperforms or matches Llama 2 70B on most benchmarks while using ~5x fewer active parameters. It notably dominates on code, mathematics, and multilingual tasks.

### Base model vs. peers

| Model | Active Params | MMLU | HellaSwag | WinoGrande | ARC | TruthfulQA | GSM8K |
|---|---|---|---|---|---|---|---|
| Mixtral 8x7B | 12.9B | 70.6% | 81.0% | 76.4% | 66.0% | 46.8% | 74.4% |
| Llama 2 70B | 70B | 69.8% | 87.3% | 83.7% | 67.3% | 44.9% | 56.8% |
| Llama 2 13B | 13B | 54.8% | 80.7% | 72.0% | 59.4% | 37.4% | 29.0% |

Mixtral matches Llama 2 70B on MMLU and TruthfulQA while using 5x fewer active parameters. It substantially exceeds Llama 2 70B on GSM8K (+17pp) and is competitive on reasoning tasks.

### Instruct model vs. peers (MT-Bench)

| Model | MT-Bench Score |
|---|---|
| GPT-4 Turbo | 9.32 |
| GPT-3.5-Turbo | 8.32 |
| Mixtral 8x7B Instruct | 8.30 |
| Claude 2.1 | 8.18 |
| Llama 2 70B Chat | ~6.3 |

Mixtral Instruct was the first open model to match GPT-3.5-Turbo on MT-Bench. On Chatbot Arena (LMSYS), it achieved an Elo of 1121, outperforming Claude 2.1 (1117) and all GPT-3.5 variants at the time of release.

### Code (HumanEval): 40.2%
### Multilingual: Strong across French, German, Spanish, Italian — significantly better than Llama 2 70B on these languages.

---

## Expert Specialization: What the Paper Found

The paper includes an empirical analysis of which experts are selected for different token types. Key findings:

- **No strong domain specialization.** Experts do not partition by topic (math vs. code vs. prose). Expert assignment proportions across different text domains (ArXiv papers, Wikipedia, code, biology, philosophy, etc.) are nearly uniform, except for marginal differences in the first and last layers.
- **Syntactic/positional patterns.** Experts do show consistent behavior with respect to token position within a word (first token vs. continuation) and syntactic role. The same expert tends to be selected for similar syntactic contexts across different runs.
- **Routing is consistent but not semantic.** The experts appear to specialize in processing structural or syntactic aspects of tokens rather than semantic domains. The paper notes this is different from the intuitive "expert in math" framing.
- **Natural balance without auxiliary loss.** Expert utilization is close to uniform (12.5% each) after training, suggesting the learned routing naturally avoids expert collapse.

---

## Memory and Inference Requirements

| Precision | VRAM Required |
|---|---|
| float16 (full) | ~90 GB |
| int8 | ~45 GB |
| int4 | ~23 GB |

At float16, the model requires two A100 80GB GPUs minimum. The sparse activation means throughput is approximately that of a 13B dense model despite the large total parameter count.

---

## Historical Significance

Mixtral 8x7B was a watershed moment for open MoE models:

1. **First widely-available open MoE at this scale.** Prior MoE work (Switch Transformer, GLaM, Gshard) was proprietary or research-only. Mixtral made a production-quality MoE accessible under Apache 2.0.
2. **Proved MoE practical beyond research settings.** The model shipped with inference tooling (Megablocks-style sparse kernels, TGI support) and quantization support from day one.
3. **Established the quality/compute paradigm.** Demonstrating that 13B active parameters could match 70B dense parameters changed how practitioners thought about deployment costs.
4. **Sparked a wave of open MoE development.** Within months, DeepSeek-MoE, Qwen1.5-MoE, DBRX, and Grok-1 all appeared, all citing or building on Mixtral's release.
5. **DPO at scale.** Mixtral Instruct demonstrated that DPO (without PPO) was sufficient for competitive instruction following at this scale, validating DPO as a practical RLHF alternative.

---

## Comparison with Later MoE Models

| Model | Total | Active | Top-K | Expert Type | Key Advance |
|---|---|---|---|---|---|
| Mixtral 8x7B (Dec 2023) | 46.7B | 12.9B | Top-2 of 8 | Full FFN | First open MoE at scale |
| DeepSeek-MoE (Jan 2024) | 16.4B | 2.8B | Top-K fine-grained | Smaller experts | Shared expert + fine-grained routing |
| Qwen1.5-MoE (Mar 2024) | 14.3B | 2.7B | Top-4 of 64 | Fine-grained | Matches 7B dense at 2.7B active |
| DeepSeek-V3 (Dec 2024) | 671B | 37B | Top-8 of 256 | Fine-grained | SOTA quality, MTP head |
| Llama 4 Maverick (Apr 2025) | ~400B | ~17B | Top-2 of 128 | Interleaved MoE/dense | Meta's first open MoE |

**Key architectural trend from Mixtral onward:** fine-grained experts (more, smaller experts with higher top-K) rather than Mixtral's coarse 8-expert design. DeepSeek-V3 uses 256 experts with top-8 routing, plus shared experts. Qwen uses 64 experts with top-4 routing. This trades routing complexity for finer-grained specialization. Mixtral's simple top-2-of-8 design remains the easiest to reason about and implement.

**Why Mixtral is in the "large" tier today:** The model quality (MMLU ~70%, MT-Bench 8.30) is still competitive for many tasks, and the 12.9B active parameter budget is genuine "large" territory. However, newer models like Qwen2.5-72B-Instruct, DeepSeek-V3, and Llama 4 Scout/Maverick deliver substantially better reasoning, coding, and instruction following at comparable or lower inference cost. Mixtral is surpassed but not obsolete — it remains a reliable baseline and is cheap to serve via API.

---

## Why MoE Works: The Core Intuition

The parameter efficiency gain in Mixtral comes from conditional computation: the model has the capacity of a ~47B dense network (wide FFN intermediate dimensions summed across 8 experts) but only executes 2/8 of that FFN capacity per token. The router learns which FFN subspace is most useful for each token. This is distinct from model ensembles (all outputs combined) or multi-head attention (all heads contribute to every token). In Mixtral's case, the 6 inactive experts contribute zero to the forward pass — there is no gradient or activation through them for that token at that layer.

The reason this works well despite seemingly random expert assignment (as noted in the specialization analysis) is that the experts still learn different weight matrices through their training trajectories, even if the router doesn't perfectly partition by domain. The diversity comes from the different data each expert sees during training due to routing decisions, not from explicit semantic partitioning.

---

## References

- Jiang et al. (2024). "Mixtral of Experts." arXiv:2401.04088. https://arxiv.org/abs/2401.04088
- Mistral AI blog post (Dec 2023). https://mistral.ai/news/mixtral-of-experts/
- Hugging Face model page and blog. https://huggingface.co/blog/mixtral
- HuggingFace Transformers documentation. https://huggingface.co/docs/transformers/model_doc/mixtral
- LMSYS Chatbot Arena leaderboard (accessed Dec 2023 snapshot).
