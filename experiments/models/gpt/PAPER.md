# Dense GPT: Research Digest

## Abstract

This document describes the dense autoregressive transformer that serves as the baseline
model in the micro arena. It is a GPT-2 style decoder-only language model with several
deliberate simplifications suited to small-scale experimentation: RMSNorm without
learnable parameters, no bias terms, and ReLU activation. The model is the root node in
the micro model lineage tree; all MoE and lifecycle variants inherit from or extend it.

---

## 1. What This Model Is

The dense GPT is a standard **decoder-only autoregressive transformer** trained with a
causal language modelling objective. Given a token sequence $(x_1, \ldots, x_T)$, the
model maximises:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})
$$

The architecture follows the standard GPT-2 block structure (Radford et al., 2019):
learned token and position embeddings, $L$ transformer blocks each containing causal
multi-head self-attention and a position-wise feed-forward network, and a linear output
head projecting to vocabulary logits. The precise mathematics are derived in `MATH.md`.

At the default configuration ($d=64$, $h=4$, $L=4$) the model has approximately 202K
parameters — intentionally micro-scale to allow rapid iteration on a laptop GPU (Apple
Silicon / MLX).

---

## 2. Key Design Choices

### 2.1 RMSNorm, No Learnable Gain or Bias

The model uses RMSNorm (Zhang & Sennrich, 2019) without learnable $\gamma$ or $\beta$
parameters, rather than the LayerNorm used in GPT-2.

**Why RMSNorm over LayerNorm?** RMSNorm drops mean-centering and the re-parameterisation
parameters, reducing compute and eliminating $2dL$ parameters of Adam state. Empirically,
the difference in training dynamics at small scale is negligible; the simplification makes
ablations cleaner because norms add no learnable state that could confound comparisons.

**Why no learnable gain/bias on the norm?** At micro scale the extra parameters are
unnecessary. The downstream weight matrices can compensate for any fixed scale mismatch.
Removing them also aligns with the trend in recent large models (LLaMA, Mistral) that
use parameter-free RMSNorm.

### 2.2 No Bias Terms

All linear layers (`wq`, `wk`, `wv`, `wo`, `fc1`, `fc2`, `lm_head`) have `bias=False`.
This follows the approach of several recent models (PaLM, LLaMA) and has two motivations:

1. Bias terms add relatively few parameters but require separate Adam moment buffers,
   creating memory overhead disproportionate to their contribution.
2. Without bias, the model is invariant to a constant offset in inputs, which can
   simplify analysis of weight norms and gradient flow.

### 2.3 ReLU, Not GELU

GPT-2 uses GELU (Hendrycks & Gimpel, 2016). This model uses plain ReLU.

**Rationale:** GELU has no closed form (it is approximated via a polynomial or the
error function), adding computation without measurable benefit at this scale. ReLU is
differentiable almost everywhere, cheap to evaluate, and well-understood. At micro scale,
the smoothness advantage of GELU over ReLU is unlikely to matter. ReLU also has a
natural sparsity property (exactly 0 for negative inputs) that is desirable when studying
MoE variants where activation patterns matter.

### 2.4 Pre-Norm Architecture

Normalisation is applied before each sublayer (pre-norm) rather than after (post-norm,
as in the original Transformer, Vaswani et al., 2017). Pre-norm is more stable during
early training and avoids the exploding/vanishing gradient issues associated with
post-norm at moderate depth.

### 2.5 Learned Absolute Positional Embeddings

Position information is injected via a learned embedding table $\mathbf{W}_{pe} \in
\mathbb{R}^{T_{\max} \times d}$, the same approach used by GPT-2. This is the simplest
option and adequate when the training and evaluation sequence lengths are both at most
`block_size`. Relative or rotary (RoPE) encodings would generalise better to longer
contexts but add complexity unnecessary at this scale.

---

## 3. Relationship to GPT-2 and Other Work

| Feature | GPT-2 (Radford et al., 2019) | This model |
|---|---|---|
| Normalisation | LayerNorm (learnable) | RMSNorm (no params) |
| Norm placement | Post-norm (original), Pre-norm (GPT-2) | Pre-norm |
| Activation | GELU | ReLU |
| Bias in linear layers | Yes | No |
| Positional encoding | Learned absolute | Learned absolute |
| Attention | Multi-head, causal | Multi-head, causal |
| MLP expansion ratio | 4x | 4x |
| Tied embeddings | Yes (input/output) | No (independent `lm_head`) |

The core block structure — attention + MLP with residual connections — is identical to
GPT-2 and traceable to Vaswani et al. (2017). The specific simplifications (RMSNorm
without parameters, no bias, ReLU) bring the model closer in spirit to LLaMA (Touvron
et al., 2023), which applies similar choices at large scale for efficiency reasons;
here the same choices are made for experimental cleanliness.

The embedding weights are **not tied** between `wte` and `lm_head`, unlike GPT-2. This
simplifies the forward pass and parameter counting at the cost of $V \cdot d$ extra
parameters. At micro scale this is immaterial.

---

## 4. Role in the Micro Arena

The dense GPT serves two roles:

1. **Baseline.** It establishes the single-network ceiling: the best loss achievable
   when all $N_{\text{params}}$ are used densely on every token. MoE and lifecycle
   variants are evaluated relative to this baseline under identical training conditions
   (same steps, same data, same optimiser).

2. **Root of the lineage tree.** The arena's `ModelTree` records parent-child
   relationships between model variants. The dense GPT has `parent=None`; it is the
   root node. The MoE variant (`moe`) is registered with `parent="gpt"` and reuses
   `RMSNorm`, `CausalSelfAttention`, and `Block` directly from this module. The frozen
   MoE variant (`moe_freeze`) is a child of `moe`. This lineage enables structured
   ablation: differences in metrics between a child and its parent isolate the effect
   of a single architectural change.

The `aux_loss()` method returns `0.0` for the dense model — it has no routing loss.
The `on_domain_switch()` hook is a no-op. Both are required by the arena's training
interface and are the baseline implementations that child models override.

---

## 5. Strengths at This Scale

- **Fast iteration.** ~202K parameters trains in seconds per hundred steps on Apple
  Silicon. Full multi-domain runs complete in under a minute.
- **No auxiliary losses.** The training objective is pure cross-entropy with no
  routing regularisation, load balancing, or distillation terms. Results are
  directly interpretable.
- **Architectural transparency.** Every component is expressed in fewer than 100 lines
  of Python. The absence of bias, learnable norm parameters, and fancy activations
  means there are no hidden degrees of freedom that complicate analysis.
- **Stable baseline.** The simplifications (pre-norm, RMSNorm, no bias) collectively
  improve training stability relative to the original GPT-2 configuration at small
  batch sizes.

---

## 6. Limitations at This Scale

- **Capacity saturation.** A single dense network shares all parameters across all
  domains. When trained sequentially on multiple domains it exhibits catastrophic
  forgetting — the primary motivation for the MoE and lifecycle variants in this
  codebase.
- **No weight tying.** Using a separate `lm_head` wastes $V \cdot d$ parameters
  relative to tied embeddings. At $V=28$, $d=64$ this is only 1,792 parameters —
  negligible, but worth noting.
- **Fixed context window.** Learned absolute positional embeddings do not generalise
  beyond `block_size`. For sequence lengths up to 32 tokens this is not a constraint,
  but the architecture cannot trivially extend to longer contexts.
- **No KV cache.** The forward pass recomputes all attention scores at each step.
  For micro-scale generation this is acceptable; for benchmarking throughput it
  understates practical inference speed.
- **ReLU dead neurons.** Unlike GELU, ReLU neurons with consistently negative
  pre-activations become permanently inactive ("dead"). At $d=64$ and 4x expansion
  the MLP has 256 hidden units per layer; dead neurons reduce effective capacity.
  This is typically not an issue at small scale but could become relevant in
  constrained-capacity experiments.

---

## References

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).
  **Language Models are Unsupervised Multitask Learners.** OpenAI Blog.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). **Attention Is All You Need.**
  NeurIPS.
- Zhang, B., & Sennrich, R. (2019). **Root Mean Square Layer Normalization.** NeurIPS.
- Hendrycks, D., & Gimpel, K. (2016). **Gaussian Error Linear Units (GELUs).** arXiv:1606.08415.
- Touvron, H., et al. (2023). **LLaMA: Open and Efficient Foundation Language Models.**
  arXiv:2302.13971.
