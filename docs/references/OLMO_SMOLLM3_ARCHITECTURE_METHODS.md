# OLMo 2/3 and SmolLM3: Complete Methods Extraction

Sources:
- OLMo 2: arxiv.org/abs/2501.00656 (7B, 13B, 32B)
- OLMo 3: arxiv.org/abs/2512.13961 (7B, 32B)
- SmolLM3: huggingface.co/blog/smollm3 (3B)

---

## 1. Architecture Comparison Table

| Feature | OLMo 2 (7B/13B/32B) | OLMo 3 (7B/32B) | SmolLM3 (3B) |
|---|---|---|---|
| Layers | 32/40/64 | 32/64 | ~32 (Llama-based) |
| Hidden Size | 4096/5120/5120 | 4096/5120 | ~3072 (est.) |
| Q Heads | 32/40/40 | 32/40 | uses GQA |
| KV Heads | 32/40/8 | 32/8 | groups=4 |
| Attention | MHA/MHA/GQA | MHA/GQA | GQA (4 groups) |
| Activation | SwiGLU | SwiGLU | SwiGLU (Llama-based) |
| Layer Norm | RMSNorm | RMSNorm | RMSNorm |
| Norm Location | **Post-norm** (outputs) | **Post-norm** (outputs) | Pre-norm (Llama default) |
| QK-Norm | Yes (RMSNorm) | Yes (RMSNorm) | Not mentioned |
| Positional Enc | RoPE (theta=500K) | RoPE (theta=500K) | RoPE + **NoPE** (every 4th layer) |
| Sliding Window | No | **Yes** (3/4 layers, 4096 window) | No |
| Z-Loss | Yes (1e-5 weight) | Yes (1e-5 weight) | Not mentioned |
| Biases | None | None | None (Llama-based) |
| Vocab Size | ~100K (cl100k) | ~100K (cl100k) | 128K (Llama 3.2 tokenizer) |
| Context (train) | 4096 | 8192 | 4096 (extended to 64K) |
| Tied Embeddings | No | No | **Yes** |
| Weight Decay on Embeddings | **No** | **No** | **No** (following OLMo 2) |

---

## 2. Normalization Strategies

### 2a. Post-Norm (OLMo 2 "Reordered Norm")
**Name**: Reordered Layer Normalization / Post-Attention Norm
**Source**: Liu et al. (2021), adopted by Chameleon Team (2024)
**Paper**: OLMo 2 Section 3.3.2

**Math (OLMo 2/3)**:
```
h  := x + RMSNorm(Attention(x))
h_out := h + RMSNorm(MLP(h))
```

**Compare to standard pre-norm (OLMo-0424, Llama, SmolLM3)**:
```
h  := x + Attention(LN(x))
h_out := h + MLP(LN(h))
```

**What it solves**: Stabilizes training by normalizing the *output* of attention/MLP blocks rather than the input. In isolation, post-norm alone did not help; it required combining with QK-norm to reduce gradient norm spikiness. Together, spike score dropped from 0.108 to 0.069.

**Where proven**: OLMo 2 Section 3.3.2, Figure 7. OLMo 3 inherits this without change.

### 2b. QK-Norm
**Name**: Query-Key Normalization
**Source**: Dehghani et al. (2023), "Scaling Vision Transformers to 22 Billion Parameters"

**Math**:
```
Q_norm = RMSNorm(Q)
K_norm = RMSNorm(K)
Attention(Q, K, V) = softmax(Q_norm @ K_norm^T / sqrt(d_k)) @ V
```

**What it solves**: Prevents attention logits from growing too large, which causes training loss divergence. Must be combined with post-norm for full effect.

**Where proven**: OLMo 2 Section 3.3.2. Critical at 32B+ scale where spikes became frequent.

### 2c. RMSNorm
**Name**: Root Mean Square Layer Normalization
**Source**: Zhang and Sennrich (2019)

**Math**:
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```
(No centering/bias, only scaling by learned gamma)

**What it solves**: Standard in modern transformers. OLMo 2 switched from non-parametric LayerNorm to parametric RMSNorm. Ablations showed no quality difference but RMSNorm is the safer/standard choice.

**Where proven**: OLMo 2 Section 3.3.1.

---

## 3. Positional Encoding Strategies

### 3a. RoPE (Rotary Position Embeddings)
**Name**: RoPE
**Source**: Su et al. (2021)
**Used by**: All three model families

**Math**:
```
RoPE(x, m) = x * cos(m * theta) + rotate_half(x) * sin(m * theta)
where theta_i = base^(-2i/d), base = theta parameter
```

**Key parameter**: theta (base frequency)
- OLMo 1: theta = 10,000
- OLMo-0424: theta = 10,000
- OLMo 2/3: theta = 500,000 (following Llama 3)
- SmolLM3: default, then extended via YaRN

**What it solves**: Encodes relative position information, critical for length generalization. Higher theta provides finer resolution, especially for longer contexts.

### 3b. NoPE (No Positional Encoding) -- SmolLM3 Innovation
**Name**: NoPE / Hybrid RoPE-NoPE
**Source**: Yang et al. (2025), "RoPE to NoRoPE and Back Again: A New Hybrid Attention Strategy"

**Implementation**: Remove RoPE from every 4th layer (1 in 4 layers has no positional encoding)

**What it solves**: Improves long context performance without affecting short context capabilities. Layers without positional encoding can attend purely based on content similarity, enabling better long-range information retrieval.

**Where proven**: SmolLM3 ablations on 3B model trained with 100B tokens from FineWeb-Edu. Combined with YaRN, enables 128K context extrapolation from 64K training.

### 3c. Sliding Window Attention (SWA) -- OLMo 3 Innovation
**Name**: Sliding Window Attention
**Source**: Beltagy et al. (2020), Longformer

**Implementation in OLMo 3**: Each token attends to previous 4096 tokens. Applied to 3 out of every 4 layers. Last layer always uses full attention.

**What it solves**: Reduces compute cost for long sequences (8K+ context during pretraining). Keeps inference manageable while maintaining global attention through the full-attention layers.

**Where proven**: OLMo 3 Section 3.2, Table 33. Enables training with 8192 context during pretraining (vs OLMo 2's 4096).

---

## 4. Attention Mechanisms

### 4a. Multi-Head Attention (MHA)
**Used by**: OLMo 2 7B/13B, OLMo 3 7B

**Math**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
```

KV cache size: n_heads * d_head * seq_len * 2 (K and V)

### 4b. Grouped Query Attention (GQA)
**Name**: GQA
**Source**: Ainslie et al. (2023)
**Used by**: OLMo 2 32B (40Q/8KV), OLMo 3 32B (40Q/8KV), SmolLM3 (4 groups)

**What it solves**: Reduces KV cache size by sharing KV heads across query heads.
- OLMo 2/3 32B: 5x reduction (40 Q heads, 8 KV heads)
- SmolLM3: significant reduction with 4 KV groups

**Where proven**:
- OLMo 2 Section 2.3: Adopted for 32B scale, inspired by Qwen 3.
- SmolLM3: Ablated on 3B model with 100B tokens from FineWeb-Edu. GQA with 4 groups **matches MHA performance** while significantly reducing KV cache during inference.

---

## 5. Training Stability Techniques

### 5a. Z-Loss Regularization
**Name**: Z-Loss / Auxiliary Loss
**Source**: Chowdhery et al. (2022) (PaLM), Chameleon Team (2024), Wortsman et al. (2023)

**Math**:
```
L_total = L_CE + alpha * log(Z)^2
where Z = sum(exp(logits)) (softmax denominator)
alpha = 1e-5 (for OLMo 2) or 1e-4 (in original PaLM formulation)
```

**What it solves**: Discourages final output logits from growing too large, preventing loss divergence. Empirically improves run stability. Also enhances invariance to learning rate choice (Wortsman et al., 2023).

**Where proven**: OLMo 2 Section 3.3.3, Figure 8. CAVEAT: Flash Attention's fused z-loss implementation produces different backward pass behavior than manual PyTorch implementation. Forward pass matches but gradients diverge.

### 5b. Weight Decay Exclusion for Embeddings
**Name**: No Weight Decay on Embeddings
**Source**: Takase et al. (2024)
**Used by**: OLMo 2, OLMo 3, SmolLM3

**Math**:
```
Standard: theta_t = theta_{t-1} * (1 - lambda * lr) - lr * grad
For embeddings: theta_t = theta_{t-1} - lr * grad  (no decay)
```

**What it solves**: Weight decay shrinks embedding norms excessively. Small embeddings produce large gradients in early layers because the Jacobian of layer_norm(x) w.r.t. x is inversely proportional to ||x||. Removing weight decay lets embedding norms stabilize naturally.

**Where proven**: OLMo 2 Section 3.4.2, Figure 10. Spike score dropped from 0.16 to 0.092. SmolLM3 validates independently.

### 5c. Initialization Strategy (OLMo 2)
**Name**: Simple Truncated Normal Initialization
**Source**: OLMo 2 (replacing scaled initialization from Zhang et al., 2019)

**Math**:
```
All parameters ~ TruncNormal(mean=0, std=0.02)
```

Previous (OLMo-0424):
```
Input projections ~ N(0, 1/sqrt(d_model))
Output projections ~ N(0, 1/sqrt(2 * d_model * layer_idx))
```

**What it solves**: The new initialization keeps growth exponents (log ratio of activation/gradient norms across layers) closer to 0 across all model widths. This means activations and gradients neither explode nor vanish across layers.

**Growth exponent**:
```
lambda = (1/n_layers) * log(||v'|| / ||v||)
Ideal: lambda ~ 0
```

**Where proven**: OLMo 2 Section 3.2, Figures 4-6. Spike score for gradient L2 norm dropped from 0.40 to 0.03. Also shows better hyperparameter transfer across widths (Yang et al., 2024b muP theory).

### 5d. AdamW Epsilon Adjustment
**Name**: Lower AdamW Epsilon
**Source**: OLMo 2

**Change**: epsilon from 1e-5 (PyTorch default) to 1e-8

**What it solves**: Lower epsilon allows larger parameter updates early in training, helping the gradient norm settle more quickly and remain permanently lower. Improves convergence speed.

**Where proven**: OLMo 2 Section 3.4.1, Figure 9.

### 5e. Repeated N-gram Filtering
**Name**: Repeated N-gram Data Filter
**Source**: OLMo 2

**Implementation**: Remove all documents with a sequence of 32+ repeated n-grams (n=1 to 13 tokens). Also mask loss for sequences containing repeated n-grams during training.

**What it solves**: Repeated sequences (e.g., encoded binary data, numerical arrays) cause gradient norm and loss spikes. Filtering reduces spike frequency (not completely eliminates).

**Where proven**: OLMo 2 Section 3.1, Figure 3.

### 5f. Intra-Document Masking
**Name**: Intra-Document Attention Masking
**Source**: SmolLM3 cites "Analysing The Impact of Sequence Composition on Language Model Pre-Training" (2402.13991). Also used in Llama 3 and OLMo 3.

**Implementation**: When packing multiple documents into a single training sequence, apply attention masking so tokens from different documents cannot attend to each other.

**What it solves**: Prevents cross-document spurious attention patterns. Enables faster and more stable long context training while maintaining short context performance.

**Where proven**: SmolLM3 architecture section. OLMo 3 Section 3.6.4 (used during long-context extension).

---

## 6. Training Methodology

### 6a. WSD Learning Rate Schedule (Warmup-Stable-Decay)
**Name**: WSD Schedule
**Used by**: SmolLM3

**Math**:
```
Phase 1 (Warmup): LR linearly increases from 0 to peak over 2000 steps
Phase 2 (Stable): LR stays at peak for ~80% of training
Phase 3 (Decay): LR linearly decays to 0 over final 10% of training
```

SmolLM3 config: peak LR = 2e-4, AdamW(beta1=0.9, beta2=0.95), weight_decay=0.1, grad_clip=1.0

### 6b. Cosine + Linear Decay Schedule (OLMo 2/3)
**Name**: Cosine Schedule with Mid-training Linear Decay
**Used by**: OLMo 2, OLMo 3

**OLMo 2 7B**: Cosine decay over 5T tokens (peak 3e-4), truncated at 4T, then linear decay to 0 during mid-training.
**OLMo 3 7B**: Modified cosine (see paper Figure 3), first half cosine, second half linear. Peak LR 3e-4, final LR 3e-5.
**OLMo 3 32B**: Full cosine schedule over 5.93T tokens, truncated at 5.5T. Peak LR 6e-4.

**Key finding**: Higher learning rates perform better early but are eventually overtaken by lower rates. The annealing step compensates exactly for worse pretraining loss at higher LR. Performance is largely invariant to LR over several multiples when combined with QK-norm + z-loss (Wortsman et al., 2023 extended).

**Where proven**: OLMo 2 Section 4.1, Table 8.

### 6c. Multi-Stage Data Curriculum
All three model families use staged training with evolving data mixtures:

**OLMo 2**: 2 stages
- Stage 1 (Pretraining): 90-95% FLOPs, mostly web data (~3.9T tokens)
- Stage 2 (Mid-training): 5-10% FLOPs, high-quality + math data (50-300B tokens), LR linearly decays to 0

**OLMo 3**: 3 stages
- Stage 1 (Pretraining): ~5.5-5.9T tokens on Dolma 3 Mix
- Stage 2 (Midtraining): 100B tokens on Dolma 3 Dolmino Mix (math, code, QA, instruction, thinking traces)
- Stage 3 (Long-context extension): 50B-100B tokens on Dolma 3 Longmino Mix

**SmolLM3**: 3 stages + mid-training + post-training
- Stage 1 (0-8T): Web 85%, Code 12%, Math 3%
- Stage 2 (8-10T): Web 75%, Code 15%, Math 10% (higher quality)
- Stage 3 (10-11.1T): Web 63%, Code 24%, Math 13% (decay phase, introduce reasoning data)
- Long Context: +100B tokens (4K->32K->64K, two 50B stages)
- Reasoning Mid-training: +140B tokens of thinking traces (4 epochs of 35B tokens)

### 6d. Model Souping / Checkpoint Averaging
**Name**: Model Soup / Weight Averaging
**Source**: Wortsman et al. (2022), Matena and Raffel (2022)

**Implementation**:
- OLMo 2 7B: Average 3 models from different mid-training data orders (50B each)
- OLMo 2 13B/32B: Average 4 models (3x100B + 1x300B)
- OLMo 3 32B midtraining: Average of 2 models from different data order seeds
- OLMo 3 32B long-context: Average of last 3 checkpoints
- SmolLM3: Linear merge of APO model soup (0.9) + mid-training checkpoint (0.1) using MergeKit

**What it solves**: Finds a better local minimum by averaging independently-trained models from the same starting point but different data orderings. Consistently improves over any individual checkpoint.

**Where proven**: OLMo 2 Section 4.5.

### 6e. Microannealing
**Name**: Microannealing
**Source**: OLMo 2

**Implementation**: Short annealing runs (~6-7B tokens each) with 50/50 mix of web + domain data (e.g., a specific math dataset). Used to evaluate individual data sources cheaply before full mid-training runs.

**What it solves**: Allows independent assessment of domain-specific data sources at ~6x lower cost than full annealing runs.

**Where proven**: OLMo 2 Section 4.4.2. 19 microanneals totaling 130B tokens = less than 3 full 50B annealing runs.

---

## 7. Long Context Extension Methods

### 7a. YaRN (Yet another RoPE extensioN)
**Name**: YaRN
**Source**: Peng et al. (2023)
**Used by**: SmolLM3 (inference extrapolation 64K->128K), OLMo 3 (best approach in ablations)

**What it solves**: Extrapolates attention beyond training context length. SmolLM3 uses it to handle 128K context from 64K training length (2x extension).

**Where proven**: SmolLM3 blog (following Qwen 2.5 methodology). OLMo 3 Section 3.6.4: ablated against adjusted base frequency scaling and position interpolation. YaRN applied only to full attention layers (not SWA layers) yields best results.

### 7b. Progressive RoPE Theta Extension (SmolLM3)
**Implementation**:
- Stage 1: 4K -> 32K context, RoPE theta = 1.5M, train 50B tokens
- Stage 2: 32K -> 64K context, RoPE theta = 5M, train 50B tokens
- Inference: YaRN extrapolation to 128K

**Key finding**: Upsampling specific long context data (code repos, books, long web pages) beyond naturally long samples in the mixture did NOT further boost RULER/HELMET performance. NoPE + decay mixture + longer sequences + increased RoPE theta was sufficient.

### 7c. Best-Fit Document Packing (OLMo 3)
**Name**: Best-Fit Decreasing (BFD) Packing
**Source**: Ding et al. (2024)

**What it solves**: Standard concatenate-then-split approach produces training instances shorter than the underlying document length distribution. BFD packing reduces split documents with negligible padding. Substantially improves long-context benchmark performance vs naive approach.

**Where proven**: OLMo 3 Section 3.6.4. Also used in SmolLM3 SFT training.

### 7d. Context Parallelism (OLMo 3)
**Implementation**: 8-way context parallelism (CP) for 65K context. Each device processes 8K tokens. Uses all-gather-based CP attention strategy (Chu et al., 2025) supporting irregular masks (SWA + intra-document masking).

---

## 8. Post-Training Methods

### 8a. Direct Preference Optimization (DPO)
**Name**: DPO
**Source**: Rafailov et al. (2024)
**Used by**: OLMo 3 (Think + Instruct), SmolLM3 (as APO variant)

**Math**:
```
r_theta(x,y) = beta * log(pi_theta(y|x) / pi_ref(y|x))

L_DPO = -E[log sigma(beta * log(pi_theta(y_w|x)/pi_ref(y_w|x)) - beta * log(pi_theta(y_l|x)/pi_ref(y_l|x)))]
```

### 8b. Anchored Preference Optimization (APO) -- SmolLM3
**Name**: APO
**Source**: SmolLM3 blog

A variant of DPO that provides a more stable optimization objective. Higher downstream performance observed.

**Preference data**: Non-reasoning from Tulu3, reasoning from Qwen3-32B (chosen) vs Qwen3-0.6B (rejected).

### 8c. Delta Learning -- OLMo 3 Innovation
**Name**: Delta Learning
**Source**: Geng et al. (2025)

**Principle**: The quality of preference data depends primarily on the delta between chosen and rejected responses, not the absolute quality of either.

**Implementation**: Pair strong model completions (Qwen 3 32B, thinking) with weak model completions (Qwen 3 0.6B, thinking). Even when SFT on the chosen completions alone would hurt performance (saturation), the contrastive signal from DPO with clear capability deltas still provides useful training signal.

**What it solves**: Overcomes SFT saturation. Extends the reasoning frontier beyond what imitation learning can provide.

**Where proven**: OLMo 3 Section 4.3. Key finding: "further supervised finetuning on thinking traces generated by Qwen3 32B outright hurts the performance of Olmo 3 Think SFT, indicating that we are approaching saturation on learning from imitation."

### 8d. RLVR (Reinforcement Learning with Verifiable Rewards)
**Name**: RLVR
**Source**: OLMo 2 (for math), extended to multiple domains in OLMo 3

**What it solves**: Enables RL training on tasks with automatically verifiable answers, without requiring human reward models.

**Domains in OLMo 3**: Math (SymPy equivalence), Code (unit test pass rate), Instruction Following (constraint checking), Chat (LLM-judge scores).

### 8e. OlmoRL (GRPO + Improvements) -- OLMo 3
**Name**: OlmoRL
**Source**: OLMo 3 Section 4.4, building on GRPO (Shao et al., 2024), DAPO (Yu et al., 2025), Dr GRPO (Liu et al., 2025b)

**Math (Final Objective)**:
```
J(theta) = (1 / sum|y_i|) * sum_i sum_t min(
    rho_hat * min(r_{i,t} * A_{i,t}, clip(r_{i,t}, 1-eps_low, 1+eps_high) * A_{i,t})
)

where:
  r_{i,t} = pi(y_{i,t}|x, y_{i,<t}; theta) / pi(y_{i,t}|x, y_{i,<t}; theta_old)
  rho_hat = pi(theta_old) / pi_vllm(theta_old)  [truncated importance sampling]
  A_{i,t} = r(x, y_i) - mean({r(x, y_j)}_j)    [no std normalization]
```

**Improvements over vanilla GRPO**:
1. **Zero gradient signal filtering**: Remove groups with identical rewards (zero std in advantages)
2. **Active sampling**: Maintain batch size despite filtering via dynamic sampling
3. **Token-level loss**: Normalize by total tokens across batch (not per-sample) to avoid length bias
4. **No KL loss**: Removed to allow less-restricted policy updates
5. **Clip higher**: Upper bound clipping > lower bound for larger positive updates
6. **Truncated importance sampling**: Adjusts for inference/training engine log-prob differences
7. **No standard deviation normalization in advantage**: Removes difficulty bias

**Where proven**: OLMo 3 Section 4.4.1. Extended RL training (750 -> 2300 steps) for Olmo 3.1 Think 32B showed continued improvement, not yet saturated.

### 8f. Model Merging for Recovery -- SmolLM3
**Tool**: MergeKit (Goddard et al., 2024)

**Two-step recipe**:
1. Create "model soup" from APO checkpoints
2. Linear merge: APO soup (weight 0.9) + mid-training checkpoint (weight 0.1)

**What it solves**: APO training degraded long-context RULER scores. Merging with mid-training checkpoint (which had strong long-context performance) recovered base model RULER scores up to 128K.

### 8g. Reasoning Mid-training -- SmolLM3
**Implementation**: 35B tokens x 4 epochs = 140B tokens of reasoning traces from:
- OpenThoughts3-1.2M
- Llama-Nemotron-Post-Training-Dataset-v1.1 subset (with R1 reasoning traces)

Using ChatML template and wrapped packing. This is a separate stage between pretraining and SFT that builds general reasoning capability without targeting a specific domain.

---

## 9. Key Architectural Insights for Composable Experts

### Relevant to adapter/composition work:

1. **Post-norm + QK-norm**: OLMo 2/3's post-norm formulation normalizes the OUTPUT of attention/MLP blocks. This means adapters modifying these blocks operate in a norm-controlled space. The residual stream carries unnormalized values, and normalization happens after each block's contribution.

2. **NoPE layers (SmolLM3)**: Layers without positional encoding attend purely on content. These layers could be particularly interesting for composition since they are position-independent -- adapters on NoPE layers would learn content-only routing.

3. **SWA layers (OLMo 3)**: 3/4 of layers have 4096-token local attention windows. Only 1/4 use full attention. This creates a natural hierarchy: local-attention layers process nearby context, full-attention layers handle long-range dependencies.

4. **GQA at scale**: Both OLMo and SmolLM3 use GQA at larger scales. The shared KV heads mean adapter modifications to KV projections have amplified effect (one KV head serves multiple Q heads).

5. **Model souping as a merging baseline**: Weight averaging of independently-trained checkpoints consistently works. This is the simplest form of model composition and serves as a strong baseline.

6. **Embedding stability**: Removing weight decay from embeddings is now consensus across all three model families. For adapter work, this suggests the embedding space is already well-conditioned without regularization.

7. **Training temperature = LR/batch_size**: OLMo 3 reports "peak training temperature" which reveals the effective step size. For 7B: ~2.15e-14; for 32B: ~5.11e-14 during pretraining. This is useful for calibrating adapter learning rates.

---

## 10. Data Curation Methods (OLMo 3)

### 10a. Token-Constrained Mixing
**Source**: OLMo 3 Section 3.4

Optimizes the proportions of data from different sources/topics using a Dirichlet distribution centered on the current mix. Trains 3B-parameter models for 5x Chinchilla tokens, sampling mixes from the Dirichlet. Handles evolving data sources during development with a conditional mixing procedure.

### 10b. Quality-Aware Upsampling
**Source**: OLMo 3 Section 3.4

Instead of flat quality filtering (top quartile), uses **upsampling curves**: higher-quality data is repeated more often (up to 7x for top 5%), while lowest quality (bottom ~40%) is discarded. Each of 24 WebOrganizer-defined topics gets its own upsampling curve, parameterized by: (1) optimal topic proportion, (2) target token count, (3) max upsampling factor of 7.

### 10c. Three-Stage Deduplication (OLMo 3)
**Source**: OLMo 3 Appendix A.2, using Duplodocus tool

1. **Exact deduplication**: Global hash-based removal of identical documents
2. **Fuzzy deduplication**: MinHash-based near-duplicate removal (32 shards + exhaustive cross-shard matching)
3. **Substring deduplication**: Novel fuzzy suffix-array-based procedure (57 shards) targeting repeated boilerplate and cross-page content

### 10d. olmOCR Science PDFs
**Source**: OLMo 3 Section 3.4.2 (Poznanski et al., 2025)

22.3M documents above 8K tokens (640B tokens total), 4.5M documents over 32K tokens (380B tokens total). Largest openly available collection for long-context research. Used for both pretraining and long-context extension.

---

## 11. SmolLM3 Dual-Mode Chat Template

SmolLM3 implements a dual reasoning/non-reasoning mode:
- `/think` flag: enables extended reasoning (default)
- `/no_think` flag: pre-fills model response with empty think blocks for direct answers
- Tool calling support: separate XML Tools and Python Tools sections
- `/system_override` flag: excludes default metadata

This design influenced OLMo 3's approach to thinking models (separate Think and Instruct model variants rather than dual-mode).
