# DeepSeek V3/R1 Architecture: Complete Method Catalog
**Date:** 2026-03-29
**Sources:** arXiv:2412.19437 (V3), arXiv:2501.12948 (R1), arXiv:2405.04434 (V2/MLA origin), HuggingFace config.json, Raschka architecture comparison

---

## Model Overview

| Property | DeepSeek-V3 | DeepSeek-R1 |
|---|---|---|
| Total params | 671B (+14B MTP = 685B on HF) | 671B (same arch) |
| Active params/token | 37B | 37B |
| Layers | 61 | 61 |
| Hidden dim (d_model) | 7168 | 7168 |
| Attention heads (n_h) | 128 | 128 |
| Vocab size | 129,280 | 129,280 |
| Context | 128K (YaRN extended from 4K base) | 128K |
| Pre-training tokens | 14.8T | N/A (RL on V3-Base) |
| Training cost | 2.788M H800 hours (~$5.6M) | Additional RL stages |

---

## Method 1: Multi-head Latent Attention (MLA)

### What it solves
Standard MHA caches full K and V projections per token per layer: `2 * n_h * d_h` elements per token. At 128 heads x 128 dim = 32,768 elements per token per layer. GQA reduces this by sharing KV across head groups but sacrifices modeling capacity. MLA compresses KV into a low-rank latent that is **smaller than GQA** while **matching or exceeding MHA** in quality.

### Math

**KV compression (down-project hidden state to shared latent):**
```
c_t^KV = W^DKV * h_t          W^DKV in R^{d_c x d}, c_t^KV in R^{d_c}
```

**Up-project to per-head keys and values:**
```
k_t^C = W^UK * c_t^KV         W^UK in R^{(n_h * d_h) x d_c}
v_t^C = W^UV * c_t^KV         W^UV in R^{(n_h * d_h) x d_c}
```

**Decoupled RoPE key (positional info kept separate):**
```
k_t^R = RoPE(W^KR * h_t)      W^KR in R^{d_R x d}, k_t^R in R^{d_R}
```

**Query compression (for training efficiency):**
```
c_t^Q = W^DQ * h_t            W^DQ in R^{d_c' x d}
q_t^C = W^UQ * c_t^Q          W^UQ in R^{(n_h * d_h) x d_c'}
q_t^R = RoPE(W^QR * c_t^Q)    W^QR in R^{(n_h * d_R) x d_c'}
```

**Combined key and query per head i:**
```
k_{t,i} = [k_{t,i}^C ; k_t^R]         dim = d_h + d_R
q_{t,i} = [q_{t,i}^C ; q_{t,i}^R]     dim = d_h + d_R
```

**Attention computation:**
```
o_{t,i} = sum_{j=1}^{t} softmax_j(q_{t,i}^T k_{j,i} / sqrt(d_h + d_R)) * v_{j,i}^C
u_t = W^O * [o_{t,1}; o_{t,2}; ...; o_{t,n_h}]
```

### KV cache: what is actually stored
Only `c_t^KV` (dim d_c) and `k_t^R` (dim d_R) per token per layer. Everything else is recomputed.

**Cache size comparison per token per layer:**
| Method | Cache elements | DeepSeek-V3 value |
|--------|---------------|-------------------|
| MHA | 2 * n_h * d_h = 2 * 128 * 128 | 32,768 |
| GQA (8 groups) | 2 * n_g * d_h = 2 * 8 * 128 | 2,048 |
| MLA | d_c + d_R = 512 + 64 | 576 |

**MLA achieves 57x compression vs MHA, 3.6x vs GQA-8, while matching MHA quality.**

### The absorption trick (inference optimization)
During inference, `W^UK` can be absorbed into `W^UQ` (merged into a single matmul), and `W^UV` can be absorbed into `W^O`. This means the up-projections for K and V never need to be explicitly computed -- only `c_t^KV` is cached and the absorbed weights handle the rest. The exception is `k_t^R` which must be cached separately because RoPE prevents absorption.

### Hyperparameters (DeepSeek-V3)
```
d_c (kv_lora_rank)      = 512       # KV compression dimension
d_c' (q_lora_rank)      = 1536      # Query compression dimension
d_h (qk_nope_head_dim)  = 128       # Content head dimension
d_R (qk_rope_head_dim)  = 64        # RoPE head dimension
v_head_dim               = 128       # Value head dimension
n_h (num_attention_heads)= 128       # Number of attention heads
```

### Where proven
- **DeepSeek-V2** (arXiv:2405.04434): First introduced. 236B total/21B active. 93.3% KV cache reduction vs DeepSeek-67B. 5.76x generation throughput improvement.
- **DeepSeek-V3** (arXiv:2412.19437): Scaled to 671B. Same MLA dims as V2.
- **Kimi K2** (1T params): Adopted MLA from DeepSeek, confirming transferability.

---

## Method 2: DeepSeekMoE (Mixture of Experts with Shared + Routed Experts)

### What it solves
Standard MoE (Mixtral-style) uses a small number of large experts (8 experts, top-2). This creates coarse-grained specialization and load balancing challenges. DeepSeekMoE uses many small experts + a shared expert that is always active, providing finer-grained specialization and a guaranteed baseline capacity.

### Math

**FFN layer output with shared + routed experts:**
```
h_t' = u_t + sum_{i=1}^{N_s} FFN_i^(s)(u_t) + sum_{i=1}^{N_r} g_{i,t} * FFN_i^(r)(u_t)
```

**Expert affinity score (sigmoid gating, NOT softmax):**
```
s_{i,t} = sigmoid(u_t^T * e_i)
```
where `e_i` is the centroid/embedding vector for routed expert i.

**Top-K selection with bias for load balancing:**
```
g_{i,t}' = s_{i,t}   if (s_{i,t} + b_i) in TopK({s_{j,t} + b_j}, K_r)
         = 0          otherwise
```

**Normalized gating (selected experts sum to 1):**
```
g_{i,t} = g_{i,t}' / sum_{j=1}^{N_r} g_{j,t}'
```

### Why sigmoid instead of softmax
Softmax creates competition between experts (zero-sum). Sigmoid allows independent scoring -- an expert's affinity is absolute, not relative. This enables better load balancing because bias adjustments don't distort the relative ordering as severely. The normalization step after selection recovers the property that weights sum to 1.

### Routed scaling factor
Config shows `routed_scaling_factor = 2.5`. This scales the routed expert contribution to compensate for the normalization across K_r active experts.

### Node-limited routing
Each token sent to at most M=4 nodes. Expert selection: pick the top K_r/M affinity scores per node, then select the M nodes with the highest sums. Reduces cross-node all-to-all communication while maintaining expert diversity.

### No token dropping
Unlike Switch Transformer and Mixtral which drop tokens exceeding expert capacity buffers, DeepSeek-V3 never drops tokens during training or inference. The auxiliary-loss-free balancing (Method 3) makes this feasible.

### First-K dense layers
```
first_k_dense_replace = 3
```
The first 3 Transformer layers use standard dense FFN (no MoE). MoE begins at layer 4. This provides stable early representations before expert routing.

### Expert group structure
```
n_group = 8        # experts divided into 8 groups of 32
topk_group = 4     # select from 4 groups per token
```
The 256 routed experts are organized into 8 groups of 32. Routing first selects 4 groups, then picks 2 experts per group = 8 total active experts.

### Hyperparameters (DeepSeek-V3)
```
N_s (n_shared_experts)    = 1        # always-active shared expert
N_r (n_routed_experts)    = 256      # sparse routed experts
K_r (num_experts_per_tok) = 8        # active routed experts per token
moe_intermediate_size     = 2048     # per-expert FFN hidden dim
intermediate_size         = 18432    # shared expert / dense layer FFN hidden dim
moe_layer_freq            = 1        # MoE every layer (after first 3 dense)
scoring_func              = sigmoid  # NOT softmax
topk_method               = noaux_tc # auxiliary-loss-free + token choice
```

### Where proven
- **DeepSeek-V2** (arXiv:2405.04434): 160 routed experts + 2 shared, K_r=6. 42.5% training cost reduction vs DeepSeek-67B.
- **DeepSeek-V3** (arXiv:2412.19437): Scaled to 256 routed + 1 shared, K_r=8. Stable training, no token dropping.
- **GLM-4.5** (355B): Also adopted initial dense layers before MoE (3 dense layers).
- **Qwen3 MoE** (235B-A22B): 256 experts, K_r=8, but NO shared expert.

---

## Method 3: Auxiliary-Loss-Free Load Balancing

### What it solves
Traditional MoE uses an auxiliary loss to encourage balanced expert utilization:
```
L_aux = alpha * N * sum_i (f_i * P_i)
```
This auxiliary loss degrades model quality because it competes with the main language modeling objective. Larger alpha = better balance but worse performance. Smaller alpha = better performance but expert collapse. DeepSeek eliminates this trade-off.

### Math

**Dynamic bias update (per training step):**
```
b_i <- b_i - gamma    if expert i is overloaded
b_i <- b_i + gamma    if expert i is underloaded
```
where gamma is the bias update speed hyperparameter.

**Critical: bias only affects routing, NOT gating weights.**
The bias `b_i` is added to `s_{i,t}` only for the TopK selection decision. The actual gating weight `g_{i,t}` uses the original unbiased `s_{i,t}`. This means:
- Routing decisions are nudged toward underloaded experts
- But the model's learned expert preferences are preserved in the output computation
- No gradient signal from load balancing pollutes the main loss

**Complementary sequence-level balance loss (tiny alpha):**
```
L_Bal = alpha * sum_{i=1}^{N_r} f_i * P_i
```
where:
```
f_i = (N_r / (K_r * T)) * sum_{t=1}^{T} 1(s_{i,t} in TopK)    # fraction routed to expert i
P_i = (1/T) * sum_{t=1}^{T} s_{i,t}'                            # mean normalized affinity
s_{i,t}' = s_{i,t} / sum_{j=1}^{N_r} s_{j,t}                   # per-token normalization
```
alpha is set to an extremely small value -- this loss prevents catastrophic within-sequence imbalance but has negligible effect on model quality.

### Where proven
- **DeepSeek-V3** (arXiv:2412.19437): First introduction. Enabled training 671B MoE with no token dropping and no irrecoverable loss spikes. "Remarkably stable" training.
- Contrast with Switch Transformer which requires capacity factor C=1.0-1.25 and drops overflow tokens.
- Contrast with Mixtral which uses standard auxiliary loss.

---

## Method 4: Multi-Token Prediction (MTP)

### What it solves
Standard next-token prediction provides a single supervision signal per position. MTP predicts D additional future tokens, providing:
1. Denser training signal per sequence
2. Better representation learning (must encode longer-range dependencies)
3. Can be repurposed for speculative decoding at inference

### Math

**Combined representation at depth k (sequential, not parallel):**
```
h_i'^k = M_k * [RMSNorm(h_i^{k-1}) ; RMSNorm(Emb(t_{i+k}))]
```
where:
- `h_i^{k-1}`: representation from previous MTP depth (k=0 is main model)
- `Emb(t_{i+k})`: embedding of the (ground-truth) token at position i+k
- `M_k in R^{d x 2d}`: linear projection merging the two normalized vectors
- `;` denotes concatenation

**Transformer block at depth k:**
```
h_{1:T-k}^k = TRM_k(h_{1:T-k}'^k)
```
Each depth has its own Transformer block `TRM_k` but shares embeddings and output head with the main model.

**Prediction at depth k:**
```
P_{i+k+1}^k = OutputHead(h_i^k)
```

**Loss per depth:**
```
L_MTP^k = -(1/T) * sum_{i=2+k}^{T+1} log P_i^k[t_i]
```

**Total MTP loss:**
```
L_MTP = (lambda / D) * sum_{k=1}^{D} L_MTP^k
```

**Combined training loss:**
```
L = L_main + L_MTP + L_Bal
```

### Key design choice: sequential not parallel
Unlike Meta's parallel MTP (arXiv:2404.19737), DeepSeek uses sequential prediction. Each depth k conditions on the ground-truth token at position i+k AND the representation from depth k-1. This maintains a complete causal chain and provides richer learning signal.

### At inference
MTP modules can be:
1. **Discarded** -- main model works independently (default)
2. **Used for speculative decoding** -- predict D draft tokens, verify in parallel

### Hyperparameters (DeepSeek-V3)
```
num_nextn_predict_layers = 1    # D=1 (predict 1 additional token)
MTP module size          = 14B  # (685B total - 671B main)
```
Note: The 14B MTP module is relatively small -- a single additional Transformer block with shared embeddings/output head.

### Where proven
- **DeepSeek-V3** (arXiv:2412.19437): D=1, "stronger performance" vs baseline without MTP.
- **Meta** (arXiv:2404.19737): Parallel MTP shown effective for code generation. DeepSeek chose sequential variant.
- **Qwen3-Next** (80B-A3B): Also adopted MTP for speculative decoding.

---

## Method 5: FP8 Mixed-Precision Training

### What it solves
BF16 training of 671B parameters requires enormous memory and compute. FP8 (E4M3 format) theoretically doubles throughput for matrix multiplications. The challenge: naive FP8 introduces too much quantization error for stable training.

### Math

**Fine-grained quantization scheme:**
```
Activations: 1 x 128 tile quantization (per-token, 128-channel groups)
Weights:     128 x 128 block quantization
```

**Per-group scaling:**
```
scale = max(|values_in_group|) / max_FP8_representable
quantized = round(values / scale)
```

**Three GEMM operations in FP8:**
- Fprop (forward pass): `Y = X * W`
- Dgrad (activation gradient): `dX = dY * W^T`
- Wgrad (weight gradient): `dW = X^T * dY`

**High-precision accumulation fix:**
H800 FP8 accumulator has only ~14 bits of precision. DeepSeek promotes partial sums to FP32 every N_C=128 elements:
```
accumulator = FP32(sum of 128 FP8 products) + running_FP32_sum
```

### What stays in BF16/FP32
- Embedding layer
- Output head (logits)
- MoE gating module
- RMSNorm operators
- Attention softmax and score computation
- Master weights, weight gradients, optimizer states (FP32)

### Where proven
- **DeepSeek-V3** (arXiv:2412.19437): First to train 671B model end-to-end with FP8. "No irrecoverable loss spikes."
- Format: E4M3 (4-bit exponent, 3-bit mantissa), dynamic activation quantization scheme.

---

## Method 6: RMSNorm (Pre-Norm)

### What it solves
LayerNorm computes both mean and variance, which is more expensive and not always necessary. RMSNorm only normalizes by the root mean square, removing the mean-centering step. Pre-Norm placement (before attention and FFN) improves training stability.

### Math
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

eps = 1e-6 (rms_norm_eps in config)
```

### Where proven
- Ubiquitous in modern LLMs: DeepSeek V2/V3, Llama 2/3/4, Qwen 2/3, Gemma 3, Mistral, OLMo 2.
- Pre-Norm is the standard placement. Post-Norm (OLMo 2) is a notable exception that showed improved training stability.
- **QK-Norm** (RMSNorm on queries and keys before attention) is a separate emerging trend (OLMo 2, MiniMax-M2) but NOT used in DeepSeek V3.

---

## Method 7: SwiGLU Activation

### What it solves
Standard ReLU/GELU activations are outperformed by gated variants. SwiGLU combines Swish (SiLU) gating with a Gated Linear Unit structure for better expressiveness.

### Math
```
SwiGLU(x) = (x * W_gate) . SiLU(x * W_up)     # element-wise product
           = (x * W_gate) . (sigma(x * W_up) * (x * W_up))

where SiLU(z) = z * sigmoid(z)
```

Note: config shows `hidden_act = silu` which is the activation inside the gated structure. The full FFN is:
```
FFN(x) = W_down * (SiLU(W_gate * x) . (W_up * x))
```
This requires 3 weight matrices per FFN instead of 2, hence the intermediate dimension is typically 8/3 * d_model (here 18432 = 2.57 * 7168).

### Where proven
- PaLM (Google, 2022): Demonstrated SwiGLU superiority over ReLU/GELU.
- Used in: DeepSeek V2/V3, Llama 2/3/4, Qwen 2/3, Mistral, Grok 2.5.
- Exception: Gemma 3 uses GELU instead.

---

## Method 8: RoPE (Rotary Position Embedding) with YaRN Extension

### What it solves
Absolute position embeddings don't generalize to longer sequences. RoPE encodes relative position through rotation matrices applied to query/key pairs, enabling length extrapolation.

### Math
```
RoPE(x, pos) = R(pos) * x

where R(pos) is a block-diagonal rotation matrix:
R(pos) = diag(R_1(pos), R_2(pos), ..., R_{d/2}(pos))

R_i(pos) = [[cos(pos * theta_i), -sin(pos * theta_i)],
             [sin(pos * theta_i),  cos(pos * theta_i)]]

theta_i = theta_base^{-2i/d}
```

### YaRN extension (for 4K -> 128K)
```
rope_scaling:
  type: yarn
  factor: 40                           # extension factor (4096 * 40 = 163,840)
  original_max_position_embeddings: 4096
  beta_fast: 32                        # high-frequency boundary
  beta_slow: 1                         # low-frequency boundary
  mscale: 1.0
  mscale_all_dim: 1.0
```

YaRN (Yet another RoPE extensioN) partitions frequency dimensions into three groups:
- High frequencies (> beta_fast): no scaling needed (already generalize)
- Low frequencies (< beta_slow): linear interpolation by factor
- Middle frequencies: smooth interpolation between the two

### Hyperparameters (DeepSeek-V3)
```
rope_theta                         = 10000
qk_rope_head_dim (d_R)            = 64
max_position_embeddings            = 163840   # 160K
original_max_position_embeddings   = 4096     # base training length
```

### In MLA: decoupled RoPE
RoPE cannot be applied to the compressed latent `c_t^KV` because the absorption trick requires position-independent keys. Solution: a separate small key `k_t^R` (dim 64) carries RoPE and is concatenated with the content key. This is cached alongside `c_t^KV`.

### Where proven
- RoPE: RoFormer (Su et al., 2021). Now ubiquitous.
- YaRN: arXiv:2309.00071. Used by DeepSeek V3 for 32x context extension.
- Alternatives emerging: NoPE (SmolLM3, every 4th layer), Partial RoPE (MiniMax-M2, Qwen3-Next).

---

## Method 9: GRPO (Group Relative Policy Optimization) -- DeepSeek-R1

### What it solves
Standard RLHF requires training a separate critic/value model (same size as the policy) to estimate advantages. This doubles memory and compute. GRPO eliminates the critic by computing advantages from a group of sampled outputs.

### Math

**Sampling:** For each prompt q, sample G completions {o_1, ..., o_G} from the current policy pi_theta.

**Reward:** Compute reward r_i = R(q, o_i) for each completion.

**Group-relative advantage (no critic needed):**
```
A_i = (r_i - mean({r_1,...,r_G})) / std({r_1,...,r_G})
```

**Policy gradient with clipping (PPO-style):**
```
L_GRPO = -E_q [1/G * sum_{i=1}^{G} (
    min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i)
    - beta * KL(pi_theta || pi_ref)
)]
```
where:
```
rho_i = pi_theta(o_i | q) / pi_old(o_i | q)     # importance ratio
```

**KL penalty:** Prevents policy from diverging too far from reference model pi_ref (the SFT checkpoint).

### R1-Zero: Pure RL (no SFT)
- Base: DeepSeek-V3-Base (pre-trained only, no instruction tuning)
- Reward: accuracy (compiler/deterministic verification) + format (<think> tags)
- Emergent: "aha moment" -- model spontaneously develops self-reflection, verification, backtracking
- Problem: readability issues, language mixing, endless repetition

### R1: Full pipeline (4 stages)
1. **Cold-start SFT**: Thousands of carefully curated reasoning examples -> SFT on V3-Base
2. **Reasoning RL**: GRPO with accuracy + format rewards -> reasoning capability emerges
3. **Rejection sampling + SFT**: Sample from Stage 2 model, filter by correctness -> 600K reasoning + 200K non-reasoning examples -> SFT
4. **Final RL**: GRPO with accuracy + format + helpfulness rewards -> alignment

### Distilled models
SFT on 800K R1-generated reasoning traces:
- DeepSeek-R1-Distill-Qwen-1.5B/7B/14B/32B (base: Qwen2.5)
- DeepSeek-R1-Distill-Llama-8B/70B (base: Llama-3.1-8B, Llama-3.3-70B)

Key finding: "Distilling R1 reasoning into Qwen-32B surpasses o1-mini on multiple benchmarks."

### Where proven
- **DeepSeek-R1** (arXiv:2501.12948, published in Nature 2025): AIME 2024 79.8% (pass@1), MATH-500 97.3%, competitive with OpenAI o1.
- GRPO first appeared in DeepSeekMath (arXiv:2402.03300).

---

## Method 10: DualPipe (Training Infrastructure)

### What it solves
Standard pipeline parallelism (1F1B schedule) has pipeline bubbles of `(PP-1) * (F+B)` time. Cross-node MoE all-to-all communication adds further latency. DualPipe overlaps computation with communication.

### Math
**Bubble comparison:**
```
Standard 1F1B:  bubble = (PP - 1) * (F + B)
DualPipe:       bubble = (PP/2 - 1) * (F&B + B - 3W)
```
where F = forward time, B = backward time, W = weight update time, F&B = fused forward+backward.

**Key idea:** Feed micro-batches from BOTH ends of the pipeline simultaneously. While one direction computes, the other communicates. Warp specialization dedicates 20 SMs to communication and remaining SMs to computation.

### Where proven
- **DeepSeek-V3** (arXiv:2412.19437): 2048 H800 GPUs, 16-way PP. Near-complete overlap of all-to-all communication with computation.

---

## Summary: Transferable Methods Ranked by Novelty

| Method | Novelty | Transferable to small models? | Key insight |
|--------|---------|-------------------------------|-------------|
| MLA | High | YES (any attention model) | Low-rank KV compression + decoupled RoPE = smaller cache than GQA with MHA quality |
| Aux-loss-free balancing | High | YES (any MoE) | Bias on routing not gating = load balance without quality loss |
| DeepSeekMoE (shared+routed) | Medium | YES (any MoE) | Shared expert as always-on baseline, many small routed experts |
| MTP (sequential) | Medium | YES (any autoregressive model) | Sequential > parallel for training signal, D=1 is sufficient |
| GRPO | Medium | YES (any RL fine-tuning) | Group statistics replace critic model, halving RL memory |
| FP8 fine-grained quant | Medium | Less relevant for small | 1x128 activation + 128x128 weight blocks, FP32 accumulator every 128 |
| Sigmoid gating | Low-Medium | YES (any MoE) | Independent expert scoring vs zero-sum softmax |
| First-K dense layers | Low | YES (any MoE) | 3 dense layers before MoE stabilizes training |
| Node-limited routing | Low | Infra-specific | M=4 nodes max per token |
| DualPipe | Low | Infra-specific | Bidirectional pipeline scheduling |

---

## Cross-Model Architecture Comparison (2025 State of Art)

From Raschka's comparison, key trends:

| Model | Attention | MoE config | Activation | Pos encoding | Notable |
|-------|-----------|-----------|------------|-------------|---------|
| DeepSeek V3 | MLA | 256R+1S, K=8 | SwiGLU | RoPE+YaRN | Aux-loss-free, MTP, sigmoid gating |
| Llama 4 Maverick | GQA | alternating MoE/dense | SwiGLU | RoPE | 2 active experts, large FFN |
| Qwen3 MoE | GQA | 256R+0S, K=8 | SwiGLU | RoPE | No shared expert (departure from DeepSeek) |
| Qwen3-Next | GatedDeltaNet+GatedAttn (3:1) | 4x more experts, +shared | SwiGLU | Partial RoPE | Hybrid linear+attention, MTP |
| Kimi K2 | MLA | more than V3 | SwiGLU | RoPE | Muon optimizer, scaled V3 design |
| Grok 2.5 | - | 8 large experts +shared | SwiGLU | - | Few-large-experts (older design) |
| GPT-OSS | GQA+sliding | 32 experts, K=4 | - | - | Attention bias, learned attention sinks |
| GLM-4.5 | GQA | +shared expert | - | - | 3 initial dense layers (like V3) |
| Gemma 3 | GQA+sliding (5:1) | dense | GELU | RoPE | 1024-token sliding window |
| OLMo 2 | MHA | dense | SwiGLU | RoPE+QK-Norm | Post-Norm (unusual), QK-Norm |
| SmolLM3 | GQA | dense | - | NoPE (every 4th layer) | No position embeddings in some layers |
| Kimi Linear | GatedDeltaNet+MLA (3:1) | - | - | NoPE in MLA layers | Channel-wise gating (Kimi Delta Attention) |
| MiniMax-M2 | Full attention | ultra-sparse (4.37%) | - | Partial RoPE | Per-layer QK-Norm |
