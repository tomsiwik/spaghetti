# ReLoRA From-Scratch Composition: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 768 |
| d_ff | FFN intermediate dimension | 3072 (4*d) |
| L | Number of transformer layers | 12 |
| n_heads | Number of attention heads | 12 |
| d_head | Head dimension | 64 (d/n_heads) |
| V | Vocabulary size | 50257 (GPT-2 BPE) |
| r | LoRA rank for experts | 16 |
| alpha | LoRA scaling | 16 (scaling=1.0) |
| r_relora | LoRA rank for ReLoRA updates | 128 |
| K | Number of ReLoRA merge cycles | 16 |
| T_warmup | Full-rank warmup steps | 2000 |
| T_relora | Steps per ReLoRA cycle | 500 |
| T_total | Total training steps | T_warmup + K * T_relora = 10000 |
| N | Number of domain experts | 5 |
| M | LoRA target modules per layer | 7 (q/k/v/o/gate/up/down for GPT-2: q/k/v/o/fc1/fc2 = 6) |

## 2. Model Architecture (~125M params)

### 2.1 Parameter Count

| Component | Shape | Parameters |
|-----------|-------|-----------|
| Token embedding | V x d = 50257 x 768 | 38,597,376 |
| Position embedding | T_ctx x d = 1024 x 768 | 786,432 |
| Per-layer attention (QKV+O) | 4 * d^2 = 4 * 768^2 | 2,359,296 |
| Per-layer FFN (fc1+fc2) | 2 * d * d_ff = 2 * 768 * 3072 | 4,718,592 |
| Per-layer LayerNorm (x2) | 2 * 2 * d | 3,072 |
| **Per layer total** | | **7,080,960** |
| 12 layers | | 84,971,520 |
| Final LayerNorm | 2 * d | 1,536 |
| LM head (tied) | 0 (tied with token emb) | 0 |
| **Total** | | **~124M** |

### 2.2 Effective Rank Accumulation

After K merge cycles with rank r_relora each, the accumulated weight
perturbation has effective rank:

    rank_eff(Delta_K) <= K * r_relora = 16 * 128 = 2048

For d=768: this means the accumulated perturbation can span the FULL
weight space (since 2048 > 768). This is the key insight from Lialin et
al.: iterative low-rank updates achieve full-rank training.

Actually, for weight matrices of shape (d_out, d_in):
- Q/K/V/O: (768, 768), rank <= 768. K*r = 2048 > 768, so full rank is achievable.
- fc1: (3072, 768), rank <= 768. Same conclusion.
- fc2: (768, 3072), rank <= 768. Same conclusion.

**Conclusion**: At K=16, r_relora=128, every weight matrix can be fully
determined by the accumulated ReLoRA updates. The model is not
rank-constrained.

## 3. FLOP-Matched Training

### 3.1 Conventional Training FLOPs

Per token, per layer, the dominant costs are:
- Attention: 4 * d^2 + 2 * T * d (for context length T)
- FFN: 2 * d * d_ff + 2 * d * d_ff = 4 * d * d_ff

Simplifying (ignoring context-dependent terms):
    FLOPs_per_token_per_layer ~ 8 * d^2 + 4 * d * d_ff
                               = 8 * 768^2 + 4 * 768 * 3072
                               = 4,718,592 + 9,437,184
                               = 14,155,776

    FLOPs_per_token = L * FLOPs_per_token_per_layer + 2 * V * d
                    ~ 12 * 14,155,776 + 2 * 50257 * 768
                    ~ 169,869,312 + 77,194,752
                    ~ 247M FLOPs/token

For batch_size=8, seq_len=512, T_total=10000:
    Total tokens = 8 * 512 * 10000 = 40.96M tokens
    Total FLOPs ~ 40.96M * 247M = ~10.1 TFLOPs (forward pass)
    Including backward: ~30.3 TFLOPs

### 3.2 ReLoRA Training FLOPs

During the warmup phase (2000 steps): same as conventional.
During ReLoRA phase (8000 steps): LoRA reduces trainable params but
forward pass is identical (LoRA adds negligible overhead). The FLOP
budget is matched because both conditions train for the same number
of steps on the same data.

The difference: during ReLoRA, only rank-128 LoRA parameters receive
gradients. This reduces backward-pass memory but not FLOPs (the
gradient computation through frozen weights still occurs).

### 3.3 Cost Estimate

A5000 throughput for 125M model: ~5000-8000 tokens/sec (mixed precision).
Total tokens: 40.96M.
Estimated time: 40.96M / 6000 ~ 6800 seconds ~ 1.9 hours.
Add expert training (5 domains x 2 conditions x 500 steps): ~30 min.
Total: ~2.5 hours. At $0.16/hr: ~$0.40.

## 4. Expert Composition Measurement

### 4.1 Expert Delta Vector Dimension

For GPT-2 style with M=6 target modules per layer:
| Module | Shape | Elements |
|--------|-------|----------|
| q_proj | d x d = 768 x 768 | 589,824 |
| k_proj | d x d | 589,824 |
| v_proj | d x d | 589,824 |
| out_proj | d x d | 589,824 |
| fc1 | d_ff x d = 3072 x 768 | 2,359,296 |
| fc2 | d x d_ff = 768 x 3072 | 2,359,296 |
| **Per layer** | | **7,077,888** |

    D = L * 7,077,888 = 12 * 7,077,888 = 84,934,656

### 4.2 Random Baseline Cosine

    E[|cos|] ~ sqrt(2 / (pi * D))
             = sqrt(2 / (pi * 84,934,656))
             = 8.66e-5

### 4.3 Expected Expert Cosine

From prior experiments:
- d=64: mean|cos| ~ 0.03 (micro)
- d=896: mean|cos| ~ 0.0002 (macro, Qwen2.5-0.5B proxy)
- d=3584: mean|cos| ~ 0.18 (macro, Qwen2.5-7B, real domain data)

At d=768, we expect mean|cos| between micro and macro values,
dominated by domain structure rather than random alignment.

### 4.4 Kill Threshold

K1: cos_ratio = relora_cos / conv_cos > 5.0 -> KILLED
K2: loss_ratio = relora_loss / conv_loss > 1.20 -> KILLED
K3: ReLoRA base perplexity / conv base perplexity > 1.20 -> KILLED

## 5. ReLoRA Merge-Restart Protocol

At each merge cycle k = 1, ..., K:

1. **Merge**: W_base += (alpha/r) * B_k @ A_k
2. **Reset LoRA**: A_{k+1} ~ N(0, sigma), B_{k+1} = 0
3. **Reset optimizer**: zero all momentum/variance for LoRA params
4. **Warmup LR**: cosine restart from 0 to lr_max over warmup_steps

After K cycles:
    W_final = W_0 + sum_{k=1}^{K} (alpha/r_k) * B_k @ A_k

Where W_0 is the warmup-phase trained weights. Each term B_k @ A_k
has rank <= r_relora. The sum has rank <= K * r_relora.

## 6. Domain Expert Training

After base training (ReLoRA or conventional), freeze all base weights
and train rank-16 LoRA adapters on domain-specific data.

Expert delta for domain i:
    Delta_i = (alpha/r) * B_i @ A_i, where A_i in R^{d_in x r}, B_i in R^{r x d_out}

Flattened across all modules and layers: Delta_i in R^D, D = 84,934,656.

Pairwise cosine between experts i, j:
    cos(Delta_i, Delta_j) = <flat(Delta_i), flat(Delta_j)> / (||flat(Delta_i)|| * ||flat(Delta_j)||)

## 7. Worked Example

### Module: fc1 of layer 0 (3072 x 768)

ReLoRA cycle 1:
- A_1 in R^{768 x 128}, B_1 in R^{128 x 3072} (initially B=0)
- After 500 steps: ||B_1 @ A_1|| ~ 0.1 (typical)
- Merge: W += B_1 @ A_1. rank(Delta_1) <= 128.

After K=16 cycles:
- rank(sum of deltas) <= 16 * 128 = 2048 > 768
- The accumulated perturbation can have full rank = 768

Expert LoRA:
- A_expert in R^{768 x 16}, B_expert in R^{16 x 3072}
- Delta_expert = B_expert @ A_expert. rank <= 16.
- Elements: 3072 * 768 = 2,359,296

Two experts on same base:
- cos(Delta_1, Delta_2) expected ~ 0.001-0.01 (d=768, real domains)
- Kill if ReLoRA cos / conv cos > 5

## 8. Assumptions

1. **Full-rank warmup is essential.** ReLoRA requires initial full-rank
   training to establish the loss landscape structure. We use 2000 steps
   (20% of total) following Lialin et al.'s recommendation.

2. **Rank-128 per cycle is sufficient.** For d=768, r=128 provides
   128/768 = 16.7% of weight space per cycle. With K=16 cycles, total
   rank coverage exceeds full rank.

3. **Domain data quality.** Expert training uses the same 5-domain
   distillation data from the pilot 50 experiment (if available on
   RunPod) or generated on-the-fly from C4/SlimPajama domain subsets.

4. **GPT-2 architecture is representative.** At d=768 with standard
   attention, the composition dynamics should be qualitatively similar
   to Qwen2.5 at the same hidden dimension. Architecture-specific
   features (GQA, SwiGLU) may quantitatively differ but should not
   change the fundamental orthogonality properties.

5. **Same-seed comparison.** Both conditions use identical random seeds,
   data order, and hyperparameters (except for the ReLoRA merge-restart
   mechanism). This isolates the effect of iterative low-rank training.
