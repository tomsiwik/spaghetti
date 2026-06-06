# ReLoRA Composition Macro: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 3584 (Qwen2.5-7B) |
| d_ff | FFN intermediate dimension | 18944 |
| d_kv | KV head dimension | 128 |
| n_heads | Number of attention heads | 28 |
| n_kv_heads | Number of KV heads | 4 (GQA) |
| L | Number of transformer layers | 28 |
| r | LoRA rank | 16 |
| alpha | LoRA scaling factor | 16 (scaling = alpha/r = 1.0) |
| K | Number of ReLoRA merge cycles | 3 |
| N | Number of domain experts | 5 |
| M | Number of LoRA target modules per layer | 7 (q/k/v/o/gate/up/down) |

## 2. Dimensionality of Expert Delta Vectors

### 2.1 Per-Module Delta Dimensions

For Qwen2.5-7B with GQA (28 heads, 4 KV heads, head_dim=128):
- hidden_size = 3584
- intermediate_size = 18944
- num_attention_heads = 28, num_key_value_heads = 4

| Module | Weight Shape (d_out x d_in) | Delta Elements |
|--------|---------------------------|----------------|
| q_proj | 3584 x 3584 | 12,845,056 |
| k_proj | 512 x 3584 | 1,835,008 |
| v_proj | 512 x 3584 | 1,835,008 |
| o_proj | 3584 x 3584 | 12,845,056 |
| gate_proj | 18944 x 3584 | 67,895,296 |
| up_proj | 18944 x 3584 | 67,895,296 |
| down_proj | 3584 x 18944 | 67,895,296 |
| **Per layer** | | **233,046,016** |

### 2.2 Full Expert Delta Vector

The flattened expert delta across all L=28 layers and M=7 modules:

    D = L * M * avg_module_size
      = 28 * 233,046,016
      = 6,525,288,448 elements

This is the dimensionality of the space in which we measure cosine similarity.

### 2.3 Random Baseline Cosine

For random vectors in R^D with D ~ 6.5 billion:

    E[|cos|] ~ sqrt(2 / (pi * D))
             ~ sqrt(2 / (pi * 6.5e9))
             ~ 3.1e-5

This is the expected cosine between two random vectors of this dimensionality.
At micro scale (D ~ 131K), we had E[|cos|] ~ 0.0039. The macro baseline is
~125x smaller, consistent with the sqrt(1/D) scaling.

## 3. ReLoRA Base Perturbation at Macro Scale

### 3.1 QLoRA Constraint

With QLoRA, base weights are quantized to 4-bit (nf4). We cannot losslessly
merge LoRA deltas into the quantized base and restart. True ReLoRA with
K merge cycles would require K quantization-dequantization roundtrips,
each introducing noise.

**Our approach:** Train a single LoRA adapter on mixed domain data for
150 steps (equivalent to 3 cycles x 50 steps). This creates a rank-16
perturbation of the base weights. For testing COMPOSITION, what matters
is that the base is perturbed; whether the perturbation came from iterative
merge-restart or continuous training is irrelevant.

When training domain experts on the "ReLoRA base":
1. Load base model (4-bit quantized)
2. Load and merge the ReLoRA adapter (PEFT merge_and_unload at fp16)
3. Add fresh LoRA for expert training

The expert sees: Q(W_0 + dW_relora) + dW_expert, where Q() is the nf4
requantization after merge.

### 3.2 Rank Analysis

The ReLoRA perturbation is a single rank-16 LoRA applied to 7 modules
per layer across 28 layers:

    rank(dW_relora) = 16 per module

For each module, this modifies 16 out of min(d_out, d_in) directions:
- k_proj, v_proj: 16/512 = 3.1% of directions
- q_proj, o_proj: 16/3584 = 0.45%
- gate/up/down: 16/3584 = 0.45%

### 3.3 Expected Impact on Expert Orthogonality

**Argument for NO (hypothesis):** Expert LoRA deltas are determined by
the gradient landscape around the base weights. A rank-16 perturbation
out of rank-3584 changes <0.5% of weight space directions. The gradient
landscape for domain-specific training is overwhelmingly determined by
the original pretrained structure. Expert deltas should be similarly
oriented on both bases.

**Argument for YES (counter):** The ReLoRA adapter is trained on mixed
domain data. If it aligns with common domain directions, it could create
correlated structure that biases subsequent expert training, manifesting
as increased inter-expert cosine similarity.

## 4. Kill Criteria Derivation

### 4.1 K1: cos_ratio > 5x

From micro (d=64): cos_ratio = 1.77x (CI [0.77, 2.64]).

If composition degradation scales with K/d:
- Micro: K=5, d=64, K/d = 0.078
- Macro: K=3, d=3584, K/d = 0.00084

The macro K/d ratio is 93x smaller. If cos_ratio is proportional to K/d,
we expect cos_ratio ~ 1.01x at macro. Even with nonlinear effects,
5x is a generous kill threshold.

### 4.2 K2: loss_ratio > 1.20

From micro: loss_ratio = 1.052 (CI [1.041, 1.074]).

The micro loss gap decomposes as:
- Base quality gap: ~4.6%
- Composition penalty: ~0.6%

At macro scale, we expect:
- Base quality gap: <2% (ReLoRA paper shows gap closes with scale)
- Composition penalty: should shrink with higher dimensionality

A 20% loss threshold (1.20x) allows substantial room for unexpected
degradation while still being meaningful.

### 4.3 K3: base quality gap > 10%

This directly tests whether ReLoRA pretraining itself is viable at 7B
scale. The ReLoRA paper reports near-parity at 1.3B. At 7B, we use
a pre-trained model + short ReLoRA continuation, so the gap should
be minimal.

## 5. Experimental Design

### 5.1 ReLoRA Base Perturbation

- Start from Qwen2.5-7B (pretrained, frozen 4-bit)
- Single LoRA adapter trained on mixed domain data, 150 steps
- Training data: mix of all 5 domains (simulates general pretraining)
- LR: 4e-4 (2x standard), cosine schedule with warmup

This creates a rank-16 perturbation. We use a single training pass
(not iterative merge-restart) because:
1. QLoRA cannot do lossless merge-and-restart
2. The experiment tests composition geometry, not ReLoRA training quality
3. A rank-16 perturbation is sufficient to test whether base modification
   changes expert composition behavior

### 5.2 Expert Training

- 5 domains: math, python, sql, medical, bash
- 100 steps per expert (composition measurement, not quality)
- Same LoRA config for both conditions
- Same training data and hyperparameters

### 5.3 Composition Metrics

1. **Pairwise cosine similarity** of flattened expert deltas
   - Expected conventional: ~0.0001-0.001 (extrapolating from d=896 result)
   - Expected ReLoRA: similar (if hypothesis holds)

2. **Per-module breakdown** (attention vs FFN)
   - From micro/ffn_only_vs_all_modules: attention cos=0.85 vs FFN cos=0.59
   - This decomposition reveals where any degradation originates

3. **Expert quality** (eval loss on held-out data)
   - Direct comparison of expert utility

## 6. Worked Example

### Setup
- Module: gate_proj (18944 x 3584)
- LoRA rank r=16, alpha=16

### Expert delta
- A in R^{3584 x 16}, B in R^{16 x 18944}
- Delta dW = (16/16) * B @ A = B @ A
- rank(dW) <= 16
- Elements in dW: 18944 * 3584 = 67,895,296

### Cosine between two experts
- Expert delta dim for gate_proj alone: 67.9M
- Across all modules and layers: ~6.5B elements
- Random baseline cos: ~3.1e-5
- Observed at d=896 (FINDINGS.md): cos ~ 0.0002
- Expected at d=3584: ~0.0001 (scaling as ~1/sqrt(D))

### Kill threshold
- K1 threshold: conventional * 5 = 0.0005
- If observed ReLoRA cos < 0.0005: SURVIVES
- If observed ReLoRA cos > 0.0005: KILLED

## 7. Assumptions

1. **QLoRA quantization noise is small.** nf4 preserves ~99.6% of
   information. A single merge (ReLoRA adapter into base) introduces
   one quantization roundtrip, <0.5% error.

2. **Short training is sufficient.** 100 steps produces measurable
   LoRA deltas even if experts don't fully converge. The cosine
   metric measures direction, not magnitude.

3. **Domain data is representative.** The 5 domains span code (python,
   sql, bash), reasoning (math), and knowledge (medical).

4. **Same-seed expert training.** Both conditions use identical seeds,
   data order, and hyperparameters. The ONLY difference is the base
   weights.

5. **Single LoRA pass approximates ReLoRA.** For testing composition
   geometry, a single-pass rank-16 perturbation is equivalent to
   iterative merge-restart. Both produce a low-rank modification
   of the pretrained weights. The difference is that true ReLoRA
   accumulates higher effective rank over K cycles; our single pass
   stays at rank-16. If anything, this makes our test conservative
   (less perturbation = easier to survive).

6. **merge_and_unload() is accurate.** PEFT's merge operation adds
   the fp16 LoRA delta to the dequantized base weights. Any
   quantization error from re-quantization is small and symmetric.
