# Falcon-E-3B LoRA Composition: Mathematical Foundations

## 1. Mechanism Definition

### 1.1 Falcon-E-3B Architecture

Falcon-E-3B is a Llama-compatible ternary language model from TII (Technology Innovation Institute). Its architecture:

- **Model type:** `LlamaForCausalLM` with `quant_method: "bitnet"`
- **Parameters:** ~3B (864M unique parameters at 1.58-bit, plus embeddings/head in bf16)
- **Hidden dimension:** d = 2048
- **Layers:** L = 32
- **Attention:** GQA with h_q = 16 query heads, h_kv = 2 KV heads, d_k = 128
  - Query: Q in R^{n x 16 x 128}, Key: K in R^{n x 2 x 128}
  - Each KV head serves h_q / h_kv = 8 query heads
  - KV cache: O(n * h_kv * d_k) = O(n * 256) vs O(n * 2048) without GQA (8x reduction)
- **MLP:** SwiGLU with intermediate = 13312
  - gate_proj: R^{2048} -> R^{13312}
  - up_proj: R^{2048} -> R^{13312}
  - down_proj: R^{13312} -> R^{2048}
  - Activation: SiLU(gate(x)) * up(x), then down()
- **Weight format:** uint8-packed ternary with per-tensor scale
  - Packed 4 ternary values per byte: w = {-1, 0, +1} + 1 -> {0, 1, 2}
  - Physical storage: ceil(out_features / 4) x in_features bytes
  - Weight_scale: scalar per matrix

### 1.2 Ternary Unpacking

The packed uint8 weight W_packed in R^{ceil(out/4) x in} encodes 4 ternary values per byte:

```
w_0 = (W_packed & 0x03) - 1        # bits [0:1]
w_1 = ((W_packed >> 2) & 0x03) - 1 # bits [2:3]
w_2 = ((W_packed >> 4) & 0x03) - 1 # bits [4:5]
w_3 = ((W_packed >> 6) & 0x03) - 1 # bits [6:7]
W_unpacked = concat([w_0, w_1, w_2, w_3], axis=0)[:out_features]
W_bf16 = W_unpacked * weight_scale  (or / weight_scale if invert)
```

Memory: 999 MB packed -> ~6.1 GB unpacked to bf16 (6.1x expansion).

### 1.3 LoRA on Ternary Weights

For a linear layer with weight W in R^{out x in}, LoRA adds a low-rank perturbation:

```
y = Wx + scale * (xA)B
```

Where A in R^{in x r}, B in R^{r x out}, and typically r << min(in, out).

For Falcon-E-3B with r = 16:
- q_proj: A in R^{2048 x 16}, B in R^{16 x 2048} -> 65,536 params
- v_proj: A in R^{2048 x 16}, B in R^{16 x 256} -> 36,864 params
- o_proj: A in R^{2048 x 16}, B in R^{16 x 2048} -> 65,536 params

Per layer: 167,936 params. Total 32 layers: 5,373,952 params (0.18% of model).

**Why LoRA works on ternary models:** The base weight W is frozen. LoRA gradients flow
through the continuous bf16 A and B matrices only. The ternary->bf16 unpack creates a
differentiable forward pass. This was validated on BitNet-2B-4T in prior experiments
(PPL improvement: -26.3% at N=5 with 200 training iterations).

### 1.4 Pre-Merge Composition

Given N domain adapters {(A_i, B_i)}_{i=1}^N with uniform weights w_i = 1/N:

```
W_composed = W_base + sum_{i=1}^N w_i * scale * B_i^T @ A_i^T
```

Note on notation: LoRALinear forward computes `y = x @ W^T + scale * (x @ A) @ B`,
so the effective delta in weight-matrix form is: dW = scale * B^T @ A^T.

Composition complexity:
- Per adapter per layer: O(r * d^2) for the matrix product B^T @ A^T
- Total: O(N * L_target * r * d^2) where L_target = number of target projections per layer
- For N=5, L_target=3 (q,v,o), 32 layers: 5 * 3 * 32 * 16 * 2048^2 = ~32B FLOPs
- This is a one-time merge cost at session start; inference is zero-overhead.

## 2. Why It Works

### 2.1 Ternary Base + Continuous Adapters

The key insight from our prior experiments: ternary bases produce lower adapter
interference than fp16 bases. Empirically on BitNet-2B-4T:

- Adapter cosine similarity at convergence: 0.00125 (vs 0.142 on Qwen-7B fp16)
- 5-adapter composed PPL: 4.16 (vs 4.35 for fp16 adapters on same base)

The mechanism: ternary weight matrices have a constrained gradient landscape.
Because W in {-1, 0, +1}^{out x in}, the frozen base provides a highly structured
projection. Adapter updates exist in a lower-dimensional effective subspace,
reducing cross-adapter interference.

### 2.2 Falcon-E-3B vs BitNet-2B-4T

| Property | BitNet-2B-4T | Falcon-E-3B |
|----------|-------------|-------------|
| Parameters | 2.4B | 3B |
| d_model | 2560 | 2048 |
| Layers | 30 | 32 |
| GQA ratio | 24:8 (3:1) | 16:2 (8:1) |
| Intermediate | 6912 | 13312 |
| Published avg | 51.54 | 53.17 |
| Model size | ~700MB | 999MB |
| Unpacked bf16 | ~5.35GB | ~6.1GB |

Falcon-E-3B has 50% more parameters but a smaller hidden dimension. The wider
MLP (13312 vs 6912) means more representational capacity in the feed-forward
layers. The aggressive GQA (8:1) reduces KV cache by 4x compared to BitNet.

### 2.3 Why Falcon-E-3B Should Improve Over BitNet

The competitive benchmark showed BitNet-2B SOLE loses 4/6 vs Qwen-3B.
Three root causes were identified:

1. **MMLU factual knowledge:** BitNet base scores 40% on medical/code.
   Falcon-E-3B is instruction-tuned (Instruct variant), should score higher.

2. **Composition hurts math/legal:** Uniform 1/N degrades specific domains.
   Same risk here, but Falcon's stronger base may absorb the perturbation better.

3. **Memory:** BitNet unpacks to 5.35GB. Falcon unpacks to ~6.1GB, actually
   worse. But Qwen-3B-4bit is only 2.45GB. Memory K3 threshold set at 6GB.

## 3. What Breaks It

### 3.1 Memory Constraint (K3: >6GB -> KILL)

Falcon-E-3B unpacked: ~6.1 GB. During composition, we load N=5 adapter files
plus the base model. Each adapter: ~0.3 MB (rank-16 on q/v/o only).
Peak memory during eval: base model (~6.1GB) + KV cache (~0.3-0.6GB for
seq_len=256) = ~6.7 GB expected. This likely FAILS K3.

However, note: the 6.1GB is the bf16 unpacked size. If we could serve without
unpacking (using native ternary kernels), base memory would be 999MB + adapters +
KV cache = ~1.5GB. The K3 failure is an engineering limitation, not architectural.

### 3.2 Quality Gap (K2: loses >4/6 -> KILL)

Qwen2.5-3B-Instruct was instruction-tuned on massive data including MMLU-style
questions. Falcon-E-3B-Instruct may have similar instruction tuning but at
ternary precision. The 1.63pp benchmark gap (53.17 vs 51.54 for BitNet) may
not be enough to close the gap vs Qwen.

Risk factor: Falcon published 53.17 avg was on their eval suite, not our
specific GSM8K + MMLU subsets. Performance on our exact benchmarks is unknown.

### 3.3 LoRA on Wider MLP

We target only attention (q/v/o). The MLP in Falcon-E-3B is 2x wider than
BitNet (13312 vs 6912). If domain knowledge resides primarily in MLP weights
(as suggested by MoE literature -- DeepSeek-V3 routes through MLP experts),
attention-only LoRA may be insufficient. Adding MLP targets (gate/up/down)
would increase trainable params by ~4x but also training time.

## 4. Assumptions

1. **Falcon-E-3B BitLinear is identical to BitNet BitLinear.**
   Justified: same `quant_method: "bitnet"`, same packed format, same
   mlx-lm `BitLinear` class. Verified empirically: unpack + LoRA trains.

2. **Instruction-tuned Falcon-E-3B retains instruction-following after LoRA.**
   Justified: we train with instruction format data, and LoRA rank-16 on
   attention only perturbs <0.2% of parameters.

3. **Uniform 1/N composition is a valid composition strategy.**
   Partially justified: proven to work on BitNet-2B (-26.3% PPL), but
   also proven to hurt math/legal on MMLU. We use it as a baseline.

4. **Qwen2.5-3B-4bit is the right competitor baseline.**
   Justified: it is the model we previously benchmarked against. 4-bit
   quantization may degrade Qwen performance (published GSM8K ~65-70%,
   we measured 36%), making this a conservative-for-Qwen comparison.

## 5. Complexity Analysis

### Training
- Per adapter: 200 iterations * O(seq_len * d^2 * L) forward/backward
- Memory: ~6.7 GB (model + optimizer state for LoRA params)
- Time: ~2-3 min per adapter on M5 Pro
- Total 5 adapters: ~10-15 min

### Composition (pre-merge)
- One-time: O(N * 3 * 32 * r * d^2) = O(N * 96 * 16 * 2048^2) ~ 10B FLOPs
- Time: <1 second (proven in latency sweep)
- No inference overhead

### Inference
- Same as base model (merged weights)
- Memory: base model + KV cache
- Throughput: same tok/s as base

## 6. Worked Example (d=2048, N=5)

For layer 0, q_proj:
- Base weight: W in R^{2048 x 2048}, ternary, stored as R^{512 x 2048} uint8
- Adapter i: A_i in R^{2048 x 16}, B_i in R^{16 x 2048}
- delta_W_i = scale * B_i^T @ A_i^T = 20.0 * R^{2048 x 16} @ R^{16 x 2048} = R^{2048 x 2048}
- Composed: W_new = W_bf16 + (1/5) * sum_{i=1}^5 delta_W_i
- Each delta has Frobenius norm O(scale * sqrt(r) * ||B_i|| * ||A_i||) << ||W||
- At 1/5 weighting: each adapter contributes 0.2 * 20.0 = 4.0x its learned delta

## 7. Connection to Architecture

This experiment directly addresses the "competitive benchmark" kill result:
BitNet-2B SOLE lost 4/6 vs Qwen-3B. Falcon-E-3B is a stronger ternary base
(+1.63pp average, +300M params, instruction-tuned).

If Falcon-E-3B + composition beats Qwen on >=3/6, it validates the thesis that
ternary base quality (not composition mechanism) was the bottleneck.

If it still loses, the implication is that ternary models fundamentally cannot
match 4-bit quantized fp16 models on factual benchmarks, and the value
proposition must shift to modularity/efficiency rather than raw accuracy.

Production comparison: DeepSeek-V3 uses 256 routed experts in MLP layers with
auxiliary-loss-free load balancing. Our approach uses 5 adapters on attention
layers with uniform routing. The mechanism is fundamentally different (per-token
expert selection vs global adapter merge), but the goal is the same: composable
specialization without full retraining.
