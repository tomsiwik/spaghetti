# MATH.md — M2P Generation Speed (exp_m2p_generation_speed)

## Overview

Measures the wall-clock latency of the M2P adapter pipeline on Qwen3-0.6B (Apple M5 Pro, MLX).
Two components: (1) isolated M2P forward pass, (2) full pipeline = hidden-state extraction + M2P forward.
The M2P network runs once per prompt (not per token), so its cost is amortized over the generation sequence.

## Definitions

Let:
- d = 1024 (d_model, Qwen3-0.6B)
- L = n_layers = 28
- r = 4 (LoRA rank)
- d_q = 2048, d_v = 1024 (q_proj and v_proj output dimensions)
- B = 400 GB/s (M5 Pro unified memory bandwidth)
- N_M2P = 356,862,976 (M2P v4 parameter count, confirmed empirically)

## Theorem 1 — M2P Forward Pass Latency Lower Bound

**Claim**: For a compute session where all M2P parameters must be read from DRAM,
the forward pass latency satisfies:

```
t_M2P ≥ N_M2P × w / B
```

where w = bytes per parameter (2 for fp16, 4 for fp32).

**Proof**: Any linear operator y = Wx requires reading all entries of W.
The M2P network consists of:
  - enc_linear1: d × 2d = 2,097,152 params
  - enc_linear2: 2d × d = 2,097,152 params
  - b_heads_q:   L × d × (r × d_q) = 234,881,024 params
  - b_heads_v:   L × d × (r × d_v) = 117,440,512 params
  Total: 356,515,840 ≈ 356.5M params (measured: 356,862,976)

Memory transfer: 356.5M × 4 bytes (fp32) = 1.427 GB
At B = 400 GB/s: t_min = 1.427 / 400 = **3.57 ms** (fp32)
At B = 400 GB/s: t_min = 1.427 / 2 = 0.713 GB → t_min = **1.78 ms** (fp16)

In practice, M5 Pro achieves ~70% of peak BW for small transfers.
Effective BW ≈ 280 GB/s → predicted t_M2P ≈ **5–10 ms** (fp32 weights).

**QED (lower bound)**

## Theorem 2 — Hidden State Extraction Latency

**Claim**: Extracting per-layer mean-pooled hidden states requires one complete
base model forward pass. For Qwen3-0.6B with input length T tokens:

```
t_extract ≥ N_base × w / B + O(T² × d × L)
```

For T = 64 and 4-bit quantized weights:
  - N_base ≈ 486M params, w = 0.5 bytes (4-bit) → 243MB
  - t_transfer ≥ 243MB / 400GB/s = **0.6 ms**
  - Attention FLOPs: T² × d × L = 64² × 1024 × 28 = 117M ops → negligible
  - With overhead: predicted t_extract ≈ **10–50 ms** for T=64

Hidden extraction cost is dominated by:
  (a) The functional_attention_forward per layer (28 × attn_fwd)
  (b) MLP forward per layer (28 × mlp_fwd)
  (c) Mean-pool accumulation

**QED**

## Theorem 3 — Pipeline Latency and K947 Feasibility

**Claim**: The total M2P pipeline (hidden extraction + M2P forward), run once per prompt,
satisfies t_pipeline < 100 ms for T ≤ 512 tokens.

**Proof**: From Theorems 1–2:
  t_pipeline = t_extract + t_M2P ≤ 50 ms + 10 ms = 60 ms ≪ 100 ms

In practice Qwen3-0.6B at inference speed (165 tok/s from VISION.md) runs ~6 ms/token.
Pipeline cost of 60 ms ≈ 10 tokens latency overhead at the start of generation.
This is acceptable for typical generation lengths (> 50 tokens).

**Prediction**: K947 PASS (t_M2P < 100 ms, total pipeline < 100 ms)

**QED**

## Quantitative Predictions

| Quantity | Predicted | Basis |
|---|---|---|
| t_M2P forward (isolated, fp32, ms) | 5–10 | BW bound + overhead |
| t_M2P forward (isolated, fp16, ms) | 3–7 | BW halved |
| t_hidden_extraction (T=64, ms) | 15–50 | Full fwd @ 4-bit |
| t_pipeline = t_extract + t_M2P (ms) | 20–60 | Sum of above |
| K947: t_M2P < 100 ms | PASS | Theorem 3 |

## Kill Criteria

- **K947**: M2P forward pass (isolated, 100-call mean) < 100 ms  
  *Fail condition*: M2P alone takes > 100 ms → M2P is unusable at inference time
