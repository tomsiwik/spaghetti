# MATH.md — Pre-merge vs Runtime LoRA on Qwen3-0.6B

## Problem

Two serving strategies for a LoRA adapter on Qwen3-0.6B-4bit:

1. **Runtime LoRA**: Keep base weights frozen (4-bit), apply `y += scale * (x @ A) @ B`
   as a side-path during every forward pass.
2. **Pre-merge**: Dequantize base weights, add `scale * B.T @ A.T`, re-quantize to 4-bit.
   No side-path at inference.

Kill criteria:
- K952: Pre-merge tok/s ≥ 1.5× runtime LoRA tok/s
- K953: |quality_premerge − quality_runtime| < 1pp on GSM8K

## Architecture Reference (Qwen3-0.6B-4bit, measured)

| Component | Dimensions | Packed bytes (4-bit, group=64) |
|-----------|-----------|-------------------------------|
| q_proj    | 2048×1024 | 2048×128 = 262 KB + scales/biases ≈ 393 KB |
| k_proj    | 1024×1024 | 131 KB + … ≈ 197 KB |
| v_proj    | 1024×1024 | 131 KB + … ≈ 197 KB |
| o_proj    | 1024×2048 | 262 KB + … ≈ 393 KB |
| gate_proj | 3072×1024 | 393 KB + … ≈ 589 KB |
| up_proj   | 3072×1024 | 393 KB + … ≈ 589 KB |
| down_proj | 1024×3072 | 393 KB + … ≈ 589 KB |
| **per layer** | | **≈ 2.95 MB** |
| **28 layers** | | **≈ 82.6 MB weight data** |
| Total model | | **≈ 340 MB measured** |

---

## Theorem 1 — Pre-merge Speedup Upper Bound

**Theorem:** For rank-r LoRA applied to q_proj and v_proj on Qwen3-0.6B (28 layers,
d_model=1024, d_q=2048, d_v=1024) in bfloat16, the decode-throughput speedup of
pre-merge over runtime LoRA is bounded by:

```
speedup ≤ 1 + LoRA_BW / Base_BW
```

where BW denotes memory bandwidth consumed per decode step (bandwidth-bound regime
on Apple Silicon M5 Pro), and:

```
LoRA_BW = n_layers × [2 × r × d_in + 2 × r × d_q + 2 × r × d_in + 2 × r × d_v] × sizeof(bf16)
         = 28 × [(2×r×1024 + 2×r×2048) + (2×r×1024 + 2×r×1024)] × 2
         = 28 × r × [6144 + 4096] × 2
         = 28 × r × 20480 bytes
```

For rank r=8:
```
LoRA_BW = 28 × 8 × 20480 = 4.59 MB
Base_BW ≈ 340 MB (measured)
speedup ≤ 1 + 4.59/340 ≈ 1.013
```

**Proof:**

Step 1 (Bandwidth-bound regime): On M5 Pro (memory bandwidth ≈ 273 GB/s peak,
~270 GB/s measured), single-token decode for a 0.6B model reads essentially all
model weights per token. Arithmetic intensity = 2 FLOPs/byte (one multiply-add
per weight byte), which is far below the GPU's compute-to-bandwidth ratio.
Therefore inference is bandwidth-bound: t_decode ∝ BW_consumed.

Step 2 (Base bandwidth): Pre-merge 4-bit forward = runtime LoRA 4-bit forward
for the base weights (same 340 MB read). The only difference is the LoRA side-path.

Step 3 (LoRA side-path bandwidth): Runtime LoRA reads A ∈ R^{d_in×r} and
B ∈ R^{r×d_out} in bf16 per projection per decode step. For Q and V:
- q_proj: A=(1024×r) + B=(r×2048) → r × (1024+2048) × 2 bytes = r × 12 KB
- v_proj: A=(1024×r) + B=(r×1024) → r × (1024+1024) × 2 bytes = r × 4 KB
- Per layer: r × 16 KB
- 28 layers: r × 448 KB

Step 4 (Speedup formula): Under bandwidth-bound assumption,
```
speedup = (Base_BW + LoRA_BW) / Base_BW = 1 + LoRA_BW/Base_BW
```
For r=8: speedup = 1 + 3.58/340 = 1.011

**Quantitative predictions (K952 kill check):**

| rank | LoRA_BW | speedup | K952 (≥1.5x) |
|------|---------|---------|--------------|
| 4    | 1.79 MB | 1.005x  | FAIL         |
| 8    | 3.58 MB | 1.011x  | FAIL         |
| 16   | 7.17 MB | 1.021x  | FAIL         |
| 32   | 14.3 MB | 1.042x  | FAIL         |
| 128  | 57.3 MB | 1.169x  | FAIL         |
| 500+ | 224 MB  | 1.66x   | PASS (impractical) |

**QED: K952 is predicted to FAIL for any practical LoRA rank (r ≤ 128).**

The break-even rank for 1.5x speedup requires LoRA_BW ≥ 0.5 × Base_BW = 170 MB,
which means r ≥ 170MB / (28 × 16KB) ≈ 380. This is larger than d_model/3 —
not a LoRA adapter but a full fine-tuned model.

---

## Theorem 2 — Pre-merge 4-bit Quality Preservation

**Theorem:** For rank-r LoRA with LoRALinear.fuse() on Qwen3-0.6B-4bit
(group_size=64, bits=4), the re-quantization error satisfies:

```
E[|W_fused_4bit - W_fused_bf16|] ≤ Δ_q / 2
```

where Δ_q = weight_range / 15 is the quantization step size. If the LoRA delta
magnitude ‖Δ‖_∞ << Δ_q, the quality degradation is negligible (< 0.5pp).

**Proof:**

Step 1 (Quantization step size): MLX 4-bit affine quantization maps each group of
g=64 values to integers {0,...,15}. The scale is:
```
scale = (max(W) - min(W)) / 15
```
For typical Qwen3-0.6B weights, max|W| ≈ 0.3, so Δ_q ≈ 0.6/15 = 0.04.

Step 2 (LoRA delta magnitude): After training with LoRALinear (scale=20, B init=0,
A~U(-1/32, 1/32)):
- Standard training dynamics: B grows to ~0.005–0.02 std per element
- Per-element delta: scale × B_ij × Σ_k A_ik ≈ 20 × 0.01 × (1/32) ≈ 0.006

Maximum delta (pessimistic): 0.006 × sqrt(rank) × 3σ ≈ 0.051 for r=8
This is comparable to Δ_q=0.04, meaning about 50% of fused weights shift by 1 step.

Step 3 (Quality impact): Re-quantization at 1 step introduces uniform noise in
[-Δ_q/2, Δ_q/2] for affected weights. Since errors are independent across
groups and layers, they average out (CLT). Empirical calibration from Finding #289
shows that LoRA scale=20 with large deltas (range ±24) destroys quality, but
for well-scaled adapters (delta < 2× Δ_q), quality loss is < 1pp.

Note: Finding #289 used ternary base (Δ_q ≈ 1.2) with delta ≈ 24 → ratio 20×.
Here 4-bit has Δ_q ≈ 0.04 and delta ≈ 0.006 → ratio 0.15×. Much safer.

**Quantitative prediction:**

| scale | E[‖delta‖_∞] | delta/Δ_q | K953 (< 1pp) |
|-------|-------------|-----------|--------------|
| 20    | 0.006       | 0.15×     | PASS         |
| 20, large B (0.1 std) | 0.06 | 1.5× | PASS (marginal) |

**QED: K953 is predicted to PASS for standard LoRA scales (≤20) on 4-bit weights.**

---

## Theorem 3 — Pre-merge Fuse Cost (One-Time Overhead)

**Theorem:** The merge overhead (LoRALinear.fuse() for all layers) is bounded by:

```
t_merge ≤ n_projs × (t_dequant + t_add + t_requant)
```

For 28 layers × 2 projections = 56 fuse operations:
- t_dequant: dequantize (output, packed_input) → bf16: ~0.5ms per op
- t_add: add delta (rank-r correction): ~0.1ms per op
- t_requant: re-quantize to 4-bit: ~0.5ms per op
- Total: 56 × 1.1ms ≈ 61ms

This is a ONE-TIME cost per request (before generation), not per-token.

**QED: Merge cost ≈ 61ms per request. Not amortized over short responses (< 60 tokens).**

---

## Summary of Predictions

| Kill Criterion | Prediction | Basis |
|----------------|------------|-------|
| K952: speedup ≥ 1.5× | **FAIL** (actual ≈ 1.01–1.02×) | Theorem 1: LoRA BW < 1% of base |
| K953: quality < 1pp | **PASS** | Theorem 2: delta/Δ_q ≈ 0.15× |

## Behavioral Implication

K952 FAIL means runtime LoRA has negligible overhead for practical ranks.
The finding will be: **use runtime LoRA** — same speed as pre-merge, adapter flexibility
(swap without re-merging), and no quantization quality risk.

This directly informs the M2P serving architecture: runtime LoRA injection at ~0.26ms
(Finding #388) and negligible decode overhead is the correct production strategy.
Pre-merge is only advantageous at ranks > 380, which is not a LoRA adapter.
