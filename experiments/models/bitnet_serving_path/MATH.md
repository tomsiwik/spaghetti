# LoTA-QAF Merge for BitNet-SOLE: Mathematical Analysis

## Setup

**Base model**: BitNet-b1.58-2B-4T with ternary weights
- Each weight matrix W has packed ternary form W_int in {-1, 0, 1}
- Per-tensor scale alpha = mean(|W_float|) (learned during pretraining)
- Physical weights: W_float = W_int * alpha (or W_int / alpha, depending on invert flag)
- Dimensions: d = 2560, L = 30 layers, 7 projections per layer (q, k, v, o, gate, up, down)

**LoRA adapters**: Trained with STE ternary quantization
- A in R^{d_in x r}, B in R^{r x d_out}, r = 16
- Forward pass: y = x @ W.T + x @ Q(A) @ Q(B) * s, where s = 20.0 (lora_scale)
- Q(W) = clip(round(W / mean(|W|)), -1, 1) * mean(|W|) (STE ternary quantization)
- Stored weights are the latent FP32 parameters, not the quantized values

## LoTA-QAF Merge Principle (arxiv 2505.18724)

Given a ternary base W_int and ternary adapter matrices A_T, B_T (both in {-1, 0, 1}):
- Adapter delta: DeltaW = B_T^T @ A_T^T * s -- this is an INTEGER matrix in [-r, r] times a scalar
- Merge: W'_float = W_float + DeltaW_float
- Requantize: alpha' = mean(|W'_float|), W'_int = clip(round(W'_float / alpha'), -1, 1)
- Serve: W'_float = W'_int * alpha' (back on the ternary grid)

**Key assumption**: DeltaW must be large enough relative to the quantization grid spacing to actually change ternary states.

## Why LoTA-QAF Fails for Our Adapters

### Scale Analysis

For a typical layer (layer 0, gate_proj):
- Weight dimensions: (6912, 2560)
- alpha (weight scale) = 1.555
- W_float values in {-1.555/1.555, 0, 1.555/1.555} = {-1/alpha, 0, 1/alpha} or {-alpha, 0, alpha}
- Actual mean(|W_float|) = 0.926

LoRA adapter statistics:
- alpha_A = mean(|A|) = 0.012
- alpha_B = mean(|B|) = 0.006
- After STE: A_q = Q(A) with values in {-0.012, 0, 0.012}, B_q in {-0.006, 0, 0.006}

Delta: DeltaW = B_q^T @ A_q^T * 20.0
- Integer part: B_int^T @ A_int^T in [-16, 16] (since r = 16)
- Scalar part: alpha_A * alpha_B * s = 0.012 * 0.006 * 20.0 = 0.00144
- Mean |DeltaW| = 0.004
- Mean |W_float| = 0.926

**Delta-to-weight ratio: 0.004 / 0.926 = 0.0043 (0.43%)**

### Requantization Grid

The requantization step computes:
```
W'_int = clip(round(W'_float / alpha'), -1, 1)
```

Where alpha' ~ alpha (since DeltaW << W). The grid spacing is alpha' ~ 0.926.

For a weight originally at W_float = alpha (the +1 state):
- W'_float = alpha + delta, where |delta| ~ 0.004
- W'_float / alpha' ~ 1 + 0.004/0.926 ~ 1.004
- round(1.004) = 1 (no change)

For a weight at W_float = 0 (the 0 state):
- W'_float = 0 + delta ~ 0.004
- W'_float / alpha' ~ 0.004/0.926 ~ 0.004
- round(0.004) = 0 (no change)

**Result: 0% of ternary states change. The adapter is completely erased.**

### Condition for LoTA-QAF to Work

A ternary state flips when |DeltaW_ij| > alpha'/2 = 0.463.

Current delta magnitude: 0.004. Required: 0.463.
**The delta must be ~116x larger for ANY weight to flip.**

This would require either:
1. alpha_A * alpha_B ~ 116x larger (adapter weights must be ~10x larger each)
2. lora_scale ~ 116x larger (s ~ 2300 instead of 20)
3. Rank r ~ 116x larger (r ~ 1856 -- defeats the purpose of low-rank)

None of these are practical. The LoTA-QAF approach is fundamentally mismatched with standard LoRA training on BitNet.

## bfloat16 Precision Limit for Float Merge

Even without requantization, merging in bfloat16 has precision issues.

bfloat16 format: 1 sign + 8 exponent + 7 mantissa bits.
For values near |W| = 0.926 (exponent ~ -1):
- ULP (unit in last place) = 2^{-1} * 2^{-7} = 0.00391

Our delta mean: 0.004 = 1.02 ULP.

**The adapter delta is at exactly the bfloat16 precision boundary.**

This means:
- ~50% of deltas are rounded away during bfloat16 addition
- Runtime LoRA avoids this by computing x @ A @ B at activation scale, then adding to output
- The output-space contribution (mean 0.228) is much larger than the weight-space delta (0.004) due to input amplification

### Quantitative Verification

Single-layer comparison (layer 0, gate_proj):
- Runtime LoRA output: mean |lora_out| = 0.228
- Base output: mean |base_out| = 47.02
- Pre-merged output difference: mean |merged - runtime| = 0.069
- Relative error: 0.069 / 47.02 = 0.15%

This 0.15% per-layer error accumulates across 30 layers and 7 projections = 210 merge points.

## Float32 Merge: The Viable Alternative

If weights are stored in float32 instead of bfloat16:
- ULP near 0.926 = 2^{-1} * 2^{-23} = 5.96e-8
- Delta (0.004) = 67,000 ULP -- no precision loss
- Memory cost: 2x bfloat16 = ~8 GB (from ~4 GB)

This is the correct serving path: merge in float32, then either:
(a) Serve directly in float32 (higher memory, no precision loss)
(b) Serve in bfloat16 with the understanding that ~50% of delta is lost
(c) Serve with runtime LoRA (original approach, no merge needed)

## Implications for BitNet-SOLE Serving

| Serving Path | PPL Quality | Memory | Latency | Feasible? |
|-------------|-------------|--------|---------|-----------|
| Runtime LoRA (unpacked bf16) | Reference | ~4 GB + adapters | ~13 tok/s | YES (proven) |
| Float merge (bf16) | Degraded (~50% delta loss) | ~4 GB | ~13 tok/s | Marginal |
| Float merge (fp32) | Near-reference | ~8 GB | ~8 tok/s (est.) | YES (trades memory) |
| LoTA-QAF requantize | Catastrophic (adapter erased) | ~0.5 GB (packed) | ~30 tok/s (est.) | NO |
| bitnet.cpp (packed ternary) | Base only (no adapters) | ~0.5 GB | ~35 tok/s | Only without adapters |

The fundamental tension: BitNet's serving advantage (packed ternary, integer arithmetic) requires staying on the ternary grid, but LoRA deltas are too small to modify the grid. Serving with adapters requires unpacking to float, which sacrifices BitNet's efficiency.

## Conclusion

LoTA-QAF merge is not viable for SOLE adapters on BitNet-2B-4T because:
1. Standard LoRA deltas are 116x too small to flip ternary states
2. Even float merge in bfloat16 loses ~50% of the delta
3. The only lossless path is runtime LoRA with unpacked float weights

This means BitNet-SOLE serving must use one of:
- Runtime LoRA in unpacked float (proven, 13 tok/s, ~4 GB)
- Specialized quantization-aware adapter training that produces grid-aligned deltas
- A fundamentally different merge strategy (e.g., scale-matching before merge)
