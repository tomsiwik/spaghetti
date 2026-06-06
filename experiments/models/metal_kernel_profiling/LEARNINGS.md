# LEARNINGS: Metal Kernel Profiling

## Status: PROVISIONAL (profiling experiment, no formal proof)

## What We Learned

### 1. BitNet-2B-4T achieves 73% bandwidth utilization on M5 Pro (1.4x gap, NOT 3.5x)

Measured: 165.6 tok/s with KV cache.
Theoretical (corrected): 228 tok/s (1.18 GB packed model / 268.6 GB/s bandwidth).
Actual gap: 1.4x.

The prior "3.5x gap" claim was wrong — it either used the wrong model size (1.7 GB unpacked
vs 1.18 GB packed) or didn't use KV cache. The inference_speed_10x experiment had already
measured 171.8 tok/s at 74.2% utilization. This experiment confirms that number.

### 2. mx.compile gives 21% speedup at seq=1 only

| Seq Length | Speedup |
|-----------|---------|
| 1 | 1.21x |
| 64 | 1.01x |
| 256 | 1.00x |

At seq=1 (token decode), dispatch overhead matters because each layer's matmul is tiny.
mx.compile fuses operations and reduces dispatches. At longer sequences, compute dominates.

**Actionable:** Use `mx.compile(model)` in the serving pipeline. It's a one-line change
that gives 21% decode speedup.

### 3. Eval boundary overhead is only 5%

32 per-layer eval sync points add only 5% overhead vs a single lazy eval.
Not worth optimizing — the lazy graph scheduler already does well.

### 4. Memory bandwidth peaks at 269 GB/s on M5 Pro

| Tensor Size | Bandwidth |
|------------|-----------|
| <10 MB | Dispatch-bound (10-120 GB/s) |
| 100 MB | 233 GB/s (87%) |
| 500 MB+ | 256-269 GB/s (95-100%) |

**Implication for adapter composition:** Individual adapter weights (~80 KB for routing
heads, ~1.9 KB for ternary LoRA) are in the dispatch-bound regime. This supports
pre-merge over runtime LoRA: merge all adapter weights into the base model at composition
time (one large tensor read) rather than reading many small adapter tensors at runtime.

### 5. Ternary unpacking costs ~125ms (prefill only, not decode)

BitLinear packed ternary → bf16 unpacking costs 0.59ms per projection, ~4.2ms per layer,
~125ms total for 30 layers. This only matters for prefill (processing the full prompt).
For token-by-token decode with KV cache, MLX uses the packed format directly.

### 6. LM head is 35% of seq=1 decode time

The tied embedding projection (vocab_size=32000 → d=2560) takes 2.5ms out of 7.3ms.
This is a single large matmul, bandwidth-bound. No optimization possible without
reducing vocabulary or model dimension.

### 7. The "component breakdown" approach is misleading

Timing each layer individually with `mx.eval` introduces serialization. Sum of parts
exceeds the whole (135% vs 100%) because the lazy graph scheduler overlaps dispatch with
compute in the unfused case. Component timing is useful for understanding proportional
contribution but not for absolute bottleneck identification.

## Confirming Evidence

- inference_speed_10x: 171.8 tok/s, 74.2% utilization (consistent)
- M5 Pro peak bandwidth ~269 GB/s (consistent with Apple specs)
- mx.compile impact is dispatch-bound (seq=1 only)

## Contradicting Evidence

- Paper initially claimed 1.05x of theory (wrong model size)
- Prior "3.5x gap" was a measurement artifact

## Follow-up

1. **Integrate mx.compile into serving pipeline** (one-line change, 21% decode speedup)
2. **Profile with adapter composition** — does pre-merged model maintain the same 73% utilization?
3. The remaining 27% gap (73% utilization) may come from:
   - Python interpreter between eval calls
   - Cache effects (1.18 GB model may not fully fit in GPU L2)
   - Attention computation overhead (small but nonzero at seq=1)
