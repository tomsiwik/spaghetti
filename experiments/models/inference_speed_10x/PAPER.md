# Inference Speed Optimization: Research Digest

## Hypothesis

The 82 tok/s measurement from exp_memory_optimized_serving understates the true
throughput of native BitLinear serving on M5 Pro. With optimized measurement
and addmm kernel fusion, adapter composition can exceed 100 tok/s.

## What This Experiment Found

### Key Discovery: The 82 tok/s Baseline Was a Measurement Artifact

The prior experiment measured speed using `time.time()` wrapping `mlx_generate()`,
which includes Python overhead (tokenizer encode, detokenizer, function call overhead).
Using `stream_generate`'s internal `time.perf_counter()` timing (measured between
GPU evaluations), the **actual base model speed is 172 tok/s** — already 2.1x the
prior measurement, at 74.2% of the M5 Pro memory bandwidth bound (231.7 tok/s).

The memory bandwidth bound (273 GB/s / 1.178 GB = 231.7 tok/s) gives a theoretical
maximum. The measured 172 tok/s is 74.2% of this bound, which is typical for
single-token autoregressive generation with kernel launch overhead.

### Speed Hierarchy

| Configuration | Internal (tok/s) | Wall-clock (tok/s) | Memory |
|---------------|-------------------|---------------------|--------|
| Base model (no adapter) | 171.8 | 153.6 | 1,179 MB |
| Adapter N=1 (naive LoRA) | 88.2 | 82.3 | 1,224 MB |
| **Adapter N=1 (addmm)** | **97.2 ± 0.0** | -- | **1,224 MB** |
| **Adapter N=1 (attn-only, addmm)** | **126.7 ± 0.2** | -- | **~1,200 MB** |
| Adapter N=2 (addmm) | 87.6 | 81.6 | 1,405 MB |
| Adapter N=5 (addmm) | 39.6 | 37.8 | 1,405 MB |

### Optimization Results

**addmm fusion (+10.2%)**: Using `mx.addmm(y, h, B, alpha=scale)` instead of
separate matmul + add saves one kernel launch per layer. 210 fewer kernel launches
per token: 88.2 -> 97.2 ± 0.0 tok/s (measured across all 4 test prompts).

**Attention-only adapters (+43.5%)**: Applying LoRA only to attention projections
(Q, K, V, O) and skipping MLP layers (gate, up, down) reduces from 210 to 120
wrapped layers and eliminates the highest-cost LoRA layers (MLP projections
are 2.7x larger). Result: 126.7 ± 0.2 tok/s (measured across all 4 test prompts).

**Precomputed A@B (WORSE, -50%)**: Materializing the full (d_in, d_out) matrix
added 4.1 GB memory, making it bandwidth-bound: 44.4 tok/s.

**Pre-merge to bf16 (WORSE, -36%)**: Unpacking ternary to bf16 increases model
from 1.18 GB to 4.83 GB, destroying the bandwidth advantage: 55.2 tok/s.

**KV cache quantization (WORSE at short contexts)**: At 100 tokens, the quant/dequant
overhead exceeds bandwidth savings: 172 -> 160 tok/s (-7%).

### Python Overhead

Wall-clock timing (including tokenizer, detokenizer, function calls) is 10.8%
slower than internal GPU timing. For 500-token generations, this narrows to 2.4%
as prefill cost is amortized.

## Key References

- BitNet b1.58 (2402.17764): ternary architecture, Metal kernel in mlx_lm
- bitnet.cpp (2502.11880): 45 tok/s on M2 CPU (our Metal kernel is 3.8x faster)
- vllm-mlx (2601.19139): 525 tok/s on 0.6B models (different architecture class)
- Prior exp_memory_optimized_serving: fixed bf16 unpack bug, established 82 tok/s baseline

## Empirical Results

### Kill Criteria
- **K1 (>50 tok/s): PASS** — Base: 171.8, Adapter (addmm): 97.2 ± 0.0, Adapter (attn-only): 126.7 ± 0.2

### Success Criteria
- **S1 FAIL (full adapter: 97.2 tok/s < 100)**. S1 PASS (attn-only: 126.7 tok/s, quality not validated).
  - Full adapter (all 7 layer types, addmm): 97.2 ± 0.0 tok/s — fails S1 by 2.8%
  - Attention-only adapter (4 layer types, addmm): 126.7 ± 0.2 tok/s — passes S1, but quality not re-evaluated
  - Quality validation of attention-only adapters is required before this counts as a full S1 PASS

### Summary Numbers

| Metric | Value |
|--------|-------|
| Base model speed | 171.8 tok/s |
| Bandwidth utilization | 74.2% of theoretical max (273 GB/s / 1.179 GB = 231.6 tok/s) |
| Adapter overhead (naive) | 48.6% |
| Adapter overhead (addmm, full) | 43.4% (97.2 ± 0.0 tok/s) |
| Adapter overhead (attn-only + addmm) | 26.2% (126.7 ± 0.2 tok/s) |
| Memory (base + 1 adapter) | 1,224 MB |
| Python overhead | 10.6% |

## Limitations

1. **Single-token generation only** — batch>1 would shift from bandwidth-bound
   to compute-bound, changing the optimization landscape entirely.

2. **Quality not re-evaluated** — attention-only adapters may degrade domain
   quality compared to full adapters (needs separate quality experiment).

3. **Short context only** — tested at 100-500 tokens. At 4K+ tokens, KV cache
   quantization and other long-context optimizations may become relevant.

4. **Single adapter tested** — the attention-only strategy needs validation
   across all 5 domains and in composition (N>1) settings.

## What Would Kill This

- If attention-only adapters show >5% quality degradation compared to full adapters,
  the speed gain is not usable and full adapter speed (97.4) becomes the true number.
- If multi-adapter addmm composition (N=2, N=5) with attention-only doesn't
  proportionally improve, the overhead scaling may be worse than measured.

## Architectural Insight for SOLE

The key finding for the SOLE architecture:

1. **Runtime LoRA is viable at 97-126 tok/s** — no need for pre-merge (which
   destroys the ternary bandwidth advantage).

2. **MLP adapters are 1.39x more expensive than attention adapters** because
   gate/up projections are 2.7x wider. If routing needs to be fast, route
   only attention adapters and keep MLP as base or always-on pre-merged.

3. **The ternary Metal kernel achieves 74.2% bandwidth utilization** (172 / 231.7 tok/s,
   using the correct M5 Pro bandwidth of 273 GB/s). This is normal for memory-bound
   single-token generation with kernel launch overhead.

4. **addmm should be the default for LoRA serving** — 10% free speedup
   from kernel fusion, no quality impact.
