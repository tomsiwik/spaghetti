# Learnings: Inference Speed Optimization

## Critical Discoveries

### 1. Prior 82 tok/s was a measurement artifact
The `time.time()` wrapping of `mlx_generate()` included Python overhead (tokenizer,
detokenizer, function calls). Internal timing via `stream_generate` shows **172 tok/s**
for the base model. This is 2.1x the prior measurement.

### 2. BitNet Metal kernel achieves 74.2% of bandwidth bound
At 172 tok/s, we achieve 74.2% of the theoretical bandwidth bound (273 GB/s / 1.178 GB = 231.7 tok/s).
The M5 Pro has 273 GB/s memory bandwidth (not 153 GB/s as previously stated).
74.2% utilization is typical for single-token autoregressive generation with kernel launch overhead.

### 3. addmm is free 10% speedup for LoRA serving
Using `mx.addmm(y, h, B, alpha=scale)` instead of `y + (h @ B) * scale` fuses the
add+scale into the matmul kernel. 88.2 -> 97.2 ± 0.0 tok/s (measured across all 4
test prompts) with zero quality impact. Should be the default for all LoRA serving code.

### 4. MLP adapters are disproportionately expensive
MLP projections (gate/up: 2560->6912, down: 6912->2560) are 2.7x wider than
attention projections (2560->2560). LoRA on MLP costs 1.39x more per block.
Attention-only LoRA achieves 126.7 ± 0.2 tok/s vs 97.2 with full LoRA (both addmm,
measured across all 4 test prompts).

### 5. Pre-merge and precomputed A@B both SLOWER
- Pre-merge (unpack to bf16): model grows from 1.18 GB to 4.83 GB, becomes
  bandwidth-limited at 55.2 tok/s. The ternary packing advantage is lost.
- Precomputed C = A@B: materializes (d_in, d_out) dense matrices, adding 4.1 GB.
  Result: 44.4 tok/s. Factored LoRA (two small matmuls) is strictly better.

### 6. KV cache quantization hurts at short contexts
At 100 tokens, KV cache is ~0.5 MB vs 1.2 GB model. The quant/dequant overhead
(quantize_cache_fn) exceeds any bandwidth savings: 172 -> 160 tok/s (-7%).

### 7. Python overhead is ~10.8% at 100 tokens, ~2.4% at 500 tokens
The async_eval pipelining in mlx_lm hides most GPU-Python synchronization,
but tokenizer/detokenizer adds fixed per-call overhead that amortizes with length.

## Design Implications for SOLE

- **Runtime LoRA > Pre-merge** for ternary models (confirmed quantitatively)
- **addmm should be default** in BitLinearWithLoRA wrapper
- **Attention-only routing** is a viable speed optimization for per-token routing
- **MLP adapters can be always-on (pre-merged)** while attention adapters are routed
  dynamically per token — this is a hybrid serving strategy
