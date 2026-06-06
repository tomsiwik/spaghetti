# MLX Adapter Inference Speed: LoRA Overhead on Apple Silicon

**Date:** 2026-03-25
**Platform:** Apple M5 Pro, MLX 0.31.1 (Metal GPU)
**Runtime:** ~8 minutes

## Abstract

We measure LoRA adapter inference overhead on Apple Silicon using MLX to complete the cross-platform serving matrix alongside prior CPU (llama.cpp) and CUDA (RTX 4090) results. Pre-merge composition is effectively free (max 0.80% overhead at N=8, within noise). Runtime adapter application incurs 33.2% overhead at k=1, failing the 15% kill criterion, but scales sub-linearly (beta=0.96). The clear recommendation for Apple Silicon serving: always pre-merge adapters offline; never apply LoRA at runtime via MLX.

**Verdict: K1 KILLED (33.2% > 15%), K2 PASS (beta=0.96 < 1.0)**

## Hypothesis

LoRA adapter composition on Apple Silicon Metal GPU (via MLX) incurs at most 15% single-adapter overhead and scales linearly or sub-linearly with the number of active adapters.

### Kill Criteria

- **K1:** Single-adapter runtime overhead > 15% vs base forward pass --> KILL
- **K2:** N-adapter overhead grows faster than O(N), i.e., power-law exponent beta > 1.0 --> KILL

## What This Experiment Tested

Two composition strategies for LoRA adapters on MLX:

1. **Pre-merge (offline):** Merge N adapters into base weights before inference. Forward pass uses the merged weight matrix W' = W + (1/N) * sum(B_i @ A_i). Inference is structurally identical to base model.

2. **Runtime activation path:** Apply k adapters dynamically during the forward pass. Each linear layer computes y = W@x + (1/k) * sum(B_j @ (A_j @ x)). This adds 2*k low-rank matmuls per linear layer per forward pass.

This fills the Apple Silicon gap in the serving cost matrix established by prior experiments on CPU (llama.cpp: 9.5% + 7.5%*N overhead) and CUDA GPU (RTX 4090: pre-merge at most +3.3% at N=50).

## Key References

- MATH.md in this directory: theoretical FLOPs analysis predicting 9.375% overhead per runtime adapter
- Prior CPU benchmark (llama.cpp): affine O(N) scaling at 7.5% per adapter
- Prior CUDA benchmark (RTX 4090): pre-merge max +3.3% at N=50
- MLX framework: lazy evaluation with Metal GPU dispatch on unified memory

## Method

### Model Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 4 |
| vocab_size | 256 |
| seq_len | 32 |
| batch_size | 1 |
| lora_rank | 8 |
| Base params | 853,120 |
| LoRA params/adapter | 73,728 |

The micro transformer has 4 layers, each with 6 LoRA-adapted linear projections (Wq, Wk, Wv, Wo, fc1, fc2). ReLU activation in MLP, RMSNorm, causal attention mask.

### Measurement Protocol

- **Warmup:** 100 iterations (compiles Metal shaders, stabilizes caches)
- **Timed:** 500 iterations with `time.perf_counter()` bracketing `mx.eval(out)` for synchronous timing
- **Metrics:** mean, std, median, p95 latency in ms; tokens/sec
- **Pre-merge:** Rebuild fresh model per N, merge adapters into weights, then benchmark
- **Runtime:** Wrap base model in `RuntimeLoRATransformer` that adds LoRA via activation path
- **N values tested:** {1, 2, 4, 8}

## Empirical Results

### Base Model

| Metric | Value |
|--------|-------|
| Mean latency | 0.706 ms |
| Std | 0.057 ms |
| Median | 0.692 ms |
| p95 | 0.758 ms |
| Throughput | 45,344 tok/s |

### Pre-Merge Strategy (Offline Adapter Fusion)

| N adapters | Mean (ms) | Overhead (%) | Tok/s |
|------------|-----------|-------------|-------|
| 1 | 0.701 | -0.69 | 45,658 |
| 2 | 0.701 | -0.67 | 45,650 |
| 4 | 0.701 | -0.67 | 45,648 |
| 8 | 0.700 | -0.80 | 45,711 |

All overhead values are negative (within measurement noise). Pre-merge is structurally identical to the base model forward pass and adds zero inference cost regardless of how many adapters are fused. Maximum |overhead| = 0.80%.

### Runtime Activation Path (Dynamic Adapter Application)

| k adapters | Mean (ms) | Overhead (%) | Tok/s |
|------------|-----------|-------------|-------|
| 1 | 0.940 | +33.2 | 34,042 |
| 2 | 1.216 | +72.4 | 26,306 |
| 4 | 1.859 | +163.5 | 17,211 |
| 8 | 2.330 | +230.1 | 13,737 |

### Kill Criteria Assessment

**K1: Single-adapter overhead = 33.2% --> FAIL (KILLED)**
The 15% threshold is exceeded by 2.2x. A single runtime LoRA adapter adds 0.234 ms to a 0.706 ms forward pass.

**K2: Scaling exponent beta = 0.96 --> PASS**
Power-law fit: overhead(k) = 36.1 * k^0.96. The exponent beta = 0.96 < 1.0 indicates sub-linear scaling, meaning each additional adapter contributes slightly less marginal overhead than the previous one (likely due to MLX's lazy evaluation batching more operations into fewer Metal dispatches).

**Pre-merge overhead: 0.80% max --> effectively zero**
Pre-merge is free. This is the expected result since merged weights have identical shape and computation as base weights.

## Discussion

### Theoretical vs Measured Gap

MATH.md predicts a theoretical overhead ratio of rho = 3r/(2d) = 9.375% per runtime adapter. The measured values exceed theory by a significant and increasing factor:

| k | Theoretical (%) | Measured (%) | Ratio (measured/theory) |
|---|-----------------|-------------|------------------------|
| 1 | 9.4 | 33.2 | 3.5x |
| 2 | 18.8 | 72.4 | 3.9x |
| 4 | 37.5 | 163.5 | 4.4x |
| 8 | 75.0 | 230.1 | 3.1x |

The consistent 3-4x gap between theory and measurement indicates substantial non-FLOP overhead. The primary suspects:

1. **Metal kernel dispatch overhead:** Each LoRA application requires two additional matmul kernel dispatches (A@x and B@(Ax)) per linear layer per adapter. At 6 layers per block and 4 blocks, k=1 requires 48 extra kernel launches. Metal kernel dispatch has a fixed-cost floor that dominates at micro model scale.

2. **Python-side overhead in MLX:** The `RuntimeLoRATransformer` replaces MLX's optimized `nn.Linear.__call__` with explicit Python-level matrix operations, losing any framework-level fusion.

3. **Memory access patterns:** Runtime LoRA requires reading adapter weights (A, B) from separate buffers, adding memory traffic beyond what FLOP counting captures.

### Cross-Platform Comparison

| Platform | Single-adapter overhead | Scaling pattern | Best strategy |
|----------|------------------------|----------------|---------------|
| CPU (llama.cpp) | 7.5% per adapter | O(N) affine | Runtime OK for small k |
| GPU (RTX 4090) | ~0% (pre-merge) | Sub-linear | Pre-merge |
| MLX (Apple Silicon) | +33.2% (runtime) | Sub-linear (beta=0.96) | Pre-merge |
| MLX (Apple Silicon) | ~0% (pre-merge) | N/A (constant) | Pre-merge |

Apple Silicon runtime overhead (33.2%) is 4.4x worse than CPU llama.cpp (7.5%) for a single adapter. This is surprising given that Metal GPU should have higher raw throughput. The explanation is that llama.cpp uses C++ with direct GGML kernel calls (minimal dispatch overhead), while MLX runtime LoRA goes through Python with per-operation lazy evaluation.

The good news: pre-merge is equally free on all platforms. Since pre-merge has zero overhead by construction (same forward pass), the platform comparison only matters for runtime application.

### Implications for the Serving Architecture

1. **Apple Silicon serving must use pre-merge.** The 33.2% runtime overhead is unacceptable for interactive serving. Pre-merge eliminates this entirely.

2. **Pre-merge is compatible with the SOLE architecture.** The router selects k experts, their LoRA deltas are summed and merged into base weights before the forward pass. For static routing (query-independent composition), this merge happens once at session start. For dynamic routing (per-token composition), the merge cost must be amortized across tokens -- but the merge itself is cheap (9.4M FLOPs for N=8, vs 50M FLOPs per forward pass).

3. **Dynamic per-token routing on MLX requires careful amortization.** If the expert set changes every token, merge overhead is non-trivial. Batching tokens with the same expert set (as in speculative decoding or prompt processing) would amortize the merge cost.

4. **The beta < 1.0 sub-linear scaling is a positive signal.** If runtime LoRA were needed (e.g., for research or debugging), scaling to k=8 costs only 230% rather than the 300%+ that linear scaling would predict.

## Limitations

1. **Micro scale only.** d=128 model is compute-bound at a different ratio than production models (d=4096+). At larger d, the FLOP ratio rho = 3r/(2d) shrinks, and kernel dispatch overhead becomes relatively smaller. The 33.2% overhead likely decreases at production scale.

2. **Batch size 1 only.** Larger batch sizes amortize kernel dispatch overhead and may reduce the theory-measurement gap.

3. **Single hardware configuration.** Tested on M5 Pro only. M5 Max/Ultra with more GPU cores may show different dispatch overhead characteristics.

4. **MLX 0.31.1 specific.** Future MLX versions may fuse LoRA operations or reduce Metal dispatch overhead.

5. **Synthetic adapters.** Random LoRA weights, not trained adapters. This does not affect latency measurements (same shapes and computation) but means the throughput numbers carry no quality signal.

6. **No amortization measurement.** We did not measure the one-time cost of pre-merging adapters. At micro scale this is negligible (~9.4M FLOPs for N=8), but at production scale it determines the minimum batch size for pre-merge to be practical with dynamic routing.

## What Would Kill This

The pre-merge strategy fails if:
- **Dynamic per-token routing is required AND merge cost exceeds per-token latency budget.** At production scale (d=4096, r=16, N=8), merge cost is ~150M FLOPs vs ~2B FLOPs per forward pass -- still only ~7.5%, likely acceptable.
- **MLX introduces fused LoRA kernels** that eliminate the dispatch overhead gap, making runtime application competitive (<15% for k=1). This would revive runtime LoRA as a viable path.
- **Memory pressure at scale** makes holding both base weights and merged weights infeasible. At micro scale this is not an issue; at 7B+ scale, the temporary double-buffering during merge would need ~14 GB, which is tight on 16 GB Apple Silicon devices.
