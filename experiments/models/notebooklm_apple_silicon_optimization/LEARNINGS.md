# LEARNINGS: MLX Optimization for Ternary Models on M5 Pro

## Core Finding

The survey consolidates known bandwidth-bound behavior into a roofline framework and produces 6 genuine new recommendations (R3/mx.compile was already done). The primary insight is that **all single-batch ternary optimization must reduce bytes read, not compute** — but the survey itself produced no new measurements, making all forward predictions unverified. ~60% overlaps with Finding #176.

## Why This Happened

The bandwidth-bound conclusion is well-established in the literature and in our own experiments:

- **Finding #176** (metal_kernel_profiling): 165.6 tok/s, 73% BW utilization, 268.6 GB/s measured peak
- **Finding #76**: mx.compile redundant for generation because async_eval already hides dispatch
- **Finding #36**: Sparse-BitNet killed — inference is BW-bound, Metal SIMD handles zeros for free
- **bitnet-mlx.rs** (leizerowicz): 82% GPU utilization, 85% ANE utilization on Apple Silicon — higher than our 73%, suggesting room for improvement via memory wiring or framework optimization
- **Williams, Waterman, Patterson (CACM 2009)**: Roofline model is standard framework, correctly applied here but uncited in MATH.md

The 73% utilization (vs. 82-85% achieved by bitnet-mlx.rs) is the actionable gap. The remaining 12% is likely split between Python/tokenizer overhead (10.8% measured) and memory management.

## Confirming Evidence

1. **bitnet.cpp** (Microsoft): Confirms ternary models are memory-bandwidth-bound, not compute-bound. ARM CPU tiling gives 1.37-5.07x speedup through better memory access patterns.
2. **bitnet-mlx.rs**: Achieves 82% GPU / 85% ANE bandwidth utilization on Apple Silicon via zero-copy unified memory — proves higher utilization than our 73% is achievable.
3. **vllm-mlx** (arxiv 2601.19139): 21-87% higher throughput than llama.cpp via continuous batching. 3.7x aggregate throughput at 16 concurrent requests. Content-hash prefix caching gives 28x on repeated queries.
4. **Orion** (arxiv 2603.06728): ANE pulls ~19 TFLOPS in fp16 but orchestration (not compute) is the bottleneck — aligns with our finding that compute headroom is irrelevant at batch=1.
5. **mlx-lm README**: Documents `iogpu.wired_limit_mb` as recommended fix for degraded generation speeds. Memory wiring is an accepted optimization, not speculative.

## Contradicting Evidence

1. **Orion ANE compilation bottleneck**: ANE bakes weights at compile time — 4.2s recompilation per weight update, 82% of wall time in compiler. This makes R5 (ANE routing offload) impractical for any scenario requiring dynamic adapter loading. The survey didn't address this.
2. **bitnet-mlx.rs 85% utilization**: If Rust can achieve 85% and we're at 73%, the gap may be more about Python/MLX framework overhead than memory wiring. R1 (memory wiring) might yield less than predicted 5-15% if the gap is primarily Python.
3. **Finding #76**: mx.compile is already redundant when async_eval is used. The survey lists mx.compile as a recommendation (R3) but our own Finding #76 says the marginal benefit over async_eval is near zero. The "21% speedup" is vs. no-async baseline, not vs. mlx-lm default.
4. **Finding #66**: Pre-merge is 39% faster than runtime LoRA for float models but **WORSE for ternary** (destroys BW advantage by unpacking to bf16). The survey correctly notes this as N4 but doesn't resolve the tension: if pre-merge is dead for ternary, runtime LoRA with ternary B is the only path — and that adds adapter bytes per token.

## Alternative Approaches

1. **Speculative decoding** (arxiv 2302.01318): mlx-lm natively supports it. Uses small draft model + batched verification. Expected 1.5-1.7x for single-user. No custom integration needed — should be tested before building fused kernels.
2. **MatMul-free LM** (ridgerchu/matmulfreellm): Eliminates FP matmul entirely using ternary + bitwise operations. Up to 2.7B params. If applicable, would further reduce arithmetic intensity and strengthen the BW-bound case.
3. **T-MAC lookup tables** (underlying bitnet.cpp): Replaces multiply-accumulate with lookup tables. The Metal equivalent would be texture-based LUT kernels — potentially simpler than full fused BitLinear (R4).
4. **bitnet-mlx.rs zero-copy approach**: Achieves 85% BW utilization through Rust + zero-copy unified memory. If MLX overhead is the gap (not memory wiring), a Rust serving layer might be higher-impact than sysctl tuning.

## Implications for Next Experiments

1. **Memory wiring (R1) should be tested, not predicted** — it's one sysctl line. The reviewer correctly flagged this. Running it would convert a prediction into a measurement and either confirm the 5-15% estimate or reveal that the gap is elsewhere (Python overhead).

2. **Speculative decoding is the lowest-risk next optimization** — mlx-lm already supports it natively. No custom kernel work. Expected 1.5-1.7x single-user throughput. Should be tested before investing in fused Metal kernels (R4).

3. **The 73% → 85% utilization gap is the key unknown** — bitnet-mlx.rs achieves 85% in Rust. If memory wiring closes the gap, great. If not, the bottleneck is Python/MLX framework overhead, and no amount of kernel fusion will fix it without switching to a lower-level serving stack.

4. **vllm-mlx integration with SOLE routing is genuinely hard** — the 2-4x multi-user prediction assumes simple batching. Per-token routing + dynamic adapter composition in a continuous batching loop is an unsolved engineering problem. This is P2 at best.

5. **ANE offload (R5) is impractical for SOLE** — Orion showed 82% of wall time is in ANE compilation. Any adapter change would trigger recompilation. The survey's "<1% impact" assessment is honest but the architectural value claim is undermined by compilation costs.

## Recommended Follow-Up

**No new experiment recommended.** The deployment track (generation quality, task accuracy, real data) remains higher priority than optimization. The actionable items from this survey are:

- Test `iogpu.wired_limit_mb` as part of any future experiment's setup (1 line, no dedicated experiment needed)
- Test speculative decoding when single-user throughput becomes a bottleneck
- vllm-mlx integration belongs on the production serving roadmap (P2), not the current research track

The survey has organized existing knowledge into an actionable priority list. That's its value — not new science.
