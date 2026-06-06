# Learnings: exp_benchmark_composition_latency_sweep

## Core Finding

Pre-merge LoRA composition scales **linearly** (not sub-linearly) with adapter count N, at ~0.30 ms/adapter on M5 Pro. The "sub-linear alpha=0.83" is a power-law fitting artifact caused by constant overhead dominating small-N measurements; the correct model is affine T = c_0 + c_1*N. **At N=25, merge latency is 3.28ms compiled — 15x under the 50ms interactive budget.** mx.compile delivers a consistent 2.3-2.4x speedup at zero memory cost. The bottleneck is memory-bandwidth-bound accumulation of d*d delta matrices (80% of merge time), not the rank-16 matmuls (20%).

## Why This Happened (Literature-Grounded)

### Linear scaling is the theoretical prediction — confirmed, not exceeded

MATH.md correctly derives T = c_0 + c_1*N (Section 2): each adapter delta B_i^T @ A_i^T is independent, producing a (2560, 2560) matrix that is scaled and accumulated into W_merged. The per-adapter cost is dominated by reading/writing the 13.1 MB delta (bf16), not the 210M FLOP rank-16 matmul. At M5 Pro's ~273 GB/s bandwidth, the theoretical floor is ~0.145 ms/adapter; empirical ~0.30 ms/adapter reflects unfused Python dispatch and Metal overhead.

The reviewer correctly identified that fitting T = a*N^alpha to data where N=1 (0.92ms) > N=2 (0.83ms) is physically impossible for a pure power law — the constant overhead c_0 pulls alpha below 1.0. At large N (50-100), the data is perfectly linear: 30.38/15.34 = 1.98x for 2x N. This is standard affine behavior with constant overhead, not genuine sub-linearity.

LoRA Soups (arXiv 2410.13025) documents linear merge cost with N for concatenate-and-average composition. Our result is consistent: no fundamental mechanism exists for sub-linear merge scaling when each adapter contributes a full (d_out, d_in) delta.

### mx.compile fuses dispatch overhead, not the mathematical operation

The 2.3-2.4x speedup from mx.compile is consistent across N >= 10. The mechanism: MLX's compilation fuses the Python loop into a single Metal compute graph, eliminating per-adapter kernel dispatch (~10us * N) and intermediate tensor allocations. This converts the dispatch-bound regime to a bandwidth-bound regime. The even lower compiled alpha (0.73) reflects the larger constant overhead from compilation setup — same fitting artifact, amplified.

### Memory bandwidth is the fundamental bottleneck at this scale

Bottleneck decomposition at N=25 shows: 80% accumulation (read/write 13.1MB W per adapter) vs 20% matmul (rank-16 @ 210M FLOP). At d=2560, r=16, the matmul arithmetic intensity is ~0.39 FLOP/byte, deeply memory-bound on M5 Pro (16.5 TFLOPS vs 273 GB/s, ridge point ~60 FLOP/byte). The vectorized alternative (stack + weighted sum) halves accumulation time but costs 328 MB at N=25 — not worth the memory trade.

EdgeMoE (arXiv 2308.14352) identifies memory bandwidth as the dominant bottleneck for on-device MoE/adapter serving. Our 80% accumulation bottleneck is the same phenomenon: reading/writing full-rank delta matrices saturates memory bandwidth before compute.

## Confirming Evidence

1. **Our exp_batched_premerge_throughput**: Runtime LoRA is 4-87x faster than pre-merge for per-token routing. This experiment complements that finding: pre-merge IS viable for always-on adapters (3.28ms at N=25), confirming the hybrid architecture recommendation.

2. **Our exp_continual_learning_adapter_growth**: Composition quality stable within ~1% across N=5-15. This experiment confirms the serving side: quality doesn't degrade with N, AND latency stays interactive. Together they validate the N=25 always-on adapter regime.

3. **LoRA Soups (arXiv 2410.13025)**: Documents linear merge cost with adapter count. Our empirical measurement of linear + constant overhead is consistent with their analysis.

4. **Naive LoRA Summation (arXiv 2508.11985)**: Proves orthogonality enables additive composition. The linearity of our merge cost is a direct consequence — each adapter's delta is independent, so no cross-adapter computation exists.

5. **EdgeMoE (arXiv 2308.14352)**: Identifies memory bandwidth as the dominant bottleneck for on-device adapter serving. Our 80% accumulation bottleneck is the same phenomenon at the merge level.

## Contradicting Evidence

1. **The "sub-linear" framing in PAPER.md is self-contradicting.** MATH.md derives T = c_0 + c_1*N (linear) on line 70, but the experiment fits T = a*N^alpha (power law). The reviewer identified N=1→N=2 latency decrease as impossible under pure power law. The correct characterization is "linear scaling with constant overhead." This does not invalidate architectural decisions but is a methodological error in the paper.

2. **DeepSeek-V3 (arXiv 2412.19437) shows communication becomes the bottleneck at extreme expert counts.** At 256 experts with Top-8 routing, all-to-all EP communication exceeds 50% of execution time. Our single-device scenario avoids this, but the affine T = c_0 + c_1*N model holds only on unified-memory devices.

3. **Our exp_competitive_benchmark showed 4.5x peak memory overhead from bf16 unpacking.** This experiment uses synthetic bf16 weights, so it does not capture the unpacking cost of real ternary weights. The 3.28ms merge latency would be additive to any ternary→bf16 conversion time (2.33x sustained generation penalty per exp_e2e_demo_pipeline_mlx).

## Alternative Approaches

### 1. Compiled runtime LoRA for always-on adapters
Apply always-on adapters via compiled runtime LoRA instead of pre-merge at session start. At 0.58% per-token overhead per adapter (proven in exp_adapter_inference_speed_mlx), N=5 always-on adapters = ~3% overhead — comparable to 3.28ms merge amortized over 128 tokens. Advantage: no session-start delay, no ternary structure destruction. Reference: MoLoRA (arXiv 2603.15965).

### 2. Grouped GEMM batching (S-LoRA pattern)
S-LoRA (arXiv 2311.03285) and Punica introduce grouped GEMM kernels that batch adapter applications across requests. For multi-tenant scenarios this replaces per-session pre-merge. Our single-user M5 Pro scenario doesn't benefit from multi-tenant optimization, but the pattern transfers if the architecture moves to runtime LoRA for all adapters.

### 3. Fused accumulate kernel via Metal
Custom Metal kernel fusing matmul + scale + accumulate into a single pass, eliminating intermediate delta materialization. Could approach the 0.145ms/adapter bandwidth floor vs current 0.30ms. Reference: mlx-bitnet (exo-explore/mlx-bitnet) custom Metal kernels; MoEShard showed 6.4x TTFT improvement via fused expert computation.

### 4. Delta caching with selective invalidation
For adapter hot-swap (change 1 of 25), incrementally update the cached merged delta instead of full re-merge. The experiment tested full delta caching (1.25x speedup, 13.1MB*N memory) — dominated by mx.compile. But incremental single-adapter updates would cost only ~0.30ms vs full 3.28ms re-merge.

## Implications for Next Experiments

1. **Pre-merge serving architecture is validated for N <= 100 on M5 Pro.** At N=100 compiled: ~12.5ms — well under 50ms. The "always-on personality blend" use case has no latency concern. The constraint is quality (uniform composition hurts factual recall per exp_competitive_benchmark), not latency.

2. **The hybrid architecture is fully characterized.** Pre-merge: 3.28ms at N=25 for always-on adapters. Runtime LoRA: 0.58% per-token for routed domain experts. This is the serving architecture for BitNet-SOLE on M5 Pro.

3. **Full-model validation is the remaining gap.** Single (2560, 2560) projection tested. Full 24-block model (168 projections) extrapolates to ~550ms compiled at N=25 — acceptable for session start but should be validated with actual model weights and varied layer dimensions.

4. **Ternary weight interaction is an open question.** Real ternary weights (42% natural sparsity per arXiv 2603.05168) may merge faster if sparsity can be exploited in accumulation. However, bf16-unpacked ternary weights have no sparsity — speedup requires native ternary kernels that MLX currently lacks.

## Recommended Follow-Up

**exp_routed_topk_composition** — Per-token top-k routing with runtime LoRA on 7 genuine domain adapters. Motivation: (1) this experiment confirmed pre-merge is viable for always-on adapters, completing the serving feasibility picture; (2) exp_competitive_benchmark showed uniform composition hurts factual recall (KILLED 3/3); (3) exp_batched_premerge_throughput proved runtime LoRA is 4-87x faster for per-token routing; (4) MoLoRA (arXiv 2603.15965) demonstrated 1.7B + 4 adapters > 8B monolithic. This is the convergent recommendation from 4 consecutive experiments.
