# Learnings: exp_memory_optimized_serving

## Core Finding

The previously-observed 10.98 GB serving memory was a **bug, not architecture**: `replace_bitlinear_with_linear()` unpacks all 210 ternary layers from uint8 to bf16, a 9.3x bloat (521 MB -> 4,827 MB) that is necessary for LoRA training but unnecessary for inference. The correct serving strategy -- native BitLinear Metal kernel + runtime LoRA as additive correction -- achieves **1.22 GB total memory** with 0.21% PPL degradation. This scales to 50 adapters at 2.79 GB (int8 B matrices), all within the 3 GB target on M5 Pro.

## Why This Happened (Literature-Grounded)

### The bf16 unpack was a training artifact carried into inference

LoRA training requires differentiable forward passes through nn.Linear (backprop through ternary operations is ill-defined without STE). The `replace_bitlinear_with_linear()` function was written for training and was inadvertently used for inference. The BitLinear Metal kernel in mlx_lm reads packed uint8 weights directly, computing the matrix-vector product in-kernel without materializing a dense bf16 intermediate. This is the standard approach documented in BitNet b1.58 (arXiv 2402.17764) and implemented in bitnet.cpp.

### Runtime LoRA is mathematically equivalent to pre-merge for single adapters

The computation `y = BitLinear(x) + (x @ A) @ B * scale` is additive. For a single adapter, this is equivalent to pre-merge `y = (W + scale * B^T @ A^T) @ x` modulo numerical differences between ternary-kernel and bf16 matmul. The measured 0.21% PPL gap (3.747 vs 3.739) conflates two variables: (a) ternary vs bf16 base weight representation, and (b) runtime vs pre-merge LoRA application. The reviewer correctly identified this conflation. The gap is an upper bound on the serving strategy's quality cost; the actual LoRA-specific cost may be zero.

### Memory-bandwidth economics favor packed ternary at inference

At d=2560, a single BitLinear layer's packed weights are 1.025 MB (uint8) vs 13.1 MB (bf16). The LoRA overhead adds 2 * 2560 * 16 * 2 = 164 KB per layer. Total memory transfer per layer: 1.19 MB (BitLinear + LoRA) vs 13.1 MB (unpacked bf16). This 11x reduction in memory traffic directly translates to the 82 tok/s throughput (vs prior 26 tok/s on the unpacked path), consistent with M5 Pro's 273 GB/s memory bandwidth: theoretical ceiling ~127 tok/s, measured 82 tok/s = 65% efficiency.

## Confirming Evidence

1. **S-LoRA (arXiv 2311.03285)**: Describes exactly this serving pattern -- base model with runtime LoRA as additive correction, adapter swapping by pointer replacement. Our implementation is the single-tenant version of their concurrent serving architecture.

2. **Our exp_benchmark_composition_latency_sweep**: Pre-merge costs 0.30 ms/adapter compiled, bottlenecked by memory bandwidth (13.1 MB read-modify-write per adapter). Runtime LoRA eliminates this merge step entirely for per-query routing, reading only 164 KB/layer instead.

3. **Our exp_falcon_e3b_composition**: Identified the same root cause -- "Peak 8.80 GB is due to bf16 unpacking of ternary weights." That experiment attributed it to missing MLX BitLinear kernels, but the kernels were present all along; the bug was using the wrong code path.

4. **EdgeMoE (arXiv 2308.14352)**: On-device MoE serving architecture that identifies memory bandwidth as the dominant bottleneck. Our 65% bandwidth efficiency on M5 Pro is within the expected range for non-fused inference.

5. **Our exp_batched_premerge_throughput**: Runtime LoRA is 4-87x faster than pre-merge for per-token routing. This experiment confirms the memory side: runtime LoRA also uses 4x less memory.

## Contradicting Evidence

1. **PPL comparison is confounded.** The 0.21% gap could be entirely from ternary vs bf16 base weights, not the LoRA serving strategy. Without an ablation (runtime LoRA on unpacked bf16 weights vs pre-merge on unpacked bf16 weights), we cannot attribute the gap. The reviewer flagged this but rated it non-blocking since the gap is well within the 5% threshold.

2. **82 tok/s is a single measurement.** No variance reported. Prior experiments in this project (e.g., bitnet_float_merge_fp32) used 5-run averages. The number is plausible given bandwidth analysis but should not be cited precisely.

3. **K2 was hardcoded True (line 867).** The verdict is correct by accident (0.21% < 5%), but the code never actually computed K2 from measurements. This is a code quality issue, not a scientific one.

4. **Phase 2 active_mb measurements are confounded.** Loading one adapter shows MORE active memory than loading five (365.7 MB vs 341.4 MB), clearly a residual memory artifact. The computed data sizes (21.9 MB, 109.4 MB) are correct; the active_mb readings are not.

5. **VISION.md inconsistency.** Prior VISION.md reported 12.3 tok/s for runtime LoRA. This experiment reports 82 tok/s. The discrepancy is likely because the old figure used the unpacked-bf16 runtime LoRA path. Not contradicting the current result, but the old figure was also wrong.

## Alternative Approaches

1. **CLA (Cross-Layer Attention, arXiv 2405.12981)**: Reduces KV cache by sharing K/V across layers. At seq=8192, KV cache would be ~629 MB; CLA could halve this. Not needed at current memory budget (1.22 GB + 629 MB = 1.85 GB, still under 3 GB), but relevant for long-context serving.

2. **Adapter quantization beyond int8.** Int4 B matrices would halve adapter memory to ~5.5 MB each. At N=50: 1,178.6 + 50*(5.5+21.4) = 2,523 MB. NF4 quantization (arXiv 2305.14314, QLoRA) is proven to maintain quality for LoRA adapters.

3. **On-demand adapter loading (S-LoRA pattern).** Instead of holding all N adapters in memory, load top-k per query from SSD. M5 Pro SSD reads at ~7.4 GB/s; a 22 MB adapter loads in ~3ms. Only matters at N>50 where memory exceeds budget.

## Implications for Next Experiments

1. **The serving pipeline is now memory-viable.** 1.22 GB for base + 1 adapter means the full SOLE stack (base + router + top-k adapters) fits in ~1.5-2 GB, leaving 46 GB for KV cache, activations, and application overhead on M5 Pro.

2. **Pre-merge is the WRONG strategy for memory-constrained deployment.** It requires unpacking all ternary weights to bf16 (4.8 GB), eliminating the core advantage of ternary models. Runtime LoRA is mandatory for memory-efficient serving of BitLinear models.

3. **The 82 tok/s throughput (if confirmed with variance) makes interactive serving feasible.** Prior 12.3-26 tok/s figures were from the buggy unpacked path. At 82 tok/s, the system can generate responses at human reading speed.

4. **Falcon-E-3B LEARNINGS.md incorrectly stated "MLX lacks native BitLinear kernels."** The kernels exist and work; the prior experiments were using the wrong inference code path. This reframes the memory constraint: it was always solvable without new kernel development.

## Recommended Follow-Up

No new experiment recommended from this work specifically. The deployment track (P0) experiments -- exp_generation_quality_test, exp_task_accuracy_real_benchmarks, exp_real_data_25_domain_adapters -- should now use the native BitLinear + runtime LoRA serving path exclusively. The memory constraint is resolved.

The convergent recommendation from 5 prior experiments (competitive_benchmark, batched_premerge, continual_learning, latency_sweep, pointer_routing) remains: **exp_routed_topk_composition** with per-sequence routing on genuine domains. This experiment confirms the serving infrastructure can support it at 1.2-2.8 GB.
