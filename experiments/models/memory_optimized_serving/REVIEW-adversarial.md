# Peer Review: Memory-Optimized Serving

## NotebookLM Findings

Skipped -- the experiment is straightforward profiling work, not a novel mechanism requiring deep mathematical review. The core claim is engineering (use the packed kernel instead of unpacking), not algorithmic.

## Mathematical Soundness

### What holds

1. **Memory accounting is internally consistent.** The dtype breakdown (332 bf16 tensors = 657.6 MB, 210 uint8 tensors = 521.0 MB, sum = 1178.6 MB) matches `mx.get_active_memory()` after load. This is a clean measurement.

2. **LoRA FLOPs overhead calculation is correct.** At r=16, d=2560: 2 * 2560 * 16 + 2 * 16 * 2560 = 163,840 FLOPs. Base: 2 * 2560 * 2560 = 13,107,200. Ratio: 1.25%. Correct.

3. **Packing ratio is correct.** 4 ternary values per uint8, so 8x compression vs bf16 (2 bytes per value / 0.25 bytes per value = 8x). The 521 MB packed vs 4827 MB unpacked is approximately 9.3x -- the discrepancy from the theoretical 8x is due to `ceil(out/4)` padding, which is correctly noted.

4. **Int8 quantization error analysis is sound.** Per-tensor symmetric quantization with worst-case max error 3.09e-04 is negligible for bf16 LoRA adapters.

5. **Scaling formula is correct.** `1178.6 + N * (21.9 + 21.4)` MB for N adapters, with the caveat that A matrices are per-domain and B matrices are per-adapter.

### What does not hold or is imprecise

1. **MATH.md Section 1 tensor shape mismatch with code.** MATH.md states `A in R^{in_features x r}` and `B in R^{r x out_features}`, implying the LoRA computation is `y += (x @ A) @ B * scale`. The code (line 527) implements exactly this: `(x @ self.lora_a) @ self.lora_b * self.lora_scale`. However, the pre-merge formula (MATH.md line 50 / code line 360) is `W_new = W + scale * B^T @ A^T`. This is mathematically consistent (the pre-merge delta should equal the runtime computation projected into weight space), but the notation is not made explicit. Not a bug, but the paper should clarify.

2. **KV cache formula in MATH.md is wrong.** Section 3 states: `30 * 8 * 80 * 256 * 2 * 2 = 19.7 MB`. Let me verify: 30 * 8 * 80 * 256 * 2 * 2 = 30 * 8 * 80 * 256 * 4 = 19,660,800 bytes = 18.75 MB. The stated 19.7 MB is approximately correct but the formula has an extra factor of 2 that is not labeled (one for K, one for V -- should be explicit). More importantly, the results.json shows `mem_after_gen_50tok` active memory of 1181.4 MB, only 2.8 MB above base (1178.6 MB), while MATH.md estimates 3.5 MB for KV cache at seq=256. This is in the right ballpark but the mismatch between formula (19.7 MB) and measured (2.8 MB) is unexplained. At seq=50, the expected KV cache is ~3.8 MB, so the active_mb increase of 2.8 MB is plausible (some may be in cache_mb, which shows 27.1 MB).

3. **The "8x" packing ratio claim is used loosely.** MATH.md says "Packing ratio: 8x compression vs bf16" but the actual ratio is 521 MB packed vs ~4168 MB bf16 equivalent (521 * 8 = 4168), not vs 4827 MB. The 4827 MB includes the bf16 non-ternary params. The paper conflates two different comparisons. The ternary layers specifically go from 521 MB to ~4168 MB (8x), while the total model goes from 1178.6 MB to 4825.6 MB (4.1x).

## Novelty Assessment

### This is not novel research -- it is a debugging finding

The core discovery is: "the previous 10.98 GB measurement was wrong because we unpacked ternary weights to bf16 unnecessarily." This is a bug fix, not a mechanism. The paper correctly identifies this (Section "Key Finding: The 10.98 GB Was a Bug, Not Architecture"), but then frames the experiment as if it demonstrates a novel serving strategy.

Using BitLinear's native Metal kernel (which was always there in `mlx_lm`) and applying LoRA as an additive runtime correction is the obvious and standard approach. S-LoRA (cited) describes exactly this pattern for concurrent adapter serving. The `BitLinearWithLoRA` wrapper (lines 515-528) is trivial -- 3 lines of actual logic.

**Delta over prior work:** Zero. The contribution is identifying that prior experiments in this project used `replace_bitlinear_with_linear()` when they should not have for inference. This is valuable project knowledge but not a publishable finding.

### Prior art already in the project

Line 223-224 of FINDINGS.md documents `bitnet_serving_path` and `bitnet_float_merge_fp32` which already established runtime LoRA as the serving path. The 1.69 GB at N=15 figure (line 223) is from prior work. This experiment refines the measurement but does not discover anything new.

## Experimental Design

### Strengths

1. **Proper `mx.eval()` barriers.** Every measurement point calls `mx.eval()` before `log_memory()`. This is correct for MLX lazy evaluation.

2. **Cleanup between phases.** The `cleanup()` function calls `gc.collect()`, `mx.clear_cache()`, and `mx.reset_peak_memory()`. This prevents cross-phase contamination.

3. **Warmup for generation speed.** Line 601 does a 5-token warmup before timing the 100-token generation. Good practice.

4. **PPL measurement methodology is standard.** Cross-entropy loss, exp-clamped, 25 samples from validation data.

### Weaknesses

1. **K2 is hardcoded to True (line 867).** The comment says "Updated after measurement" but it never is -- it is always `True` regardless of actual PPL measurements. This is a code bug. The actual K2 criterion ("quality degrades >5% from compression") is tested in `phase_memory_budget` but only logged, never wired into the result. The correct comparison for K2 should be runtime LoRA PPL (3.747) vs pre-merge PPL (3.739), which is 0.21% degradation -- clearly passes the 5% threshold. So the verdict is correct by accident, but the code does not actually compute or enforce K2.

2. **PPL comparison is between different code paths, not same-code-path with different compression.** Pre-merge PPL (3.739) is computed after unpacking all ternary weights to bf16, merging the adapter, and running inference through nn.Linear. Runtime LoRA PPL (3.747) is computed through BitLinear + additive correction. These are *numerically different operations* (the BitLinear Metal kernel's ternary matmul vs bf16 matmul), so the 0.21% PPL difference could come from the ternary quantization noise of the base weights, not from the LoRA serving strategy. The paper claims "zero quality loss compared to bf16 pre-merge" but the comparison conflates two variables: (a) ternary vs bf16 base weights, and (b) runtime vs pre-merge LoRA. To isolate the LoRA serving strategy, you would need to compare runtime LoRA on unpacked bf16 weights vs pre-merge on unpacked bf16 weights. As it stands, the 0.21% difference is an upper bound that includes both effects.

3. **Phase 2 adapter memory measurements are unreliable.** `all_5_bf16_active_mb` is 341.4 MB but `one_bf16_active_mb` is 365.7 MB. Loading one adapter uses MORE active memory than loading five? This is clearly a residual memory artifact from sequential phase execution. The `cleanup()` between Strategy B and Strategy C did not fully release memory. The computed data sizes (109.4 MB and 21.9 MB) are correct, but the `active_mb` readings are confounded.

4. **Generation speed measurement has no variance.** The 82.0 tok/s figure is a single measurement. Prior experiments in this project (e.g., `bitnet_float_merge_fp32`) measured 5 runs with stddev. At minimum, 3-5 runs should be reported.

5. **Only 25 validation samples for PPL.** This is acknowledged in limitations but the paper treats the PPL values to 3 decimal places (3.747 vs 3.739). With 25 samples, the standard error is substantial. No confidence intervals are reported.

6. **"89% reduction" is misleading framing.** The reduction is from 10.98 GB (the buggy measurement) to 1.22 GB. But the 10.98 GB was never the correct serving memory -- it was a bug. Framing a bug fix as an "89% reduction" implies an optimization achievement. More honest: "the correct serving memory is 1.22 GB; prior measurements were incorrect."

## Logic Bugs in results.json

1. **`k2_quality_preserved: True` is hardcoded**, not computed from measurements (as noted above).

2. **Phase 6 `best_serving_mb` computation.** Line 775-778 takes `min(base_gen_mem, runtime_lora_gen_mem)`. The `base_gen_mem` comes from `native_results["mem_after_gen"]["active_mb"]` = 1224.2 MB. But `mem_after_gen` is measured AFTER runtime LoRA wrapping and generation -- it includes the adapter. So `base_gen_mem` is not actually the base-only generation memory. The correct base-only generation memory would be from Phase 1's `mem_after_gen_50tok` (1181.4 MB). The `min()` of two identical values (both 1224.2 MB) is meaningless. The code should compare distinct configurations.

3. **Phase 6 `base_only` config reports `gen_mb: 1224.2`.** This is from `native_results["mem_after_gen"]` which includes the adapter (Phase 4 loads adapters before generation). The base-only generation memory should come from Phase 1.

## Memory Accounting Verification

| Component | Claimed | Verified |
|-----------|---------|----------|
| Packed ternary (uint8) | 521.0 MB | 521.0 MB (210 tensors, measured) |
| Non-ternary (bf16) | 657.6 MB | 657.6 MB (332 tensors, measured) |
| Base total | 1178.6 MB | 1178.6 MB (matches active_mb) |
| 1 adapter B (bf16) | 21.9 MB | 21.9 MB (data size, measured) |
| 1 domain A (bf16) | 21.4 MB | 21.4 MB (data size, measured) |
| With adapter total | 1224.2 MB | 1224.2 MB (1178.6 + 21.9 + 21.4 + ~2.3 overhead) |
| Pre-merge unpacked | 4825.6 MB | 4827.3 MB (slight rounding, consistent) |

The memory accounting adds up. The core claim (1.22 GB serving) is supported by the data.

### One concern: skeleton memory is double-counted

MATH.md reports "1 domain A matrices (bf16): 21.4 MB" but Phase 2 measures `skeleton_one_domain_mb: 42.8 MB`. The 42.8 MB is the numpy fp32 skeleton; the 21.4 MB is after bf16 cast. This is consistent but the skeleton is stored as fp32 on disk (42.8 MB) and cast to bf16 at load time (21.4 MB). The PAPER memory budget table correctly uses the bf16 figure. No issue.

## Macro-Scale Risks (advisory)

1. **KV cache at long sequences.** At seq=8192, MATH.md estimates 629 MB of KV cache. This is a known concern but the experiment only tests seq=256 (where KV cache is negligible). The claim "still under 2 GB" at seq=8192 should be validated.

2. **Multiple concurrent adapters.** The scaling formula is linear but untested. At N=25 adapters (claimed 2.26 GB), peak memory during the wrapping phase may be higher than steady-state.

3. **The 82 tok/s claim needs context.** VISION.md reports 16.7 tok/s for bf16 merge and 12.3 tok/s for runtime LoRA at N=5. This experiment reports 82 tok/s for runtime LoRA at N=1. The 82 tok/s likely comes from the BitLinear Metal kernel being much faster than unpacked bf16 matmul (consistent with BitNet's design). But VISION.md was not updated with this finding, and the discrepancy (82 vs 12.3 tok/s) needs reconciliation. The 12.3 tok/s may have been from the old unpacked-bf16 runtime LoRA path.

4. **On-demand adapter loading latency.** The paper claims "swap from disk in ~2ms per adapter, per S-LoRA measurements" but S-LoRA operates on GPU with NVMe storage. Apple Silicon disk-to-unified-memory latency may differ. Untested.

## Verdict

**PROCEED**

### Justification

The core measurement is sound: BitLinear native serving with runtime LoRA achieves 1.22 GB total memory, a genuine and useful finding for the project. The memory accounting is tight and verified by `mx.get_active_memory()` with proper eval barriers. The finding directly advances VISION.md's goal of serving on Apple Silicon with minimal memory.

### Required fixes before citing in FINDINGS.md (4 items):

1. **Fix K2 hardcoding (line 867).** Compute `k2_quality_preserved` from actual PPL measurements: `abs(adapted_ppl - merged_ppl) / merged_ppl < 0.05`. Currently always `True`.

2. **Fix Phase 6 `base_only` config.** Use Phase 1's generation memory for `base_only`, not Phase 4's (which includes adapters). The `gen_mb` for `base_only` should be ~1181 MB, not 1224.2 MB.

3. **Reframe the "89% reduction" claim.** The honest framing is: "prior memory measurements were inflated by unnecessary bf16 unpacking; correct serving memory is 1.22 GB." The reduction is from a bug, not an optimization.

4. **Add PPL confidence intervals or at minimum note the comparison conflates ternary-vs-bf16 base weights with runtime-vs-premerge LoRA.** The 0.21% difference is within noise but the paper presents it as a clean comparison of serving strategies. Acknowledging this conflation in the limitations section is sufficient.

### Advisory (non-blocking):

- Add 3-5 run variance for the 82 tok/s throughput measurement.
- Reconcile 82 tok/s with VISION.md's 12.3 tok/s figure (update VISION.md if the old figure was from the buggy unpacked path).
- Phase 2 `active_mb` values (341.4 MB, 365.7 MB) are confounded by residual memory; flag or remove from the paper.
