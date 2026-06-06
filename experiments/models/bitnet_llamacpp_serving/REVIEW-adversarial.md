# Peer Review: exp_bitnet_llamacpp_serving (Re-Review)

## Previous Review Status

The first review issued REVISE with 1 blocking and 4 non-blocking fixes. This re-review verifies all 5 fixes.

## Fix Verification

### Fix 1 (BLOCKING): Adapter size unit error (KB to MB)

**PAPER.md**: All four occurrences now correctly read "82.5 MB" (lines 62, 105, 110, 134). No residual KB references. **FIXED.**

**MATH.md**: Lines 119 and 121 correctly read "82.5 MB" and the memory budget calculation `floor(1024/82.5) = 12 adapters` is correct. **FIXED.**

**results.json**: Field `size_per_adapter_mb: 82.5` was already correct. **OK.**

**PAPER.md total memory**: Updated from 1.16 GiB to 1.53 GiB (line 108). Verification: 1.11 GiB base + 5 * 82.5 MB + 24 MiB buffer = 1.11 + 0.384 + 0.023 = ~1.52 GiB. The 1.53 GiB figure is close enough (GiB/GB rounding). **FIXED.**

**HYPOTHESES.yml line 3526**: Still reads "82.5KB each". **NOT FIXED.** This is the evidence claim for the experiment node. It propagates the old unit error to anyone reading the hypothesis graph.

**FINDINGS.md line 103**: Still reads "82.5KB each", "Per-adapter marginal ~9.4%", and "Memory: 1.16 GiB with 5 adapters". **NOT FIXED.** Three stale values remain.

### Fix 2 (Non-blocking): K3a rename

PAPER.md line 87 now reads "K3a -- Deterministic reproducibility" with an explicit note that this tests seed-deterministic reproducibility across separate runs, not in-process hot-swap. **FIXED.**

### Fix 3 (Non-blocking): MATH.md unit

All MATH.md references corrected to MB. Memory budget calculation updated. **FIXED.** (Covered in Fix 1 above.)

### Fix 4 (Non-blocking): Affine overhead model

MATH.md lines 80-90 now present the fitted affine model `overhead(N) = 9.5% + 7.5% * N` with a table showing predicted vs measured at N=1,3,5. Verification:
- N=1: 9.5 + 7.5 = 17.0% (measured 17.0%) -- exact fit
- N=3: 9.5 + 22.5 = 32.0% (measured 34.4%) -- 2.4pp residual
- N=5: 9.5 + 37.5 = 47.0% (measured 47.1%) -- 0.1pp residual

The fit is anchored by the two endpoints (N=1, N=5) with a 2.4pp residual at N=3, suggesting mild super-linearity. With only 3 data points this is not statistically distinguishable from linear. The model is honest.

The N_max calculation: floor((50 - 9.5)/7.5) = floor(5.4) = 5 adapters at rank-16. Correct. **FIXED.**

### Fix 5 (Non-blocking): K2 confidence interval

MATH.md lines 94-99 now present a delta method CI. Verification:
- SE(tg_5x) = 0.6/sqrt(3) = 0.346, SE(tg_base) = 2.3/sqrt(3) = 1.328
- overhead = 1 - X/Y where X=33.8, Y=63.9
- SE(overhead) = sqrt((SE_X/Y)^2 + (X*SE_Y/Y^2)^2) = sqrt((0.346/63.9)^2 + (33.8*1.328/63.9^2)^2) = sqrt(2.93e-5 + 1.21e-4) = 0.0123 = 1.23%
- With t_{0.025, df=2} = 4.303: CI = 47.1% +/- 5.3%
- Range: [41.8%, 52.4%]

The math checks out. The paper correctly notes that the 50% threshold falls within the 95% CI, making the K2 PASS marginal. **FIXED.**

## Mathematical Soundness

All calculations in MATH.md are arithmetically correct:
- Per-layer LoRA FLOPs: 1,441,792 (verified each projection)
- Per-token per-adapter: 43.3M FLOPs (30 layers * 1.44M)
- Predicted overhead: 0.9% per adapter (43.3M / 4.8G)
- 10x gap explanation (bandwidth-bound) is qualitatively sound
- Affine model fit is correct
- Delta method CI is correct
- Memory model: 87.5M bytes = 83.4 MB, measured 82.5 MB (consistent)

One minor note: the MATH.md line 42 "Wait -- that seems too high" is informal notebook-style prose left in the final document. Not blocking, but unusual for a paper artifact.

## Novelty Assessment

Engineering validation experiment, not a research contribution. This is appropriate -- it tests whether llama.cpp works for a specific use case. The three converter patches and the MLX-to-GGUF adapter converter are useful tooling contributions. No prior art conflict.

## Experimental Design

**K1 (Loading)**: Sound. All 5 adapters load. Converter patches documented.

**K2 (Throughput)**: Sound. Server API benchmarking with proper separation of prompt eval and token generation. 3 runs per config. True baseline without adapters loaded. The adapter size issue from the first review is resolved -- results.json confirms `size_per_adapter_mb: 82.5`, and the total memory of ~1.53 GiB is consistent with 5 real 82.5 MB adapters.

**K3 (Hot-swap)**: K3a is now honestly labeled as a deterministic reproducibility test (not a hot-swap test). K3b is a genuine hot-swap test. Both pass.

## Remaining Issue: Stale References Outside PAPER.md/MATH.md

The unit fix was applied to PAPER.md and MATH.md but NOT propagated to:

1. **HYPOTHESES.yml** (line 3526): "82.5KB each" -- should be "82.5 MB each"
2. **FINDINGS.md** (line 103): Three stale values:
   - "82.5KB each" -- should be "82.5 MB each"
   - "Per-adapter marginal ~9.4%" -- should be "Per-adapter marginal ~7.5% (plus 9.5% fixed overhead)"
   - "Memory: 1.16 GiB with 5 adapters" -- should be "Memory: ~1.53 GiB with 5 adapters"

Additionally:
3. **results.json** (line 41): `"per_adapter_marginal_overhead_pct": 9.4` is now stale given the affine model. The 9.4% is the average overhead per adapter (47.1/5), not the marginal cost. This is a data file so less critical, but the label "marginal" is misleading.

## Macro-Scale Risks (advisory, not blocking)

1. **CPU-only limitation**: Metal/GPU support for TQ types would change the overhead equation. The 7.5%/adapter marginal is CPU-specific.
2. **Per-request routing**: llama-server sets adapters globally, not per-request. Production multi-tenant serving needs custom solution.
3. **K2 margin is razor-thin**: 47.1% vs 50% with CI [41.8%, 52.4%]. Any additional overhead source (larger rank, more projections, longer sequences) could push past threshold.
4. **Rank-8 extrapolation unverified**: The claim of ~10 adapters at rank-8 assumes linear cost halving. Not tested.

## Verdict

**PROCEED** (with non-blocking fixes)

The core experiment is sound. All 5 requested fixes were applied to the primary documents (PAPER.md and MATH.md). The math is correct, the experimental design is adequate, and the conclusions are supported by the data within stated uncertainty. The K2 margin is thin but honestly reported with a proper confidence interval.

The remaining issues are **non-blocking** propagation fixes in ancillary files, not errors in the experiment itself:

1. Update HYPOTHESES.yml evidence claim: "82.5KB" to "82.5 MB"
2. Update FINDINGS.md entry: fix all three stale values (KB to MB, 9.4% to 7.5%+9.5%, 1.16 GiB to 1.53 GiB)
3. Consider relabeling results.json `per_adapter_marginal_overhead_pct` to `per_adapter_average_overhead_pct` or updating to the affine model parameters

These should be fixed for consistency but do not affect the experiment's validity or its SUPPORTED status.
