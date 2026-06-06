# Peer Review: benchmark_composition_latency_sweep

## Mathematical Soundness

### FLOP Analysis (Section 1.1-1.2): Correct
The derivation of 223M FLOPs per adapter merge (210M matmul + 13M scale/add) is correct. The forward pass FLOP count (13.1M per token) is correct. The linear-in-N prediction follows directly from the independence of adapter deltas.

### Memory Bandwidth Analysis (Section 1.3): Minor Error
The per-adapter bandwidth calculation claims ~39.5 MB per adapter. Breaking this down:
- Read B_i: 81.9 KB (correct)
- Read A_i: 81.9 KB (correct)
- Write delta: 13.1 MB (correct)
- Read/write W: 26.2 MB (correct -- read 13.1MB + write 13.1MB)

However, this assumes the delta is materialized to memory before being added to W. MLX with `mx.compile` may fuse these operations, so the bandwidth bound of 0.145 ms/adapter is a conservative lower bound. The actual measured uncompiled time (~0.3 ms/adapter at large N) is roughly 2x the bandwidth floor, which is reasonable for unfused Python-loop dispatch.

### Scaling Law Fit: Methodologically Questionable

This is the most important critique. The claimed sub-linear scaling (alpha=0.83) is an artifact of fitting a power law across a range where constant overhead dominates the small-N regime.

**Evidence:**
- N=1: 0.918 ms, N=2: 0.831 ms. Latency DECREASED from N=1 to N=2. This is physically impossible for a pure power law T = a*N^alpha. It reveals that a significant constant term c_0 exists (as MATH.md Section 2 correctly predicts: T = c_0 + c_1*N).
- The correct model is **affine**, not power-law: T = c_0 + c_1*N. The paper's own MATH.md derives this on line 70 but then the code fits a power law instead.
- At large N (50-100), the data is nearly perfectly linear: 15.34 -> 22.77 -> 30.38. The ratios 30.38/15.34 = 1.98x for 2x N, and (30.38-15.34)/(100-50) = 0.301 ms/adapter.

**What would an affine fit show?** Fitting T = c_0 + c_1*N to the N >= 5 data would give c_1 approximately 0.30 ms/adapter (from the large-N slope) and c_0 approximately 0.2-0.4 ms. This is LINEAR scaling with a constant offset, not sub-linear scaling. The power-law exponent alpha < 1 is a fitting artifact caused by including small-N points where the constant overhead c_0 dominates.

**Impact on claims:** This does NOT invalidate the kill criterion (K1 asks for superlinear, alpha > 1.05, which is clearly not present). The absolute latencies are real and well-measured. But the headline claim "sub-linear scaling" is misleading. The correct characterization is "linear scaling with constant overhead that makes it appear sub-linear when fit to a power law."

### Compiled Merge Alpha (0.73): Same Artifact, Amplified
The compiled merge shows even lower alpha (0.73) because mx.compile has a larger constant overhead (compilation cost amortized into warmup, but Metal graph setup remains). The same affine-vs-power-law confusion applies.

### LORA_SCALE Unused
`LORA_SCALE = 20.0` is defined but never used. The alphas are set to uniform 1/N, not incorporating the LoRA scale factor. This does not affect timing measurements (scale is a scalar multiply, negligible), but it is a code hygiene issue and means the merge does not exactly match the architecture's actual merge equation.

## Novelty Assessment

This is a straightforward engineering benchmark, not a novel mechanism. That is appropriate -- the architecture needs this data point. No novelty claim is made.

**Prior art:** The experiment correctly cites Naive LoRA Summation (2508.11985) and LoRA Soups (2410.13025) for the composition mechanism. It correctly builds on the project's own batched_premerge_throughput results.

**Delta over prior work:** This experiment adds the N-scaling characterization and mx.compile speedup measurement on M5 Pro. The batched_premerge_throughput experiment showed runtime LoRA dominates pre-merge for per-token routing but did not sweep N systematically for the pre-merge path.

## Experimental Design

### Strengths
1. **Timing methodology is correct.** Uses `mx.eval()` to force synchronous execution before timing, which is the right approach for MLX lazy evaluation. Warmup rounds (5-8) are adequate.
2. **Multiple strategies tested.** Uncompiled, compiled, and cached delta approaches provide useful comparison.
3. **Bottleneck decomposition (Phase 4).** Separating matmul from accumulation time is genuinely informative and well-designed.
4. **Adequate measurement count.** 20 measurements per point with p50/p95 reporting. Standard deviations are small (< 3% relative for most points), indicating stable measurements.
5. **Forward pass constant verification.** Confirming ~0.19ms regardless of N validates that pre-merge pays cost at merge time only.

### Weaknesses
1. **No confidence intervals on alpha.** A single point estimate alpha=0.83 with no uncertainty quantification. Given the N=1/N=2 anomaly, bootstrap or leave-one-out analysis would reveal how sensitive the fit is.
2. **Synthetic weights only.** Acknowledged in limitations. Random bf16 weights have different memory access patterns than ternary {-1, 0, +1} weights (which have ~42% sparsity and could potentially be faster to merge). However, the bottleneck is accumulation of the full 2560x2560 delta, not the adapter matmul, so this is unlikely to change conclusions.
3. **Single-thread, no contention.** Acknowledged. In production, Metal would be shared with the forward pass of other layers, potentially increasing latency.
4. **Multi-layer extrapolation is naive.** Phase 3 measures 7 layers but does not use mx.compile. The paper extrapolates "~550 ms compiled for 24 blocks" by applying the single-layer compile factor (2.35x) to the uncompiled multi-layer measurement. This is reasonable but not validated -- mx.compile across multiple layers may have different behavior than single-layer compilation.

## Kill/Success Criteria Assessment

### K1 (Superlinear scaling, alpha > 1.05): PASS -- Correctly Evaluated
The data clearly shows no superlinear scaling. Even with my critique of the power-law fit, an affine fit T = c_0 + c_1*N is strictly sub-linear in the O-notation sense (it is O(N), alpha=1.0 exactly). The kill criterion threshold is 1.05, and no reasonable fit gives alpha anywhere near that. **K1 PASS is valid.**

### S1 (Interactive at N=25, <50ms): PASS -- Correctly Evaluated
N=25 uncompiled: 7.72 ms. N=25 compiled: 3.28 ms. Both well under 50 ms. **S1 PASS is valid.**

### S1 (Sub-linear scaling): Overclaimed
As argued above, the scaling is linear with constant overhead, not genuinely sub-linear. The per-adapter marginal cost is approximately constant at ~0.30 ms for large N. However, this does not affect the practical conclusion -- linear scaling is perfectly fine for the architecture.

## Macro-Scale Risks (advisory)

1. **Full-model merge at N=25:** The extrapolated ~550 ms (compiled) for 168 projections is acceptable for session-start but should be validated with an actual 24-block model, not extrapolation. Different layer dimensions (e.g., GQA head projections smaller than 2560x2560) would change the picture.

2. **Concurrent Metal workloads:** When serving tokens while re-merging (e.g., adapter hot-swap), the merge will contend for memory bandwidth with the forward pass. The 3.28 ms per-projection could increase significantly.

3. **Memory pressure at scale:** At N=25 with 168 projections, the adapter parameters themselves are small (168 * 25 * 163.8 KB = 687 MB), but if intermediate delta allocations are not freed promptly, peak memory could spike.

4. **Ternary weight interaction:** Real ternary base weights stored in packed format may require unpacking before merge, adding latency not captured here.

## Verdict

**PROCEED**

The experiment is a well-executed engineering benchmark that provides actionable data for the architecture. The absolute latency measurements are sound, the timing methodology is correct, and both kill (K1) and success (S1) criteria are legitimately passed.

The main flaw is the "sub-linear scaling" narrative, which is a fitting artifact. The correct characterization is "linear scaling with constant overhead." This does not change any architectural decisions or invalidate any claims about interactive feasibility.

### Recommended Fixes (non-blocking)

1. **Recharacterize the scaling law.** Fit T = c_0 + c_1*N (affine) in addition to the power law. Report c_1 as the per-adapter marginal cost. Note that the power-law alpha < 1 is driven by constant overhead at small N, not genuine sub-linearity.

2. **Add confidence interval on alpha.** Bootstrap the 8-point fit or report sensitivity to dropping N=1 and N=2 from the fit.

3. **Remove or use LORA_SCALE.** Either incorporate the 20.0 scale factor into the merge or remove the unused variable.
