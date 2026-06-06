# Peer Review: adapter_hot_cache_mlx

## NotebookLM Findings

Skipped -- the experiment is a KILL with strong quantitative evidence across all conditions. The mathematical claims are verifiable from the code and results directly.

## Mathematical Soundness

### Cache hit rate model (MATH.md Section 2-3): Partially correct, partially misleading

The Zipf hit rate model `hit_rate(K, alpha) = H_{K,alpha} / H_{M,alpha}` in MATH.md Section 2 assumes pairs are independently Zipf-distributed and ordered by decreasing frequency. This is the correct formula for *static* top-K coverage under Zipf, but MATH.md presents it as a prediction for LRU cache performance, which it is not. The paper and results correctly distinguish these (static coverage vs LRU hit rate), showing a ~2.5x gap. No mathematical error, but the MATH.md framing could mislead a reader into thinking these are the same thing.

**Specific numbers check:** MATH.md claims "K=20, M=1225, alpha=1.0: hit_rate = H_{20,1}/H_{1225,1} = 3.55/7.11 = 49.9%". The actual measured *static* top-20 coverage at alpha=1.0 is 27.2% (from results.json phase1). The discrepancy: the Zipf model assumes pure Zipf on pairs, but the simulation applies Zipf to *domains* and then derives pairs through secondary selection. MATH.md acknowledges this ("routing is NOT random -- it's deterministic given the input embedding") but still presents the pure-Zipf pair formula as if it were predictive. The measured values are substantially worse than the formula predicts, which means the MATH.md formula is optimistic, making the kill *more* justified than the math alone would suggest.

### Combinatorial argument: Sound

C(50,2) = 1225 is correct. The argument that top-k routing expands the effective cache space from N to C(N,k) is mathematically precise and represents the core insight. This is well-reasoned.

### Memory analysis: Correct

Per-pair delta at d=2560, bf16: 28 layers * 7 targets * 2560 * 2560 * 2 bytes = 2,569 MB. Verified. The conclusion that K=20 pairs at production scale requires 50+ GB is arithmetic fact.

### Latency threshold derivation: Sound

The break-even condition `P(hit) > T_overhead / (T_merge - T_lookup + T_overhead)` is correctly derived from the expected latency comparison. The conclusion that hit rate must exceed only ~1.2% to be latency-beneficial is correct -- but this is mooted by the memory infeasibility at production scale.

### Weight sensitivity analysis: Methodologically sound but trivial

The L2 distance between deltas at weights (0.7, 0.3) vs (0.9, 0.1) is measured at 37.2%. This is mathematically expected: the delta is `w1*D1 + w2*D2`, so changing (w1,w2) by (0.2, -0.2) changes the output by `0.2*(D1-D2)`. For random D1, D2, `||D1-D2|| / ||0.7*D1+0.3*D2||` is approximately `0.2 * sqrt(2) / sqrt(0.49+0.09) ~ 0.37`. The measured value matches the theoretical expectation exactly. This confirms the analysis is correct but also means it was predictable without running the experiment.

## Novelty Assessment

No novelty claim is made -- this is a feasibility study for a serving optimization. The references to S-LoRA and EdgeMoE are appropriate. The conclusion (runtime LoRA supersedes pair caching) is well-established in the multi-LoRA serving literature. S-LoRA specifically keeps adapters in factored form for exactly this reason.

The experiment's value is in quantifying the failure for the specific Composable Ternary Experts architecture (N=50, top-2, ternary LoRA). This is legitimate engineering due diligence.

## Experimental Design

### Strengths

1. **Multi-phase design is excellent.** Separating access pattern analysis (Phase 1), LRU simulation (Phase 2), latency measurement (Phase 3), end-to-end (Phase 4), memory (Phase 5), and weight sensitivity (Phase 6) provides clear evidence at each level. Any single phase failing would be sufficient for a kill.

2. **Range of Zipf alphas tested.** Testing alpha from 0.0 to 1.5 plus domain-balanced covers the full realistic range. The kill is not conditional on a specific alpha value.

3. **Kill criteria are well-designed.** K1 (50% hit rate on balanced traffic) is the right criterion -- if caching fails on balanced traffic, it fails for the common serving scenario.

### Issues

**Issue 1: Secondary selection model is synthetic and biased by index ordering.**

The secondary adapter selection uses `1/(1+|i-j|)` where i,j are adapter *indices*. This means adapter 0 is always "close" to adapter 1 and "far" from adapter 49. Real router confusion clusters (from exp_softmax_router_scaling) are semantic, not index-ordered. This proximity model:

- Creates artificial locality in the pair space (adjacent indices always co-occur more)
- Could either help or hurt caching depending on whether real clusters are tighter or looser

However, the domain-balanced traffic (which is the K1 criterion) applies this same proximity model uniformly across all domains, so the bias is symmetric. The key result -- that even under Zipf(1.5) with concentrated traffic, LRU K=20 only achieves 22.3% -- is driven by the combinatorial argument, not the secondary selection model. The specific secondary model affects the exact numbers but not the conclusion.

**Verdict on Issue 1:** Not kill-invalidating. The combinatorial argument (1225 pairs >> 20 cache slots) dominates regardless of the secondary selection distribution.

**Issue 2: E2E simulation caches pair deltas with fixed weights from the first miss.**

In Phase 4 (lines 519-529), on a cache miss, the delta is computed with the query's specific routing weights and cached. On a subsequent hit with the *same pair but different weights*, the stale delta is used. This is the bug that Phase 6 identifies (37% L2 error from weight variation). However, this bug makes the cache *look better* than it actually is (hits return wrong deltas without penalty), not worse. It does not invalidate the kill.

**Issue 3: The S2 criterion is misleading.**

S2 reports "2938x speedup" which is the ratio of merge latency to dict lookup latency. This measures Python dict performance, not the cache mechanism. The actual end-to-end speedup is 1.01-1.07x (Phase 4), which is the meaningful number. The PAPER.md correctly notes this ("the mechanism works perfectly -- the problem is that hits are too rare") but listing S2 as PASS is misleading in isolation. The criterion should have been end-to-end speedup, not per-hit speedup.

**Issue 4: No cache warming test.**

The PAPER.md Limitations section acknowledges this. A pre-warmed cache (populated with the K most frequent pairs from historical traffic) would eliminate the cold-start penalty and the LRU churn problem for the top-K pairs. Under Zipf(1.5), static top-20 coverage is 41.4% vs LRU 22.3% -- warming could nearly double the hit rate. This still fails K1 (41.4% < 50%) but the gap is meaningful. Not a kill-invalidating omission, but the LEARNINGS.md claim that "LRU eviction makes things worse" is only true for *unpopulated* LRU, not for warmed caches.

## Kill Justification Assessment

The kill is justified on multiple independent grounds:

1. **Combinatorial**: C(50,2) = 1225 >> K=20. This alone is sufficient.
2. **Empirical**: 3.9% hit rate on balanced traffic (K1 threshold: 50%). Margin: 12.8x below threshold.
3. **Memory**: 2.5 GB per pair at production scale. K=20 requires 50 GB. Exceeds hardware.
4. **Superseded**: Runtime LoRA eliminates the need entirely (O(k*r*d) vs O(d^2) per merge).

Any single one of these would justify the kill. Together, they make revival essentially impossible at N=50, top-2.

## LEARNINGS.md Assessment

The learnings correctly capture the key implications:

- Pair space combinatorics as the fundamental barrier: correct
- Domain Zipf does not imply pair Zipf: correct and important
- Runtime LoRA as the correct alternative: consistent with prior experiments
- The "fundamental asymmetry" observation (caching trades memory for compute, but factored form is already optimal): this is the most valuable insight and is correctly articulated

**One missing nuance:** The LEARNINGS.md says "Cache is useful only at small N" and gives N<=10 as the threshold. But it then immediately notes "at production scale: still 29 GB, so factored form still wins." This means caching is not useful at *any* N at production dimensions -- the small-N advantage exists only at micro scale. This should be stated more clearly.

## Macro-Scale Risks (advisory)

Not applicable -- the experiment is killed. The macro-scale implication (use runtime LoRA, not pair caching) is already the architecture's serving strategy per VISION.md.

If someone revisited this for a different regime (N<10, top-1 routing), the key macro risk is that the weight sensitivity problem (37% L2 error from weight variation) means pair-level caching requires either quantized weight bins (expanding the key space) or per-adapter delta caching with runtime weighted sum (which is just runtime LoRA with an extra cache layer).

## Verdict

**PROCEED** (kill confirmed)

The kill is justified on four independent grounds (combinatorial, empirical, memory, superseded). The experimental methodology is sound -- the synthetic secondary selection model and the stale-weight cache bug both bias results in favor of the cache (not against it), making the measured failure rates conservative lower bounds on the actual failure. The MATH.md analysis is correct in its derivations though optimistic in its Zipf pair predictions (measured hit rates are worse than the formula predicts). The LEARNINGS.md correctly identifies runtime LoRA as the definitive alternative.

No revisions needed. This is a clean, well-executed negative result that closes a design path and strengthens the case for the existing runtime LoRA serving architecture.
