# Peer Review: Merge Order Dependence

## NotebookLM Findings

Skipped -- manual deep review was sufficient given the experiment's scope and the clarity of the mathematical claims.

## Mathematical Soundness

### What holds

1. **Gram-Schmidt is order-dependent (Section 2.1).** Correct. The orthogonalized vectors d_k' depend on the subspace spanned by predecessors. Different orderings produce different decompositions of each vector into "kept" and "projected-out" components. The sum S(sigma) genuinely changes.

2. **Subspace invariance (Section 2.2).** Correct. GS is a QR decomposition. The column space of the Q matrix is the same regardless of column ordering. Only the individual Q columns change.

3. **Synthetic expert construction (Phase 2).** The formula d_k = alpha * shared + beta * unique_k with alpha = sqrt(c), beta = sqrt(1-c) correctly produces pairwise cosine approximately c when unique vectors are orthogonal to shared. The code properly removes the shared component from unique vectors (line 343 of test_merge_order.py), making the construction clean.

4. **SVD alternative analysis (Phase 3).** The Hungarian assignment approach correctly identifies the order-invariant single-basis-vector assignment. The signal retention penalty (2.5x worse) is a genuine consequence of projecting multi-component vectors onto single basis vectors.

### Issues found

**Issue 1: The "O(epsilon) bound" in MATH.md Section 2.3 is hand-waved, not proven.**

The theorem states: if all pairwise |cos(d_i, d_j)| < epsilon, then ||A(sigma) - A(tau)||_2 / ||A(sigma)||_2 = O(epsilon).

The "sketch" argues the projection removed is O(N * epsilon^2) in norm, then claims reallocation produces O(epsilon) variation. This skips a critical step. The variation between two orderings sigma and tau involves how much projection mass is redistributed, which depends on the specific permutation structure, not just on the magnitude of projections. The bound should be O(N * epsilon) in the worst case (each of N experts can have its projection redirected by O(epsilon)), and the paper's claimed O(epsilon) would only hold if the 1/N averaging cancels the N factor -- which it does for the average but not for the sum.

For the average A(sigma) = S(sigma)/N, this cancellation is plausible but unstated. The paper should make clear whether the bound is on the sum or the average, and the N-dependence should be explicit.

**Severity: Low.** The empirical results (30x below even the loose O(epsilon) bound) make the tightness of this bound irrelevant for the verdict. But the MATH.md should not claim a bound it hasn't proven.

**Issue 2: The "variation ~ 80 * cos" linear fit in Section 3 uses (1 - merged_cos_min) * 100 as the variation metric.**

This metric (line 462 of test_merge_order.py: `variation_pct = (1.0 - merged_cos_min) * 100`) measures the maximum angular deviation of the merged vector across orderings. This is a geometric deviation metric, not a quality variance metric. The paper implicitly equates "merged vector deviation" with "quality variance" when comparing to the 5% CV kill criterion. These are different quantities:
- CV measures std/mean of a scalar loss.
- (1 - cos_min) * 100 measures worst-case angular deviation of a high-dimensional vector.

The connection between angular deviation and loss CV is not established. A 1% angular deviation in a high-dimensional space could produce anywhere from 0% to much more than 1% loss variation depending on the loss landscape curvature.

**Severity: Medium.** The Phase 1 results (which use actual loss CV) are the ones that matter for kill criteria. The Phase 2 synthetic results are a stress test using a proxy metric. The paper should clearly label (1 - cos_min) as a geometric proxy, not conflate it with quality CV. This does not affect the Phase 1 verdict.

**Issue 3: The scaling claim to d=896 and d=4096 (Section 2.5) relies on cosine values from other experiments.**

MATH.md states: "At d=896 (Qwen 0.5B), measured cos = 0.0002." This experiment did not measure this -- it comes from the structural_orthogonality_proof experiment. The chain of reasoning is: (a) cos scales as ~d^(-0.673), (b) at d=896 cos = 0.0002, (c) the linear relationship variation ~ 80*cos gives variation ~ 0.016%. This is a valid extrapolation but should be clearly labeled as depending on external results, not as an internal finding.

**Severity: Low.** The extrapolation is reasonable and properly referenced elsewhere in the project.

## Novelty Assessment

**Prior art check:** No specific prior work was found analyzing GS merge order dependence for LoRA adapters. MDM-OC (arXiv:2507.20997) uses GS with fixed ordering but does not analyze order sensitivity. The experiment correctly identifies this gap.

**Delta over existing work:** This is a thorough empirical investigation of a known mathematical property (GS order dependence) applied to a specific context (LoRA composition). The novelty is in the quantitative characterization (the variation ~ 80*cos relationship, the threshold analysis) rather than in fundamental new mathematics.

**Reinvention check:** The GS implementation in `gram_schmidt.py` is standard Classical Gram-Schmidt. No reinvention of existing tools from `references/`. The SVD alternative and Symmetric GS are straightforward implementations of well-known techniques.

## Experimental Design

### Strengths

1. **Three-phase design is well-structured.** Phase 1 tests the actual hypothesis with real experts. Phase 2 characterizes the scaling relationship. Phase 3 evaluates alternatives. Each phase answers a distinct question.

2. **Kill criteria are testable and well-chosen.** CV < 5% and worst/best < 15% are concrete, falsifiable thresholds with clear connection to practical significance.

3. **The margins are overwhelming.** 175x below kill threshold is not a borderline result. Even with concerns about the bound tightness or metric proxies, the conclusion is robust.

4. **Multiple conditions tested.** Two seeds for N=5, one seed for N=8, seven overlap levels for synthetic. The coverage is adequate for the claim.

### Weaknesses

**W1: Phase 2 uses a proxy metric, not the kill criteria metric.**

The kill criteria are defined in terms of quality variance (CV of NTP loss). Phase 2 measures merged vector cosine deviation, not loss. The paper should acknowledge this gap more clearly. Phase 1 is the authoritative test; Phase 2 is supplementary.

**W2: N=8 has only 1 seed.**

The paper notes this (Limitation 4) and argues the 175x margin makes additional seeds unnecessary. This is reasonable but worth flagging: the N=8 condition is the one where order effects should be largest, and a single seed means we cannot assess seed-to-seed variability at higher N.

**W3: The "GS is unnecessary" conclusion (Section 5.2) is borrowed from the parent experiment.**

This experiment tests order dependence, not GS necessity. The conclusion that "GS is unnecessary" and "simple averaging is equivalent" is inherited from gram_schmidt_composition. It is correctly attributed but should not be presented as a finding of this experiment.

**W4: 20 orderings out of N! possible orderings.**

For N=5, there are 120 possible orderings. Testing 20 covers 1/6. For N=8, there are 40,320 possible orderings. Testing 20 covers 0.05%. The paper does not discuss whether 20 orderings is sufficient to capture the worst case. Given the extreme margins, this is unlikely to matter, but for N=8 it is theoretically possible that a pathological ordering exists in the untested 99.95%. The synthetic Phase 2 stress test partially mitigates this concern by showing the relationship is smooth and predictable.

### Hypothesis Graph Consistency

The HYPOTHESES.yml entry for `exp_merge_order_dependence` lists:
- K1: quality variance across 10 random orderings > 5%
- K2: worst ordering > 15% worse than best ordering

The experiment tests 20 orderings (more than the 10 specified) and evaluates both K1 and K2 on actual NTP loss. The kill criteria are the ones actually tested. The status "proven" is appropriate: the hypothesis that "GS merge order affects quality" is proven to be true in principle but irrelevant in practice for SOLE's operating regime.

## Macro-Scale Risks (advisory)

1. **Attention-layer cosines are much higher (cos=0.85 for related domains).** The paper's Section 5.3 acknowledges this. At macro scale with layer-wise cosines this high, order dependence within attention layers could exceed the threshold. Layer-wise GS (rather than flattened-vector GS) should be tested at macro.

2. **The linear relationship (variation ~ 80*cos) was calibrated with uniform-cosine synthetic experts.** Real macro experts have heterogeneous per-layer cosines. Layers with cos=0.85 (attention) coexist with layers at cos=0.001 (FFN). The flattened-vector analysis may mask layer-specific order sensitivity.

3. **The 1/N dilution effect means order dependence decreases with N.** At N=500 (production scale), each expert contributes 0.2% of the merge, making ordering even more irrelevant. This is a positive macro signal.

## Verdict

**PROCEED**

The experiment cleanly answers its hypothesis with overwhelming margins (175x below kill threshold). The three-phase design is thorough, the code is correct, and the conclusions are properly scoped.

Two non-blocking improvements recommended:

1. **Clarify the O(epsilon) bound in MATH.md Section 2.3.** Either prove it rigorously (showing the 1/N cancellation of the N-factor) or label it as a conjecture supported by empirical evidence. Currently it reads as a proven theorem but the sketch has a gap.

2. **Label the Phase 2 variation metric as a geometric proxy.** The (1 - cos_min) * 100 metric is not the same as quality CV. Add a sentence noting this distinction and that Phase 1 (which uses actual loss CV) is the authoritative test for the kill criteria.

Neither fix affects the verdict. The core result -- GS merge order is mathematically real but practically irrelevant for SOLE -- is sound.
