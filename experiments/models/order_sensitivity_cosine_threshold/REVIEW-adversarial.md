# Peer Review: Order Sensitivity Cosine Threshold

## NotebookLM Findings

Skipped -- the experiment is straightforward synthetic linear algebra with no deep conceptual ambiguity requiring external review. The math is verifiable by hand.

## Mathematical Soundness

### Power-Law Fits

**Threshold fit (cos_5pct):** Verified by hand. The formula `cos_5pct(N) = 0.616 * N^(-0.935)` reproduces all four interpolated data points to within 1.4%. R2=0.9999 is credible given the near-perfect match. This is the formula used for all safety predictions, and it is sound.

**Slope fit:** The formula `slope(N) = 13.8 * N^0.79` (R2=0.971) is notably less accurate. Hand-checking:
- N=5: predicted 53.4 vs actual 43.8 (22% error)
- N=50: predicted 346.7 vs actual 277.1 (25% error)

The R2=0.971 is from log-log regression, which compresses residuals. This fit is used only for intuition (Section 3.2 of MATH.md), not for safety predictions, so the inaccuracy is non-blocking. However, the theoretical argument in Section 3.4 predicts variation ~ O(N * cos), i.e., slope ~ O(N), while the empirical exponent is 0.79 -- the paper correctly notes this as "sub-linear due to averaging." The theoretical argument is informal but directionally correct.

**INCONSISTENCY (blocking):** PAPER.md line 127 states `cos_5pct(N) = 0.37 * N^(-0.80)` in the Verdict section, while the rest of the paper and all of MATH.md use `cos_5pct(N) = 0.616 * N^(-0.935)`. These give substantially different predictions (e.g., at N=100: 0.0060 vs 0.0083, a 38% difference). The 0.616 formula fits the data; the 0.37 formula appears to be a stale artifact from the no-intercept slope fit (5/slope rather than interpolation). This must be corrected.

### Variation Metric vs CV Terminology

The kill criteria and hypothesis refer to "CV exceeds 5%" but the actual metric is `(1 - min_pairwise_cosine) * 100`, which is a worst-case angular deviation, not a coefficient of variation. The code computes actual norm CV separately (which is negligible). The metric is well-defined and consistently applied in the code and results. The terminology mismatch between "CV" (in HYPOTHESES.yml kill criteria) and "variation%" (in the code/paper) is confusing but does not affect the scientific validity. The experiment does answer the hypothesis it claims to test.

### Synthetic Vector Construction

The construction `d_k = alpha * shared + beta * unique_k` with `alpha = sqrt(cos)`, `beta = sqrt(1-cos)` produces vectors with pairwise cosine approximately equal to `target_cosine`. This is a standard construction. The code correctly verifies actual cosines and uses them for fitting. The actual cosines systematically exceed target cosines (e.g., target 0.005 -> actual 0.010, target 0.01 -> actual 0.015). This is expected: the shared component contributes `alpha^2 = target_cosine`, but the unique components have nonzero inner products at order `1/sqrt(D)`, adding a positive bias. At D=4096 this is `~0.016`, explaining the offset at low target cosines. The fits use actual cosines, so this is correctly handled.

### Safety Margin Extrapolation

The paper claims cos=0.0002 at d=896 (production). This comes from the structural orthogonality characterization experiment. The safety margin calculation (e.g., N=5000: margin 1.1x) is a pure arithmetic consequence of the fit, which I verified. The concern is the extrapolation: the fit is calibrated at N in {5, 10, 20, 50} and extrapolated to N=5000. Power laws can break at extreme extrapolation. However, the near-unity exponent (-0.935 approximately -1) has a clean theoretical interpretation (threshold ~ 1/N), which gives some confidence.

The paper does NOT claim precision at N=5000 -- it says "approximately" and "~". The 1.1x margin at N=5000 is appropriately flagged as tight. This is honest.

## Novelty Assessment

This is an internal calibration experiment, not a novel research contribution. It corrects an earlier finding (fixed cos=0.06 threshold) with an N-dependent formula. The GS order-sensitivity analysis is standard linear algebra. The novelty is in its application to the SOLE architecture's safety guarantees, which is project-specific.

No prior art concerns -- this is original experimental work within the project's hypothesis tree.

## Experimental Design

**Does it test what it claims?** Yes. The experiment directly tests whether cos=0.06 is a universal threshold by sweeping N. K1 is clearly killed (6 violations). K2 clearly passes. The corrected formula is derived from the same data.

**Controls:** 3 seeds, 50 orderings, 15 cosine points, 4 N values = 180 conditions with 3-seed averaging. Adequate for a synthetic experiment.

**Confounds:** The linear interpolation for exact thresholds is simple but adequate given the fine cosine grid (15 points from 0.005 to 0.30). The interpolated thresholds are very close to the 5/slope estimates, providing cross-validation.

**Could a simpler mechanism explain the result?** The result IS the simple mechanism: GS sensitivity scales with N because more vectors means more accumulated projection loss. The O(N * cos) scaling is the simplest possible explanation.

**Norm CV dead metric finding:** Well-supported (CV < 0.01% everywhere). The distinction between magnitude preservation and direction change is useful for understanding what order sensitivity actually affects.

## Hypothesis Graph Consistency

The experiment is correctly linked to `exp_layerwise_order_sensitivity` and `exp_merge_order_dependence` as dependencies. Status is "killed" which matches K1 being violated. The `blocks: []` is correct since the corrected formula replaces the old threshold without blocking anything -- SOLE safety is preserved with the new formula.

The evidence entry in HYPOTHESES.yml uses the corrected formula (0.616 * N^(-0.935)), matching MATH.md.

## Macro-Scale Risks (advisory)

1. **Extrapolation to N > 50:** The power law is fitted on N = {5, 10, 20, 50} and extrapolated to N = 5000. The near-unity exponent provides theoretical grounding, but macro validation at N = 100+ with real LoRA deltas would increase confidence.

2. **Non-uniform cosines:** Real expert ensembles have non-uniform pairwise cosines. The paper correctly notes the threshold applies to "max pairwise cosine," but within-cluster cosines (e.g., 5 math experts) may be much higher than 0.0002. The domain-clustering concern is properly flagged.

3. **Flattened vs per-layer:** D=4096 flattened vectors may differ from per-layer behavior. The killed sibling experiment showed slope ~ 62 vs 80 at N=10, so the coefficients differ but the N-dependence pattern should transfer.

## Verdict

**REVISE**

The experiment is sound, the math checks out, and the kill criteria assessment is correct. One blocking fix:

1. **Fix the stale formula in PAPER.md Verdict section (line 127).** Replace `cos_5pct(N) = 0.37 * N^(-0.80)` with `cos_5pct(N) = 0.616 * N^(-0.935)`. The stale formula gives 38% different predictions at N=100 and could mislead anyone reading only the Verdict section. This is a copy-paste error, not a mathematical disagreement.

Non-blocking (recommended but not required for PROCEED):

2. **Clarify the "CV" vs "variation%" terminology** in either the paper or HYPOTHESES.yml. The kill criteria say "CV" but the metric is worst-case cosine deviation, not coefficient of variation. The distinction matters because the actual norm CV is a separate (dead) metric in this experiment.

3. **Add a note on the slope fit quality.** The slope power law (R2=0.971) has 20-25% errors at the endpoints. The paper presents it without caveats. A sentence noting the threshold fit (R2=0.9999) is the canonical formula for safety predictions would help.
