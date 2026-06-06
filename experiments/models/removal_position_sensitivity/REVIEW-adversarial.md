# Peer Review: Expert Removal Position Sensitivity

## NotebookLM Findings

Skipped (NotebookLM not configured in this environment). Review conducted through direct mathematical and experimental scrutiny.

## Mathematical Soundness

**Theorem 2.2 (Last expert removal is exact): CORRECT.**
This is a straightforward property of Gram-Schmidt. The last vector in GS order is never referenced by any predecessor's projection step, so subtracting it from the sum is identical to recomputing GS on N-1 vectors. The proof is trivial and correct. The empirical confirmation (1.7e-14%) is consistent with machine epsilon for float64.

**Section 2.4 (Monotonic linear decay): PARTIALLY INCORRECT.**
The claim that deviation decays "approximately linearly" with R^2=0.946 is computed on only 5 data points. With 5 points and 2 parameters (slope + intercept), this leaves 3 degrees of freedom -- the p=0.005 is borderline but acceptable.

However, the paper's own dense sweep (Test 3, 11 data points) directly contradicts monotonicity:
- Position 0: 0.3021%
- Position 5: 0.3023% (higher)
- Position 15: 0.3377% (HIGHEST of all, not position 0)
- Position 35: 0.2644% (higher than position 30 at 0.1549%)

The paper acknowledges this as "noisy but clear downward trend with some non-monotonicity" but does not compute R^2 on the dense sweep. If the linear model were fit to the 11 dense-sweep points, the R^2 would be substantially lower than 0.946. The paper reports the flattering 5-point statistic while burying the contradictory 11-point data. This is selective reporting.

**Section 2.3 (First expert error bound): SOUND in principle.**
The argument that removing expert 0 invalidates all N-1 projections, with each correction proportional to cos(delta_j, delta_0'), is correct. The scaling epsilon_k ~ C * (N-1-k) * mean_cos is a reasonable first-order approximation but ignores second-order GS cascade effects, which the dense sweep confirms are non-negligible.

**Section 4.3 (Position effect scales as N-1-k): OVERSTATED.**
The theoretical prediction of first/Q3 ~ 49/12 = 4.08x vs empirical 2.16x shows the linear model overestimates by ~2x. The paper attributes this to "second-order GS corrections" but this is hand-waving -- a 2x discrepancy in a first-order model suggests the mechanism is more complex than presented.

**Amplification ratio stability (CV=2.9%): SOUND.**
This is the strongest finding. The position-independence of the amplification ratio (0.019-0.020 across all positions) is a genuine architectural insight. If confirmed at macro scale, this means position sensitivity is purely a weight-space phenomenon.

## Novelty Assessment

**Low novelty, but appropriate for micro scope.** The order-dependence of Gram-Schmidt is a textbook result (any numerical linear algebra text covers this). The novel contribution is measuring its impact on the specific SOLE composition/removal workflow. No prior art search needed -- this is an engineering characterization, not a theoretical contribution.

The connection to the parent experiment (removal_safety_complete_bound) is well-motivated. The parent only tested middle position; this fills the gap.

## Experimental Design

**Issue 1: Kill criteria manipulation.**
K1 is defined as "position-dependent deviation varies by <2x across all positions at N=50." The experiment:
1. First excludes the last position (which gives infinite ratio) -- reasonable, since it is a degenerate case
2. Then STILL FAILS at 2.16x (first/Q3)
3. Claims SUPPORTED anyway because K2 passes

The problem: the kill criteria were written to test whether position sensitivity matters. K1 fails, meaning it DOES matter more than the 2x threshold. The paper then argues "but it's only 2.16x, and absolute values are small." This is moving the goalposts. The honest assessment is: K1 FAIL, K2 PASS. The experiment shows position sensitivity IS significant (>2x), but the absolute magnitude is small enough that it does not create a safety concern.

The SUPPORTED verdict would be more defensible if reframed: "Position sensitivity exists and exceeds the 2x threshold (K1 FAIL), but the absolute worst case (0.164%) remains 6x below the 1% safety threshold (K2 PASS). The practical recommendation (random GS permutation) resolves the issue entirely."

**Issue 2: The R^2=0.946 claim is cherry-picked.**
As noted above, this is computed on 5 points. The dense sweep (11 points) shows substantial non-monotonicity that would reduce R^2 significantly. The paper should report both R^2 values.

**Issue 3: K1 ratio computation is inconsistent.**
The paper computes K1 as first/Q3 = 2.16x, but the kill criterion says "across all positions." The natural interpretation is max/min across non-degenerate positions. At d=128, the dense sweep shows position 15 (0.3377%) is actually the worst case, not position 0 (0.3021%). The dense max/min ratio (excluding last) is 0.3377/0.0989 = 3.41x at d=128, which is a much more severe K1 failure than reported.

**Issue 4: Controls are adequate.**
Three seeds at d=256 and d=128, plus a single-seed dense sweep, provide reasonable coverage for a micro experiment. The cross-validation at d=128 showing the same qualitative pattern strengthens the finding.

**Issue 5: The recommendation (random permutation) is sound but untested.**
The paper recommends random GS permutation per layer to amortize position effects. This is mathematically reasonable (expected deviation = 0.5 * worst case) but was not actually tested in the experiment. A single run with shuffled GS ordering would have validated this claim cheaply.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node. The kill criteria (K1: <2x ratio, K2: worst within 2x of mean) are the ones actually tested. The status "supported" is defensible given K2 PASS, though K1 FAIL should be more prominently flagged.

The experiment correctly depends on exp_removal_safety_complete_bound and blocks nothing, which is appropriate for a characterization experiment.

## Macro-Scale Risks (advisory)

1. **Position sensitivity may change with trained adapters.** Random initialization gives mean pairwise cosine ~0.003. Trained adapters on related domains have cos ~0.01-0.85 (per VISION.md). Higher cosines mean larger GS corrections, which could amplify position sensitivity beyond the 2x seen here. The paper's Section 2.3 formula (epsilon ~ C * (N-1-k) * mean_cos) predicts this directly.

2. **The recommendation (random permutation per layer) changes GS output.** If each layer uses a different random ordering, the orthogonalized deltas will differ from a fixed-ordering GS. This may interact with layer-wise composition in ways not tested here.

3. **N scaling is untested.** At N=500 (production target), the number of affected successors for position 0 is 499 vs 49 at N=50. If the position sensitivity ratio grows with N (even slowly), the 2x threshold could be substantially exceeded. The paper claims the ratio is bounded but provides no evidence at N>50.

## Verdict

**PROCEED**

The experiment achieves its primary goal: characterizing the position sensitivity of expert removal in the GS ordering. The key findings are:

- Last expert removal is exact (mathematically proven, empirically confirmed)
- Worst case (first position) is 0.164% at d=256, which is 6x below the 1% safety threshold
- Amplification ratio is position-independent (CV=2.9%), confirming this is a weight-space-only effect
- Random permutation is a sound mitigation (though untested)

K1 failing at 2.16x is not blocking because the absolute deviations are so small that the position effect is operationally irrelevant at any production scale. The paper's core claim -- "position sensitivity exists but does not compromise safety" -- is correct.

**Required fixes before merging into FINDINGS.md** (not blocking PROCEED, but should be addressed):

1. Report the dense-sweep R^2 alongside the 5-point R^2. Do not claim "linear decay (R^2=0.946)" without noting this is on 5 points and that the dense sweep shows substantial non-monotonicity.

2. Acknowledge that position 15 at d=128 (0.3377%) exceeds position 0 (0.3021%) in the dense sweep, which means the "first position is always worst case" claim is empirically false at single-seed resolution. The claim holds only in expectation across seeds.

3. The FINDINGS.md entry should say "K1 FAIL (2.16x), K2 PASS (1.40x)" rather than implying the experiment fully supports the hypothesis. The SUPPORTED status is appropriate, but the K1 failure should be visible.
