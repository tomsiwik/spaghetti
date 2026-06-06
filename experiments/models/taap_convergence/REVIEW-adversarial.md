# Peer Review: TAAP Convergence

## NotebookLM Findings

Skipped -- the mathematical content is tractable enough for direct verification without external tooling.

## Mathematical Soundness

### The sqrt(r) identity: VERIFIED

The central claim is c_eq / mu_W = sqrt(r). Step-by-step verification:

1. **Welch bound derivation.** mu_W = sqrt(r(Nr - d) / (d(Nr - r))). This is the standard generalized Welch bound for Grassmannian packings (Dhillon et al. 2008, equation 3). Correct.

2. **Equidistributed coherence derivation.** The Gram matrix G is (Nr x Nr), has rank at most d (since frames live in R^d), and trace Nr (since diagonal blocks are I_r). By Cauchy-Schwarz on eigenvalues: sum(lambda_k^2) >= (sum lambda_k)^2 / d = (Nr)^2 / d. Expanding ||G||_F^2 = Nr + N(N-1)c^2 and setting >= (Nr)^2/d gives c^2 >= r(Nr-d)/(d(N-1)). Correct.

3. **Equality condition.** Cauchy-Schwarz is tight when all d nonzero eigenvalues are equal (= Nr/d). This is the equidistributed fixed point. The claim that AP converges to this configuration is **empirically demonstrated** (ratio = 2.828427... across all settings) but **not formally proved** in this paper. The paper does not prove that AP must converge to the uniform-eigenvalue solution; it observes that it does. This is a minor gap -- the fixed-point structure of AP for Grassmannian packings is known in the literature (Dhillon et al. establish convergence to equidistributed arrangements when they exist), but the paper should be more careful about distinguishing "proven algebraically" from "verified empirically for AP."

4. **Ratio computation.** c_eq/mu_W = sqrt((Nr-r)/(N-1)) = sqrt(r(N-1)/(N-1)) = sqrt(r). The key step Nr - r = r(N-1) is trivially correct. The identity holds exactly. **Verified.**

5. **Hidden assumption check.** The derivation assumes Nr > d (otherwise the Welch bound is 0). All test configurations satisfy this (Nr/d = 1.25 to 1.5). The identity also requires N >= 2 (denominator N-1). Both conditions are met. No hidden assumptions found.

### The capacity formula: MINOR ERROR

MATH.md line 105-106 gives: "N <= 1 + r(d - r*N_max) / (d * mu_max^2)." This formula has N_max on the right-hand side in a bound for N -- it should be solved explicitly. Inverting c_eq(N) <= mu_max:

c_eq^2 = r(Nr - d) / (d(N-1)) <= mu_max^2

r(Nr - d) <= d(N-1) * mu_max^2

rNr - rd <= dN * mu_max^2 - d * mu_max^2

N(r^2 - d * mu_max^2) <= rd - d * mu_max^2

N <= d(r - mu_max^2) / (r^2 - d * mu_max^2) ... when r^2 > d * mu_max^2

The formula in PAPER.md (line 233-234) is different again: "N_max = 1 + d * mu_max^2 / (r - d * mu_max^2 / r)." Simplifying the denominator: r - d*mu_max^2/r = (r^2 - d*mu_max^2)/r, so N_max = 1 + d*r*mu_max^2 / (r^2 - d*mu_max^2). Let me verify: from c_eq^2 <= mu_max^2, we get N(N-1) term. Actually, let me redo this cleanly:

r(Nr - d) / (d(N-1)) <= mu_max^2
r*Nr - r*d <= d*mu_max^2*(N-1)
r^2*N - rd <= d*mu_max^2*N - d*mu_max^2
N*(r^2 - d*mu_max^2) <= rd - d*mu_max^2
N <= d*(r - mu_max^2) / (r^2 - d*mu_max^2)

The PAPER.md formula gives N_max = 1 + dr*mu_max^2/(r^2 - d*mu_max^2). These are not the same expression. The PAPER.md version appears to be an incorrect rearrangement. However, this is a secondary formula not used in any experiment, so it does not affect the core results. Flag as minor.

## Novelty Assessment

### The sqrt(r) identity

This result is **not novel in the Grassmannian packing literature**. The relationship between the Welch bound and equidistributed (equiangular) subspace packings is well-known. Specifically:

- Conway, Hardin, Sloane (1996) established that equiangular tight frames (ETFs) achieve the Welch bound for r=1 but not for r>1 in general.
- The gap between equidistributed packings and the Welch bound for subspaces (r>1) is a known consequence of the rank constraint. The specific factor sqrt(r) follows directly from the standard frame potential analysis.

The paper does not claim this as a novel mathematical contribution to Grassmannian theory, and it is appropriate that it does not. The novelty is in **applying this known result to diagnose the SOLE architecture**, which is legitimate.

### TAAP reference

The paper cites "Meszaros et al., TAAP: Grassmannian Frame Computation via Accelerated Alternating Projections, SampTA 2025." I cannot verify this reference exists. If it is fabricated or hallucinated, it should be removed or replaced with the actual accelerated projection literature (e.g., Bauschke & Borwein's acceleration of alternating projections, or Nesterov-accelerated variants in the optimization literature).

### Delta over parent experiments

The experiment provides genuine value over its parents:
- **grassmannian_expert_init** observed a 2.8x gap but left it as an open question
- **minimax_grassmannian_packing** proved AP is equidistributed but warned the gap "may be fundamental"
- This experiment proves the gap IS fundamental, gives the exact formula, and shows convergence is complete at 500 iterations

This is a clean resolution of a dangling question.

## Experimental Design

### Strengths

1. **Good controls.** Standard AP at both 500 and 2000 iterations provides both a baseline and a convergence check. Same random seed ensures identical initialization across methods.

2. **Three dimensions tested.** d=64, 128, 256 with appropriate N scaling shows the ratio is dimension-independent, as the theory predicts.

3. **Kill criteria are appropriate.** K1 (TAAP closer to Welch bound) directly tests the hypothesis. The reinterpretation as "gap is fundamental" is honest and well-justified.

4. **TAAP-Selective serves as a useful contrast.** It demonstrates that breaking equidistribution can slightly improve mean coherence at the cost of much worse max coherence, confirming the equidistributed configuration is a genuine trade-off, not just a local minimum.

### Weaknesses

1. **Only 2 seeds.** For a result that is algebraically exact, 2 seeds suffice to demonstrate it empirically. But for the TAAP-Selective results (which show variation across seeds), more seeds would strengthen the trade-off claim. Minor concern since the paper correctly identifies this as a negative result.

2. **The r=1 validation case is missing.** The paper acknowledges this (Limitation 3). Testing r=1 to confirm ratio=1.000 would be a trivially cheap validation of the identity. This is a missed opportunity, not a flaw.

3. **No Riemannian gradient descent comparison.** The paper tests AP variants only. A Riemannian conjugate gradient optimizer on the Grassmannian (e.g., Manopt/PyManopt) would provide a methodologically different baseline. However, since the identity is algebraic and applies to any equidistributed configuration regardless of how it was found, this is not a flaw in the conclusion.

### Could a simpler mechanism explain the results?

No -- the result IS the simple mechanism. The sqrt(r) identity is a three-line algebraic derivation. The experiment confirms it empirically.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry correctly reflects:
- Status: killed
- Kill criteria match what was tested (K1: TAAP not closer, K2: runtime)
- Evidence accurately summarizes the finding
- The "positive kill" framing is appropriate -- the question is definitively answered

The experiment properly resolves the open question listed in VISION.md ("TAAP convergence: Standard AP stops 2.8-3x above Welch bound. TAAP may close the gap for N=500+.").

## Integration Risk

**None.** This experiment closes a question rather than introducing a new component. It confirms that the existing AP skeleton is provably optimal and no further optimization is needed. The only integration action is updating capacity formulas to use c_eq instead of mu_W, which the paper correctly identifies.

## Macro-Scale Risks (advisory)

1. **The capacity formula matters at scale.** If someone uses the Welch bound mu_W for capacity planning instead of c_eq = sqrt(r)*mu_W, they will overestimate N_max by a factor of r. At r=16, this is a 16x overestimate. The corrected formula should be propagated to any capacity planning code.

2. **The identity holds for uniform rank.** Mixed-rank experts (from adaptive rank selection) break the uniform Gram matrix structure. The sqrt(r) identity does not directly apply to mixed-rank packings. This is already flagged as an open question in VISION.md.

## Verdict

**PROCEED**

The experiment cleanly resolves a dangling question from the Grassmannian infrastructure line of research. The mathematical derivation is correct and verified step-by-step. The empirical results exactly match the theory. The kill verdict is honest and well-justified. The finding has clear practical implications for SOLE capacity planning.

Minor issues that do not block PROCEED:
1. The capacity formula in MATH.md/PAPER.md appears to have an algebraic error in the rearrangement. Should be corrected before it is used in production code.
2. The Meszaros et al. TAAP reference should be verified or replaced.
3. The paper should be slightly more precise about distinguishing "AP converges to equidistributed fixed point" (empirical) from "sqrt(r) identity" (algebraic proof).
