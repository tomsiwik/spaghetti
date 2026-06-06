# Peer Review: Expert Removal Graceful

## NotebookLM Findings

Skipped -- the experiment is straightforward enough (linear algebra on synthetic vectors) that a deep NotebookLM review would add marginal value over direct mathematical verification.

## Mathematical Soundness

### Derivations verified

1. **GS cascade error analysis (Section 3.4):** The Case 1 / Case 2 decomposition is correct. For i < k, the GS-orthogonalized delta is independent of expert k because GS processes sequentially forward. For i > k, removing k from the basis changes all subsequent orthogonalized vectors. This is standard linear algebra.

2. **Error bound (Section 3.5):** The bound `||E||_F <= sum_{i>k} ||proj(delta_i, delta_k')||` is a first-order approximation that ignores the cascade: removing delta_k' changes delta_{k+1}', which changes delta_{k+2}', etc. The true error includes second-order and higher cross-terms. However, these higher-order terms are products of cosines (cos^2, cos^3, ...), which at cos ~ 0.0002 are negligible (< 10^{-7}). **The bound is valid in the near-orthogonal regime but understates the error in the high-overlap regime.** This is consistent with the experimental results: the bound predicts ~0.48% for N=50, k=25 at cos=0.0002, while the actual measurement is 0.096% (bound is conservative, as expected).

3. **Position dependence (Section 3.6):** Correct. GS position k affects N-k-1 subsequent experts. Position 0 is worst case. Position N-1 is zero error. This is confirmed experimentally.

4. **Timing extrapolation (Section 4):** O(N^2 * D) scaling is correct for classical GS. The extrapolation to N=1000 (~370s) and N=2000 (~1480s) follows from the quadratic fit. The claim that the 10-min threshold is hit at N~1600 is plausible from the data.

### Issues found

**Issue 1: The error bound is not tight for the high-overlap regime.** The paper states the bound but then reports experimental errors (7-10%) that significantly exceed the first-order bound. This is expected (cascade amplification), but the MATH.md does not derive or even mention the higher-order terms. The bound in Section 3.5 is presented as if it were the full story, when it is actually only the leading term. **Severity: low.** The paper correctly identifies the two regimes experimentally, so the loose bound does not affect conclusions.

**Issue 2: "PPL is Lipschitz in weights for bounded activations" (Section 6, Assumption 3).** This claim is stated without proof or citation. While transformer outputs are Lipschitz in weights under reasonable assumptions (bounded inputs, bounded layer norms), the Lipschitz constant can be exponential in depth. A 0.18% weight-space error at a single layer could amplify through L layers. The paper acknowledges this is a proxy but does not bound the amplification. **Severity: medium for macro transfer, low for the micro conclusion.** The micro experiment correctly identifies this as a limitation (Section 5 of PAPER.md), and the claim is only that weight-space error is a "conservative upper bound" -- but it is actually an *optimistic* proxy if the Lipschitz constant is large.

**Issue 3: The K1 kill criterion switching.** The kill criterion in HYPOTHESES.yml is ">3% PPL regression on remaining experts." The experiment measures *reconstruction error* (Frobenius norm ratio) and *per-expert cosine alignment regression*, not PPL. The paper uses reconstruction error as a PPL proxy, which is reasonable at micro scale, but the K1 verdict in PAPER.md is "CONDITIONAL PASS" while HYPOTHESES.yml records it as "proven." The status should be "supported" given that the actual kill criterion (PPL) was never measured. **Severity: medium** for status accuracy.

## Novelty Assessment

### Prior art

The paper correctly cites Ilharco et al. 2022 (Task Arithmetic) as the foundation for the naive subtraction approach. Task Arithmetic showed that negating a task vector removes the corresponding capability with minimal collateral damage. This experiment extends that finding to the GS-orthogonalized composition setting, which is a valid incremental contribution.

MDM-OC (referenced in REFERENCES.yml) performs reversible GS composition with learned alpha coefficients. The reversibility in MDM-OC is by construction (store the alpha coefficients and the orthogonalized deltas, then subtract to reverse). This experiment validates that even *without* explicit reversibility machinery, naive subtraction works when cosines are small. This is a distinct contribution.

### Delta over existing work

The novelty is in the **regime characterization**: the clean boundary between "naive subtraction sufficient" (cos < 0.01) and "recomputation required" (cos > 0.1), combined with the observation that SOLE's structural orthogonality places it firmly in the sufficient regime. This is useful operational knowledge, not a theoretical breakthrough.

## Experimental Design

### Does it test the hypothesis?

The hypothesis is: "Removing an expert from a Gram-Schmidt-composed merged model does not break remaining experts." The experiment tests this by measuring reconstruction error (weight-space) and per-expert cosine alignment after removal. **Yes, it tests the stated hypothesis**, within the acknowledged limitation that PPL is not measured.

### Controls

- **Ground truth baseline:** GS recomputation on N-1 experts. This is the correct gold standard.
- **Multiple seeds:** 3 seeds per configuration. Adequate for a micro experiment.
- **Multiple regimes:** Near-orthogonal, moderate overlap (cos=0.3), high overlap (cos=0.5). Good stress testing.
- **Position sensitivity:** Tested 5 positions. Good.
- **Sequential removal:** Tested removing 1, 3, 5 experts. Good stress test.

### Concerns

**Concern 1: Synthetic experts only.** All experts are generated synthetically. The near-orthogonal regime uses random matrices (which are near-orthogonal by concentration of measure), and the clustered regime uses a controlled mixing scheme. Real LoRA experts have structured gradients from training on actual data. The paper acknowledges this (Limitation 1) and cites the cos=0.0002 measurement from macro experiments as evidence that real experts fall in the near-orthogonal regime. This is reasonable.

**Concern 2: Single linear layer.** The experiment simulates a single d x d layer. Real LoRA operates across multiple layers with potentially different cosine structures per layer. The paper claims this generalizes via flattening (Limitation 2), which is mathematically correct -- GS on concatenated flattened deltas is identical to GS on the full parameter vector. However, layer-specific effects (e.g., attention layers having higher cosine than FFN layers, as noted in VISION.md with cos=0.85 for related domains in attention) are not captured. **This is a valid limitation but within micro scope.**

**Concern 3: The per-expert quality metric is weak.** The `per_expert_quality` function measures cosine alignment between the merged weight vector and each expert's raw delta. This captures directional preservation but not magnitude. An expert whose contribution is present in direction but attenuated by 50% would show high cosine alignment but degraded quality. A better metric would be the projection magnitude: `|<w_merged, delta_i>| / ||delta_i||^2` compared to the expected value of 1.0. **Severity: low** -- the reconstruction error metric (Frobenius norm ratio) is the primary metric and does capture magnitude differences.

**Concern 4: The K1 assessment logic in the code is wrong.** Line 561-562 of `run_experiment.py`: `k1_pass = max_recon_overall < 3.0`. This checks reconstruction error against the 3% threshold across ALL test configurations including the stress tests (cos=0.3, cos=0.5). The PAPER.md correctly reports "CONDITIONAL PASS" (pass in SOLE regime, fail in high-overlap), but the code's aggregate K1 check would report KILL because the clustered tests have >3% reconstruction error. The code's printed verdict at the end (`'PASS' if (k1 and k2) else 'KILL'`) would print KILL even though the experiment's actual conclusion is conditional pass. This is a code/presentation inconsistency, not a scientific error -- the detailed per-regime analysis in the paper is correct.

## Macro-Scale Risks (advisory)

1. **Attention layer cosines.** VISION.md notes cos=0.85 between related domains in attention layers. If expert removal is applied per-layer (removing the attention component of an expert), the high-overlap regime applies to attention layers even in SOLE. The fallback (GS recomputation) is cheap, but the naive subtraction shortcut would not apply to attention layers of related-domain experts.

2. **Multi-layer cascade.** Removing an expert modifies the merged weights at every layer. Even if per-layer reconstruction error is 0.18%, the cumulative effect across L=24 layers could be larger. The Lipschitz concern from Issue 2 above applies here.

3. **Trained expert cosines.** The structural orthogonality proof covers MLP layers at large d. Attention layers and gradient-aligned experts (which may drift from random initialization) need macro validation.

## Verdict

**PROCEED**

The experiment is well-designed within micro constraints, the math is sound (with the caveats noted), the code correctly implements the stated methodology, and the results provide clear operational guidance: naive subtraction works at SOLE production cosines, GS recomputation is cheap when needed.

The two-regime characterization is the key contribution and is well-supported by the data.

### Required fixes (non-blocking, should be addressed before citing as "proven"):

1. **Downgrade HYPOTHESES.yml status from "proven" to "supported."** The kill criterion specifies "PPL regression" but only reconstruction error was measured. "Proven" should require PPL measurement (macro experiment). The weight-space evidence is strong enough for "supported."

2. **Fix the Lipschitz claim in MATH.md Section 6.** Either remove the parenthetical "(PPL is Lipschitz in weights for bounded activations)" or add a caveat that the Lipschitz constant may be large (exponential in depth). As written, it implies the weight-space error directly bounds PPL change, which overstates the guarantee.

3. **Acknowledge the per-layer attention cosine risk.** The paper discusses cos=0.0002 as the production regime, but VISION.md documents cos=0.85 in attention for related domains. Add a note in the limitations that naive subtraction may not apply to attention-layer deltas of semantically related experts, and GS recomputation would be the correct strategy there.
