# Peer Review: Symmetric GS Cost-Benefit

## NotebookLM Findings

Skipped -- this is a clean negative result with straightforward geometry. The mathematical argument and code are compact enough to verify directly.

## Mathematical Soundness

**The core argument is correct.** The destructive interference mechanism is geometrically sound:

1. Each permutation sigma produces M_sigma = alpha_s * s_hat + r_sigma, where r_sigma depends on the ordering. This decomposition is valid by projection onto the shared direction.

2. The residuals r_sigma point in different directions because GS assigns the "credit" for shared variance to whichever expert comes first, leaving different residual shapes for different orderings. This is a direct consequence of GS being order-dependent.

3. Averaging vectors that share a common component but have diverse orthogonal components produces a result with preserved shared component but reduced orthogonal component (by Jensen's inequality on norms, or equivalently the triangle inequality). The paper states this correctly.

**Minor issue with the quantitative prediction.** The MATH.md states:

> ||M_sigma|| approx (1/N) * sqrt(1 + (N-1)(1-c))

This formula is not derived -- it appears as an assertion. For unit-norm experts with uniform pairwise cosine c, the sum of GS-orthogonalized vectors has a norm that depends on the specific ordering through the Gram matrix eigenstructure. The formula appears to be an approximation for the "uniform overlap" model (single shared direction + independent unique directions), which is the synthetic model used. In that specific model, it holds because the GS process preserves the first vector fully and progressively shrinks later vectors by factor sqrt(1-c). The mean of these has norm approximately as stated. This is fine for the synthetic setting but should be flagged as model-dependent, not general.

**The kill criteria logic has a subtle inversion that the paper correctly identifies.** K1 was designed as "symmetric GS within 0.1% of best" (meaning: if true, no benefit, so kill). K2 was "improvement < 1%" (same logic). Both are met trivially because symmetric GS is actually WORSE, not merely "not better." The paper handles this inversion transparently. No deception.

## Novelty Assessment

**Low novelty, but that is appropriate for a kill experiment.** The purpose was not to discover something new but to close an open question in the SOLE architecture about whether the 100x compute of symmetric GS is justified.

**Prior art alignment:** The result is consistent with well-known properties of Gram-Schmidt. Averaging over orderings of GS is not a standard technique in the numerical linear algebra literature precisely because it destroys norm. The closest related concept is "democratic orthogonalization" (equal-weight combinations), but that operates differently. The paper does not overclaim novelty.

**Internal consistency:** The experiment correctly builds on merge_order_dependence (proven: variation ~ 80*cos), layerwise_order_sensitivity (killed: no layer-type effect), and gs_random_permutation_validation (proven: random permutation equalizes). The lineage is clean.

## Experimental Design

**Strengths:**

1. Three-phase design (sweep, deep dive, practical alternatives) is thorough for a micro experiment.
2. Three seeds with 500 single-ordering samples provides adequate statistical power for the claimed 0.12% CV.
3. The P convergence analysis (P=1 through P=200) correctly shows symmetric GS converging to a floor below all single orderings.
4. Testing multiple deterministic orderings (canonical, reverse, norm-sorted, random-fixed) and showing they are all within 0.1% of each other strengthens the "any ordering is fine" conclusion.

**Weaknesses:**

1. **Synthetic-only evaluation.** The uniform overlap model (single shared direction + independent unique directions) is the simplest possible overlap structure. Real LoRA deltas at d=4096 have multi-dimensional shared subspaces. The paper acknowledges this in limitations but the argument that "the destructive interference mechanism is geometric and applies to any overlap structure" is hand-waved. In principle, if real deltas had correlated residuals across orderings (e.g., because real overlap is structured, not random), the averaging could be less destructive. However, this would require a very specific correlation pattern, and the paper is correct that this is unlikely.

2. **Norm as sole quality metric.** The paper uses merged vector norm as the quality proxy. This is reasonable for synthetic vectors but not validated against downstream loss. The justification given (norm tracks quality, parent experiment showed norm CV tracks quality CV) is indirect. However, at 0.12% CV among single orderings, this concern is academic -- no downstream metric could distinguish them. And the 9% gap for symmetric GS is large enough that any reasonable quality metric would show it.

3. **N=5 only.** With 5 experts, there are only 120 total permutations. The experiment samples 100 of them for "symmetric" GS, covering most of the permutation space. At larger N, the permutation space explodes and the residual diversity should increase, making averaging even more destructive. This limitation actually strengthens the conclusion (the effect would be worse at scale).

**No confounds detected.** The code correctly implements Modified Gram-Schmidt, the synthetic expert construction produces the target cosines (verified by actual_cos in results), and the averaging is straightforward numpy operations. The random seeds are properly separated across phases (seed+1000, seed+2000, etc.) to avoid correlation.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches the experiment precisely:
- Kill criteria K1 ("within 0.1% of best") and K2 ("< 1% improvement at cos=0.85") are exactly the criteria tested.
- Status is "killed" with clear evidence citation.
- Dependencies (merge_order_dependence, layerwise_order_sensitivity) are correct.
- The experiment blocks nothing (blocks: []), which is appropriate -- this was an optimization question, not a blocking dependency.

The FINDINGS.md entry accurately summarizes: "Symmetric GS KILLED (strictly dominated)."

## Macro-Scale Risks (advisory)

Minimal, because:
1. The conclusion is "do NOT implement symmetric GS" -- there is nothing to scale.
2. The positive finding (any deterministic ordering is fine, CV=0.12%) de-risks the composition pipeline by confirming order-invariance at production cosines.
3. The only scenario that could overturn this at macro is if real LoRA deltas at d=4096 have correlated residual structure across GS orderings that makes averaging constructive rather than destructive. This is a theoretically possible but practically unlikely edge case. If it occurs, it would manifest as the symmetric GS norm exceeding single-ordering norms -- easy to check in a 5-minute sanity test during macro.

## Verdict

**PROCEED**

This is a clean, well-executed kill experiment. The mathematical argument is sound (averaging over GS orderings causes destructive interference in residual components). The experimental design is adequate for the claim. The kill criteria are correctly applied. The result is unsurprising from a linear algebra perspective but needed empirical confirmation to close the investigation thread.

The conclusion -- use any deterministic ordering, do not implement symmetric GS -- is well-supported and directly actionable for SOLE.

No revisions needed. The only improvement would be adding the explicit derivation of the norm approximation formula in MATH.md (currently stated without proof), but this is a documentation nicety, not a blocking issue.
