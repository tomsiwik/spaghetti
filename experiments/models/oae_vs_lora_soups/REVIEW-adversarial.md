# Peer Review: SOLE vs LoRA Soups

## NotebookLM Findings

Skipped. The experiment is primarily a literature positioning study with a thin empirical component. The documents are clear enough to review directly without external summarization.

## Mathematical Soundness

### What holds

1. **Shared composition form (MATH.md Section 1.2)** is correct. All three methods are indeed special cases of W_composed = W_s + sum(c_i * dW_i), differing only in coefficient selection. This is a clean and accurate framing.

2. **Convergence analysis (Section 4.1)** is directionally correct. The gradient dL/dw_i = dL/dW : dW_i follows from the chain rule. The claim that orthogonal experts make the optimal weight approximately 1.0 (when experts are well-trained) is sound reasoning.

3. **The CAT-SOLE gap bound (Section 4.2)** has the right scaling behavior: O(k^2 * cos_max^2 * ||dW||^4 / d^2). At cos_max ~ 0.0002 and large d, this is genuinely negligible.

4. **Orthogonality measurements** (mean |cos| = 0.0023) are consistent with prior experiments and theory.

### What does not hold or is imprecise

1. **Section 4.1 has a sign error or ambiguity.** The expression for the optimal weight at convergence:

   ```
   w_i^{(l)} = -[sum_j w_j * <dW_j, dW_i>]^{-1} * <dW_i, dL/dW_s>
   ```

   This is not a closed-form solution -- it is an implicit equation (w_i appears on both sides via the sum). The simplification to the orthogonal case is valid, but the general form is misleading as written. It looks like it was derived from setting the gradient to zero and rearranging, but the matrix inversion is over a system, not a scalar. This should be written as a linear system: Gram_matrix * w = -gradient_vector, where Gram_{ij} = <dW_i, dW_j>.

2. **The interference bound (Section 3.1)**:

   ```
   |interference| <= C * k^2 * (r/sqrt(d))^2 * max ||dW_i||^2
   ```

   The constant C is undefined. This is presented as a bound on SiLU nonlinearity cross-terms but no derivation is given. For a positioning paper this is acceptable, but calling it a "bound" without proof or citation is an overstatement. It is better described as a scaling argument.

3. **Section 4.2 gap bound** also lacks derivation. The ||dW||^4 / d^2 term has no justification. The dimensional analysis is plausible (higher-order cross-terms), but this is a heuristic, not a proven bound.

### Severity: LOW

These are presentational issues in what is fundamentally a positioning/framing study. The qualitative conclusions do not depend on these bounds being tight.

## Novelty Assessment

### Prior art check

The paper surveys the right literature: LoRA Soups (Prabhakar et al., COLING 2025), Modular LLMs / Arrow (Ostapenko et al., 2024), Task-Aware LoRA Composition (2025), Model Soups (Wortsman et al., 2022), TIES-Merging, DARE, and LoRAHub. The references/lora-soups/ folder confirms this is grounded in the actual paper.

### What is actually novel

The claim that "no prior LoRA composition work analyzes or guarantees orthogonality" appears correct based on the survey. The specific contribution is framing SOLE as an architecture (routing + composition + evolution) rather than just a composition technique. This is a legitimate intellectual contribution.

### What is not novel

1. **Unit-weight LoRA addition** is not new. It is the simplest possible composition and has been used informally in the LoRA community (merge adapters by summing). The novelty is in the *justification* (orthogonality guarantees make it optimal), not the mechanism itself.

2. **FFN-only adapters** -- while the paper claims "no prior work makes this architectural choice," this needs verification. Parameter-efficient methods that target specific module types exist (e.g., adapter placement studies). The specific claim about orthogonality improvement from FFN-only may be novel, but the restriction itself is not unprecedented.

### Missing prior art

- **LoRA-Flow (Wang et al., 2024)** -- uses dynamic per-layer LoRA weights based on input, making it a superset of both SOLE (fixed weights) and CAT (learned static weights). Should be cited and compared.
- **mLoRA / Multi-LoRA serving** -- S-LoRA (Sheng et al., 2023) and Punica (Chen et al., 2023) enable serving multiple LoRA adapters efficiently. These are operational comparisons that strengthen SOLE's serving story and should be referenced.

## Experimental Design

### Critical problem: the quality comparison is entirely vacuous

The paper acknowledges this clearly and repeatedly, which is good. But the implications deserve closer scrutiny:

1. **Expert specialization is zero.** Individual expert losses match base losses to 4+ decimal places (e.g., code_a base: 3.3831 vs expert: 3.3830). The LoRA deltas are essentially noise -- they have not learned anything meaningful.

2. **CAT weights converge to 1.0 trivially.** The learned weights are all within [0.9999, 1.0001] of 1.0. This does NOT validate the theoretical prediction that "orthogonality makes optimal weights equal to 1." It validates that "when expert deltas are negligible, any weighting produces the same result." These are very different statements.

3. **The timing comparison is the only meaningful empirical result.** CAT at 52-75x overhead is real and meaningful. But this is expected by construction: finite-difference optimization of 2*k*L scalars versus zero optimization. The overhead ratio tells us about the implementation (100 optimization steps with numpy finite differences), not about CAT's inherent cost at production scale. A production CAT implementation with analytical gradients on GPU would have much lower overhead (though still nonzero).

4. **The dynamic addition test is vacuous.** The stale-retrained gap is ~1e-11 nats. This is floating-point noise, not evidence that CAT requires retraining when adding experts. It is evidence that when deltas are zero, nothing matters.

### What the experiment actually tests vs what it claims

**Claims**: SOLE has 5 distinct advantages over LoRA Soups.

**What the experiment tests**: Whether three composition methods produce different results on a toy model where experts have not specialized.

**What the experiment actually shows**: When expert deltas are negligible, all composition methods are equivalent, and the one with optimization is slower. This is a tautology, not evidence for SOLE's positioning.

### The 5 advantages are argued, not demonstrated

Every claimed advantage is a **logical argument from architecture**, not an empirical finding:

1. Zero setup cost -- true by definition (no optimization step)
2. Instant expert addition -- true by definition
3. N-independence -- shown in a separate experiment (inference_latency_vs_N), not here
4. Evolution support -- not tested
5. Determinism -- true by definition

These are valid architectural arguments. But they could have been made without running any experiment at all. The empirical component adds essentially nothing to the positioning.

### Controls

The Uniform Averaging baseline (1/k) is a good inclusion. It helps show that all three methods collapse to the same result. No further controls needed given the study's nature.

## Hypothesis Graph Consistency

**Kill criteria from HYPOTHESES.yml:**

- K1: "SOLE provides no measurable advantage over LoRA Soups on any metric"
- K2: "LoRA Soups already achieves orthogonal composition without FFN-only constraint"

**Assessment:**

- K1 is not testable at micro scale because the quality comparison is vacuous. The paper correctly claims K1 SURVIVES based on operational advantages, but these are architectural properties, not empirical measurements from this experiment.
- K2 SURVIVES correctly -- LoRA Soups indeed does not analyze orthogonality.

The kill criteria are appropriate for a positioning study. The status "proven" in HYPOTHESES.yml is somewhat misleading -- this is a literature analysis that happens to include a vacuous empirical component, not an empirical proof.

## Macro-Scale Risks (advisory)

1. **The core prediction is falsifiable at macro:** If CAT at d=896 with real expert specialization learns weights significantly different from 1.0 (e.g., w_i ~ 0.7), the entire SOLE positioning collapses. The prediction that orthogonality forces w_i -> 1.0 is testable and important.

2. **LoRA Soups' binary limitation may not hold.** The original paper acknowledges challenges at k>2, but this does not mean it is impossible. If someone extends CAT to k=10 efficiently, one of SOLE's key advantages disappears.

3. **The FFN-only claim needs macro validation.** At d=896, the 15% orthogonality improvement from FFN-only (from ffn_only_vs_all_modules) may or may not hold. If it does not, that advantage weakens.

4. **Timing comparison is misleading for macro.** CAT with analytical gradients on GPU would be much faster than numpy finite differences on CPU. The 52-75x overhead is an artifact of the implementation, not the algorithm. At macro, the overhead might be 5-10x (still nonzero, but much less dramatic).

## Verdict

**PROCEED** -- with caveats noted below.

This is fundamentally a literature positioning study, and a competent one. The survey is thorough, the comparison table is useful, the framing of SOLE as architecture vs LoRA Soups as technique is intellectually sound and strategically valuable.

The empirical component is acknowledged as vacuous by the authors, which demonstrates honest assessment. The timing comparison is directionally valid even if the magnitude is implementation-dependent.

The mathematical framework in MATH.md provides the right scaffolding for future macro-scale validation. The key falsifiable prediction -- that CAT weights converge to 1.0 when experts are orthogonal and well-calibrated -- is clearly stated and testable.

**What should be fixed before citing this in a paper (non-blocking for PROCEED):**

1. Rewrite MATH.md Section 4.1 to present the optimal CAT weights as a linear system (Gram matrix inversion), not as a misleading implicit equation.
2. Label the interference bound (Section 3.1) and gap bound (Section 4.2) as "scaling arguments" rather than "bounds" -- they lack derivations.
3. Add LoRA-Flow (Wang et al., 2024) to the comparison table as a dynamic per-layer weight method.
4. Explicitly state in the paper that the timing comparison reflects a naive implementation and that production CAT overhead would be lower (though still nonzero).
5. Change the HYPOTHESES.yml status descriptor from "proven" to something like "supported-by-analysis" -- this is a literature study with a vacuous empirical component, not an empirical proof.
