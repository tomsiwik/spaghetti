# Peer Review: Self-Contrast Decoding

## Experiment Type
Frontier extension (correctly identified)

## Hack Detector
- Fix count: 1 (contrastive decoding with single hyperparameter alpha). Clean, no stacking.
- Is MATH.md a proof or a description? **Description dressed in equations.** There are "Propositions" but no proofs -- Proposition 1 and Proposition 2 state conditions under which the mechanism works or fails, but neither is proved. The "Proof Sketch" section (D) is algebraic manipulation of the contrastive formula, not a theorem-proof-QED structure. This is acceptable for a frontier extension but must be noted.
- Metric used as evidence: Behavioral eval scores (domain-specific rubrics from Finding #238). These are reasonable proxies for generation quality.
- Kill criteria source: K1 threshold (3/5 domains worse) is arbitrary. K2 threshold (3x latency) is reasonable engineering constraint. Neither is derived from the proof.

## Self-Test Audit

1. **One-sentence impossibility property:** Honest answer -- acknowledges this does NOT make failure impossible. Appropriate for frontier extension. PASS.
2. **Cited theorems:** Li et al. (2210.15097) and SCMoE (2405.14507) are real papers. However, the "SCMoE Theorem (informal)" cited in Section C is not an actual theorem from the paper -- it is a paraphrase of the paper's empirical finding. The key condition ("amateur must be reasonably competent but less specialized") is not a theorem condition but an empirical observation. Minor FLAG -- the distinction between empirical finding and theorem is blurred.
3. **Predicted numbers:** P1-P5 are specific and falsifiable. PASS.
4. **Falsification condition:** "If self-contrast degrades ALL domains including math and code" -- this is well-targeted. PASS.
5. **Hyperparameter count:** 1 (alpha), with honest acknowledgment that optimal value depends on SNR. PASS.
6. **Hack check:** Clean -- standalone inference-time technique. PASS.

## Mathematical Soundness

**Section D (Proof Sketch) -- step-by-step verification:**

The algebra is correct. Starting from:
- z_E = (W + s_P * Delta_P)(x)
- z_A = (W + (1/(K-1)) * sum_j s_{Q_j} * Delta_{Q_j})(x)
- z_CD = (1+alpha) * z_E - alpha * z_A

Expanding:
- z_CD = (1+alpha)*W(x) + (1+alpha)*s_P*Delta_P(x) - alpha*W(x) - alpha*(1/(K-1))*sum(...)
- z_CD = W(x) + (1+alpha)*s_P*Delta_P(x) - alpha*(1/(K-1))*sum(...)

**Observation 1** (base model preserved with coefficient 1): Correct. The (1+alpha) - alpha = 1 cancellation is valid.

**Observation 2** (primary amplified by 1+alpha): Correct but misleading. This is mathematically equivalent to increasing adapter scale, yes, but it also scales the adapter's noise/errors by the same factor. The worked example in Section F selectively shows cases where this helps, but does not address the case where the adapter's own errors are also amplified.

**Observation 3** (non-primary contribution is small): Correct arithmetic. For K=5, alpha=0.5, each non-primary contributes -0.125x. However, this observation is misleading -- the issue is not the magnitude of the non-primary contribution but its correlation structure. Even a small random perturbation to logits can flip discrete decisions (greedy argmax), especially for math where the correct token has a narrow margin.

**Proposition 1 (Signal condition):** Stated but not proved. The claim that subtracting avg(Delta_{Q_j}) suppresses generic tokens requires showing that non-primary adapters have higher activation on generic tokens than domain-specific tokens. No evidence or proof is offered for this condition holding. In fact, the Grassmannian orthogonality of A-matrices makes this condition unlikely to hold -- orthogonal subspaces have no preferential alignment with "generic" vs "domain-specific" directions.

**Proposition 2 (Noise condition):** Also stated but not proved. However, the prediction is correct: if the non-primary signal is uncorrelated, subtraction adds variance proportional to alpha^2 * Var(avg(Delta_Q)). This is the condition that actually holds, as confirmed by the experiment.

**Critical gap in the analysis:** MATH.md identifies the correct failure mode (Section A) and even cites Finding #242 (H^1=3 topological obstructions). But it does not connect this to a formal prediction about WHEN the failure mode dominates. The Grassmannian orthogonality of A-matrices is mentioned in the PAPER.md root cause analysis but is not incorporated into the MATH.md predictions. This is the key missed opportunity -- the kill could have been derived analytically.

## Prediction vs Measurement

PAPER.md contains the prediction table. Assessment:

| Prediction | Measured | Verdict |
|------------|----------|---------|
| P1: math >= 0.80 | 0.80 at alpha=0.1 only | PARTIAL -- holds at weakest alpha only, degrades to 0.0 at alpha=1.0 |
| P2: code >= 0.62 | 0.553 at best | FAIL -- 11% below prediction even at alpha=0.1 |
| P3: finance degrades | -0.6% at alpha=0.1 | PASS |
| P4: <= 2 domains worse | 0 domains better (vacuous) | Technically PASS but reveals the prediction was wrong in spirit |
| P5: latency ~2x | 2.02x | PASS |

The predictions were optimistic. P1 and P2 assumed the signal condition (Proposition 1) would hold for strong adapters. It did not. The measurement falsifies the signal condition, not just the experiment.

## NotebookLM Findings

Skipped -- the experiment is already killed and the analysis is straightforward. The core insight (orthogonality prevents contrastive signal) does not require external review to validate.

## Novelty Assessment

The negative result itself is novel and valuable. No prior work (that I can find in the references) has tested contrastive decoding on orthogonal LoRA adapters. The conclusion that Grassmannian orthogonality and contrastive value extraction are dual (one prevents the other) is a genuine insight worth recording.

SCMoE (2405.14507) operates on co-trained MoE experts. DExperts (2105.03023) uses separately trained expert/anti-expert pairs but for controlled generation (toxicity reduction), not domain composition. Neither addresses the orthogonal adapter setting.

## Macro-Scale Risks (advisory)

Not applicable -- the mechanism is killed. The duality insight (orthogonality prevents contrastive extraction) scales by construction and does not need macro verification.

## Post-Mortem: Could This Have Been Killed Analytically?

**Yes.** The kill was justified but late. Here is the argument that should have appeared in MATH.md before running code:

1. The Grassmannian skeleton guarantees A_i^T A_j is near zero for i != j.
2. For LoRA, Delta_W = B * A^T. The adapter's logit contribution on input x is proportional to B * A^T * h, where h is the hidden state.
3. The non-primary adapters project h onto orthogonal subspaces (A_j^T h is in a different r-dimensional subspace than A_i^T h).
4. Therefore, avg(Delta_{Q_j}(x)) captures components of h that are orthogonal to the primary adapter's projection. These components carry no information about the primary domain's specialization.
5. Subtracting a signal that is orthogonal to the domain-specific signal cannot sharpen that signal. It can only add noise proportional to the projection of h onto the complement of the primary subspace.
6. QED: Proposition 2 (noise condition) holds by construction whenever adapters have Grassmannian-orthogonal A-matrices. No experiment needed.

This argument takes 5 minutes and would have saved ~43 minutes of compute (2566 seconds).

## Conclusions About the Kill

**The kill is correct and well-documented.** The root cause analysis in PAPER.md (lines 99-115) correctly identifies the fundamental duality: orthogonality enables interference-free composition but prevents contrastive value extraction. The key takeaway (lines 137-145) is properly scoped and actionable.

**Minor issues to address before accepting:**

1. PAPER.md K1 reporting is confusing. K1 technically "PASS"es (0 domains worse) because the best alpha per domain defaults to baseline (no contrast applied). The paper correctly notes this is vacuous, but the results.json records it as "PASS." This should be recorded as "PASS (vacuous -- mechanism provides no benefit)" in the finding.

2. The prediction-vs-measurement table in PAPER.md reports P1 as "PARTIAL" but the spirit of the prediction (self-contrast maintains math quality) is falsified -- it only holds at the weakest possible alpha where the mechanism barely operates.

3. MATH.md Proposition 1 should be explicitly marked as refuted, not just unproven.

## Verdict

**PROCEED (accept the kill)**

The kill is justified on both empirical and analytical grounds. The duality insight (Grassmannian orthogonality prevents contrastive extraction) is a valuable negative result that should be recorded as a finding. The experiment is well-documented with clear root cause analysis.

Recommended fixes before closing (non-blocking):

1. Record a finding with `--impossibility-structure "Grassmannian-orthogonal A-matrices guarantee that non-primary adapter outputs are in orthogonal subspaces to primary, making their average output uncorrelated noise -- Proposition 2 of MATH.md holds by construction"`.
2. Mark K1 result as "PASS (vacuous)" in the experiment record to avoid confusion.
3. Add a one-paragraph "analytical kill" note to MATH.md Section D explaining why Proposition 2 holds by construction given the Grassmannian skeleton, so future researchers do not repeat this experiment.
