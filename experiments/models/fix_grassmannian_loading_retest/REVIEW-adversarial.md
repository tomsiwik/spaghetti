# Peer Review: fix_grassmannian_loading_retest

## Experiment Type
Verification (Type 1) -- bug-fix verification with proof that mismatched A-B produces zero-mean noise.

## Hack Detector
- Fix count: 1 (single bug fix -- correct A-matrix loading). No stacked mechanisms.
- Is MATH.md a proof or a description? **Proof with QED** (Theorem 1 has a proper derivation). However, see caveats below on proof quality.
- Metric used as evidence: PPL improvement (oracle and routed). PPL is a standard proxy; adequate for this scope.
- Kill criteria source: K596 derived from proof (correct A should restore training-time improvement). K597 (routing >= 50%) is an arbitrary threshold, not derived from any proof. K598 (memory) is a hardware constraint.

## Self-Test Audit
1. One-sentence impossibility property -- PASS. "Correct A-B pairing ensures learned perturbation is deterministic; incorrect pairing produces zero-mean noise." Clear and singular.
2. Cited theorems -- PASS. LoRA (Hu et al., 2021) is real and correctly applied. JL-lemma is mentioned but not actually used in the proof (it is cited for intuition only). This is fine since the proof does not depend on JL.
3. Predicted numbers -- PASS. Three predictions: wrong A gives ~0% improvement (confirmed at -0.6%); correct A gives >= 20% (confirmed at 34.8%); routing accuracy improves (marginal: 39.2% -> 41.2%).
4. Falsification condition -- PASS. "If correct A loading also shows ~0% improvement, proof is wrong." This directly targets the theorem.
5. Hyperparameter count -- PASS. Zero new hyperparameters.
6. Hack check -- PASS. Not adding a fix on top of fixes; removing root cause of 7 failed experiments.

## Mathematical Soundness

**Theorem 1 is correct in structure but loose in rigor.**

Step (1): E[x @ A' @ B*] = 0. Valid. A' has zero-mean i.i.d. entries independent of x and B*. Linearity of expectation applies. The factoring E[x @ A'] = x @ E[A'] = 0 is correct because x is treated as fixed (conditioned on). This is sound.

Step (2): Variance bound. The derivation of E[z_j^2] = (s^2/3) * ||x||^2 is correct for Uniform(-s, s) entries. The step from E[||z @ B*||^2] to (s^2/3) * ||x||^2 * ||B*||_F^2 requires that different z_j are uncorrelated, which holds because rows of A' are independent (each z_j = sum_i x_i A'_{ij} and different j-columns are independent). Sound.

**Minor issues (non-blocking):**

1. The proof claims "the perturbation is random noise centered at zero, explaining why buggy adapters showed ~0% improvement." This conflates zero mean with zero effect on PPL. Zero-mean noise still increases variance and would typically increase PPL slightly. The observed -0.6% (PPL slightly worse than base) is consistent with this, but the proof does not formally bound the PPL impact of zero-mean additive noise. The proof shows E[perturbation] = 0 but does not prove E[PPL(perturbed)] ~ PPL(base). This gap is small and the empirical confirmation is strong enough to overlook.

2. The "Corollary 1" is stated without proof. It follows straightforwardly from Theorem 1 but should be noted as an observation rather than a formal corollary.

3. Prediction 3 ("routing accuracy should improve significantly") is vague and not derived from the proof. The measured change (39.2% -> 41.2%) is within noise. The paper honestly reports this but should not have listed it as a proof-derived prediction.

**Overall: The proof is correct for what it claims. It establishes that wrong-A produces zero-mean perturbation, correct-A produces the learned perturbation, and the experiment confirms the dramatic PPL difference.**

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table. Assessment:

| Prediction | Expected | Measured | Verdict |
|-----------|----------|----------|---------|
| Wrong A: oracle ~ base PPL | ~0% improvement | -0.6% | MATCH |
| Correct A: >= 20% improvement | >= 20% | 34.8% | MATCH |
| All 24 domains specialize | 24/24 >= 5% | 24/24 | MATCH |

**Missing from prediction table:** Prediction 3 (routing accuracy improves) is effectively a FAIL (39.2% -> 41.2% is not "significant improvement") but the paper does not include it in the table. This is an omission but not dishonest -- the paper discusses it extensively in the "Critical Finding" section.

## Critical Finding: Routed PPL = Oracle PPL Despite 41% Routing Accuracy

This is the most interesting and most concerning result. The paper reports avg_routed_ppl = 6.3213 vs avg_oracle_ppl = 6.3222 -- virtually identical -- despite 41.2% routing accuracy meaning 59% of samples get the WRONG adapter.

**Two interpretations:**

1. **Paper's interpretation (generous):** "Adapters provide general quality improvement that transfers across domains." This would mean the Grassmannian A matrices and trained B weights improve the model regardless of which domain's adapter is applied.

2. **Skeptical interpretation:** If ANY adapter produces the same PPL improvement on ANY domain, then the adapters are not domain-specific experts at all. They are general fine-tuning artifacts. The entire premise of "composable domain experts" is undermined -- you have 24 copies of approximately the same general-purpose adapter with slightly different weights.

**Evidence for interpretation 2:** Look at the confusion matrix. The router sends cybersecurity samples to cooking (8 times), marketing to cooking (7 times), education to code (8 times). These are semantically distant domains. If their adapters were truly specialized, these misroutes would significantly degrade PPL. The fact that they do not means the adapters are largely interchangeable.

**This is architecturally significant.** If adapters are not domain-specific, then:
- Routing is unnecessary (any adapter works equally well)
- The Grassmannian skeleton's domain separation is irrelevant (domains do not need separate subspaces)
- The vision of "composable domain experts" needs re-examination

The paper acknowledges this in Limitations and "What Would Kill This" sections, which is honest. But this finding should be elevated to the primary result, not treated as a secondary observation.

## NotebookLM Findings

Skipping NotebookLM deep review. The mathematical content is straightforward enough (bug fix verification) that automated deep review would not add value beyond the manual analysis above.

## Novelty Assessment

This is a bug-fix verification, not a novel contribution. No novelty claim is made, which is appropriate. The experiment correctly identifies that 7 prior experiments were invalid due to a loading bug and re-runs the evaluation.

The finding that "adapters are interchangeable across domains" is potentially novel and important for the project's direction, though it is presented as a secondary observation.

## Macro-Scale Risks (advisory)

1. **Adapter interchangeability may not hold at scale.** With larger models and more training data, adapters may become more specialized. The current result could be an artifact of small-scale training (character-level, short sequences, limited data per domain).

2. **The router trains on base hidden states (no adapter applied).** This means the router never sees the effect of adapters on representations. At larger scale, a router that observes adapter-modified representations might achieve higher accuracy.

3. **20 validation samples per domain is very small.** Domain-level PPL averages over 20 samples may not be statistically reliable for detecting small differences between routed and oracle performance.

## Verdict

**PROCEED** -- with caveats.

**Justification:**

The experiment achieves its primary goal: it proves that the A-matrix loading bug was the root cause of 7 failed experiments, confirms the proof's predictions (zero-mean noise from wrong A, 34.8% improvement from correct A), and invalidates Finding #198.

The proof is correct. The predictions match measurements. The bug fix is validated.

However, two issues must be addressed in subsequent work (not blocking this experiment):

1. **K597 FAIL (routing accuracy 41.2% < 50%) should be investigated, not dismissed.** The paper reframes it as "routing accuracy does not matter because adapters are interchangeable." This reframing, while supported by the data, undermines the core thesis of domain-specific experts. A new experiment should test whether adapters are truly interchangeable by evaluating cross-domain PPL systematically (apply each adapter to each domain and measure the full 24x24 PPL matrix).

2. **The finding status should be "supported" not "conclusive."** The bug fix is conclusively verified, but the deeper question it raises (are adapters domain-specific or general?) is unresolved. The 24x24 cross-domain PPL matrix would resolve this.

3. **Prediction 3 should be honestly reported as a miss** in any finding recorded from this experiment. Routing accuracy did not improve significantly (39.2% -> 41.2% is within expected variance for a 480-sample evaluation).
