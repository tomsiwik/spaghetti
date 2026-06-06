# Peer Review: Cross-Domain PPL Matrix N=24

## Experiment Type
Guided exploration (Type 2). The experiment operates within the proven LoRA perturbation framework and Grassmannian skeleton. The unknown being narrowed is the *degree* of domain specialization, not whether it exists in principle.

## Hack Detector
- Fix count: 0. This is a pure measurement experiment. No mechanisms, losses, or tricks.
- Is MATH.md a proof or a description? **Description dressed in equations with a "proof sketch" label.** The "Theorem (Adapter Specificity)" on lines 57-76 is labeled QED (sketch), which is an honest admission that it is NOT a complete proof. The argument is informal: "B_j was optimized for P_j, not P_i, therefore L(i,j) > L(i,i)." This hand-waves over: non-convexity of the loss landscape, finite training steps (convergence is assumed not demonstrated), the fact that LoRA rank-16 may lack capacity to distinguish distributions, and the role of shared A-matrices constraining the optimization landscape.
- Metric used as evidence: PPL (perplexity). PAPER.md itself acknowledges (Limitation #1) that PPL does not predict task quality (r=0.08 correlation from Finding #200). This is stated honestly but weakens the finding.
- Kill criteria source: DDR > 1.05 is derived from the theorem's prediction + DES-MoE reference range. Reasonable for a guided exploration. The 12/24 diagonal wins threshold and 20% improvement are reasonable lower bounds.

## Self-Test Audit
1. **One-sentence impossibility property:** "If domains are distributionally distinct and adapters are trained to convergence, domain-specific adapters MUST outperform mismatched adapters." This is a valid single property. However, it conflates two assumptions (distributional distance AND convergence) into what is claimed as one property. Minor issue.
2. **Cited theorems:** "Optimality of gradient-descent solutions for convex surrogate losses." The cross-entropy loss over a neural network is NOT convex. The cited theorem does not apply directly. The proof sketch acknowledges "local minimum" but the self-test claims convex optimization theory, which is a mismatch. **FLAG: cited theorem conditions do not hold.**
3. **Predicted numbers:** DDR > 1.05, diagonal wins >= 18/24, wrong-adapter PPL in [6.5, 9.0]. These are specific and falsifiable. Pass.
4. **Falsification condition:** "If DDR < 1.05 and fewer than 12/24 domains show diagonal dominance." This targets the prediction, which is appropriate. But the falsification threshold (12/24) differs from the prediction threshold (18/24), creating a gray zone between 12 and 18 that is neither confirmed nor falsified. Minor issue.
5. **Hyperparameter count:** 0. Correct -- this is a measurement experiment.
6. **Hack check:** "No fixes being added. Pure measurement." Correct.

## Mathematical Soundness

**The "proof" is a plausibility argument, not a proof.** Specifically:

1. **Non-convexity.** The argument assumes that because B_i minimizes L(i,i), applying B_j (optimized for P_j) must yield higher L(i,j). This follows trivially for convex problems but neural network loss landscapes are non-convex. It is entirely possible that B_j sits in a basin that happens to also be good for P_i.

2. **Finite capacity escape.** With rank-16, the adapter has very limited capacity. If the 16-dimensional subspace cannot capture the domain-specific features, all adapters might converge to the same generic improvement. The theorem does not bound the specialization gap as a function of rank.

3. **Shared A-matrix constraint.** The Grassmannian A matrices impose structure that the proof ignores. The effective perturbation is s * A_i * B_i. Even if B_i is optimized for domain i, the composition A_i * B_i may have limited expressivity if A_i was designed for decorrelation rather than domain-specific projection.

4. **The proof makes no quantitative prediction.** It says E[L(i,j)] > E[L(i,i)] but does not bound the gap. The quantitative predictions (DDR > 1.05, etc.) are derived from prior empirical results (Finding #200, DES-MoE), not from the proof. This is acceptable for a Type 2 guided exploration but should be stated explicitly.

**Verdict on math:** For a Type 2 (guided exploration), the mathematical framework is adequate. The experiment correctly identifies that the DEGREE of specialization is the unknown, and the proof sketch provides the theoretical motivation. The self-test's claim of convex optimization theory is incorrect but the experiment does not depend on it being a formal proof.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. All six predictions match:

| Prediction | Threshold | Measured | Match |
|-----------|-----------|----------|-------|
| DDR > 1.05 | 1.05 | 1.126 | YES |
| Diagonal wins >= 18/24 | 18 | 24/24 | YES (exceeds) |
| Wrong-adapter PPL in [6.5, 9.0] | range | ~7.0 avg | YES |
| Avg improvement >= 20% | 20% | 34.8% | YES |
| Some domain pairs near-interchangeable | qualitative | 14/24 non-diagonal best | YES |
| Specificity correlates with domain distance | qualitative | math DDR=1.31, linguistics DDR=1.06 | YES |

The predictions are directionally correct and the measurements are within expected ranges.

**Key concern: the 24/24 diagonal wins exceeds the 18/24 prediction by a wide margin.** This is good news empirically but raises a question about calibration -- the prediction was too conservative. The 34.8% average improvement also exactly matches the oracle PPL from Finding #200, which is expected since diagonal = own adapter.

## Code Verification

I verified the DDR computation in `run_experiment.py` (lines 418-428). The code correctly computes column-centric DDR: for each evaluation domain, average PPL from all non-matching adapters divided by PPL from the matching adapter. This matches the MATH.md definition.

The `set_lora_a()` function (lines 209-225) correctly uses domain index for skeleton lookup, matching the fix from Finding #201.

One note: the base PPL computation (Phase 1, lines 282-322) applies LoRA structure with zeroed B weights rather than computing without any LoRA at all. This means "base" includes the computational path through the LoRA module with zero output, which should be mathematically equivalent to no LoRA but adds a minor assumption.

## Novelty Assessment

**Low novelty, high utility.** Cross-domain evaluation matrices are standard practice (cited: LoRAuter arXiv:2601.21795). The contribution is applying this methodology to Grassmannian-initialized adapters and discovering the "simultaneous specificity and generality" phenomenon. The finding that DDR = 1.13 (modest) while diagonal wins = 24/24 (perfect) is genuinely informative for routing architecture decisions.

## NotebookLM Findings

Skipped -- the experiment files are straightforward enough for direct review. The key insights are apparent from the documents themselves.

## Macro-Scale Risks (advisory)

1. **PPL specificity may not translate to behavioral specificity.** The paper acknowledges this (r=0.08 correlation). At macro scale, the question is whether adapters produce different *outputs*, not different PPL values. A DDR of 1.13 on PPL could correspond to DDR of 1.0 or 2.0 on task accuracy.

2. **Rank scaling.** At higher ranks (32, 64), adapters may show stronger or weaker specialization. The rank-16 finding may not extrapolate.

3. **The "routing doesn't matter" conclusion is PPL-specific.** If behavioral specialization is stronger than PPL specialization (plausible for domains like code vs creative writing), routing accuracy could still matter for task quality.

## Verdict

**PROCEED**

Justification:

1. For a Type 2 guided exploration, the mathematical framework is adequate. The proof sketch provides correct motivation even though it is not a formal proof. The experiment honestly identifies itself as guided exploration (MATH.md line 78).

2. All quantitative predictions match measurements. The prediction-vs-measurement table is clear and complete.

3. The experiment produces a genuinely useful finding: adapters are simultaneously specific (DDR=1.13, 24/24 diagonal wins) AND general (every adapter helps every domain). This directly explains the Finding #200 paradox (41% routing accuracy = oracle PPL).

4. The code is correct. DDR computation matches the mathematical definition. A-matrix loading uses the verified fix from Finding #201.

5. Limitations are honestly stated (PPL is not task quality, small eval set, rank constraint).

**Minor revisions recommended (non-blocking):**

1. Self-test item 2 should cite non-convex optimization or first-order optimality conditions, not "convex surrogate losses." The loss landscape is non-convex.

2. The "QED (sketch)" should be relabeled as "Plausibility Argument" or "Motivation" to avoid confusion with a formal proof. Proof sketches in mathematics still carry proof structure; this is closer to an informal argument.

3. PAPER.md should explicitly note that the quantitative predictions come from prior empirical results, not from the proof itself. The proof only predicts the sign of the gap (DDR > 1), not the magnitude.
