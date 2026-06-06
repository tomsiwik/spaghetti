# Peer Review: Norm-Bounded Adapter Training (Revision 2)

## Experiment Type
Guided exploration (Type 2)

## Prior Review Status
The prior review (REVISE) raised 4 blocking issues. This review verifies whether all 4 are adequately addressed.

## Hack Detector
- Fix count: 3 strategies tested as alternatives (not stacked). Each is a single mechanism replacing the prior post-hoc equalization stack. NO FLAG.
- Is MATH.md a proof or a description? **Mixed but adequate for Type 2.** Theorem 1a-b is a correct proof with QED (energy fraction bounds under Grassmannian orthogonality). Theorem 1c is clearly labeled as FALSIFIED. Theorem 2 is valid for convex, honestly caveated for non-convex, with post-experiment addendum noting empirical refutation. The framework is proven (Pythagorean energy decomposition, projected gradient descent convergence); the unknown (whether training-time constraints preserve adapter quality) is properly identified for guided exploration.
- Metric used as evidence: Composed spectral Gini, mixed PPL, convergence rate. Correct for the hypothesis.
- Kill criteria source: K709 derived from Theorem 1c (now acknowledged as falsified). K710 derived from Theorem 2. K711 derived from PGD convergence theory. Properly grounded, with post-hoc acknowledgment that K709's threshold was based on a faulty bound.

## Self-Test Audit

1. **One-sentence impossibility property:** "Uniform Frobenius norm constraint on B during training makes cross-domain energy imbalance impossible because ||Delta_i||_F = s * ||B_i||_F * sqrt(r) with s=1 and ||B_i||_F = tau for all i." Single property, precisely stated. PASS.

2. **Cited theorems:** Projected gradient descent convergence (Bertsekas 1999) -- real, correctly applied. Pythagorean theorem for Frobenius norms -- standard, conditions verified (|cos|=0.026). Gini scale-invariance -- standard. PASS.

3. **Predicted numbers:** P1: Gini <= 0.15. P2: PPL <= 6.508. P3: >= 3/5 converge. P4: B-norm ratio < 2.0. Specific and falsifiable. PASS.

4. **Falsification condition:** Updated post-experiment to note that this actually happened -- Theorem 1c's Gini decomposition was structurally incomplete (missing Pyatt 1976 overlap term). Correctly distinguishes the falsification of Theorem 1c from the still-valid Theorems 1a-b. PASS.

5. **Hyperparameter count:** 2 (tau, lambda). Acknowledged as unknowns for Type 2 exploration. PASS.

6. **Hack check:** "No. This REPLACES the per-domain scale + post-hoc equalization stack with a single training-time constraint." Accurate -- the mechanism count goes from 3 to 1. PASS.

Self-test complete. No blanks or evasions.

## Blocking Issue Verification

### Issue 1: Correct the 7%/93% decomposition claim -- ADEQUATELY ADDRESSED

The prior review identified that comparing baseline Gini (0.490, adapters trained at per-domain optimal scales) vs Strategy C Gini (0.456, DIFFERENT adapters trained at uniform s=10) was confounded. The revision corrects this throughout:

- MATH.md Section "Corrected Gini Decomposition" (lines 457-498): Explicitly calls out the original 7%/93% as wrong, provides the correct decomposition using Finding #279 (same adapters, full equalization -> Gini 0.267), yielding ~45% between-domain / ~55% within-domain.
- PAPER.md "CRITICAL FINDING" section (lines 104-123): Uses Finding #279 as the correct comparison baseline.
- PAPER.md Limitation 5 (line 180): Explicitly flags the original claim as confounded.
- PAPER.md Key Structural Finding #2 (lines 209-218): Correct decomposition presented consistently.

The methodology is now sound: isolating the between-domain energy effect by equalizing the SAME adapters (Finding #279 full equalization) rather than comparing adapters trained under different conditions.

**Minor factual inconsistency (non-blocking):** MATH.md line 50 states "50% log-compression yields Gini 0.267" but this is incorrect. Finding #279 shows that 50% log-compression (partial equalization) yields Gini 0.393. The Gini 0.267 comes from FULL equalization (100%). The rest of the document correctly references Finding #279's full equalization for the 0.267 figure, so this is an isolated copy error that does not affect the conclusions.

### Issue 2: Acknowledge Theorem 1c Gini union bound falsified -- ADEQUATELY ADDRESSED

- MATH.md Theorem 1c (lines 159-169): Clearly labeled "[CONJECTURE -- EMPIRICALLY FALSIFIED]" with explanation of the missing overlap term from Pyatt (1976).
- MATH.md lines 189: Proof explicitly "[WITHDRAWN -- bound falsified experimentally.]"
- MATH.md Post-experiment addendum (lines 407-428): Full analysis including the specific numbers (bound predicts <= 0.316, measured 0.456, exceeds by 44%).
- MATH.md correctly notes that Finding #279's measurement (0.267) falling below the bound was coincidental, not validating.
- PAPER.md Limitation 4 and Key Structural Finding #3 both state the falsification clearly.

The treatment is thorough and scientifically honest.

### Issue 3: Strategy B miscalibration -- ADEQUATELY ADDRESSED

- PAPER.md Training Convergence table (line 70): Strategy B explicitly labeled "MISCALIBRATED -- results not meaningful for the norm-bounding hypothesis."
- PAPER.md lines 71-76: Detailed calculation showing WD loss ~57 vs CE loss ~1.0 (57x domination).
- PAPER.md Limitation 1 (lines 156-159): Explains what went wrong and what might fix it.
- Strategy B is excluded from all conclusions in Key Structural Findings.
- PAPER.md "What Would Kill This" section (line 199): Explicitly notes that Strategy B's "best PPL" of 6.652 "should not be credited."

### Issue 4: Reframe findings correctly -- ADEQUATELY ADDRESSED

- PAPER.md Key Structural Finding #1 (lines 202-207): "Training-time norm constraints produce WORSE composition quality than post-hoc equalization" with quantitative comparison (0.456 vs 0.267).
- PAPER.md Key Structural Finding #2 (lines 209-218): "Adapter quality matters as much as energy balance" -- proven by Strategy C having near-perfect energy equalization yet worse Gini than Finding #279.
- PAPER.md Key Structural Finding #3 (lines 222-224): Gini bound falsified.
- MATH.md addendum (lines 439-453): Theorem 2 explicitly noted as "partially falsified for non-convex case" with the correct observation that the non-convex caveat in the proof is the operative case.

The finding is framed as a negative result that narrows the search space, which is the correct framing for a Type 2 guided exploration.

## Mathematical Soundness

### Theorem 1a (Energy Fraction Bounds) -- CORRECT
The Pythagorean decomposition ||Delta_comp||_F^2 = sum_i s^2 ||B_i||_F^2 * r follows from verified Grassmannian orthogonality (|cos|=0.026). Energy fraction bounds f_i in [1/(NR^2), R^2/N] are direct arithmetic consequences. QED valid.

### Theorem 1b (Perfect Equalization) -- CORRECT (trivially)
If ||B_i||_F = tau for all i, then f_i = 1/N. Arithmetic identity. QED valid.

### Theorem 1c (Gini Union Bound) -- CORRECTLY WITHDRAWN
Falsified experimentally. The MATH.md properly identifies the structural gap: omitted overlap/interaction term from the standard Pyatt (1976) Gini decomposition. The withdrawal is the right scientific response.

### Theorem 2 (Training vs Post-Hoc) -- ADEQUATELY CAVEATED
Valid for convex loss (L(B*) <= L(B_post) by optimality). The non-convex extension is acknowledged as informal, and the post-experiment addendum explicitly notes the empirical refutation. The authors do not overclaim.

### Type 2 Framework Compliance
- Proven framework: Frobenius norm governs composed energy allocation (verified in Finding #279). Grassmannian orthogonality decouples domains (|cos|=0.026). Both are established.
- Unknown identified precisely: "Can training-time norm constraints produce scale-balanced adapters that retain domain capability?"
- Exploration narrows the unknown: Answer is NO -- training-time constraints produce worse results than post-hoc equalization on the same adapters, because within-domain SV structure (determined by training conditions) matters as much as between-domain energy balance.

This satisfies Type 2 requirements.

## Prediction vs Measurement

| Prediction | Source | Measured | Match? | Assessment |
|---|---|---|---|---|
| P1: Gini <= 0.15 | Thm 1c (falsified) + assumption about SV compression | 0.440 (A), 0.456 (C) | NO | Bound was structurally incomplete. Both strategies exceed even the corrected bound of 0.316. |
| P2: PPL <= 6.508 | Theorem 2 (convex optimality) | 6.839 (A), 7.129 (C) | NO | Non-convex landscape defeats constrained optimization within 200 steps. Strategy B (6.652) excluded as miscalibrated. |
| P3: >= 3/5 converge | PGD theory (Bertsekas) | A: 4/5, B: 0/5, C: 1/5 | PARTIAL | Strategy A meets threshold. C does not (4/5 fail). |
| P4: B-norm ratio < 2.0 | Constraint definition | A: 5.2:1, C: 1.2:1 | PARTIAL | C achieves goal. A fails due to asymmetric projection (clips down only). |
| P5: Training-time better than post-hoc | Theorem 2 | All valid strategies worse | NO | Clear refutation for non-convex setting. |

Score: 0 clear passes, 2 partial, 3 clear failures. The experiment comprehensively refutes the hypothesis, which is a valid and valuable outcome for a Type 2 guided exploration.

## Novelty Assessment

**Prior art:** NB-LoRA (arXiv:2501.19050) bounds singular values during LoRA training; DeLoRA (arXiv:2503.18225) decouples magnitude and direction; DO-Merging (arXiv:2505.15875) handles direction-only merging. All cited. None specifically address whether training-time norm equalization can replace post-hoc equalization for Grassmannian-orthogonal ternary adapters in a composition setting.

**Novel contributions:**
1. Empirical evidence that training-time norm constraints produce worse composition quality than post-hoc equalization for ternary LoRA adapters (this specific comparison is new).
2. Falsification of the Gini union bound (Theorem 1c) -- the missing overlap term is identified and diagnosed.
3. Correct Gini decomposition for this architecture: ~45% between-domain / ~55% within-domain (correcting the confounded 7%/93% from the original analysis).
4. Resolution of the spectral arc: post-hoc partial equalization (Finding #279) remains the practical ceiling.

## Macro-Scale Risks (advisory)

1. The 200-step training budget may be too short for the constrained optimizer to find good B-directions. At macro scale with longer training, constrained optimization might eventually close the gap with post-hoc equalization. This does not invalidate the micro finding but limits its extrapolation.

2. With per-token softmax routing (VISION.md: matches oracle at N=24), uniform summation of all N adapters at equal weight is not the deployment scenario. The scale equalization question matters primarily for always-on composition or pre-merge scenarios.

3. The Gini decomposition (45%/55% between/within) is specific to N=5, r=16, and these particular domains. The ratio could shift at different scales.

## Verdict

**PROCEED**

All 4 blocking issues from the prior review are adequately addressed:

1. The 7%/93% confounded decomposition has been replaced with the correct 45%/55% decomposition using Finding #279 (same adapters, full equalization).
2. Theorem 1c's Gini union bound is explicitly labeled as falsified, with proper structural diagnosis (missing Pyatt 1976 overlap term).
3. Strategy B is clearly labeled as miscalibrated (WD loss 57x CE loss) and excluded from all conclusions.
4. The finding is correctly reframed as a negative result: training-time norm bounding does not outperform post-hoc equalization, and Theorem 2 is explicitly noted as refuted for the non-convex case.

The experiment satisfies Type 2 (guided exploration) requirements: the proven framework is stated (Frobenius energy allocation + Grassmannian orthogonality), the unknown is precisely identified (whether training-time constraints can replace post-hoc equalization), and the exploration conclusively narrows the unknown (answer: no, post-hoc is better because it preserves within-domain SV structure).

The finding status of "supported (negative result)" is appropriate. The negative result is well-characterized, the mathematical framework is mostly sound (with the falsified Theorem 1c properly withdrawn), and the spectral arc is meaningfully resolved.

**Non-blocking issue for the record:**

1. MATH.md line 50 states "50% log-compression yields Gini 0.267" but Finding #279 shows 50% log-compression yields Gini 0.393. The Gini 0.267 comes from full (100%) equalization. This is a minor copy error that does not affect conclusions, since the rest of both documents correctly reference Finding #279's full equalization for the 0.267 figure.
