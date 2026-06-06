# Peer Review: Frobenius-Norm Equalized Composition

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 1 (Frobenius norm equalization before summation). Clean, no stacking. This explicitly replaces the DC-Merge and spectral surgery approaches that were killed. Good discipline.
- Is MATH.md a proof or a description? Mixed. The Proposition (Frobenius Norm of Orthogonal Sum) is a correct proof. Theorem 1 (Gini Reduction) asserts an unproven inequality as its critical step. Theorem 2 (Behavioral Preservation) is a logical framework, not a proof. Theorem 3 is a heuristic interpolation dressed as a theorem.
- Metric used as evidence: Gini coefficient (K703), PPL (K704), generation quality (K705). Gini is geometrically motivated. PPL is acknowledged as weak proxy (r=0.08). Generation samples provide behavioral grounding. Reasonable for Type 2.
- Kill criteria source: K703 is derived from Theorem 1's prediction. K704 is derived from Theorem 2's two-scenario framework. K705 is a behavioral check. Properly grounded for guided exploration.

## Self-Test Audit

1. **One-sentence impossibility property:** "Equal Frobenius energy across domains makes it impossible for any one domain to dominate the composed spectrum, by the Pythagorean property of orthogonal sums." Correct and specific. PASS.

2. **Cited theorems:** Pythagorean theorem for Frobenius norms (standard linear algebra), Grassmannian orthogonality (Finding #225), Gini scale-invariance (standard). All real theorems, all conditions apply. PASS.

3. **Predicted numbers:** P1: Gini <= 0.29. P2: two falsifiable scenarios for PPL. P4: Gini ~ 0.35. P5: norm ratio ~20:1. These are specific and falsifiable. PASS.

4. **Falsification condition:** "If equalized composition produces Gini > max individual Gini despite confirmed Grassmannian orthogonality." This targets the proof, not just the experiment. PASS.

5. **Hyperparameter count:** Full equalization: 0. Partial: 1 (compression exponent). Acknowledged, and the choice between full/partial is correctly identified as the Type 2 unknown. PASS.

6. **Hack check:** "No. This replaces the scale-imbalanced raw sum with energy-balanced raw sum. One mechanism: normalize, then sum." Accurate. PASS.

Self-test is complete and honest. No blanks or evasions.

## Mathematical Soundness

### Proposition (Frobenius Norm of Orthogonal Sum) -- CORRECT
The proof that ||Delta_comp||_F^2 = sum_i ||Delta_i||_F^2 under Grassmannian orthogonality is correct. The trace manipulation is valid: Tr(B_j^T A_j^T A_i B_i) = 0 when A_i^T A_j = 0. The claim that ||B^T A^T||_F = ||B||_F when A has orthonormal rows follows from the unitary invariance of the Frobenius norm: ||B^T A^T||_F^2 = Tr(A B B^T A^T). Since A has orthonormal rows (A A^T = I_r), this equals Tr(B B^T) = ||B||_F^2 only when the trace cycle gives us A^T A = I, which requires A^T to have orthonormal columns. But A is r x d_in with orthonormal rows, so A^T has orthonormal columns, and A A^T = I_r. The trace: Tr(A B B^T A^T) = Tr(B^T A^T A B) = Tr(B^T B) = ||B||_F^2. Correct.

**Caveat:** This holds exactly only when A_i^T A_j = 0. At |cos| = 0.026, the cross-term contribution is 0.026^2 = 0.000676, or 0.07% of energy. Negligible. ACCEPTABLE.

### Theorem 1 (Gini Reduction under Full Equalization) -- GAP IN PROOF
The proof rests on the inequality:

  Gini(union) <= max(Gini(group_i)) + Gini(group_norms)

This is asserted without proof or citation. It is not a standard result in the Gini coefficient literature. The standard Gini decomposition (Pyatt 1976, Mookherjee & Shorrocks 1982) decomposes total Gini into within-group, between-group, and overlap terms:

  G = sum_i w_i * G_i + G_between + G_overlap

where w_i are population-share-weighted income shares, and G_overlap captures the contribution from overlapping group distributions. The asserted bound Gini(union) <= max(G_i) + Gini(group_norms) does not follow directly from this decomposition. It appears to be a reasonable conjecture (verified numerically in the worked example), but it is not proven.

**Mitigation:** The experimental measurement (Gini = 0.267 after full equalization) is consistent with the bound (max_i G_i ~ 0.29). The bound appears to hold in practice. For Type 2 guided exploration, an unproven but empirically verified bound is acceptable. However, calling this a "Theorem" with "Proof" and "QED" is misleading when the critical inequality is asserted without derivation.

### Theorem 2 (Behavioral Preservation Condition) -- FRAMEWORK, NOT PROOF
This is correctly labeled as identifying the Type 2 unknown. It provides a logical if-and-only-if framework for interpreting the results. The reasoning is sound: if scales encode genuine capability requirements, equalization destroys them; if scales are artifacts, equalization corrects them. The experiment resolves the unknown (scales encode both). This is appropriate for Type 2.

### Theorem 3 (Partial Equalization Gini Prediction) -- INTERNAL INCONSISTENCY
There is a clear internal contradiction between Theorem 3 and prediction P4:

- Theorem 3 derives: at R = sqrt(20) ~ 4.5, Gini ~ 0.28 + 0.148 = **0.43**
- Prediction P4 states: Gini ~ **0.35** (30% reduction from 0.49)
- Measurement: **0.393**

Neither prediction is accurate. The measurement sits between them.

Furthermore, Theorem 3 is not a theorem in any meaningful sense. It is a heuristic interpolation formula calibrated to the measured Gini at R=20. Specifically, w_between is solved from the R=20 data point:

  0.49 - 0.28 = (20-1)/(20+1) * w_between => w_between = 0.232

This means the "theorem" has zero predictive power at R=20 (it is calibrated to that point) and limited predictive power at other R values (it is an interpolation, not derived from first principles). The (R-1)/(R+1) formula for between-group Gini applies to two groups with ratio R:1, but with 5 groups having different ratios, the effective R is not simply the max/min ratio. FLAG.

### Worked Example -- CORRECT
The N=2, r=4 worked example is arithmetically correct. I verified the Gini calculations: raw composed Gini of [10, 6, 4, 2, 0.4, 0.3, 0.25, 0.15] and equalized Gini of [2.154, 1.857, 1.393, 1.292, 1.161, 0.862, 0.696, 0.431]. The 66% reduction is consistent with the two-group case. PASS.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Assessment:

| Prediction | Derived From | Measured | Match? | Notes |
|---|---|---|---|---|
| P1: Gini <= 0.29 | Theorem 1 | 0.267 | YES | Bound holds (8% margin) |
| P2a: All 5 within 5% PPL (if artifacts) | Theorem 2 | 0/5 within 5% | NO | Correctly falsifies "scales are artifacts" |
| P2b: High-scale hurt, low-scale helped (if signal) | Theorem 2 | med +18.5%, math +16.2%, legal -9%, fin -6% | YES | Confirms scales encode signal |
| P3: Generation coherent on >=2 domains | Behavioral | 5/5 coherent | YES | |
| P4: Partial Gini ~ 0.35 | MATH.md P4 (not Theorem 3!) | 0.393 | NO | 12.3% off; Theorem 3 gives 0.43 (8.6% off) |
| P5: Norm ratio ~20:1 | Corollary | 21.6:1 | YES | |
| P5: Top-3 energy ~99% | Corollary | 98.7% | YES | |
| B-norms similar | Assumption | 29.1-31.5 (8% spread) | YES | Validates key assumption |

Score: 5/8 clear matches, 1 correctly falsified, 2 misses (P2a scenarios resolved by P2b, P4 inaccurate).

**Critical assessment:** The experiment's core value is NOT in verifying Theorem 1 (which has the unproven bound). The value is in resolving the Type 2 unknown: scales encode both signal and artifact. This is a genuine discovery from a well-designed two-scenario test. The Gini predictions are secondary to the behavioral finding.

## NotebookLM Findings

NotebookLM deep review was not executed due to the tool requiring interactive authentication. Review conducted through direct document analysis.

## Novelty Assessment

**Prior art:** FroM (arXiv:2506.02478) already describes Frobenius-norm adaptive merging. DO-Merging (arXiv:2505.15875) decouples magnitude and direction. This experiment applies a simpler version (uniform normalization, not adaptive) to the specific context of Grassmannian-orthogonal ternary LoRA adapters.

**Genuine novelty:** The finding that per-domain optimal scales encode BOTH training artifact (energy imbalance) and genuine capability (domain-specific perturbation magnitude) is a useful empirical result. The prior DC-Merge experiment (#277) identified the disease (cross-domain scale imbalance), spectral surgery (#278) showed post-hoc correction is structurally inverted, and this experiment shows that the cure must be partial, not total. This three-experiment arc is well-structured.

**What was already known:** Frobenius-norm normalization before model merging is standard practice in the merging literature (TIES-Merging, DARE, SLERP all handle scale). The innovation here is applying it within the Grassmannian composition framework and discovering the signal/artifact duality of the scale factors.

## Macro-Scale Risks (advisory)

1. **Routing makes this potentially moot.** With per-token routing selecting top-k adapters, the composition is over 1-2 adapters, not all 5. The 20:1 scale ratio only causes problems when summing ALL adapters uniformly. If routing is working (and VISION.md says softmax router matches oracle at N=24), this equalization may be unnecessary at production time. The PAPER.md acknowledges this in "What Would Kill This" item 2.

2. **The 50% log-compression factor is unprincipled.** It is a single data point (N=5, these specific adapters, this specific base model). There is no theory predicting the optimal compression factor. At macro scale with different domain combinations, this number could be completely different. The PAPER.md acknowledges this in Limitation 3.

3. **Statistical power at N_eval=20.** The PPL differences for partial equalization (2.3%, -1.3%, 4.6%, -5.6%, -3.6%) are small relative to the sampling noise at N=20 per domain with MAX_SEQ_LENGTH=256. No confidence intervals are reported. The 1.2% mixed PPL improvement could easily be within noise. For micro-scale this is acceptable but should not be treated as definitive.

4. **B-norm similarity assumption.** The 8% spread (29.1-31.5) is reassuringly small for these 5 adapters, but this is a consequence of similar training procedures and data volumes. At macro scale with heterogeneous adapter sources (different trainers, different data sizes, different hyperparameters), B-norms could vary much more, changing the equalization dynamics.

## Specific Issues

### Issue 1: Theorem 1 bound is unproven
The Gini union bound `Gini(union) <= max(G_i) + Gini(group_norms)` is the central mathematical claim but is neither proven nor cited. It is plausible (I verified it numerically on several examples) but presenting it with Theorem/Proof/QED structure when the critical step is an assertion is misleading. For a verification experiment (Type 1), this would be blocking. For Type 2 guided exploration, it is a weakness but not fatal, since the measurement independently confirms the bound holds.

### Issue 2: Theorem 3 vs P4 internal inconsistency
MATH.md contains two different predictions for partial equalization Gini:
- Theorem 3: Gini ~ 0.43 (for R=4.5)
- P4: Gini ~ 0.35 (30% reduction)

Neither matches the measurement (0.393). This inconsistency should be corrected. The Theorem 3 derivation (0.43) is at least derived from a formula; the P4 prediction (0.35) appears to come from a different calculation or is simply a desired target.

### Issue 3: K704 assessed for full equalization, but finding pivots to partial
The kill criterion K704 is "at least 3/5 domains PPL within 5% of raw sum." This formally FAILS for full equalization (0/5). The PAPER.md then evaluates partial equalization (4/5 within 5%) and treats this as an effective pass. This is acceptable as a Type 2 pivot -- discovering that full equalization is too aggressive IS the finding -- but the formal K704 status should be clearly reported as FAIL (full) / PASS (partial), which the paper does.

### Issue 4: "Dual nature" claim needs caution
The claim that scales encode "both signal and artifact" is the right interpretation of the data, but it is an inference from N=1 experiment with 5 domains. The signal component could also be explained by:
- Different validation set difficulties (medical/math eval sets may be harder for the base model, making adapter contribution more noticeable)
- Different adapter training convergence (some domains may have been undertrained)
- Evaluation metric sensitivity (PPL may be more sensitive to scale for some domains)

The paper does not control for these alternatives. For a micro-scale finding this is fine, but the "dual nature" framing should be presented as an interpretation, not a confirmed mechanism.

### Issue 5: Gini standard deviations are heterogeneous
The standard deviations of Gini across sampled (layer, key) pairs differ substantially:
- Raw sum: 0.022 (4.5% CV)
- Full equalization: 0.037 (14% CV)
- Partial: 0.026 (6.6% CV)

Full equalization has 3x the relative variability of the raw sum. This suggests equalization affects different layers/keys differently, possibly because the B-matrix Gini varies more than reported. The per-sample Ginis for full equalization range from 0.228 to 0.365 -- the upper end exceeds the max individual adapter Gini of 0.29 stated in Theorem 1. This means Theorem 1's bound is violated on 2 of 14 samples (0.329 and 0.365). The bound holds for the mean but not uniformly. This is not acknowledged.

## Verdict

**PROCEED** (as supported finding, with minor revisions)

The experiment is well-designed, answers a genuine question from two prior killed/provisional experiments, and produces a useful finding. The three-experiment arc (DC-Merge -> spectral surgery -> Frobenius equalization) demonstrates disciplined investigation of a root cause. The core finding -- that per-domain scales encode both artifact and signal, making partial equalization the practical approach -- is valuable for the composition pipeline.

The mathematical framework is adequate for Type 2 guided exploration, despite the unproven Gini union bound and the Theorem 3 inconsistency. The exploration successfully narrows the Type 2 unknown (from "do scales matter?" to "scales partially encode capability, partial compression is needed") and the behavioral evidence (generation quality) supports the metric results.

**Revisions required (non-blocking, should be done before recording finding):**

1. **Fix Theorem 3 vs P4 inconsistency.** Either remove the P4 prediction (Gini ~ 0.35) or reconcile it with Theorem 3's derivation (Gini ~ 0.43). State which prediction was actually tested and report the mismatch honestly.

2. **Acknowledge Theorem 1 bound is unproven.** Add a note that the Gini union bound is conjectured, not proven. It holds empirically (on the mean) but is violated on 2/14 individual (layer, key) samples (Gini = 0.329 and 0.365 > 0.29). Downgrade from "Theorem/Proof/QED" to "Conjecture (empirically supported)."

3. **Report the per-sample bound violations.** The full equalization produces Gini > 0.29 on some individual (layer, key) pairs. This is relevant because it means the theoretical guarantee does not hold uniformly. State this and note it does not affect the overall finding.

4. **Finding status: "supported" is correct for Type 2.** The exploration narrowed the unknown (full equalization too aggressive, partial equalization works, scales are dual-natured). The status should not be "conclusive" (no formal proof of optimality) or "provisional" (the finding is well-evidenced for the specific setup). "Supported" is right.

5. **Note that routing may make this moot.** The PAPER.md does mention this, but it should be more prominent. If the softmax router selects top-1 or top-2 adapters per token (as VISION.md indicates is the serving model), uniform summation of all 5 adapters is not the production configuration. The equalization finding applies to the pre-merge / always-on composition path, not the routed path.
