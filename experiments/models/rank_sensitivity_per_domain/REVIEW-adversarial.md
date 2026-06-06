# Peer Review: rank_sensitivity_per_domain

## Experiment Type
Guided exploration (Type 2) within the Eckart-Young-Mirsky framework.
Unknowns: whether PPL improvement from SVD truncation is directional (H1) or
magnitude-based (H2), and the fine-grained rank-quality curve.

## Hack Detector
- Fix count: 0. This is a diagnostic/analysis experiment, no new mechanisms.
- Is MATH.md a proof or a description? **Description dressed in theorems.** Eckart-Young and Davis-Kahan are cited as *context*, not applied to derive predictions. The predictions (P1-P5) are interpolations from Finding #325, not derived from the cited theorems. Acceptable for Type 2 exploration, but should be stated honestly.
- Metric used as evidence: PPL ratio (primary), token-overlap behavioral score (secondary). PPL-behavioral correlation (rho=-0.849) is demonstrated within this experiment.
- Kill criteria source: K836 is derived from a practical threshold ("within 20% of full merge"), not from the proof. Acceptable for exploration.

## Self-Test Audit
1. One-sentence impossibility property: "Eckart-Young guarantees the truncated SVD is optimal." This is not an impossibility property about the *experiment's mechanism* -- it is a statement about SVD generally. Marginal pass: the experiment does not claim a new mechanism, so "optimal approximation" as the grounding property is adequate.
2. Cited theorems: Eckart-Young-Mirsky (1936) -- real, applied correctly (optimal rank-r approximation). Davis-Kahan (1970) -- real, connection sketched loosely (operator norm vs Frobenius norm conflation, see below). Pass with caveat.
3. Predicted numbers: Scale factor c in [0.68, 0.75], H2 threshold 10%. Measured c in [0.648, 0.695]. The predicted range was based on Finding #325 first-module-only spectra; the actual 252-module averages yielded slightly lower values. The prediction was close but not derived from theory. Marginal pass.
4. Falsification condition: Clear. If scale-control matches rank-4, H1 is false. If it diverges, H2 is false. This is well-constructed. Pass.
5. Hyperparameter count: 1 (SVD rank). Correct. Pass.
6. Hack check: No fixes being stacked. Pass.

## Mathematical Soundness

**What holds:**
- Eckart-Young is correctly applied: truncated SVD is optimal rank-r approximation in Frobenius norm.
- The scale-control design is sound: matching ||c * delta_16||_F = ||delta_r||_F by setting c = sqrt(E(r)) is algebraically correct.
- The H1/H2 framing is a genuine contribution: this is the right question to ask.

**Issues:**

1. **Davis-Kahan misapplication (minor).** Section C claims "reducing ||delta||_F by factor c reduces ||delta||_op by at most factor c." This is true for uniform scaling (c * delta), but the Davis-Kahan theorem bounds subspace rotation by operator norm divided by spectral gap. The claim that "directional selection does NOT reduce operator norm" is correct (sigma_1 is preserved at rank >= 1), but the conclusion that H1 operates via "second-order perturbation effects" is hand-waving, not a derivation. No quantitative prediction follows from Davis-Kahan.

2. **Energy mismatch between spectral analysis and scale-control.** The 252-module-averaged energy (spectral_analysis) and the aggregate energy (rank sweep, used for scale-control) differ by 2-3pp. Example: medical mean_energy_r4 = 0.477 (spectral) vs 0.443 (rank sweep used in control). PAPER.md reports spectral analysis numbers in one table and rank-sweep numbers in another without noting the discrepancy. The scale-control used the correct quantity (aggregate energy from SVD extraction), but MATH.md's predictions used Finding #325 first-module-only numbers. This means the prediction "c in [0.68, 0.75]" was based on stale data; actual c was [0.648, 0.695]. Not fatal but sloppy.

3. **The scale-control does NOT match Frobenius norm correctly.** Critical issue. The scale-control multiplies the *adapter scale* (applied globally as `scale * B`) by c = sqrt(E(r)). But E(r) is computed from the *delta* = scale * B @ A. Scaling the adapter scale by c gives ||c * scale * B @ A||_F = c * ||scale * B @ A||_F = c * ||delta||_F. This equals sqrt(E(r)) * ||delta||_F = ||delta_r||_F. So the norm matching IS correct for the aggregate. However, this is *uniform* scaling across all 252 modules. SVD rank-4 truncation retains different energy fractions per module (some modules lose 30%, others lose 70%). The scale-control applies the same scalar c to ALL modules. This means the scale-control is not a true Frobenius-norm-matched control at the per-module level -- it is an average-matched control. The H1/H2 discrimination is weaker than claimed: the scale-control is a different perturbation structure than rank-4 SVD (uniform shrinkage vs heterogeneous truncation).

4. **Medical anomaly undermines the clean narrative.** Medical has 27.6% gap (scale-control PPL = 6.774 vs SVD r=4 PPL = 9.361). PAPER.md claims this is "in the WRONG direction" because scale-control is *better* than SVD. But this means for medical, the full-rank adapter at reduced scale dramatically outperforms the rank-4 SVD. This is actually evidence FOR H2 (magnitude reduction helps more than direction selection), not against it. The paper's framing is correct here despite the confusing direction of the gap.

## Prediction vs Measurement

PAPER.md contains the prediction table. Results from results.json:

| Prediction | Predicted | Measured | Match? | Verified in results.json? |
|------------|-----------|----------|--------|---------------------------|
| P1: r2 < r4 | r2 in [0.65, 0.75] | r2=0.747, r4=0.766 | YES | YES (0.7473 < 0.7661) |
| P2: r1 < r2 | r1 in [0.55, 0.70] | r1=0.710 | r1=0.710 > 0.70 (edge) | YES (0.7103 < 0.7473) |
| P3: H2 within 10% | <10% if H2 | 10.4% mean | BORDERLINE | YES (10.4 in results.json) |
| P4: same rank all domains | all same | 4/5 rank=1, math rank=2 | MOSTLY | YES |
| P5: behavioral tracks PPL | rho > 0.5 | rho=-0.849 | YES (negative) | YES (-0.849, p=0.0) |

**Critical note on P3:** The code declares H2 "supported" at the overall level (4/5 domains) despite the mean gap being 10.4% (above the 10% threshold). The verdict "H2 (magnitude reduction)" in the code is triggered by `h2_count >= 4`, not by the mean gap. This is legitimate (per-domain majority vote), but PAPER.md also reports "mean gap=10.4%, 4/5 domains <10%" and marks it "BORDERLINE." The paper is honest about this.

**Numbers verified against results.json:** All PPL values, behavioral scores, ratios, and scale-control values in PAPER.md match results.json. No fabrication detected.

## NotebookLM Findings

NotebookLM review was not run (authentication not verified). Manual deep review performed instead.

## Novelty Assessment

**Prior art:**
- FlexMoRE (cited) predicts knowledge=low rank, reasoning=high rank. This experiment finds no such differentiation, which is a genuine negative result at micro scale. Properly qualified: "does NOT replicate for rank-16 LoRA adapters."
- The H1/H2 discrimination via scale-control is a reasonable methodological contribution. Similar ablations exist in the adapter pruning literature (e.g., AdaLoRA, LoRA-drop), but the specific framing as a scale-control experiment is clean.
- The finding that lower rank = better PPL is consistent with the general LoRA over-parameterization literature (Aghajanyan et al., 2021, "intrinsic dimensionality").

**Delta over existing:** The H1/H2 discrimination is the primary contribution. The conclusion (magnitude reduction dominates) is important for the project but not novel in the broader literature -- it is well-known that LoRA adapters are often over-parameterized and lower-rank or lower-scale often helps.

## Critical Weaknesses (Non-Blocking)

1. **Behavioral evaluation is near-meaningless.** The `behavioral_score` function is token-overlap recall (stop words removed) for 4/5 domains, and 70% syntax-check + 30% token-overlap for code. One generation per domain, greedy decoding. The base model code score = 0.0 because it outputs reasoning text ("Let me think about what a palindrome is...") that does not parse as Python syntax, not because it cannot write code. SVD rank=1 code score = 0.85 because truncation forces the model toward rote template output that happens to parse. This metric rewards degenerate repetitive output that contains the right tokens. Example: rank=1 finance generation is "A stock is a share of ownership in a company. A bond is a debt that you own. A bond is a debt that you own. A bond is a debt..." -- this is clearly degenerate but scores 0.135 on factual recall.

2. **PPL on domain validation data is not a behavioral outcome.** The experiment correctly notes the project's overall PPL-task correlation is r=0.08, then argues within-adapter correlation should be higher. The rho=-0.849 finding is over 25 points (5 domains x 5 ranks), but these are not independent: the 5 ranks share the same adapter and same data. The effective degrees of freedom are much lower.

3. **The "raw LoRA degrades behavioral" finding (F5) is likely an artifact of scale=20.** Scale=20 was presumably chosen for training, not for inference quality. The fact that scale=20 is too high for generation does not mean the adapters are "harmful" -- it means the inference scale needs tuning. This is not a surprising finding.

## Macro-Scale Risks (advisory)

1. The H2 conclusion (scale > rank) may not hold at macro scale where adapters are trained with appropriate scales. The scale=20 over-parameterization drives most findings here.
2. FlexMoRE's knowledge-vs-reasoning differentiation was observed with 2896-rank adapters on 7B models. The absence at rank-16 on a 4B model is expected, not contradictory.
3. The uniform spectral profiles across domains (s1/s16 ratio 2.79-3.74) could change with longer training or different data distributions.

## Verdict

**PROCEED** (as provisional, Type 2 exploration)

The experiment is well-designed for its scope. The H1/H2 discrimination via scale-control is the right question and the answer (H2 dominates) is directionally supported. The core numerical claims are verified against raw data.

**Caveats that must appear in the finding:**

1. The scale-control is *uniform* shrinkage, not per-module Frobenius-norm-matched. The H1/H2 discrimination is weaker than claimed: it shows uniform scaling beats heterogeneous truncation, not that direction selection is irrelevant. Re-label: "uniform magnitude reduction matches or beats rank-4 SVD truncation."

2. P3 technically fails (10.4% mean gap > 10% threshold). The H2 verdict is based on per-domain majority vote (4/5), not the stated threshold. Finding status should note this borderline result.

3. Behavioral scores are not reliable enough to support F4 (rho=-0.849) or F5 (raw LoRA degrades). These should be flagged as directional observations, not findings.

4. The FlexMoRE non-replication is properly qualified in the paper but should explicitly note the rank-16 vs rank-2896 difference as the likely explanation, not present it as a contradiction.

5. Finding title should be: "SVD PPL improvement at scale=20 is primarily from magnitude reduction, not directional selection" (scoping to the specific scale regime tested).
