# Peer Review: Information Preservation Predictor

## NotebookLM Findings

Skipped (authentication not configured). Review conducted through direct analysis of MATH.md, PAPER.md, code, and project context.

## Mathematical Soundness

### What holds

1. **IP_avg metric definition is correct.** The formula `IP_avg(m) = 1 - ||tau_m - tau_avg||_F / ||tau_avg||_F` is well-defined and equals 1.0 for simple average by construction, as stated.

2. **Parallel axis theorem decomposition is correct.** The derivation `sum_k ||tau_avg - tau_k||^2 = sum_k ||tau_k||^2 - N ||tau_avg||^2` is the standard variance decomposition. The orthogonal-case specializations for IP_orig and NR are correct and match empirical values (NR(avg) = 0.715 at N=2 vs predicted 0.707; IP_orig = 0.296 at N=2 vs predicted 0.293).

3. **Spearman implementation is correct.** The code handles ties via average ranks and uses the Pearson-of-ranks formula (not the shortcut d^2 formula which breaks on ties). Verified step-by-step.

4. **DARE noise amplification argument is sound.** Rescale factor 1/(1-p) amplifies both signal and noise. At p=0.9 the 10x rescale produces NR=2.264, correctly captured by the metric. The paper's explanation of why this hurts is mechanistically clear.

### What does not hold or needs qualification

5. **Norm ratio correlation is non-monotonic in principle but treated as monotonic.** The code (line 507) computes `norm_vals = [-agg[m]["norm_ratio_mean"] for m in methods]`, assuming "lower NR = better." This happens to work because all good methods have NR < 1 and all bad methods have NR > 1 in this dataset. But the MATH.md correctly notes NR < 1 means signal attenuation, and one could construct a method with NR = 0.01 (massive attenuation) that would rank as "best" under this metric while being terrible. The monotonic assumption holds empirically here but is not principled. The paper should state this more explicitly: the Spearman correlation benefits from the fact that no tested method has pathologically low NR.

6. **Concat+cal metrics are hardcoded, not measured.** Lines 402-406 of the code assign `ip_vs_avg = 1.0, ip_vs_orig = 1.0, norm_ratio = 1.0` to concat+cal by fiat because it has no single merged delta. This is defensible for IP (it truly preserves all information) but NR=1.0 is meaningless -- concat+cal does not produce a merged vector whose norm can be compared. With 8 methods, one hardcoded data point is 12.5% of the Spearman computation. The paper acknowledges concat+cal breaks monotonicity at N=5 but does not disclose the hardcoding. **Impact: modest.** At N=5, removing concat+cal (computing over 7 methods) would change rho_norm slightly but likely still exceed 0.8 given the strong separation between tiers. Still, this should be disclosed and a sensitivity analysis (rho with and without concat+cal) should be reported.

7. **Confidence intervals on Spearman rho are wide and unreported.** With n=8 methods, a single rank swap changes rho substantially. The Fisher transformation 95% CI for rho=0.952 at n=8 is approximately [0.79, 0.99]. The paper mentions this limitation ("8 methods is a small sample") but does not compute the actual confidence interval. At the lower bound (0.79), KC2 would marginally fail. This is important context for the "KC2 passes" claim.

8. **MATH.md claim: "simple average is the unique lossless linear merge."** This is imprecise. The simple average is the minimizer of sum_k ||tau_m - tau_k||^2 over all tau_m (the centroid). Calling it "lossless" is misleading -- it loses all information about inter-delta variance. It is the best single-point summary under squared loss, not "lossless."

## Novelty Assessment

### Prior art

The idea that weight-space distance from the average predicts merging quality is implicit in Task Arithmetic (Ilharco et al., ICLR 2023) and explicit in model merging folklore. The DARE paper (Yu et al., 2023) already discusses rescaling-induced noise as a failure mode. The specific contribution here -- formalizing this as Spearman correlation with explicit metrics, and finding the NR > 1.5 threshold -- is a modest but useful formalization.

No published work was found that formally establishes norm ratio as a zero-cost failure detector for LoRA merging. The practical rule (NR > 1.5 predicts failure) has clear utility even if the underlying observation is not deeply novel.

### Delta over existing work

The experiment correctly traces its lineage to the lora_merging_bakeoff reviewer's observation. The formalization is the contribution. The delta is: (1) three specific metrics defined and compared, (2) norm ratio identified as best single predictor, (3) NR > 1.5 threshold validated across N=2 and N=5. This is incremental but properly scoped for a diagnostic micro-experiment.

## Experimental Design

### Strengths

1. **The experiment tests exactly what it claims.** Three metrics, two kill criteria, two scale conditions, three seeds each. Clean design.

2. **Kill criteria are well-calibrated.** KC1 (rank mismatches <= 1) is appropriately strict. KC2 (Spearman >= 0.8) is standard. The experiment honestly reports KC1 as killed rather than cherry-picking the passing criterion.

3. **Multi-seed aggregation is correct.** Averaging quality and IP across 3 seeds before ranking prevents single-seed flukes from driving the result.

4. **The TIES counterexample is genuinely informative.** TIES has near-zero Frobenius distance from the average but introduces correlated errors through sign election. This is a real insight about the limitations of weight-space metrics and is well-explained.

### Weaknesses

5. **The NR > 1.5 threshold is fit to the same data that "validates" it.** All 7 zero-shot methods at both scales are used to both find and validate the threshold. There is no held-out test. The claim "correctly classifies all 7 zero-shot methods" is tautological -- the threshold was chosen to do exactly that. The paper should say "consistent with all observed data" rather than "correctly classifies." A true validation would require new methods or new data.

6. **TIES at NR=1.238 (N=2) is below the 1.5 threshold but is still "bad" (+6.85%).** The paper notes this but the rule would misclassify TIES at N=2 as "good." The proposed NR > 1.1 tightening is mentioned casually but not tested. This is an edge case that weakens the practical rule.

7. **The Spearman correlation for norm ratio has a directional problem.** At N=2, the quality-ranked list is: concat_cal (NR=1.000), simple_avg (NR=0.715), dare_p0.3 (NR=0.852), dare_p0.5 (NR=1.010). The NR ranking would be: simple_avg < dare_p0.3 < concat_cal < dare_p0.5, but the quality ranking puts concat_cal first. The high Spearman is driven by the catastrophic tail (dare_p0.9, dare_ties with NR > 2), not by fine discrimination in the good tier. The paper already states this ("IP correctly separates catastrophic tier from good tier") but the aggregate Spearman number oversells the predictive power within the good tier.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_information_preservation_predictor` correctly lists:
- KC1: ranking mismatches > 1 -- KILL (3 at N=2, 5 at N=5)
- KC2: Spearman < 0.8 -- PASS (0.929-0.952)
- Status: partial

This is consistent with the paper's reporting. The evidence entries accurately reflect the findings.

## Macro-Scale Risks (advisory)

1. **Non-orthogonal deltas at macro scale.** The MATH.md correctly identifies this: with correlated deltas, TIES's sign resolution could genuinely help, and simple average NR would increase (closer to 1.0). The NR > 1.5 threshold may need recalibration.

2. **NR as a failure detector depends on the absence of pathologically low NR methods.** If a macro merging method strongly attenuates norms (NR << 1), it would pass the NR < 1.5 check while potentially destroying information. The metric catches amplification but not attenuation.

3. **The 8-method universe is small.** At macro scale with more merging variants (SLERP, model soups, RegMean, Fisher-weighted merging), the correlation could weaken as more methods occupy the "good tier" with different NR profiles.

## Verdict

**PROCEED**

This is a well-executed diagnostic experiment with honest kill-criteria reporting. KC1 properly killed, KC2 properly passed. The practical finding (NR > 1.5 as failure detector) is useful, and the analysis of why IP fails at fine resolution (TIES correlated errors, concat+cal router noise) adds genuine understanding.

The issues identified are documentation-level, not mechanism-level:

1. **Disclose that concat+cal IP/NR values are hardcoded** (code lines 402-406) and report Spearman rho with and without concat+cal as a sensitivity check.
2. **Report the NR > 1.5 threshold as "consistent with observed data," not "correctly classifies."** It was fit and validated on the same data. Note the TIES edge case at N=2 (NR=1.238, still bad).
3. **Replace "unique lossless linear merge" with "minimum-MSE single-point summary"** in MATH.md line 20-21. The simple average is not lossless; it is the centroid.
4. **Add a note about Spearman confidence intervals** at n=8, even if approximate (Fisher z-transform gives ~[0.79, 0.99] for rho=0.952). The KC2 pass is real but not as definitive as the point estimate suggests.

None of these change the conclusion. The experiment advances the vision by providing a zero-cost diagnostic for filtering merging methods before evaluation -- directly useful in the contribution protocol. The "partial" status is appropriate and the findings are correctly recorded in FINDINGS.md and HYPOTHESES.yml.
