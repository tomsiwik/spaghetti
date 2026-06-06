# Peer Review: adaptive_rank_snr_fallback

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that code+math inspection suffices. The core mechanism is a two-line branching heuristic on top of existing energy rank computations.

## Mathematical Soundness

### What holds

1. **Weyl's perturbation theorem application (Section 2.2).** The claim that noise inflates spectral tails is standard and correct. At low SNR, the 99% energy threshold must capture noise dimensions that the 95% threshold can safely ignore. The intuition is sound.

2. **The ratio R = r_99/r_95 as a noise diagnostic (Section 2.3).** The argument that R separates clean from noisy spectra is empirically validated. The ratio is dimensionless and has clear physical interpretation. The empirical table (SNR=5: R=3.8-13.0; SNR>=10: R=1.3-1.5) shows a wide gap between regimes, making the threshold choice insensitive to the exact value.

3. **Monotonicity in T (Section 3.2).** Correct. T=infinity gives r_99; T=1 gives r_95. The sweep confirms this.

4. **Computational complexity (Section 6).** The O(d) overhead claim is correct -- one additional cumulative sum scan on top of the SVD that dominates anyway.

5. **Worked example (Section 7).** I verified the arithmetic. The numbers are internally consistent with the domain generation code (signal norm ~3.0, noise norm ~0.6 at SNR=5).

### What does not hold or is imprecise

1. **The noise inflation formula in Section 2.2 is hand-waving.** The expression:
   ```
   r_99(Delta) ~ r_99(S) + (d - r_99(S)) * (1 - eta^2 / (eta^2 + 1)) * correction
   ```
   is labeled as an approximation but the "correction" factor is undefined. This is not a derivation -- it is a qualitative sketch dressed up as a formula. The Weyl bound itself is tight, but the energy rank inflation formula is ad hoc. **Non-blocking:** the experiment does not depend on this formula; the empirical results stand on their own.

2. **NaN in results at d=256, SNR=5.** The r_99 predictor produces `spearman_rho: NaN` at this condition. Inspecting the code: `stats.spearmanr` returns NaN when one input has zero variance. This means all 15 domains have the same snapped r_99 prediction at d=256, SNR=5. The within_2x metric (26.7%) is still valid, but the NaN indicates that r_99 is so badly inflated that it predicts the same (maximum) rank for every domain. **Non-blocking** for the compound heuristic, but PAPER.md reports Spearman rho values for r_99 without noting this NaN. The MATH.md Section 5 table reports +60.0pp improvement for d=256 SNR=5 which is calculated from within_2x (80% vs 26.7% = 53.3pp from results, not 60.0pp). Let me recheck: MATH.md says d=256 SNR=5 improvement is +60.0pp (86.7% - 26.7%), which matches the results.json (compound_r95_t2.0: 0.867, r_99: 0.267). That checks out.

3. **K1 criterion is weak.** The kill criterion reads: "fallback heuristic does not improve over r_99 alone at SNR<=10." The K1_improvements array in results is `[0.0, 0.533, 0.0, 0.600, 0.0, 0.267]`. Three of the six low-SNR conditions show exactly 0.0pp improvement (all the SNR=10 conditions). K1 passes because the SNR=5 conditions show massive improvement and the mean is positive (+23.3pp). However, this means the fallback contributes nothing at SNR=10. The claimed "+23.3pp improvement at SNR<=10" is accurate but slightly misleading -- it is entirely driven by SNR=5. At SNR=10, the compound heuristic is bit-identical to r_99 because the fallback never triggers. This is honestly documented in the paper ("0.0pp delta, fallback never triggers") but the K1 criterion lumps SNR=5 and SNR=10 together. A fairer statement: "+23.3pp mean improvement at SNR<=10, driven entirely by SNR=5; zero improvement at SNR=10 (fallback inactive)."

4. **PAPER.md claims "95% within-2x accuracy across 12 conditions."** The actual number from results.json is 0.9500 mean, which rounds to 95.0%. However, this is an average of 12 per-condition accuracies, each computed from only 15 domains. With 15 domains per condition, the within_2x metric has a granularity of 1/15 = 6.67 percentage points. A single domain flipping from within-2x to outside-2x changes the condition accuracy by 6.67pp. The 95.0% mean has no confidence interval reported. **Non-blocking** but the precision ("95.0%") overstates the statistical resolution.

### Hidden assumptions

1. **The "within-2x" metric is generous.** A predicted rank of 32 for a true optimal rank of 16 (or vice versa) counts as success. For SOLE capacity planning, a 2x overprediction means 2x more parameters per expert (wasted budget), and a 2x underprediction means losing up to 75% of signal energy (Eckart-Young). The parent adversarial review already flagged this but it is worth repeating: the accuracy at a tighter tolerance (1.5x) would be informative.

2. **Assumption 5 is the real risk.** "Real LoRA training has effective SNR in the range tested (5-100)." If real training always produces SNR >= 10 (likely for well-tuned training with adequate data), the entire fallback mechanism is correct but vacuous. The paper honestly acknowledges this.

## Novelty Assessment

**This is engineering, not research.** The compound heuristic is a two-line if/else statement on top of standard spectral energy thresholds. The r_99/r_95 ratio as a noise diagnostic is an obvious consequence of energy rank definitions -- it is not a new theoretical contribution.

**Prior art:** Adaptive rank selection has been studied in AdaLoRA (Zhang et al., 2023), which uses SVD importance scoring for per-layer rank budgets. The specific r_99/r_95 ratio heuristic appears to be novel as a diagnostic, but the concept of using multiple energy thresholds for noise detection is standard in signal processing (Gavish & Donoho, 2014, optimal hard thresholding for singular values is a more principled approach to the same problem).

**Delta over parent:** The parent experiment already worked at SNR >= 10. This experiment patches the SNR=5 failure mode with a simple heuristic. The contribution is a practical fix, not a conceptual advance. That said, within the micro-experiment framework, this is an appropriate response to an adversarial review finding.

## Experimental Design

### Does it test the hypothesis?

Yes, directly. The hypothesis is "compound heuristic beats r_99 at low SNR without regressing at high SNR." The experiment tests exactly this across 12 conditions (3 dimensions x 4 SNR levels).

### Controls

- **Null baseline** (always predict rank 16): present and appropriate.
- **Unconditional r_99 and r_95**: both present as baselines.
- **Effective rank alternative**: tested as alternative fallback.
- **Threshold sweep**: 4 values tested (1.5, 2.0, 2.5, 3.0).

Controls are adequate for the claim being made.

### Could a simpler mechanism explain the result?

Yes. At SNR=5, unconditional r_95 achieves the same within-2x accuracy as compound_r95_t2.0 in all three SNR=5 conditions (80%, 86.7%, 86.7%). The compound heuristic's value is entirely in the "no regression at high SNR" property -- it selects r_99 when r_99 is better and r_95 when r_95 is better. This is by design and clearly stated.

The simplest alternative would be: "always use r_95." This achieves 82.2% mean vs compound's 95.0%. The 12.8pp gap comes from SNR >= 10 conditions where r_99 dominates r_95. The compound heuristic correctly identifies when to use each.

### Concerns

1. **Threshold T=2.0 and T=2.5 give identical results.** This means the experiment has not identified the precise boundary -- it only shows that any threshold in [2.0, 2.5] works. The paper acknowledges this ("the trigger gap between SNR=5 and SNR=10 is wide enough that both thresholds fall in the same bin"). This is actually a strength: the heuristic is robust to threshold choice. But it also means the threshold was not stress-tested in the transition zone (e.g., SNR=7 or SNR=8, where R values might fall near 2.0).

2. **15 domains per condition.** With 8 exact-rank + 7 spectral-decay = 15 domains, each condition has only 15 data points for within_2x computation. The per-condition accuracy has 6.67pp granularity. This limits the precision of the comparison.

3. **Synthetic spectral structure only.** Both exact-rank and geometric-decay are idealized spectra. Real LoRA deltas may have more complex spectral structures (e.g., power-law tails, multi-scale structure). The heuristic's behavior on such spectra is untested.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry correctly:
- Depends on exp_adaptive_rank_selection (the parent)
- Lists kill criteria matching the code's K1 and K2
- Status is "proven" with appropriate evidence

One inconsistency: the K1 kill criterion in HYPOTHESES.yml reads "r_99 if r_99 <= 2*r_95 else r_95" which is the condition for the fallback, not the kill criterion itself. The actual kill criterion (K1 does not improve over r_99 at SNR<=10) is also stated. This is a cosmetic issue.

## Macro-Scale Risks (advisory)

1. **SNR mapping is unknown.** The most likely outcome at macro scale is that well-tuned LoRA training produces effective SNR >> 10 for most domains, making the fallback unnecessary (but harmless). Testing this requires measuring r_99/r_95 ratios on real trained LoRA deltas from the pilot 50 experts.

2. **Real spectra may trigger false fallbacks.** If a real LoRA delta has legitimate signal spread across many dimensions (e.g., a highly complex domain), the r_99/r_95 ratio could exceed 2.0 even without noise, causing the heuristic to underpredict rank. This would lose signal.

3. **Per-layer vs per-adapter.** The heuristic is applied per-adapter (one Delta per domain). In practice, each LoRA has per-layer A and B matrices with different spectral properties. The heuristic should be applied per-layer, but the experiment tests only a single matrix per domain.

4. **The Gavish-Donoho optimal hard threshold** (2014) provides a principled, non-heuristic approach to separating signal from noise singular values when the noise level is known or estimable. At macro scale, this would be a more theoretically grounded alternative to the r_99/r_95 ratio heuristic.

## Verdict

**PROCEED**

The experiment does what it claims: it patches a known failure mode (r_99 at low SNR) with a simple, well-characterized heuristic that never regresses. The math is sound (modulo the hand-wavy inflation formula, which is cosmetic). The experimental design is clean with appropriate controls. The claims match the evidence. The limitations are honestly documented.

This is a minor engineering contribution, not a breakthrough -- but that is appropriate for a follow-up micro experiment fixing a specific adversarial-review finding. The compound heuristic is the correct recommendation for SOLE's automated rank selection pipeline.

**Non-blocking issues:**

1. The noise inflation formula in MATH.md Section 2.2 (the one with "correction") should be removed or explicitly labeled as "qualitative intuition, not a derivation." It adds nothing beyond the Weyl bound statement.

2. PAPER.md should note the NaN Spearman rho for r_99 at d=256 SNR=5 (all predictions identical due to extreme inflation).

3. The "mean +23.3pp improvement at SNR<=10" should be qualified as "driven entirely by SNR=5; zero improvement at SNR=10 (fallback inactive)." This is already implicit in the per-condition table but the summary headline could mislead.

4. Consider testing one intermediate SNR value (e.g., SNR=7) to validate that the R=2.0 threshold boundary is correct in the transition zone. This is optional given that T=2.0 and T=2.5 give identical results, suggesting robustness.
