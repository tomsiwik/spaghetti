# Peer Review: Profiling Noise Quantification

## NotebookLM Findings

Skipped. The experiment is a targeted robustness check with straightforward statistics. The materials are self-contained and do not require external deep-dive tooling to assess rigor.

---

## Mathematical Soundness

### Binomial noise model (MATH.md Section 3.1): CORRECT

The derivation P(false dead | p_i) = (1-p_i)^N_prof is standard binomial survival probability. The numerical examples are verified:
- (0.999)^640 = 0.527 (correct to 3 sig figs)
- (0.995)^640 = 0.041 (correct)
- (0.990)^640 = 0.0016 (correct)

The critical insight -- only capsules with p_i < 0.005 are susceptible -- follows directly and is sound.

### Consensus correction (MATH.md Section 3.4): CORRECT

P(consensus false dead | p_i) = ((1-p_i)^N)^2 = (1-p_i)^(2N) is correct under the assumption that runs A and B are independent. The independence assumption is discussed in Section 9.1 and is reasonable given that 640 samples represent a minority of the validation set.

### Noise attribution logic (MATH.md Section 5.2): CORRECT BUT WITH A SUBTLETY

The paper defines noise_fraction = (DA_single - DA_consensus) / DA_single. This is a sensible definition. However, the interpretation of negative noise_fraction deserves scrutiny.

**The negative noise fraction (-6.2%) is mathematically expected and does NOT mean "noise underestimates revival."** Here is why:

When you take the consensus dead set (intersection of two dead sets), you produce a SMALLER dead set. Fewer capsules start as "dead," so even if the same absolute number revive, the D->A count could change in either direction because:
- Some capsules that were "dead in A but alive in B" at checkpoint S1 are removed from the dead set. If those capsules are alive at S2, that removes a D->A transition (noise reduction, the intended effect).
- BUT some capsules that were "dead in A but alive in B" at checkpoint S1 are removed from the dead set. If those capsules are DEAD at S2, this has no effect on D->A but reduces D->D, potentially changing the base from which revival rate is computed.
- Additionally, at checkpoint S2, the consensus mask also changes. A capsule dead in consensus at S1 might be alive in run A at S2 but dead in run B at S2. Under single-run-A tracking, this is D->A. Under consensus tracking, the capsule is alive at S2 in consensus (not dead in both), so it is STILL D->A.

The PAPER.md explanation (Section "Finding 3") attributes the negative noise fraction to "consensus shrinks the dead set...so the remaining dead capsules have a higher revival rate." This is an INCORRECT causal explanation. The negative noise fraction arises because consensus changes BOTH endpoints of the transition (the dead set at S1 AND the classification at S2), and these changes do not cancel in a simple direction. The numbers are correct; the narrative explanation is imprecise.

**Severity: LOW.** The conclusion (noise does not inflate revival) is still supported by the data. The explanation of WHY is hand-wavy but does not affect the validity of the kill-criterion analysis.

### Expected disagreement under null model (MATH.md Section 4): ACKNOWLEDGED AS INTRACTABLE

The paper correctly notes that computing expected disagreement requires knowing the distribution of p_i values, which is unknown. The empirical measurement sidesteps this. This is honest and appropriate. No hidden assumptions here.

### Kill criteria thresholds: REASONABLE

- 20% disagreement: a sensible cutoff. At 20%, one in five capsules would be unreliably classified.
- 50% noise D->A: if more than half of revivals are noise, the finding is substantially weakened.
- 5% corrected revival rate: below this, the practical implication ("prune after training") becomes marginal.

These are pre-registered (stated in MATH.md) and not tuned post-hoc. Good practice.

---

## Novelty Assessment

### Prior art

Dual-profiling / split-half reliability is a standard psychometric technique (Cronbach's alpha, split-half correlation). Applying it to neural network activation profiling is straightforward but appropriate. This is not a novel method -- it is a standard robustness check applied to a specific experimental claim.

### Delta over existing work

This is not intended to be novel research. It is a methodological control for Exp 18. Evaluated as such, it is well-designed and thorough.

---

## Experimental Design

### Does this test what it claims? YES, with minor caveats.

The core design -- profile the same checkpoint twice with different random seeds, measure disagreement -- directly tests the noise hypothesis. The consensus mask comparison directly quantifies how much of the D->A signal survives noise correction. This is exactly the right experiment.

### Controls: ADEQUATE

- Same training seeds as Exp 18 (42, 123, 7)
- Same profiling parameters (20 batches x 32)
- Same step counts and checkpoints
- The only difference is the dual profiling, which is the independent variable

### Potential weaknesses in the design:

**1. Validation set overlap between profiling runs.**

The paper acknowledges this in Limitations (Section "Micro-Scale Limitations" item 1) and in MATH.md Assumption 1. At micro scale, the validation set may be small enough that 640+640=1280 samples from a small pool have substantial overlap. If runs A and B share many samples, their disagreement is artificially LOW, making the profiling appear more reliable than it truly is.

This is a legitimate concern but is correctly identified as a micro-scale limitation. At macro scale with larger validation sets, this would be less of an issue. Within the micro/macro contract, this is acceptable.

**2. Only 2 profiling runs per checkpoint.**

Two runs establish a point estimate of disagreement. Ten runs would give a distribution. The paper acknowledges this (Limitation 2). For a micro experiment, 2 is sufficient to get directional evidence.

**3. Consensus cohort tracking compares different base populations.**

The single-run cohort tracks capsules dead at S=100 in run A. The consensus cohort tracks capsules dead at S=100 in BOTH runs. These are different sets (consensus is a strict subset of the single-run dead set). The comparison in the cohort table is therefore not apples-to-apples -- the consensus cohort excludes borderline capsules that were in the single-run cohort. Since borderline capsules are the ones most likely to show "revival" (they were never truly dead), removing them could INCREASE the apparent revival rate of the remaining cohort, which is exactly what is observed (28.2% vs 26.7%).

This actually STRENGTHENS the paper's conclusion: the consensus cohort (which excludes borderline capsules) shows equal or higher revival, meaning the revival signal comes from capsules that are reliably dead, not from borderline flickering.

**4. The experiment retrains from scratch for each seed rather than loading Exp 18 checkpoints.**

The paper uses the same seeds and training procedure as Exp 18, so the models should be statistically equivalent. But they are not guaranteed to be IDENTICAL (MLX evaluation order, floating point non-determinism). This means the death rates and transition counts may differ slightly from Exp 18's reported numbers. The paper reports single-run cohort revival of 26.7% vs Exp 18's 28.1% -- a 1.4pp difference that could be explained by this.

**Severity: NEGLIGIBLE.** The comparison is seed-matched, and the small differences are within the reported standard deviations.

### Hypothesis Graph Consistency

The experiment is listed as item 12 in VISION.md ("Profiling noise quantification (from Exp 18 review)"). It is not explicitly tracked as a node in HYPOTHESES.yml. This is a minor bookkeeping gap -- the experiment should be added to the hypothesis graph with its kill criteria and result.

The kill criteria in the code match those in MATH.md Section 7. The test file implements all three criteria correctly.

---

## Code Review

### `profiling_noise.py` (model class)

Trivial wrapper around ReLURouterGPT. No concerns. Correct to make this a no-op model class for lineage tracking.

### `test_profiling_noise.py` (experiment runner)

**Line 146:** Disagreement counting uses `if a != b` on boolean values. Correct.

**Lines 150-157:** Flickering count iterates over raw frequency values with `0 < fa_val < 0.05`. This threshold is hardcoded. The MATH.md discusses borderline capsules with p < 0.005 as the noise-susceptible population, but the code uses 0.05 (10x larger). This means the "flickering" count includes capsules that are reliably alive (p > 0.01) but have low frequency. This inflates the reported flickering percentage (22-39%) relative to the truly noise-susceptible population. However, the flickering count is a diagnostic metric, not used in any kill criterion, so this does not affect the conclusions.

**Lines 178-181:** Consensus mask construction uses `a and b` (logical AND). This correctly implements "dead only if dead in both runs" since True=dead, False=alive in the mask convention. Verified against `get_dead_mask` which returns True for dead capsules.

**Lines 349-359:** Noise fraction calculation. The denominator is `total_da_a_all` (total single-run D->A). If this is zero, the code handles it with a conditional. The sign convention is correct: positive noise_fraction means consensus has FEWER D->A (noise inflated them), negative means consensus has MORE D->A.

**Overall code quality:** Clean, well-commented, follows established patterns from prior experiments. No bugs identified that would affect results.

---

## Macro-Scale Risks (advisory)

1. **SiLU activation changes the entire framework.** With SiLU, there is no hard zero. The f=0 threshold becomes a magnitude threshold, introducing an arbitrary parameter. The dual-profiling methodology transfers, but the noise characteristics will be different (continuous activation magnitudes have different sampling properties than binary fire/not-fire). The paper correctly flags this.

2. **Larger validation sets reduce overlap concern.** At macro scale, 640 samples from a much larger validation set will have less overlap between runs A and B, potentially increasing measured disagreement. This could go either way: more distinct batches might reveal more true noise, or they might be more representative and thus agree more.

3. **The profiling budget (640 samples) may be insufficient at macro scale.** With larger models and more diverse data, the activation probability distribution may have a heavier tail of borderline capsules, requiring more samples for reliable classification.

---

## Verdict

**PROCEED**

### Justification

This is a well-designed, well-executed methodological control experiment that directly addresses a specific concern raised in Exp 18's adversarial review. The results are clear and unambiguous:

1. **Profiling disagreement is low (2.6-3.8%).** This is well below the 20% kill threshold and establishes that the existing profiling protocol is reliable.

2. **Noise does not inflate revival.** The noise fraction is -6.2% (negative), meaning consensus correction INCREASES reported revival. This definitively answers the adversarial concern.

3. **All three kill criteria pass with wide margins.** No threshold is close to being triggered.

4. **Exp 18 findings are reproduced within noise.** Cohort revival (26.7%/28.2% vs 28.1%), Jaccard (0.676/0.654 vs 0.669), and max revival rate (16.3%/17.4% vs 15.9%) all match Exp 18 within reported standard deviations.

### Minor issues (not blocking):

1. **HYPOTHESES.yml:** Add this experiment as a node with status=proven.
2. **Narrative imprecision:** The explanation of why noise fraction is negative (PAPER.md Finding 3) conflates two effects (smaller dead set and different consensus classifications at S2). The numbers are correct; the causal story could be tightened.
3. **Flickering threshold:** The 0.05 threshold in the code is 10x the theoretically noise-susceptible threshold of 0.005. The flickering count is a diagnostic, not a kill metric, so this does not affect conclusions, but the paper should note the discrepancy or use the theoretical threshold.

### Assessment of Exp 18 revival finding

This experiment provides strong evidence that Exp 18's revival finding is not an artifact of profiling noise. The "prune after training completes" recommendation from Exp 18 is validated. The profiling protocol (20 batches x 32 samples) is confirmed sufficient for reliable dead/alive classification at the binary f=0 threshold.
