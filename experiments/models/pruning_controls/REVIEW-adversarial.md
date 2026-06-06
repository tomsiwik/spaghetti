# Peer Review: Pruning Controls (Experiment 10)

## NotebookLM Findings

Skipped (no NotebookLM session created; manual deep review performed instead with full code and mathematical verification).

## Mathematical Soundness

### 1. Decomposition Inconsistency (Moderate -- Presentation Error)

MATH.md Section 3.3 defines a three-way additive decomposition:

```
delta_composed = delta_training + delta_domain + delta_shift
```

The code (`test_pruning_controls.py` line 408) computes:

```python
delta_shift = delta_composed - delta_training
```

This is a TWO-way decomposition: delta_shift absorbs both delta_domain and the true distribution-shift component. The code never subtracts delta_domain from delta_shift. This means:

- Code's delta_shift = MATH.md's delta_domain + MATH.md's delta_shift
- delta_shift_reported (7.7%) = delta_domain (3.0%) + true_shift (~4.7%)

The PAPER.md decomposition table (lines 114-119) is misleading. It presents four rows:

| Component | Rate | Fraction |
|-----------|------|----------|
| delta_training | 54.3% | 87.4% |
| delta_shift | 7.7% | 12.4% |
| delta_domain | 3.0% | 4.8% |
| delta_composed | 62.1% | 100% |

The "Fraction of Composed" column sums to 87.4 + 12.4 + 4.8 = 104.6%, not 100%. The table visually implies the first three rows are additive components summing to the fourth, but delta_domain is a separate measurement that OVERLAPS with delta_shift. The actual additive partition is only two-way: delta_training (54.3%) + delta_shift (7.7%) = 62.0% (rounding accounts for the 0.1% gap to 62.1%).

**Fix required**: Either (a) make the three-way decomposition in the code by computing `delta_shift = delta_composed - delta_training - delta_domain`, or (b) remove delta_domain from the decomposition table and present it as a separate diagnostic. Option (b) is more honest since delta_domain as defined measures a property of the SINGLE-domain model, not a property of the composed model.

### 2. delta_domain Definition Problem (Minor -- Conceptual)

MATH.md Section 3.4 defines delta_domain as capsules that are "alive on own domain, dead on cross domain" in the single-domain model:

```
delta_domain ~ (1/P) * sum_{i=1}^P 1{f_i^{single,A}(D_A) > 0 AND f_i^{single,A}(D_B) = 0}
```

But this measures a property of the PRE-composition model. After composition, the hidden states change (attention processes both pools' outputs). A capsule that is alive-own/dead-cross in the single-domain model might become alive or dead on BOTH domains in the composed model due to distribution shift. So delta_domain estimated from single-domain profiling is only an approximation of the true domain-mismatch contribution.

The paper acknowledges this implicitly by treating delta_shift as a residual that captures "everything else including distribution shift." But the conceptual gap between "measured pre-composition" and "contributing to composed death" is not made explicit.

### 3. Derivations That Hold

- The exact zero-change theorem for dead capsule pruning (from Exp 9) is correct and not re-derived here. The pruning_controls experiment correctly reuses the same pruning machinery.
- The random pruning error analysis (MATH.md Section 4) is directionally correct: E[n_alive_pruned] = r * P_total * (1 - delta_composed) = 0.57 * 256 * 0.43 ~ 63. This is a valid expectation calculation.
- The independence assumption for error bound (Section 4.2) is standard but noted as an assumption. Capsule outputs are correlated through shared input x, so the actual variance may be higher or lower depending on correlation structure.

### 4. Numerical Consistency Check

- Joint model: 0.5251 loss. Concat: 0.5690. (0.5690 - 0.5251)/0.5251 = 8.36% -- PAPER says +8.4%. Consistent.
- Targeted prune: 0.5690 (identical to concat, correct by zero-change theorem). PAPER says +0.0%. Consistent.
- Random prune: 0.5524. (0.5524 - 0.5690)/0.5690 = -2.92%. PAPER says -2.9%. Consistent.
- Targeted+cal: 0.5223. (0.5223 - 0.5690)/0.5690 = -8.21%. PAPER says -8.2%. Consistent.
- Random+cal: 0.5262. (0.5262 - 0.5690)/0.5690 = -7.52%. PAPER says -7.5%. Consistent.
- Random+cal vs targeted+cal: (0.5262 - 0.5223)/0.5223 = +0.75%. PAPER says +0.8%. Consistent (rounding).
- delta_training: 54.3%, delta_composed: 62.1%, difference: 7.8%. PAPER says 7.7%. Rounding difference, acceptable.
- Kill criterion 1: 54.3% > 45%: KILL. Correct.
- Kill criterion 2: random vs targeted = -2.9% (random BETTER). |-2.9%| > 2%: PASS. BUT: this criterion is checking whether profiling is unnecessary. The original intent was that random-within-2%-of-targeted means profiling is unnecessary. Random being 2.9% BETTER than targeted means the criterion direction is confusing. The code checks `abs(random_vs_targeted) < 2.0`, which would also flag if random were 2.9% WORSE. The pass/kill semantics are inverted from what the hypothesis expected. See "Experimental Design" below.
- Kill criterion 3: 7.7% < 10%: KILL. Correct.

## Novelty Assessment

### Prior Art

The core finding -- that ~50% of ReLU neurons in trained transformers are dead -- is well-documented:

1. **Li et al. (2023), "Lazy Neuron Phenomenon"**: Reports exactly this ~50% natural sparsity. The paper cites this appropriately.
2. **Lottery Ticket Hypothesis (Frankle & Carlin, 2019)**: Random pruning preserving quality is a known phenomenon in overparameterized models. The regularization-through-pruning effect is also documented.
3. **Dropout literature (Srivastava et al., 2014)**: Random pruning at inference time acting as regularization is structurally similar to test-time dropout. Cited.

### Delta Over Existing Work

The novel contribution is NARROW but REAL: this is the first measurement (at any scale) of the death-rate decomposition for composed models specifically. Knowing that 87% of dead capsules in composed models were already dead pre-composition is a useful empirical finding for the composition protocol. No prior work measures this because model composition by weight concatenation is itself novel in this project's context.

The random pruning baseline is not novel in isolation but is a necessary control for the dead-capsule-pruning claim from Exp 9.

### Assessment

This is a CONTROLS experiment, not a novelty-claiming experiment. Its value is in validating or falsifying assumptions from Exp 9. It succeeds at this purpose: it falsifies the composition-induced-death hypothesis and validates that targeted profiling has value (with calibration). Novelty is not the relevant criterion here.

## Experimental Design

### 1. Kill Criterion 2 Semantics Are Inverted (Moderate Issue)

The hypothesis stated: "targeted dead-capsule identification is necessary (random pruning at the same rate should degrade quality significantly)."

Kill criterion 2: "Random within 2% of targeted" with the expectation that random would be WORSE. The measured result: random is 2.9% BETTER than targeted (not worse).

The code checks `abs(random_vs_targeted) < 2.0` and marks this as PASS because the absolute difference exceeds 2%. But the DIRECTION is opposite to the hypothesis. The experiment predicted random pruning would degrade quality; instead it improved quality. The PAPER correctly explains this (implicit regularization from removing alive capsules in an overparameterized model), but the kill criterion as written does not capture this inversion.

The more informative comparison is with calibration: targeted+cal (-8.2%) vs random+cal (-7.5%), where targeted wins by 0.7pp. This is within noise (0.7pp on losses with std of 0.0093 and 0.0078). Effect size = (0.5262 - 0.5223) / sqrt((0.0093^2 + 0.0078^2)/2) = 0.0039 / 0.0086 = 0.45 standard deviations. Not statistically significant at p < 0.05 with n=3 seeds vs n=15 draws.

**The honest conclusion**: there is no statistically significant evidence that targeted profiling outperforms random pruning, even with calibration. The paper's claim that "profiling provides value for calibration quality" is directionally plausible but not supported at the measured significance level.

### 2. Random Pruning Variance Comparison Is Misleading (Minor)

PAPER.md Finding 3 states: "random pruning is UNRELIABLE (std=0.0217 vs 0.0349 for targeted)." But targeted pruning has HIGHER std (0.0349) than random (0.0217). The std for targeted comes from seed-to-seed variation (3 values), while the std for random comes from 15 values (5 draws x 3 seeds). These are measuring different things:

- Targeted std (0.0349): variance across seeds in the UNDERLYING MODEL, not in the pruning itself (targeted pruning is deterministic given the model).
- Random std (0.0217): variance across random draws AND seeds combined.

Comparing these two std values is apples-to-oranges. The relevant comparison would be: within a single seed, what is the std of random pruning across its 5 draws? That number is not reported but would isolate the pruning-decision variance from the model-training variance.

### 3. The Experiment Correctly Tests Its Stated Hypotheses

Despite the issues above, the experiment IS well-designed for its primary purpose:

- Phase 1 (pre-composition profiling) directly addresses the Exp 9 adversarial review's concern about missing the single-domain death rate baseline. The measurement is clean: same profiling procedure, same data, same architecture, before vs after composition.
- Phase 2 (random pruning) directly addresses the missing random baseline. The protocol (5 draws per seed, 3 seeds, same prune rate) is adequate for a micro-scale experiment.
- The controls (joint training, weight averaging, targeted+calibration) are appropriate.

### 4. Could a Simpler Mechanism Explain the Results?

Yes, and the paper acknowledges this: the high single-domain death rate (54.3%) is entirely consistent with general ReLU dying neuron dynamics under short training (200 steps). The Li et al. "Lazy Neuron" reference at ~50% is the complete explanation. No composition-specific mechanism is needed to explain the observed death rates. This is actually the paper's FINDING, not a critique -- it correctly identifies the simpler explanation.

### 5. Profiling Data Concern

Both single-domain and composed models are profiled on the same validation sets (20 batches x 32 samples = 640 samples). This is adequate for binary dead/alive classification (a capsule that fires on even 1 of 640 samples is "alive"). The concern would be if a capsule fires on 0.1% of the true distribution but 0/640 profiling samples -- this would misclassify it as dead. At P=128 capsules x 4 layers, with 640 samples, the false-dead rate for a capsule with true frequency 0.001 is (1 - 0.001)^640 = 0.527, meaning ~53% chance of misclassifying a very-rare-firing capsule. However, since the paper uses tau=0.0 (strict dead), and the main finding is about the RATE of death (54% vs 62%), not individual capsule identity, this sampling limitation affects both single-domain and composed measurements equally and does not bias the decomposition.

## Macro-Scale Risks (advisory)

### 1. Training Duration Confound

The 54.3% single-domain death rate is measured after only 200 fine-tuning steps. At macro scale (100K+ steps), the death rate could be very different:
- **Lower** if longer training allows capsules to find useful directions (recovering from early dying ReLU)
- **Higher** if dying ReLU accumulates monotonically

If macro single-domain death drops to <20%, the entire decomposition reverses and composition becomes the dominant death mechanism. This is the paper's most important macro risk and it is correctly identified in the Limitations section.

### 2. Domain Similarity

a-m vs n-z character names share the same character distribution (all 26 letters). With truly distinct domains (Python vs JavaScript at macro scale), cross-domain death (currently 3.0%) could be much higher. This could shift the decomposition significantly.

### 3. SiLU Activation

Qwen (the target macro model) uses SiLU, not ReLU. SiLU has no hard zero -- neurons are never strictly "dead." The entire dead-capsule-pruning framework requires redefinition for SiLU (magnitude thresholds instead of zero thresholds). The paper's VISION.md already notes this as Exp 15. This is not a critique of the micro experiment (which uses ReLU by design) but a reminder that the technique does not transfer trivially.

### 4. Statistical Power at Scale

The +0.8% advantage of targeted+cal over random+cal is not statistically significant at micro scale. At macro scale with larger models and more evaluation data, this effect (if real) should become detectable. If it remains within noise at macro scale, the profiling step can be skipped in favor of random pruning (which is computationally cheaper -- no forward pass needed to identify dead capsules).

## Additional Issues

### 1. Decomposition Table Presentation (Fix Required)

As detailed in Mathematical Soundness section 1, the decomposition table in PAPER.md has delta_domain listed alongside delta_training and delta_shift as if it were an additive component. The "Fraction of Composed" column sums to 104.6%, not 100%. This will confuse any reader.

### 2. Missing alive_own_dead_cross in aggregate stats

The `profile_single_domain` function (pruning_controls.py line 102-114) computes `alive_own_dead_cross` per layer but the aggregate dict (line 108-113) only includes `pct_dead_own`, `pct_dead_cross`, and `pct_dead_both`. The aggregate `alive_own_dead_cross` is computed later in the test script by summing per-layer counts. This works but the aggregate stat should be added to the `profile_single_domain` return value for consistency.

### 3. The "87% of dead capsules were already dead" Claim

This is the key headline number. Let me verify: 54.3 / 62.1 = 87.4%. This is the fraction of COMPOSED dead rate explained by TRAINING death. But this implicitly assumes all training-dead capsules in the single-domain model are also dead in the composed model. Is this true?

The experiment does NOT verify this assumption directly. It compares RATES across different models (single-domain vs composed) but does not track INDIVIDUAL capsules. A capsule dead in the single-domain model COULD come alive in the composed model (different hidden-state distribution) while a different capsule that was alive becomes dead. The aggregate rates could match even if the individual capsule identities differ.

To make the "87% were already dead" claim precisely, you would need to track capsule indices: profile capsule i in the single-domain model AND in the composed model (where it occupies the same index in the first half of the concatenated pool), and check if dead_single[i] implies dead_composed[i]. The current experiment does not do this -- it only compares aggregate rates.

The 54.3% single-domain rate does establish that training produces roughly the same AMOUNT of death as observed in composed models, which is the directional finding the experiment needs. But the "87% of dead capsules were already dead before composition" phrasing implies identity-level tracking that was not performed.

## Verdict

**PROCEED** (with minor revisions)

The experiment achieves its primary purpose: it provides the two missing controls from the Exp 9 adversarial review and correctly identifies that dead capsule pruning is a general ReLU phenomenon, not composition-specific. The kill criteria are honestly applied (2 of 3 triggered) and the implications are sound. The finding that pruning is general rather than composition-specific is a STRONGER result than the original hypothesis -- it means the technique applies universally.

### Required Revisions

1. **Fix the decomposition table in PAPER.md** (lines 114-119). Either remove delta_domain from the table and present it as a separate diagnostic below, or recompute delta_shift as `delta_composed - delta_training - delta_domain` in the code (which would make delta_shift = 62.1 - 54.3 - 3.0 = 4.8%, and the three components would sum correctly). The current presentation where "Fraction of Composed" sums to 104.6% is incorrect.

2. **Restate the "87% were already dead" claim** to clarify it is an aggregate-rate comparison, not identity-level tracking. Suggested wording: "The single-domain death rate (54.3%) accounts for 87% of the composed death rate (62.1%), indicating that training-induced ReLU death is the dominant mechanism." This avoids implying that specific capsules were tracked.

3. **Report within-seed random pruning variance** separately from cross-seed variance. The current std comparison (0.0217 for random vs 0.0349 for targeted) conflates different sources of variance and should not be used to argue random pruning is "unreliable."

4. **Acknowledge that targeted vs random with calibration is not statistically significant.** The 0.7-0.8pp advantage (effect size ~0.45 SD) does not reach significance with n=3/15. Either collect more seeds or state "directional advantage, not statistically confirmed."

### Non-blocking Notes

- Kill criterion 2's semantics should be clarified for future reference: the hypothesis predicted random would be WORSE, but random was BETTER. The criterion as coded (absolute difference) catches this but for the wrong reason.
- The `alive_own_dead_cross` aggregate should be added to the `profile_single_domain` return dict for API consistency.
- MATH.md Section 8 (worked example) uses different numbers than the actual results, which is fine for illustration but could note the discrepancy ("illustrative values; see Section 5 for actual measurements").
