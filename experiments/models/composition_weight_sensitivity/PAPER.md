# Composition Weight Sensitivity: Research Digest

## Hypothesis

Pre-merged SOLE quality degrades gracefully as expert count N grows
(dilution test). Per-expert marginal contribution remains detectable
above the noise floor up to N=100.

## What This Study Is

A systematic sweep of N = 2, 5, 10, 20, 50, 100 experts composed via
pre-merge, measuring:

1. **Aggregate quality**: composed model loss vs base model loss
2. **Per-expert marginal contribution**: delta loss from removing one
   expert (leave-one-out analysis)
3. **Noise floor**: random perturbation control with matched norm and
   symmetric LOO reweighting (20 samples per N value)
4. **Signal-to-noise ratio (SNR)**: whether expert signals are detectable

Two composition modes are compared:
- **SUM**: W + sum(dW_i) -- each expert contributes its full delta
- **AVG**: W + (1/N) * sum(dW_i) -- averaged, each expert at 1/N weight

Architecture: 4-layer SiLU MLP, d=64, d_ff=256, rank-8 LoRA, 100 Markov
chain domains, 3 seeds. Reuses model from lora_flow_comparison.

## Lineage in the Arena

```
oae_vs_lora_soups --> lora_flow_comparison --> composition_weight_sensitivity (this)
```

Spawned from lora_flow_comparison adversarial review, macro risk #3:
"Pre-merge dilutes each expert to 1/N strength. At N=500, each expert
is 0.2% of the weight space."

## Key References

| Paper | Relevance |
|-------|-----------|
| LoRA-Flow (Wang et al., 2024) | Pre-merge assumed, dilution concern motivates |
| Model Merging at Scale (2024) | Weight averaging dilution in broader context |
| TIES-Merging (Yadav et al., 2023) | Sign-conflict resolution for merging |
| AdapterSoup (Chronopoulou et al., 2023) | Weight averaging across adapters |

## Empirical Results

### Configuration

d=64, d_ff=256, r=8, L=4, V=32, 100 domains (Markov chains),
150 training steps/expert, 3 seeds. Noise floor: 20 random replacements
per N value (up from 3 in v1, per adversarial review fix #2).

### Expert Specialization (Micro-Scale Limitation)

Individual expert improvement over base: **0.00%** (< 5e-5%).
Mean |cos| between experts: **0.0022** (measured among 20 of 100 experts,
i.e. 190 of 4950 possible pairs; see "Orthogonality" note below).

This means expert deltas are negligibly small -- the quality comparison
operates at floating-point precision limits (~1e-6 to 1e-11). The
dilution test is structurally correct but numerically vacuous.

### Aggregate Quality: SUM vs AVG vs Base

| N | SUM gap (%) | AVG gap (%) |
|---|-------------|-------------|
| 2 | -5.0e-5 +/- 0.7e-5 | -2.5e-5 +/- 0.4e-5 |
| 5 | -3.4e-5 +/- 0.6e-5 | -6.7e-6 +/- 1.1e-6 |
| 10 | -6.6e-5 +/- 2.6e-5 | -6.6e-6 +/- 2.6e-6 |
| 20 | -8.4e-5 +/- 5.0e-5 | -4.2e-6 +/- 2.5e-6 |
| 50 | -1.4e-4 +/- 6.2e-5 | -2.8e-6 +/- 1.2e-6 |
| 100 | -2.1e-4 +/- 1.4e-4 | -2.1e-6 +/- 1.4e-6 |

**All gaps are negative** (composed model is marginally BETTER than base).
This is consistent with experts slightly regularizing the model, not
degrading it.

Key observation: **AVG gap is N-independent** (stays ~2-6e-6% regardless
of N). SUM gap grows linearly with N (total perturbation increases).

### Marginal Contribution vs Noise Floor

| N | Marginal (AVG) | Noise floor (20 samples) | SNR |
|---|----------------|--------------------------|-----|
| 2 | 6.2e-11 | 1.4e-7 | 0.00 |
| 5 | 8.3e-12 | 1.4e-8 | 0.00 |
| 10 | 1.4e-12 | 2.5e-9 | 0.00 |
| 20 | 1.2e-9 | 5.0e-10 | 2.2 |
| 50 | 4.2e-10 | 5.1e-11 | 13.6 |
| 100 | 3.9e-11 | 1.5e-11 | 16.0 |

SNR increases with N, going from <1 at small N to ~16 at N=100.

**Signal/noise symmetry (v2 fix)**: Both signal and noise measurements
now use the same LOO reweighting operation. The signal is
loss(compose_avg(N-1 real)) - loss(compose_avg(N real)). The noise is
loss(compose_avg(N-1 real)) - loss(compose_avg(N-1 real + 1 random)).
Both see the same 1/N -> 1/(N-1) reweighting when one expert is removed.
See MATH.md Section 2.6 for why this symmetry matters.

### SNR Scaling: Theoretical PREDICTION (Untested)

**IMPORTANT**: The SNR ~ O(N) scaling law presented in MATH.md Section
2.5 is a THEORETICAL PREDICTION derived from first-order Taylor analysis,
NOT a validated finding from this experiment. The micro-scale data cannot
validate it for two reasons:

1. **The power law exponent is +0.57, not -1.0.** The theory predicts
   marginal contribution scales as O(1/N) (exponent -1.0). The measured
   exponent is +0.57 (marginal contribution INCREASES with N). This
   directly contradicts the theoretical prediction and is the single
   most important discrepancy in this experiment.

2. **The zero-signal regime invalidates scaling analysis.** With marginal
   contributions of 1e-11 to 1e-9 and expert improvement of 0.00%, the
   measured values are floating-point noise, not dilution physics. The
   clean-looking SNR increase from 0 to 16 could equally reflect how
   float32 rounding errors scale with averaging operations.

The theory may well be correct at macro scale where expert deltas have
real magnitude. But the micro-scale evidence neither supports nor refutes
it. Any use of the SNR ~ O(N) scaling in downstream reasoning must treat
it as a prediction requiring macro validation.

### Power Law Fit

The fitted power law exponent for marginal contribution is **+0.57**
(positive -- increasing with N). The theoretical prediction is **-1.0**
(decreasing as 1/N). The +1.57 discrepancy is not a minor numerical
issue -- it represents a complete qualitative disagreement between theory
and measurement. The most likely explanation is that at expert
improvement of 0.00%, the marginal contributions are dominated by
floating-point arithmetic patterns rather than actual dilution physics.
This makes the power law fit meaningless as evidence for or against the
theoretical scaling.

## Kill Criteria Assessment

### K1: Per-expert signal drops below noise floor at N < 50

**Status: INCONCLUSIVE (not KILL)**

The K1 metric formally triggers (SNR < 1.0) at N = 2, 5, 10. However,
this is entirely an artifact of both signal and noise being at machine
epsilon (~1e-8 to 1e-12). The experts have zero specialization at micro
scale, so there is no signal to detect or dilute.

At N = 20-100, SNR is above 1.0 and increasing. The structural analysis
predicts SNR ~ O(N), meaning larger N improves detectability under
averaged composition. However, this prediction is untested (see
"SNR Scaling" section above).

**Honest assessment**: K1 cannot be meaningfully evaluated at micro scale.
The trigger is a false positive caused by floating-point arithmetic in
the zero-signal regime, not by dilution.

### K2: Pre-merge quality degrades >20% at N=100 vs N=10

**Status: PASS (trivially)**

Degradation: +0.000% (both are within 3e-6% of base).

This result is trivially true because expert deltas contribute 0.00%
improvement over base. When experts contribute nothing, adding or
removing them cannot degrade quality. You could compose N random
perturbations of matched (near-zero) norm and obtain the same result.

The PASS is valid in the narrow sense that the composition operation
itself does not introduce degradation. But it provides no evidence
about whether composition is stable when experts carry real signal.

### Overall: SUPPORTED (K2 only)

K1: cannot evaluate (false positive from zero-signal regime).
K2: PASS (trivially, due to zero expert specialization).

The experiment confirms that the composition operation itself is
numerically stable up to N=100. It cannot test whether meaningful expert
signals survive dilution, because no meaningful expert signals exist at
micro scale.

## Structural Findings (Valid Independent of Expert Specialization)

Despite the vacuous quality comparison, several structural findings hold:

1. **AVG composition is N-independent**: quality gap stays constant as
   N grows from 2 to 100. The 1/N scaling in the weight does NOT cause
   degradation because each expert's contribution is diluted equally.

2. **SUM composition grows linearly but stays stable**: total perturbation
   increases as O(N * ||dW||) but remains bounded and does not diverge.

3. **SNR increases with N under AVG (PREDICTION, not finding)**: noise
   floor drops as O(1/N^2) while signal drops as O(1/N), predicting
   SNR ~ O(N). This is a theoretical prediction from first-order Taylor
   analysis. The micro-scale measurements are consistent with increasing
   SNR but cannot validate the O(N) scaling law because the power law
   exponent (+0.57) contradicts the predicted signal scaling (-1.0).

4. **Orthogonality measured at d=64**: mean |cos| = 0.0022, measured
   among 20 of 100 experts (190 pairwise cosines out of 4950 possible
   at N=100). This is consistent with structural orthogonality at d=64
   but does not confirm orthogonality across all 100 experts.
   Structural orthogonality is a property of the training procedure
   (see structural_orthogonality_proof), so the subsample is
   directionally informative.

5. **Both SUM and AVG produce identical quality**: at micro scale with
   near-zero deltas, W + sum(dW_i) and W + (1/N)*sum(dW_i) are
   indistinguishable.

## Theoretical Prediction for Macro Scale

At macro scale where experts have real specialization (42% PPL improvement
from pilot-50), the per-expert marginal contribution would be:

- N=10: each expert contributes ~4.2% improvement (10% weight per expert)
- N=50: each expert contributes ~0.84% improvement (2% weight per expert)
- N=100: each expert contributes ~0.42% improvement (1% weight per expert)
- N=500: each expert contributes ~0.084% improvement (0.2% weight per expert)

Under structural orthogonality, the SNR ~ O(N) PREDICTION (not yet
validated) suggests that even at N=500, the marginal contribution should
be detectable. The noise floor drops as 1/N^2 from the averaging.

The risk is not dilution (which is predicted to be graceful) but whether
0.084% per-expert is sufficient for shadow scoring to discriminate expert
quality. At the pilot-50 level (0.811 correlation between answer-conditioned
PPL and accuracy), a 0.084% PPL difference may be within the scoring
noise floor.

## Micro-Scale Limitations

1. **Zero expert specialization**: experts do not specialize at d=64
   with Markov chain data. This makes quality comparisons vacuous and
   renders K2 trivially passing.

2. **Floating-point precision**: all measurements are at 1e-6 to 1e-12
   scale, making numerical artifacts indistinguishable from real effects.

3. **Small d limits orthogonality margin**: d=64 with r=8 gives
   sqrt(r/d) = 0.354 bound. At d=4096 with r=16, bound drops to 0.063
   (5.6x lower).

4. **Transition matrix domains are similar**: Markov chains with
   overlapping vocabulary regions do not produce the strong domain
   specialization seen with real text data.

5. **K1 false positive**: the "kill" at small N is a measurement artifact,
   not a genuine dilution failure.

6. **Power law exponent contradicts theory**: the measured +0.57 exponent
   versus the predicted -1.0 means the SNR ~ O(N) scaling law is
   entirely untested at this scale.

## What Would Kill This

### At Micro Scale (already tested)
- K2 PASS: no quality degradation at N=100 vs N=10 (trivially, due to
  zero expert specialization)

### At Macro Scale (not yet tested)
- Per-expert marginal contribution drops below shadow scoring noise floor
  at N < 50 (needed for clone-and-compete evolution)
- Pre-merge quality degrades >20% at N=100 vs N=10 on held-out MMLU
- Quality gap between AVG and SUM composition becomes significant at
  N > 50 (would motivate weighted/selective composition)

## Conclusion

Pre-merge composition via averaging is numerically stable up to N=100
at micro scale. Quality degradation is zero, but this is trivially true
because expert deltas contribute 0.00% improvement (there is nothing
to dilute).

The SNR ~ O(N) scaling prediction remains theoretically motivated but
empirically untested. The micro-scale power law exponent (+0.57)
directly contradicts the predicted -1.0, and this discrepancy cannot be
resolved without macro-scale experiments where experts carry real signal.

**Recommended status**: SUPPORTED (K2 only). K2 passes trivially because
expert deltas contribute 0.00% improvement. K1 is inconclusive. The
SNR ~ O(N) scaling is a prediction, not a finding. Macro validation is
required to test whether composition stability holds when experts carry
meaningful signal.

## Revision History

- **v1** (2026-03-15): Initial experiment with 3 noise samples.
- **v2** (2026-03-15): Adversarial review revision. 6 fixes:
  (1) SNR~O(N) relabeled as prediction, not finding; +0.57 vs -1.0
  contradiction prominently discussed.
  (2) Noise floor samples increased from 3 to 20.
  (3) Signal/noise asymmetry fixed: noise experiment now uses symmetric
  LOO reweighting matching the signal experiment.
  (4) Orthogonality scope qualified (20 of 100 experts measured).
  (5) Status downgraded to "supported (K2 only)".
  (6) FINDINGS.md caveat updated to note trivial K2 PASS.

## Artifacts

- `micro/models/composition_weight_sensitivity/PAPER.md` -- this document
- `micro/models/composition_weight_sensitivity/MATH.md` -- formal analysis
- `micro/models/composition_weight_sensitivity/composition_weight_sensitivity.py` -- experiment
- `micro/models/composition_weight_sensitivity/results.json` -- raw results
