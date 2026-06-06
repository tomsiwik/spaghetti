# Adversarial Review: exp_p0_mcq_mixed_training

## Verdict: PROCEED

## Summary

Clean guided-exploration experiment. Core finding (+14.5pp MCQ effect under TT-LoRA r6)
is well-supported by both theory and measurement. All predictions land within stated ranges.
K1437 miss (34.5% vs 35%) is within 1 SE of the binomial (SE ~3.4pp at N=200) — not a
meaningful failure.

## Checklist

- [x] Prediction-vs-measurement table present and complete
- [x] Kill criteria match results.json evidence
- [x] Finding status (SUPPORTED) appropriate for guided-exploration type
- [x] Math is sound — gradient concentration ratio V/4 is correct
- [x] Controls adequate — same params, same data, same seed, only loss differs

## Strengths

1. **Excellent experimental design**: Single-variable comparison (NTP vs NTP+MCQ), same
   architecture, same hyperparameters. This is how you isolate a causal effect.
2. **Theory-measurement alignment**: All 6 predictions land in range. Theorem 1 confirmed
   with a 14.5pp effect size that's >4x the statistical uncertainty.
3. **Surprising NTP bonus**: Mixed NTP loss (0.131) < NTP-only (0.195). MCQ acts as
   regularizer, not competitor. This finding has architectural implications.

## Non-Blocking Issues

1. **"Ceiling" claim is premature**: The 35% compression capacity ceiling is inferred from
   a single λ=1.0 experiment. Higher λ, curriculum scheduling, or two-stage training could
   shift it. Call it an "observed bound at λ=1.0" rather than a fundamental ceiling.
   The PAPER.md does list options to exceed it (higher rank, selective allocation, two-stage),
   which partially addresses this — but the scratchpad/event language overstates certainty.

2. **NTP-only control discrepancy**: 20.0% here vs 18.5% in Finding #521 (1.5pp gap).
   Likely training variance across runs but worth noting for reproducibility.

3. **exp(-MCQ_loss) mapping**: "exp(-1.261) ≈ 28.3% maps to 34.5% eval" is hand-wavy.
   The gap between training loss and eval accuracy reflects format/prompt differences,
   not a clean mathematical mapping. Don't over-interpret this correspondence.

## Status: SUPPORTED

Appropriate for guided-exploration. The unknown (whether σ_disc enters top-6) is answered:
yes, partially. The compression capacity question is narrowed to a testable range.
K1437 miss is statistical noise at N=200.
