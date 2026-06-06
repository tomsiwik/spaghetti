# PPL-Probe Temperature Sensitivity: Research Digest

## Hypothesis

PPL-probe weighting quality is robust across softmax temperatures tau in {0.1, 0.5, 1.0, 2.0, 5.0}.

**Status: KILLED** -- tau has a strong, statistically significant effect on composition quality.

## What This Experiment Is

A synthetic micro-scale test (d=32, r=4, N=5 experts, 50 seeds) measuring how
softmax temperature affects PPL-probe composition quality. For each trial:
random LoRA expert matrices are generated, a noisy target composition is created,
and "loss" scores are computed as Frobenius distance from each expert to the target.
The gap improvement measures how much better PPL-probe weighting is vs equal-weight,
expressed as percentage points of Frobenius distance reduction.

## Key References

- exp_cross_domain_dilution_vs_k (proven): PPL-probe at tau=1.0, +9.34pp
- exp_ppl_probe_k3_scaling (supported): PPL-probe scales to K=3,5 at tau=1.0

## Empirical Results

### Gap Improvement by tau and K

| tau | K=2 mean (pp) | K=3 mean (pp) | K=5 mean (pp) |
|-----|---------------|---------------|---------------|
| 0.1 | +29.9 (std 35.0) | +26.8 (std 27.0) | +31.3 (std 16.7) |
| 0.5 | +37.5 (std 16.2) | +36.0 (std 8.0) | +25.8 (std 6.7) |
| 1.0 | +22.8 (std 9.7) | +19.4 (std 4.0) | +12.6 (std 2.9) |
| 2.0 | +12.1 (std 5.1) | +9.8 (std 1.9) | +6.2 (std 1.3) |
| 5.0 | +5.0 (std 2.1) | +3.9 (std 0.7) | +2.4 (std 0.5) |

### Kill Criteria

- **K1 KILLED**: Max std across tau = 11.76pp (threshold: 5.0pp). Range up to 32.6pp.
  ANOVA highly significant (p < 0.0001 for all K values).
- **K2 KILLED**: Optimal tau = 0.5 (K=2,3), 0.1 (K=5). All differ from 1.0 by >=2x.

### Key Findings

1. **Lower tau is better** -- more discriminating temperatures yield larger gap
   improvements. The relationship is monotonic: tau=0.5 gives ~60-100% more
   improvement than tau=1.0 across all K.

2. **Very low tau (0.1) has high variance** -- while mean gap is high, std is
   also very high (35pp at K=2) with some negative outcomes (12/50 trials worse
   than equal-weight at K=2). The weight distribution becomes nearly one-hot,
   making composition unstable.

3. **tau=0.5 is the sweet spot** -- highest mean gap for K=2,3 with moderate
   variance and zero negative outcomes (50/50 positive). For K=5, tau=0.1 wins
   on mean but tau=0.5 is safer (0 vs 3 negative outcomes).

4. **tau=1.0 is conservative but safe** -- always positive, moderate improvement,
   lowest variance among the "useful" range. Never hurts, always helps.

5. **All tau values beat equal-weight** -- even tau=5.0 (nearly uniform) shows
   2-5pp improvement, confirming PPL-probe adds value regardless of temperature.

6. **tau sensitivity increases with score spread** -- when expert losses have
   higher variance, the temperature matters more (spread=5.0: range=45pp across tau).

## What This Means for SOLE

The kill says tau=1.0 is NOT optimal -- tau=0.5 is consistently better. However:

1. **tau=1.0 is not wrong**, just suboptimal. It always improves over equal-weight.
2. **Recommendation: use tau=0.5 as default** for PPL-probe weighting. This is a
   free improvement (no additional compute).
3. **Adaptive tau** could be even better: set tau proportional to std(losses) to
   normalize the temperature relative to the score spread.

## Limitations

1. **Synthetic losses**: Real PPL losses may have different scale/distribution than
   Frobenius distances. The optimal tau in production depends on the actual loss scale.
2. **No model**: This uses random matrices, not trained LoRA experts.
3. **Oracle definition**: Uses unconstrained least-squares (allows negative weights).
   Real PPL-probe is constrained to non-negative softmax weights.
4. **Micro scale**: d=32, r=4. Real scale (d=4096, r=16) may have different
   score distributions.
5. **The "right" tau depends on loss scale** -- the key finding is really that
   tau should be calibrated to the score spread, not that 0.5 is universally optimal.

## What Would Kill This

- If real PPL losses at macro scale have such uniform scores that tau does not
  matter (all experts equidistant from any probe), then temperature is irrelevant.
- If the optimal tau at d=4096 with real adapters is actually tau=1.0 (because
  real loss scales differ from synthetic Frobenius distances).

## Actionable Outcome

Despite being "killed" (tau IS sensitive), this experiment produces a clear
actionable recommendation: **lower tau from 1.0 to 0.5** in PPL-probe weighting.
This gives ~60% more gap improvement with zero additional cost. Even better:
normalize tau by std(losses) for adaptive temperature scaling.
