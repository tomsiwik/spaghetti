# LEARNINGS: LoRA Scale Ablation

## What We Learned

### 1. rho < 0.15 at all scales — overscaling hypothesis falsified
Measured perturbation-to-base ratio (rho) at scale=20 is 0.144, not the predicted 2.22.
Even MLX's aggressive default scale=20 keeps the LoRA update as a modest perturbation
on Falcon-E-3B ternary weights. The "overwrite regime" (rho > 1) does not occur at
any tested scale.

**Calibrated values:**
- ||W||_F (base weight Frobenius norm, per-layer average) = 83.1
- ||B^T A^T||_F (unscaled LoRA update norm, 300 steps, lr=1e-4) = 0.6-1.2
- Critical scale (where rho=1): ~660-1380 depending on adapter

### 2. MATH.md norm estimates were off by 8-15x
Two sources of error:
- ||W||_F was 1.8x larger than estimated (83 vs 45) — per-channel scales in
  Falcon-E-3B are not O(1/sqrt(d_in)) as assumed
- ||B^T A^T||_F was ~7x smaller than estimated (0.6-1.2 vs 5) — 300 Adam steps
  at lr=1e-4 from B=0 initialization are much more conservative than estimated

**Lesson:** Always measure norms before making scaling predictions. The toy
worked example (d=16) in MATH.md was not representative of actual model dimensions.

### 3. Degradation is domain specialization, not scale damage
~50-60% of MMLU sub-benchmarks degrade after adapter training, but this rate is
constant across all scales (50% at scale=1 through 62% at scale=20). The degradation
is from domain specialization (training on medical hurts math), not from overscaling.

**Caveat (from adversarial review):** MMLU at n=20 per domain has 95% CI of ±0.22.
Most "degradation" claims are within noise. Need larger evaluation sets to
distinguish domain specialization from measurement noise.

### 4. SFT > NTP for task accuracy
SFT (response-only loss) consistently produces higher GSM8K accuracy than NTP
(all-token loss) at scales 1-4 (0.500-0.620 vs 0.473-0.527). SFT focuses gradient
signal on task-relevant tokens.

### 5. GSM8K shows an inverted-U with scale
Base=0.440, peak at scale=8 (0.593), decline at scale=20 (0.523). The effect is
underpowered (n=50) but the trend is consistent. If confirmed, this suggests scale=4-8
is optimal for this model/training regime.

### 6. Prior findings are NOT invalidated (but not "validated" either)
The adversarial review correctly notes that "rho < 0.15 means no overwrite" does NOT
equal "prior findings are validated." It means the specific concern about overscaling
is resolved. Other confounds may exist.

## What Went Wrong

1. **MATH.md predictions were poorly calibrated.** Should have measured one adapter's
   norms as a pilot before designing the full experiment.

2. **MMLU evaluation is too small.** 20 questions per domain is useless for detecting
   effects smaller than ±22%. Need 100+ per domain.

3. **Experiment took too long.** 40+ evaluations at ~5-20 min each = many hours.
   Should have done a smaller pilot (3 scales × 1 loss type × 1 domain) first.

4. **No checkpoint from start.** Wasted time when the experiment was interrupted.
   Always build checkpointing into multi-hour experiments.

## Actionable Follow-ups

- **Use scale=4-8 as default** for future adapter training (optimal GSM8K range)
- **Increase MMLU evaluation to 100+ per domain** in future experiments
- **Always measure norms first** before making scaling predictions
- **The deployment track is unblocked** — scale was not a confound, so proceed with
  exp_generation_quality_test and exp_task_accuracy_real_benchmarks
