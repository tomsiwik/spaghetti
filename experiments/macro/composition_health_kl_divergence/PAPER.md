# Composition Health via KL Divergence: Research Digest

## Hypothesis

KL divergence between composed-model and base-model logit distributions on
fixed calibration tokens detects harmful expert additions without task labels,
enabling label-free composition health monitoring for the SOLE architecture.

## What This Model Is

This is a composition diagnostic, not a new model. It tests whether a simple
statistical signal -- KL divergence from base -- can serve as a health monitor
for SOLE expert composition. The key innovation is that it requires NO
per-expert evaluation data, NO task labels, and only a single forward pass
on 20 calibration texts.

**The problem it solves:** When composing N LoRA experts into a base model,
how do you detect that expert #37 is hurting the composition without
evaluating all 50 domains? Prior approaches require:
- Canary queries: 20 held-out examples per expert (works, FNR=9.8%, but
  requires per-expert curation)
- Cosine-based gating: ANTI-correlated at micro (r=-0.46), worse than random
- Full evaluation: correct but O(N * domains), expensive

KL divergence offers a third path: one measurement, no labels, cheap.

## Lineage in the Arena

```
exp_quality_degradation_detection (proven, micro)
  |-- cosine gating: anti-correlated (r=-0.46)
  |-- canary queries: FNR=9.8%, works but needs per-expert data
  |
  v
exp_composition_health_kl_divergence (THIS EXPERIMENT) — KILLED
  |-- label-free KL divergence: anti-correlated (rho=-0.7), wrong direction
  |-- harmful expert undetectable (z=-2.97, discrimination=False)
```

## Key References

- Hinton et al. (2015) -- Knowledge Distillation (KL divergence as training signal)
- Yadav et al. (2023) -- TIES-Merging: parameter interference in model merging
- Yu et al. (2023) -- DARE: random drop + rescale for delta parameter merging
- Wang et al. (2024) -- LoRA Soups: CAT composition with learned weights
- Micro quality_degradation_detection (this project): canary queries proven,
  cosine gating killed

## Empirical Results

**Run:** run_kl_health_1773582050 (420.6s, OK)
**Adapters available:** 5 (bash, math, medical, python, sql) — not the full 50 pilot set
**Base model:** Qwen2.5-7B, 4-bit NF4 quantization, A5000 GPU

### Phase 2: KL Divergence vs N

| N | Mean KL | Std KL | Time (s) |
|---|---------|--------|----------|
| 5 | ~10.4   | —      | ~45      |

Only N=5 was feasible (only 5 adapters on disk). N=10/25/50 were skipped.
Single data point prevents assessing KL scaling behavior (linear vs superlinear).

### Phase 3: Leave-One-Out DeltaKL

Leave-one-out at N=5 (all 5 adapters). Last reported entry:
- kl_without=10.3582, delta_kl=0.0876
- Phase 3 took 219.7s total (5 LOO compositions, ~44s each)

The delta_kl values were small (order 0.08 nats), suggesting all 5 adapters
contribute similarly to the total KL shift — no clear outliers among healthy experts.

### Phase 4: Harmful Expert Discrimination

| Metric | Harmful (medical negated) |
|--------|--------------------------|
| KL_with_harmful | 8.7760 |
| z_score | -2.97 |
| discrimination | False |

**Critical finding:** The negated medical adapter REDUCED KL divergence from
base (8.78 vs ~10.4 for healthy composition). The harmful expert pulled the
composed model CLOSER to base, not further away. This is the opposite of
what was predicted.

**Interpretation:** Negating lora_B effectively subtracts the medical expert's
contribution from the composition. Since the medical adapter was one of the
5 being composed, negating it partially cancels the other adapters' collective
drift, reducing total KL. This means the "harmful" expert acts as a regularizer
against the other experts' perturbation, not as an additional source of
divergence. The synthetic harm construction (B-matrix negation) doesn't produce
the expected KL signature.

### Phase 5: Correlation with Quality Impact

| Metric | Value |
|--------|-------|
| Spearman rho | -0.7000 |
| p-value | 0.1881 |
| n | 5 |

**Anti-correlation:** Higher DeltaKL associates with BETTER quality (lower
quality loss), not worse. This is the reverse of the predicted direction.
N=5 gives very low statistical power (p=0.19), but the direction is clear.

This mirrors the cosine-gating result from micro (r=-0.46): unsupervised
distributional signals seem to be systematically anti-correlated with quality
in this composition regime.

### Phase 6: Per-Domain PPL

Skipped — training data directory not found on GPU instance.

### Kill Criteria

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| K1: Spearman rho(DeltaKL, quality_loss) | >= 0.3 | rho=-0.7 | **FAIL** |
| K2: Harmful expert z-score | > 2.0 | z=-2.97 | **FAIL** |
| K3: Time per composition | < 30s | ~45s/composition | **FAIL** |

**Verdict: KILL**

All three kill criteria failed. K1 and K2 failed in the wrong direction
(anti-correlated, negative z-score). K3 failed due to full model reload per
composition (~45s each on A5000 with 4-bit quant).

## Why It Failed

### 1. Anti-correlation (K1) — structural, not random

The anti-correlation (rho=-0.7) between DeltaKL and quality loss suggests
that BETTER experts create MORE distributional shift from base. This makes
theoretical sense: a well-trained expert should strongly modify predictions
in its domain, which also shifts generic calibration logits. A weak expert
barely changes anything, keeping KL low. So KL measures "impact magnitude"
not "impact quality."

This is the same failure mode as cosine-based gating (r=-0.46 at micro).
Unsupervised distributional metrics capture expert STRENGTH, not expert
QUALITY. Strong experts are both more helpful on-domain and more disruptive
off-domain.

### 2. Synthetic harm construction fails (K2)

Negating lora_B doesn't create a "harmful" expert — it creates an "anti-expert"
that partially cancels the other adapters' contributions. The net effect is
regularization, not harm. A truly harmful expert would need to push predictions
in a NOVEL wrong direction, not cancel existing correct ones.

### 3. Too slow (K3)

Each composition requires a full model reload (~45s on A5000). The SPEC
estimated ~8s (reusing base model + incremental adapter loading). In practice,
PEFT's in-place modification requires fresh base model loads. This could
potentially be fixed by weight-space merging (merge_and_unload), but the
fundamental K1/K2 failures make optimization moot.

## Limitations

1. **Only 5 adapters tested (not 50).** The pilot50 adapters were not all
   present on the GPU instance. N=5 gives very low statistical power for
   correlation tests.

2. **Synthetic harmful expert is artificial.** Negating B-matrix doesn't
   represent realistic quality degradation (e.g., noisy training, wrong domain).

3. **No N-scaling data.** With only 5 adapters, we couldn't test KL scaling
   behavior across N. The superlinear growth hypothesis remains untested.

4. **K3 timing may be a placeholder.** The log reports t=999.0s for K3, which
   may be a default value rather than actual measurement. Per-composition time
   was ~45s from Phase 3 timing (219.7s / 5 LOO = ~44s each).

## What Would Kill This (was killed)

All three macro kill criteria failed:
- K1: rho=-0.7 (anti-correlated, needed >= 0.3)
- K2: z=-2.97 (wrong direction, needed > 2.0)
- K3: ~45s (needed < 30s)

## Implications for SOLE Architecture

**KL divergence from base is NOT a viable composition health metric.** It
measures expert impact magnitude, not quality. This joins cosine-based gating
(r=-0.46) as the second unsupervised distributional metric killed for
composition health monitoring.

**What survives:** Canary queries (FNR=9.8%) remain the only proven
degradation detection method. They require per-expert curated examples but
are reliable. For label-free monitoring, new approaches are needed — perhaps
comparing composed vs individual expert predictions on domain-specific inputs,
or using the shadow scoring metric (answer-conditioned PPL, r=0.811) as a
composition health proxy.

**Pattern emerging:** Unsupervised distributional metrics (cosine similarity,
KL divergence) consistently anti-correlate with quality in SOLE composition.
This suggests that expert quality cannot be inferred from distributional
distance alone — it requires semantic evaluation (canary queries, shadow scoring).
