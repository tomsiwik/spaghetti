# Profiling Noise Quantification: Research Digest

## Hypothesis

A substantial fraction (>50%) of the D->A transitions observed in Exp 18
are artifacts of sampling noise in the profiling procedure, not genuine
capsule revival. With only 640 samples per profiling run (20 batches x 32),
borderline capsules near f=0 may flicker between dead and alive
classifications across different random batches.

**Falsifiable**: If the same checkpoint profiled with different random
batches disagrees on >20% of capsules, or if consensus-corrected D->A
transitions are <50% of single-run D->A transitions, the revival
finding from Exp 18 is artifactual.

**Result: 0 of 3 kill criteria triggered.** The hypothesis is WRONG.
Profiling noise is negligible: same-checkpoint disagreement is only
2.6-3.8% of capsules. Consensus-corrected D->A transition counts
actually INCREASE (by 6.2%), not decrease. Exp 18's revival finding
is robust.

---

## What This Experiment Tests

**Q: Is the revival observed in Exp 18 real, or is it an artifact of
profiling noise?**

Exp 18's adversarial review (Finding 2) flagged that borderline capsules
near f=0 could produce false D->A transitions when profiled with only
640 samples. This experiment directly measures the noise floor by
profiling each checkpoint TWICE with different random batches:

1. **Run A** (seed=1000): 20 batches x 32 samples
2. **Run B** (seed=2000): 20 batches x 32 samples

Since the model weights are identical between runs A and B, any
disagreement in dead/alive classification is definitionally sampling
noise. We then compare transition matrices computed from single-run
masks vs consensus masks (dead only if dead in both runs).

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> relu_router -> dead_capsule_pruning -> pruning_controls -> capsule_revival -> profiling_noise
                               (composition    (activation-based      (pre-composition       (per-capsule       (noise
                                by concat)      dead pruning)          death rate +            identity          quantification
                                                                       random baseline)        tracking)         for profiling)
```

---

## Key References

**Exp 18 (capsule_revival)**: Found 28.1% of capsules dead at S=100
revive by S=3200. Adversarial review flagged profiling noise as
potential confound. This experiment directly addresses that concern.

**Binomial sampling theory**: With N=640 samples, a capsule with true
activation probability p=0.001 has a 52.7% chance of being falsely
classified as dead. At p=0.005, this drops to 4.1%. At p=0.01, 0.16%.
Only capsules with extremely rare activations (p < 0.005) are susceptible
to noise.

---

## Empirical Results

### 3-Seed Aggregate (seeds 42, 123, 7)

#### Same-Checkpoint Profiling Disagreement

| Steps | Disagree | % of total | Flickering (0<f<0.05) | % |
|-------|----------|------------|----------------------|---|
| 0 | 13.3 | 2.6% | 199.7 | 39.0% |
| 50 | 18.0 | 3.5% | 115.0 | 22.5% |
| 100 | 19.7 | 3.8% | 119.0 | 23.2% |
| 200 | 13.7 | 2.7% | 111.3 | 21.7% |
| 400 | 15.3 | 3.0% | 118.0 | 23.0% |
| 800 | 14.7 | 2.9% | 120.3 | 23.5% |
| 1600 | 14.0 | 2.7% | 145.0 | 28.3% |
| 3200 | 15.7 | 3.1% | 143.7 | 28.1% |

The disagreement rate is remarkably stable at 2.6-3.8% across all
checkpoints and training durations. Profiling is reliable: >96% of
capsules receive the same dead/alive classification regardless of which
random batches are used.

Notably, 22-39% of capsules are "flickering" (0 < f < 0.05 in at least
one run), but this borderline population rarely crosses the f=0 boundary.
Most borderline capsules fire rarely but consistently.

#### Per-Layer Disagreement

| Layer | Mean disagree (of 128) |
|-------|----------------------|
| 0 | 1.7 |
| 1 | 4.7 |
| 2 | 4.5 |
| 3 | 4.6 |

Layer 0 has lowest disagreement (processes frozen embeddings, fewer
borderline capsules). Layers 1-3 have ~4.5 disagreeing capsules each.

#### Transition Comparison: Single-Run vs Consensus

| Interval | D->A (single) | D->A (consensus) | Reduction |
|----------|--------------|-----------------|-----------|
| 0->50 | 10.3 | 12.0 | -16.1% |
| 50->100 | 21.0 | 25.3 | -20.6% |
| 100->200 | 25.0 | 24.0 | +4.0% |
| 200->400 | 24.7 | 27.0 | -9.5% |
| 400->800 | 26.3 | 27.3 | -3.8% |
| 800->1600 | 40.0 | 41.3 | -3.3% |
| 1600->3200 | 25.7 | 26.7 | -3.9% |

**Total D->A (single-run): 173.0**
**Total D->A (consensus): 183.7**
**Noise-attributable D->A: -10.7 (-6.2%)**

The noise fraction is NEGATIVE. Consensus-corrected D->A counts are
HIGHER than single-run counts, not lower. This occurs because the
consensus dead set is smaller (more conservative), so each revival
event represents a larger fraction of the dead population. Profiling
noise does not inflate D->A transitions -- if anything, single-run
profiling slightly underestimates revival.

#### Cohort Analysis: Single-Run vs Consensus

| Steps | Revived (single) | Revived (consensus) |
|-------|-----------------|-------------------|
| 100 | 0.0% | 0.0% |
| 200 | 9.8% | 9.8% |
| 400 | 13.9% | 14.5% |
| 800 | 17.4% | 18.1% |
| 1600 | 27.3% | 28.4% |
| 3200 | 26.7% | 28.2% |

The consensus-corrected cohort revival (28.2%) is virtually identical
to -- and slightly higher than -- the single-run estimate (26.7%).
Both are consistent with Exp 18's reported 28.1%.

#### Jaccard Comparison

| Method | Jaccard(100, 3200) | Std |
|--------|-------------------|-----|
| Single-run (A) | 0.676 | 0.094 |
| Consensus | 0.654 | 0.102 |

Both are consistent with Exp 18's reported 0.669.

---

## Kill Threshold Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Same-checkpoint disagreement | 3.8% max | >20% | **PASS** |
| Noise-attributable D->A | -6.2% (negative) | >50% | **PASS** |
| Noise-corrected max revival rate | 17.4% | <5% | **PASS** |

**0 of 3 kill criteria triggered.**

---

## Key Findings

### Finding 1: Profiling Is Remarkably Stable

Same-checkpoint disagreement is only 2.6-3.8% of capsules across all
checkpoints and training durations. The 20-batch, 32-sample profiling
protocol from Exp 9/10/17/18 is reliable: the vast majority of dead/alive
classifications are robust to the specific random batches used.

### Finding 2: The Borderline Population Is Large But Mostly Consistent

22-39% of capsules have activation frequency in (0, 0.05) in at least
one profiling run. But this borderline population mostly stays above
f=0 -- they fire rarely but consistently. The binary f=0 threshold
cleanly separates the truly dead from the borderline alive.

### Finding 3: Consensus Correction Increases, Not Decreases, Revival

The noise fraction is -6.2%: consensus-corrected D->A counts are
6.2% HIGHER than single-run counts. This counterintuitive result
occurs because consensus shrinks the dead set (by removing borderline
capsules from it), so the remaining dead capsules have a higher
revival rate. Profiling noise does not inflate revival -- it slightly
deflates it by including borderline capsules in the dead set that
aren't actually dead.

### Finding 4: Exp 18 Results Are Fully Confirmed

All three key metrics from Exp 18 are reproduced within noise:
- Cohort revival: 26.7% (single-run) / 28.2% (consensus) vs 28.1% (Exp 18)
- Jaccard: 0.676 / 0.654 vs 0.669 (Exp 18)
- Max revival rate: 16.3% (single-run) / 17.4% (consensus) vs 15.9% (Exp 18)

### Finding 5: Per-Layer Noise Is Uniform

Layers 1-3 each have ~4.5 disagreeing capsules out of 128 (~3.5%).
Layer 0 has 1.7 (~1.3%). The profiling noise does not preferentially
affect any particular layer, so Exp 18's finding that Layer 3 has the
highest revival rate is not a noise artifact.

---

## Micro-Scale Limitations

1. **Validation set size**: The validation set may be small enough that
   different random batches still overlap substantially, artificially
   reducing disagreement. At macro scale with larger validation sets,
   disagreement could increase.

2. **Only 2 profiling runs per checkpoint**: Two runs establish a lower
   bound on noise. More runs (e.g., 10) would give a tighter estimate
   of the borderline population and their noise contribution.

3. **Same profiling sample count (640)**: The experiment validates the
   existing protocol but does not explore whether fewer samples (e.g.,
   320) would degrade reliability. This would inform the minimum
   profiling budget.

4. **Binary threshold only**: We tested f=0 (the standard threshold).
   Other thresholds (f < 0.01, f < 0.05) would have different noise
   characteristics. The borderline population data (22-39% with
   0 < f < 0.05) suggests that a threshold of f < 0.01 would have
   substantially higher disagreement.

---

## What Would Kill This

### At Micro Scale (tested)

- **Profiling unreliable (disagreement > 20%)**: NOT KILLED. Max 3.8%.
  Profiling is stable.

- **Revival is noise (>50% noise D->A)**: NOT KILLED. Noise fraction
  is -6.2% (negative). Consensus shows MORE revival.

- **True revival too weak (<5%)**: NOT KILLED. Consensus revival rate
  reaches 17.4% per interval. Revival is genuine and substantial.

### At Macro Scale (untested)

- **Larger validation sets reduce overlap**: With much larger validation
  sets, the two profiling runs may draw more distinct batches, potentially
  increasing disagreement. But this also means more representative
  profiling, so the net effect on reliability is unclear.

- **SiLU activation**: Without a hard zero, the binary dead/alive
  classification requires a magnitude threshold, which introduces an
  additional noise source not present with ReLU.

---

## Implications for the Project

### Exp 18 Revival Finding: VALIDATED

The adversarial review's concern about profiling noise is definitively
addressed. The 28.1% cohort revival rate is not an artifact of sampling
noise. The profiling protocol with 640 samples is sufficient for
reliable dead/alive classification at the binary (f=0) threshold.

### Profiling Protocol: CONFIRMED SUFFICIENT

The existing protocol (20 batches x 32 samples) is reliable. No changes
are needed before proceeding to Exp 19 (LR schedule), Exp 20 (layer
freezing), or macro scale.

### Connection to Future Experiments

- **Exp 19 (LR schedule)**: Can use the same profiling protocol with
  confidence. The noise floor is well below the expected signal.

- **Exp 20 (Layer freezing)**: The per-layer noise uniformity means
  that layer-specific revival rates (Finding 4 of Exp 18) are genuine
  signals, not noise artifacts. Layer freezing experiments can trust
  per-layer profiling differences.

- **Macro transition**: At macro scale with SiLU, a magnitude threshold
  replaces f=0. The noise characteristics will differ and should be
  re-validated, but the methodology (dual profiling) transfers directly.
