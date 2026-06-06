# Shared Layer 0 at N=5: Research Digest

## Hypothesis

Sharing a single Layer 0 capsule pool across 5 domains degrades quality
by less than 2% compared to per-domain Layer 0 pools, and Layer 0
cross-domain Jaccard remains above 0.40.

**Falsifiable**: If all three sharing strategies (base, average, first)
degrade quality by more than 2% vs full concatenation, OR if pairwise
Layer 0 Jaccard drops below 0.40, the hypothesis is killed.

**Result: KILL (quality).** All three sharing strategies degrade quality
7.8-12.0% vs full concatenation at N=5. This reverses the N=2 finding
where sharing improved quality 1.7-3.0%. Layer 0 Jaccard passes (0.853,
well above 0.40) -- the problem is not that Layer 0 pools diverge, but
that sharing breaks the residual stream magnitude balance when D is large.

---

## What This Experiment Tests

**Q: Does the shared Layer 0 benefit from N=2 persist when scaling to
N=5 domains?**

The parent experiment (shared_layer0_pool) proved that sharing Layer 0
IMPROVES quality at N=2 by eliminating "double counting" -- when two
similar pools are concatenated at Layer 0, their combined output is ~2x
the trained magnitude, distorting the residual stream. At N=5, the
question is whether the 5x distortion from full concatenation is still
worse than the alternative: sharing Layer 0 (1x) while deeper layers
contribute at 5x.

The answer is no. At N=5, full concatenation preserves the ratio between
layers (all at 5x), while shared Layer 0 starves Layer 0 relative to
deeper layers (1x vs 5x). The magnitude balance matters more than
eliminating double counting at higher D.

---

## Lineage in the Arena

```
gpt -> relu_router -> ... -> behavioral_dedup -> shared_layer0_pool -> shared_layer0_n5
                                                  (N=2, PASS: -3.0%)    (N=5, KILL: +7.8%)
                               capsule_identity -> n5_identity_scaling
                                                    (N=5, J=0.792)
```

---

## Key References

**Shared Layer 0 Pool (this project, N=2)**: All strategies improve
quality 1.7-3.0% vs full concatenation. Double counting at Layer 0
is the dominant problem at N=2. Cross-pool Jaccard = 0.544.

**N=5 Identity Scaling (this project)**: Combined Jaccard = 0.792
at N=5. Linear degradation ~0.026/domain. This experiment uses the
same quintary split (a-e, f-j, k-o, p-t, u-z).

**Yosinski et al. (2014)**: Early layers learn general features.
Confirmed at N=5: pairwise Layer 0 Jaccard = 0.853. Layer 0 IS
domain-invariant -- the kill is not about representation quality.

---

## Empirical Results

### 3-Seed Aggregate Quality (seeds 42, 123, 7)

| Method | Avg Val Loss | Std | vs Joint | vs Full Concat |
|--------|-------------|-----|----------|----------------|
| joint (baseline) | 0.5228 | 0.0069 | -- | -33.2% |
| full_concat (control) | 0.7829 | 0.1288 | +49.8% | -- |
| weight_avg | 0.5434 | 0.0205 | +4.0% | -30.6% |
| **shared_L0_base** | **0.8692** | **0.2241** | **+66.3%** | **+11.0%** |
| **shared_L0_average** | **0.8766** | **0.1326** | **+67.7%** | **+12.0%** |
| **shared_L0_first** | **0.8436** | **0.1935** | **+61.4%** | **+7.8%** |

All shared strategies are WORSE than full concatenation at N=5. The
best strategy ("first") still degrades quality by +7.8%. The magnitude
of degradation is 2.6-4.0x the 2% kill threshold.

For comparison, at N=2 the parent experiment showed -1.7% to -3.0%
(improvement). The direction reverses between N=2 and N=5.

### Kill Criterion 1: Quality Degradation

| Strategy | vs Full Concat | Threshold | Verdict |
|----------|---------------|-----------|---------|
| base | +11.0% | >2% degrades | **KILL** |
| average | +12.0% | >2% degrades | **KILL** |
| first | +7.8% | >2% degrades | **KILL** |

### Kill Criterion 2: Layer 0 Cross-Domain Jaccard

| Metric | Value | Threshold | Verdict |
|--------|-------|-----------|---------|
| Mean pairwise Jaccard | 0.853 | <0.40 | **PASS** |
| Min pairwise pair (any seed) | 0.690 | | Above 0.40 |
| Co-activation Jaccard (composed) | 0.482 | | Above 0.40 |

Layer 0 features remain highly similar across 5 domains. The Jaccard
kill criterion is not triggered. This confirms that the quality
degradation is NOT caused by Layer 0 pools diverging -- it is caused
by the magnitude imbalance.

### Per-Seed Quality Detail

| Seed | Strategy | Shared L0 | Full Concat | Delta |
|------|----------|-----------|-------------|-------|
| 42 | first | 1.0656 | 0.9304 | +14.5% |
| 123 | first | 0.7110 | 0.7256 | -2.0% |
| 7 | first | 0.7543 | 0.6927 | +8.9% |

Seed 123 shows a small improvement (-2.0%), but seeds 42 and 7 show
large degradations. The benefit is seed-dependent at N=5, unlike N=2
where all seeds improved.

### Layer 0 Pairwise Jaccard at N=5 (Seed 42)

| Pair | Jaccard |
|------|---------|
| a_e vs f_j | 0.824 |
| a_e vs k_o | 0.793 |
| a_e vs p_t | 0.774 |
| a_e vs u_z | 0.824 |
| f_j vs k_o | 0.825 |
| f_j vs p_t | 0.790 |
| f_j vs u_z | 0.870 |
| k_o vs p_t | 0.744 |
| k_o vs u_z | 0.795 |
| p_t vs u_z | 0.776 |

All pairwise Jaccards well above 0.40 (minimum 0.744). Layer 0 IS
learning domain-invariant features across all 5 domains.

### Parameter Savings (Scales Linearly with N)

| Configuration | N=2 (parent) | N=5 (this) |
|---------------|-------------|------------|
| Full concat params | 202,112 | 398,720 |
| Shared L0 params | 185,728 | 333,184 |
| Saving (abs) | 16,384 | 65,536 |
| Saving (%) | 8.1% | 16.4% |
| Saving ratio | 1.0x | 4.0x |

Parameter savings scale as (D-1)/(L*D), which is 4/20 = 20% of capsule
params at N=5. The savings are real and substantial -- but the quality
cost makes them unusable.

---

## Why the Finding Reverses at N=5

### The Residual Stream Balance Explanation

In single-domain training, each layer contributes roughly equally to
the residual stream. After full concatenation of D domains:

```
Full concat:    all layers contribute at ~D*magnitude
                Layer ratios preserved: y_0/y_l = constant

Shared Layer 0: Layer 0 at 1x, Layers 1+ at D*magnitude
                Layer 0 is starved: y_0/(sum y_1..3) = 1/(3D) instead of 1/3
```

At N=2: the starvation (1/6 vs 1/3) is modest and offset by the
benefit of eliminating 2x double counting. Net: improvement.

At N=5: the starvation (1/15 vs 1/3, i.e., Layer 0 contributes
6.25% instead of 25% of the MLP delta) is severe. Full concatenation
preserves the layer ratios (all layers at 5x), which turns out to be
less harmful than the 5x distortion at every layer.

### Crossover Point

The benefit reverses somewhere between N=2 and N=5. At N=2, sharing
helps. At N=5, it hurts. The crossover likely occurs around N=3-4,
where the magnitude balance cost begins to outweigh the double
counting fix.

---

## Micro-Scale Limitations

1. **Same limitations as parent experiment**: toy domains, character-level
   tokenization, small model (d=64, P=128, L=4).

2. **High seed variance at N=5**: Std ranges from 0.13 to 0.22 for shared
   strategies. The kill is clear in aggregate (+7.8% to +12.0%) but
   seed 123 shows -2.0% for the "first" strategy, suggesting the effect
   magnitude is not fully stable.

3. **No calibration tested**: Calibration (200 steps on mixed data) might
   partially recover the shared Layer 0 quality at N=5. The parent
   experiment noted this as an open question. However, a >7.8% gap is
   much harder to close with calibration than the <3% gap at N=2.

4. **Full concatenation baseline is also degraded**: Full concat at N=5
   shows +49.8% vs joint (compared to +8.5% at N=2). The composition
   gap grows substantially with N. Weight averaging (-30.6% vs full
   concat) remains far superior at N=5.

---

## What Would Kill This (Already Killed)

### At Micro Scale (tested, killed)

- **Quality degradation >2%**: KILLED. All strategies degrade 7.8-12.0%.
  The shared Layer 0 protocol does not scale from N=2 to N=5.

- **Layer 0 Jaccard <0.40**: NOT KILLED. Jaccard = 0.853 (well above).
  The representation similarity is preserved; the problem is magnitude
  balance, not feature divergence.

### What This Kill Means for the Project

The shared Layer 0 protocol is NOT a general solution. It works at N=2
(where it was validated) but fails at N=5. The composition protocol
should NOT share Layer 0 at N>=5.

However, the Jaccard finding is informative: Layer 0 features ARE
domain-invariant even at N=5. A corrected protocol might:

1. **Share Layer 0 with per-layer magnitude scaling**: Add a learned
   scalar per layer during calibration to rebalance the residual stream.
   This would fix the 1x vs 5x imbalance while preserving the parameter
   savings.

2. **Scale the shared Layer 0 output by D**: Multiply the shared pool
   output by D to match the magnitude of full concatenation. This
   preserves the layer ratio while using fewer parameters.

3. **Only share at small N (N<=3)**: Use the protocol selectively based
   on domain count. At N=2, share. At N>=5, concatenate.

Option 2 is the most promising: it is equivalent to "share the function
but preserve the magnitude" and requires zero additional training.

---

## Implications for the Project

1. **The shared Layer 0 protocol has a scaling limit.** It improves
   quality at N=2 but degrades at N=5. The crossover is between N=2
   and N=5, likely around N=3-4.

2. **Magnitude balance is the mechanism.** The quality degradation is
   NOT from feature divergence (Jaccard stays high) but from breaking
   the per-layer contribution ratios in the residual stream.

3. **Weight averaging remains superior at N=5.** With -30.6% vs full
   concat (and only +4.0% vs joint), weight averaging is the best
   zero-shot composition strategy at any N.

4. **A magnitude-corrected variant may rescue the approach.** Multiplying
   shared Layer 0 output by D could restore the layer balance while
   keeping the parameter savings. This is a natural follow-up experiment.

5. **Parameter savings scale linearly with N but are wasted.** 16.4%
   savings at N=5 (65,536 params) vs 8.1% at N=2, but the quality
   cost makes it unusable without correction.
