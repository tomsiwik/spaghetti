# Gap-as-Signal Practical Regime: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| Delta_i | (d, d') | LoRA delta for expert i: (alpha/r) * A_i @ B_i |
| cos(i,j) | scalar in [0,1] | Cosine similarity between flattened Delta_i, Delta_j |
| G_CE(c) | scalar | CE gap at cosine level c |
| Q(c) | scalar | Quality gap (% above joint) at cosine level c |
| sigma_within | scalar | Within-cosine-level standard deviation of Q |
| sigma_between | scalar | Between-cosine-level standard deviation of mean(Q) |
| d_Cohen | scalar | Cohen's d effect size |
| SNR | scalar | Signal-to-noise ratio: range(Q) / sigma_within |

## The Question

The parent experiment (gap_as_signal) showed:

```
Q(c) = quality gap at cosine c
r^2(cos, Q) = 0.74   across cos in {0.0, 0.1, ..., 0.9}
```

The adversarial review identified a **leverage effect**: the two highest
cosine levels (0.7, 0.9) produce quality gaps of +4.8% and +12.1%,
which dominate the regression. In the practical regime where real LoRA
adapters live (cos < 0.3, since natural cos ~ 0.000), the quality
differences were only +2.1% to +2.3% -- a 0.2pp spread.

**Question**: Is there a meaningful gradient within [0.0, 0.3], or is
the quality surface essentially flat in this regime?

## Prediction Under the Gap-as-Signal Hypothesis

If gap-as-signal is informative at all cosine levels, then within the
practical regime [0.0, 0.3]:

1. **Q(0.3) - Q(0.0) > 0.5pp** (meaningful quality difference)
2. **sigma_between > sigma_within** (between-level variation exceeds noise)
3. **r^2(cos, Q) > 0.1** within the practical regime

## Kill Criteria (Formal)

### KC1: Quality Difference

```
H0: Q(0.3) - Q(0.0) >= 0.5pp
Kill if: Q(0.3) - Q(0.0) < 0.5pp
```

Additionally, compute Cohen's d:
```
d_Cohen = (mean(Q_0.3) - mean(Q_0.0)) / pooled_std
```
where pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2)).

Interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large.

### KC2: Gap Variation vs Noise

```
SNR = range(mean(Q)) / mean(std(Q))
```
where range is taken over cosine levels in [0.0, 0.3] and std is within
each cosine level across seeds.

Kill if: SNR < 1.0 (signal does not exceed noise floor).

## Experimental Design

### Fine-Grained Cosine Sweep

7 levels in the practical regime:
```
cos_practical = {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
```

Plus 2 anchors for full-range comparison:
```
cos_anchor = {0.50, 0.90}
```

### Statistical Power

5 seeds (42, 123, 7, 2024, 999) instead of 3, because:
- Expected effect size is small (parent showed only 0.2pp in this range)
- Need more replicates to distinguish signal from noise
- With 5 seeds, minimum detectable effect at d=0.5 is approximately
  achievable (though still underpowered for very small effects)

### Protocol

Identical to parent experiment:
1. Train joint model (baseline) and base model
2. Train two LoRA experts (domains a-m, n-z)
3. Project expert B to target cosine levels via Gram-Schmidt
4. Measure function-space gap (CE, KL) before calibration
5. Calibrate softmax router for 300 steps
6. Measure final quality relative to joint model

## Worked Example

At d=64, r=8, n_layer=4:
- Delta dimension D = 131,072
- Expected natural cosine: cos ~ r/sqrt(D) ~ 0.022
- At cos=0.0: quality ~ +1.6% vs joint
- At cos=0.3: quality ~ +2.1% vs joint
- Difference: ~0.5pp
- Within-seed std at any given cosine: ~1.9pp
- The 0.5pp difference is buried in ~1.9pp of seed-to-seed variance

## Computational Cost

Per seed: 9 cosine levels x (projection + gap measurement + 300 calibration steps)
- Projection: O(D) = negligible
- Gap measurement: 20 batches x forward pass = ~40M FLOPs
- Calibration: 300 steps x forward+backward = ~600M FLOPs
- Total per level: ~640M FLOPs
- Total per seed: 9 x 640M = ~5.8G FLOPs
- Total experiment: 5 seeds x 5.8G = ~29G FLOPs

Wall clock: ~3 minutes total on Apple Silicon.

## Assumptions

1. **Projection faithfully simulates cosine differences.** The projected
   expert B is synthetic, not a naturally-trained expert at that cosine.
   This is the same assumption as the parent experiment.

2. **The quality metric (vs_joint_pct) is the right measure.** At micro
   scale, calibration speed manifests as final quality, not convergence
   time. This is inherited from the parent experiment.

3. **5 seeds is sufficient.** For a 0.5pp threshold with ~2pp within-seed
   variance, we would need ~50 seeds for 80% power at alpha=0.05 for a
   two-sample t-test. With 5 seeds, we can only detect large effects.
   However, the kill criteria are designed so that failure to detect IS
   the meaningful finding: if 5 seeds cannot distinguish cos=0.0 from
   cos=0.3, the effect is too small to be practically useful.
