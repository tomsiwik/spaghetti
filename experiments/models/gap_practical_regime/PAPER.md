# Gap-as-Signal Practical Regime: Research Digest

## Hypothesis

The function-space gap between composed and jointly-trained models provides
meaningful discrimination between expert quality levels even in the practical
cosine regime (cos < 0.3) where real LoRA adapters live.

**Falsifiable prediction:** Quality difference between cos=0.0 and cos=0.3
exceeds 0.5pp, and gap magnitude variation across [0.0, 0.3] exceeds
seed-to-seed noise.

## What This Model Is

This experiment zooms into the practical regime of the gap-as-signal finding.
The parent experiment (gap_as_signal) demonstrated r^2=0.74 across
cos={0.0, ..., 0.9}, but the adversarial review identified a leverage
effect: cos>=0.7 drives most of the correlation, while the practical
regime (cos < 0.3) shows only ~0.2pp quality differences. Since real LoRA
adapters are naturally near-orthogonal (cos ~ 0.000 at macro scale), the
practical question is whether gap-as-signal provides useful discrimination
in the regime that actually occurs in deployment.

**Protocol:** Identical to parent, but with:
- 7 cosine levels in [0.0, 0.3] at 0.05 increments (vs 4 levels in parent)
- 2 anchor points (0.5, 0.9) for full-range comparison
- 5 seeds instead of 3 (more statistical power for small effects)

## Lineage in the Arena

```
gpt (base)
 `-- lora_gpt (LoRA adapters on MLP)
      `-- gap_as_signal (controlled orthogonality sweep, PROVEN)
           `-- gap_practical_regime (fine-grained [0.0, 0.3] sweep, KILLED)
```

## Key References

- **Parent experiment (gap_as_signal):** r^2=0.74 across full cosine range.
  Established the gap-quality correlation but with noted leverage effect.

- **Adversarial review of gap_as_signal:** Identified that cos>=0.7
  drives most of the signal. The practical regime (cos<0.3) shows only
  0.2pp quality spread. This experiment directly tests that concern.

## Empirical Results

### Summary Table (5 seeds, mean values)

| Cosine | CE Gap     | KL Gap     | Final VL | vs Joint | Std    |
|--------|-----------|-----------|----------|----------|--------|
| 0.00   | 0.00974   | 0.03850   | 0.5152   | +1.64%   | 2.047  |
| 0.05   | 0.00919   | 0.03871   | 0.5153   | +1.69%   | 2.029  |
| 0.10   | 0.00864   | 0.03904   | 0.5155   | +1.74%   | 1.958  |
| 0.15   | 0.00814   | 0.03954   | 0.5159   | +1.78%   | 1.957  |
| 0.20   | 0.00777   | 0.04024   | 0.5164   | +1.89%   | 1.902  |
| 0.25   | 0.00781   | 0.04118   | 0.5183   | +2.27%   | 1.717  |
| 0.30   | 0.00798   | 0.04238   | 0.5176   | +2.10%   | 1.841  |
| 0.50   | 0.01026   | 0.05040   | 0.5206   | +2.72%   | 1.776  |
| 0.90   | 0.02951   | 0.08749   | 0.5358   | +9.21%   | 3.001  |

### Kill Criteria Results

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| Quality diff cos=0.0 vs cos=0.3 | >= 0.5pp | 0.47pp | **KILL** |
| Gap variation > noise | SNR > 1.0 | SNR = 0.33 | **KILL** |

**Both kill criteria fail. Hypothesis KILLED.**

### Detailed Kill Analysis

**KC1: Quality difference is below threshold.**
- cos=0.0 quality: +1.64% vs joint (std=2.05)
- cos=0.3 quality: +2.10% vs joint (std=1.84)
- Difference: +0.47pp (just under the 0.5pp threshold)
- Cohen's d: 0.24 (small effect)
- The quality difference from cos=0.0 to cos=0.3 is barely one-quarter
  of a standard deviation. A practitioner could not reliably distinguish
  an expert at cos=0.0 from one at cos=0.3 based on composition quality.

**KC2: Gap variation is buried in noise.**
- Quality range across [0.0, 0.3]: 0.64pp
- Noise floor (within-level seed-to-seed std): 1.92pp
- Signal-to-noise ratio: 0.33 (signal is 3x smaller than noise)
- F-ratio (between/within variance): 0.01
- The between-cosine variation is completely dominated by seed-to-seed
  randomness. Any observed quality difference at adjacent cosine levels
  could be entirely due to initialization variance.

### Correlation Analysis

| Regime | Relationship | r | r^2 |
|--------|-------------|---|-----|
| [0.0, 0.3] (practical) | Cosine vs Quality | 0.112 | **0.013** |
| [0.0, 0.3] (practical) | CE Gap vs Quality | 0.165 | **0.027** |
| [0.0, 0.9] (full range) | Cosine vs Quality | 0.710 | 0.504 |

The within-regime r^2 = 0.013 is essentially zero. The gap-as-signal
correlation evaporates completely when restricted to the practical regime.

### Monotonicity

Despite the lack of statistical significance, the mean quality trend is
5/6 monotonic within [0.0, 0.3] (only the 0.25->0.30 step reverses).
This suggests a real but extremely weak underlying signal that is
overwhelmed by noise at 5 seeds.

### Practical Regime Fraction

The [0.0, 0.3] regime accounts for only **8.4%** of the total quality
range across [0.0, 0.9]. Nearly all of the gap-as-signal's discriminative
power comes from the cos >= 0.5 regime.

## Key Finding

**The gap-as-signal is a binary classifier, not a continuous predictor.**

In the regime that matters for real LoRA adapters (cos < 0.3), the gap
provides effectively zero discrimination. The function-space gap becomes
a useful signal only for detecting adapters with cos >= 0.5, which should
never occur with independently trained LoRA adapters (natural cos ~ 0.000).

This means:
1. **No gap measurement is needed** for independently trained LoRA adapters,
   because they are always deeply in the "good" regime.
2. **The gap diagnostic is only useful for detecting pathological cases:**
   adapters trained on nearly identical data, adapters that have been
   manually aligned, or adapters at very low rank where cos is nontrivially
   nonzero.
3. **The VISION.md framing is correct:** orthogonality is free and
   guaranteed by dimensionality. The gap-as-signal is a safety check,
   not a quality predictor.

## Micro-Scale Limitations

1. **d=64 limits capacity.** At micro scale, the model has limited ability
   to differentiate experts. At d=896+, the quality surface within [0.0, 0.3]
   might show more structure. However, the macro results (r^2=0.22 because
   everything is cos ~ 0.000) suggest the opposite.

2. **Projected experts are synthetic.** Real experts at cos=0.2 (from
   overlapping training data) might behave differently from projected
   experts at cos=0.2.

3. **Character-level names.** The domains (a-m vs n-z) are structurally
   similar. More diverse domains might amplify within-regime differences.

## What Would Kill This

This experiment itself IS a kill result. The hypothesis "gap-as-signal
provides meaningful discrimination in the practical regime" is killed.

The parent finding (gap-as-signal across the full range) remains PROVEN --
it just has a narrower practical utility than originally claimed. The
gap IS informative, but only for detecting expert pairs with cos >= 0.5,
which is a pathological case that rarely occurs with independent training.

## Implications for the Project

1. **Drop gap measurement from the contribution protocol.** Since natural
   LoRA experts are always at cos ~ 0.000, and the gap provides no
   discrimination in that regime, measuring it adds cost without value.

2. **Keep gap as a safety check only.** If a submitted expert has
   cos > 0.5 with any existing expert, reject it. This is a simple
   cosine computation, not a full gap measurement.

3. **The "gap-as-signal" framing should evolve.** The original claim was
   "gap IS the routing signal." The refined understanding is: "orthogonality
   guarantees good composition; the gap is relevant only when orthogonality
   fails." Since orthogonality never fails with independent training,
   the gap framing is theoretically correct but practically vacuous.
