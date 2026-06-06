# Revival Dynamics Under Composition: Research Digest

## Hypothesis

Composition suppresses or amplifies capsule revival compared to
single-domain fine-tuning by at least 5 percentage points,
changing the practical importance of pruning timing.

**Falsifiable**: If |revival_composed - revival_single| < 5 pp at
S_post=3200, composition does not meaningfully change revival
dynamics. The "prune after training" recommendation applies equally
regardless of composition status.

**Result: PASS.** Composition SUPPRESSES revival by 8.6 pp
(17.1% single vs 8.5% composed+joint, 3 seeds). The kill criterion
(< 5 pp difference) is not triggered. The suppression is consistent
across all checkpoint intervals and both composed conditions.

---

## What This Experiment Tests

**Q: Does composition change the revival rate of dead capsules?**

Exp 18 showed 28.1% revival in single-domain models over 3100 training
steps, driven by inter-layer coupling (Exp 20: freezing upstream
reduces revival 79-94%). Exp 16 showed the SAME capsules die in
single-domain and composed models (Jaccard=0.895). But none of these
measured whether the revival RATE changes when capsules exist inside
a composed model.

In a composed model, capsule pools from two domains are concatenated.
During continued training (calibration), the model sees mixed-domain
inputs. This could:
- SUPPRESS revival: cross-domain gradients partially cancel, weakening
  the upstream weight updates that drive inter-layer coupling
- AMPLIFY revival: diverse inputs provide more pathways for dead
  capsules to see positive pre-activations

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> relu_router -> dead_capsule_pruning -> pruning_controls -> capsule_revival -> revival_under_composition
                                              (composition           (activation-based      (per-capsule        (per-capsule         (composition
                                               by concat)             dead pruning)          identity)           revival tracking)    revival effects)
```

---

## Key References

**Exp 18 (capsule_revival)**: Established 28.1% cohort revival under
single-domain training (S=100 to S=3200). Revival accelerates with
training (5.8% to 15.9% per interval). This experiment uses the same
per-capsule tracking methodology.

**Exp 20 (death_recovery_mechanism)**: Proved inter-layer coupling
drives 79-94% of revival. Freezing upstream layers suppresses downstream
revival. The gradient competition mechanism proposed here is a
composition-specific variant of the same phenomenon.

**Exp 16 (capsule_identity)**: Same capsules die in single-domain and
composed models (Jaccard=0.895). This establishes that the dead SET
is preserved across composition. Our experiment shows the revival
DYNAMICS change even though the initial dead set is similar.

**Gurbuzbalaban et al. (2024)**: >90% of revived neurons eventually
re-die. Our composed revival rate (8.5%) being lower than single-domain
(17.1%) is consistent: in composed models, the weaker inter-layer
coupling means fewer capsules cross the revival boundary, and those
that do are more likely to be transient.

---

## Empirical Results

### 3-Seed Aggregate (seeds 42, 123, 7)

#### Revival Rates (fraction of anchor dead cohort that revived)

| Post-Compose Steps | Single-Domain | Composed+Joint | Composed+Own | Diff (Joint) |
|-------------------|--------------|----------------|-------------|-------------|
| 100 | 5.1% | 2.9% | 3.7% | -2.2 pp |
| 400 | 6.2% | 4.0% | 4.4% | -2.2 pp |
| 800 | 8.7% | 4.6% | 6.4% | -4.1 pp |
| 1600 | 13.3% | 5.0% | 6.8% | -8.2 pp |
| 3200 | 17.1% | 8.5% | 9.5% | -8.6 pp |

Key pattern: suppression GROWS with training duration. At S=+100 the
difference is only 2.2 pp (below kill threshold). By S=+3200 it reaches
8.6 pp. This is consistent with accumulating gradient competition:
longer training means more cross-domain gradient cancellation events.

#### Death Rate Trajectories

| Post-Compose Steps | Single-Domain | Composed+Joint | Comp A | Comp B |
|-------------------|--------------|----------------|--------|--------|
| 0 (anchor) | 54.8% | 58.2% | 58.8% | 57.7% |
| 100 | 61.2% | 67.6% | 68.4% | 66.9% |
| 400 | 60.5% | 67.3% | 67.6% | 67.0% |
| 800 | 58.4% | 66.4% | 66.6% | 66.3% |
| 1600 | 55.6% | 65.8% | 66.3% | 65.2% |
| 3200 | 52.5% | 63.0% | 64.1% | 61.9% |

Composed models have consistently higher death rates (~10 pp higher
plateau). Both show the characteristic "spike then slow decay" pattern
from Exp 17, but the composed decay is slower (less revival to drive it).

#### New Death (alive-to-dead transitions from anchor)

| Steps | Single A->D | Composed A->D |
|-------|------------|--------------|
| 100 | 47 | 113 |
| 400 | 46 | 116 |
| 800 | 43 | 112 |
| 1600 | 41 | 107 |
| 3200 | 36 | 98 |

Composed models produce 2-3x more new deaths. This combines two effects:
(1) 2x more total capsules, and (2) cross-domain inputs push in-domain
capsules past the death boundary. The composed model's higher plateau is
driven by BOTH suppressed revival AND elevated new death.

#### Per-Domain Revival in Composed Model

| Steps | Domain A | Domain B |
|-------|----------|----------|
| 100 | 2.9% | 2.9% |
| 400 | 4.0% | 3.9% |
| 800 | 5.0% | 4.2% |
| 1600 | 4.9% | 5.1% |
| 3200 | 7.9% | 9.0% |

Both domain halves show similar revival rates (no asymmetry), suggesting
the suppression is symmetric and not driven by domain-specific effects.

---

## Kill Threshold Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| \|revival_composed_joint - revival_single\| | 8.6 pp | < 5 pp | **PASS** |
| \|revival_composed_own - revival_single\| | 7.6 pp | < 5 pp | **PASS** |

**Result**: Composition suppresses revival by 8.6 pp (joint training)
and 7.6 pp (own-domain training). Both exceed the 5 pp kill threshold.

---

## Key Findings

### Finding 1: Composition Suppresses Revival

Single-domain revival at S=+3200: 17.1%. Composed revival: 8.5%.
The suppression is 8.6 pp, roughly halving the revival rate. This is
the largest effect we have measured on capsule revival dynamics.

The suppression grows monotonically with training: 2.2 pp at S=+100
to 8.6 pp at S=+3200. The accumulating nature is consistent with
gradient competition: each training step with cross-domain inputs
slightly weakens the inter-layer coupling that drives revival.

### Finding 2: Suppression Is Structural, Not Just Data-Driven

Condition C (composed model trained on own-domain data only) shows
7.6 pp suppression -- almost as large as the 8.6 pp from joint training.
This means most of the suppression comes from the STRUCTURAL change
(having 2x capsules in the composed model), not from seeing cross-domain
inputs. The composed model's wider weight matrices mean inter-layer
coupling must shift a higher-dimensional space, diluting the effect on
any individual capsule's input distribution.

### Finding 3: Composed Models Have Higher Death Plateaus

Composed equilibrium death: 63.0% vs single-domain 52.5% at S=+3200.
This 10.5 pp gap comes from both suppressed revival (fewer dead
capsules returning) AND elevated new death (more alive capsules being
killed by cross-domain input distribution shifts).

### Finding 4: Pruning Timing Is LESS Critical in Composed Models

The practical implication: in composed models, dead capsules are MORE
likely to STAY dead. The 8.5% revival rate means only ~1 in 12 dead
capsules will revive (vs ~1 in 6 for single-domain). Therefore:

- Pruning mid-composition-training is SAFER than pruning mid-single-domain-training
- The "prune after training completes" recommendation still holds but
  is less critical: composed models waste fewer capsules on transient death
- The pruning yield is higher in composed models (more permanently dead)

### Finding 5: Per-Domain Revival Is Symmetric

Both domain halves show similar revival rates (7.9% vs 9.0% at S=+3200).
The suppression is not asymmetric. Contributors from different domains
can expect similar revival dynamics for their capsules after composition.

---

## Micro-Scale Limitations

1. **Only D=2 domains**: With more domains (N=5+), gradient competition
   should increase. Suppression may be stronger at higher N. Alternatively,
   more diverse inputs could provide more revival pathways.

2. **Maximum 3200 post-compose steps**: Revival rate was still increasing
   at S=+3200 in both conditions. The gap may widen or narrow with longer
   training. The monotonic growth of the gap up to S=+3200 suggests it
   would continue growing.

3. **Anchor at S=200 only**: Different anchor points (earlier or later
   in fine-tuning) may show different composition effects. A later anchor
   would have more mature capsule specialization.

4. **Same-domain profiling asymmetry**: Single-domain models are profiled
   on their own domain val data. Composed models are profiled on joint
   val data. The composed profiling sees cross-domain inputs that may
   classify some capsules differently. However, Exp 16 showed Jaccard=0.895
   between these profiling conditions, so the effect is small.

5. **Small model (d=64, P=128)**: At larger d, the dead/alive boundary
   margin may differ, changing both single-domain and composed revival
   rates. The relative suppression effect could be scale-dependent.

6. **Only 3 seeds**: Standard deviations at S=+3200 are 4.6% (single)
   and 2.4% (composed). The 8.6 pp gap is ~1.9 sigma, directionally
   strong but not statistically bulletproof.

---

## What Would Kill This

### At Micro Scale (tested)

- **Composition changes revival < 5 pp**: NOT KILLED. The difference
  is 8.6 pp, well above threshold.

### At Macro Scale (untested)

- **Suppression reverses at scale**: If larger models have narrower
  dead/alive boundaries (smaller margin), the wider input distribution
  from composition might push MORE capsules past the boundary, flipping
  the effect to amplification.

- **Warmup + cosine decay changes dynamics**: Exp 19 showed these LR
  schedules dramatically change revival. Under standard macro training
  with warmup, the baseline revival rate may be lower (~20% death vs
  ~47%), and the composition effect could shrink to below 5 pp.

- **Calibration-only training (100 steps)**: Real composition uses
  ~100-200 steps of calibration, not 3200. At S=+100 the suppression
  is only 2.2 pp (below threshold). For the practical recommendation,
  the short-calibration regime matters more than long training.

---

## Implications for the Project

### Revised Understanding of Revival Under Composition

**Old (from Exp 18)**: "Prune after training completes, not during,
because 28% of dead capsules revive over 3100 steps."

**New**: "The 'prune after training' recommendation still holds but
is LESS critical for composed models. Composition suppresses revival
to 8.5% (from 17.1% single-domain). In the practical regime (100-200
calibration steps), the revival rate is only ~3% for composed models.
Pruning at the moment of composition (before calibration) wastes
very few capsules that would have revived."

### Updated Protocol Consideration

The composition protocol can optionally be simplified:

```
Option A (current, conservative):
1. Fine-tune per domain -> compose -> calibrate 200 steps -> profile -> prune

Option B (aggressive, supported by this finding):
1. Fine-tune per domain -> profile each single-domain model -> prune -> compose -> calibrate

Option B is safe because:
- Same capsules die in single-domain and composed (Exp 16, J=0.895)
- Revival during calibration is low (~3% at 100 steps composed)
- Pre-composition pruning already validated (prune_before_compose, +0.01%)
```

This experiment provides additional evidence that Option B (pre-composition
pruning) is safe: even if you prune capsules that would revive during
calibration, the revival rate is so low under composition (~3%) that
the loss is negligible.
