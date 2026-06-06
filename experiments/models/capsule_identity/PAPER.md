# Capsule Identity Tracking Across Composition: Research Digest

## Hypothesis

The per-capsule dead/alive identity is preserved across composition:
the SAME capsules that die in single-domain models also die in composed
models, not just at the same aggregate rate.

**Falsifiable**: If per-capsule death identity Jaccard overlap < 50%
between single-domain and composed settings, composition creates
fundamentally different death patterns and pre-composition profiling
is unreliable.

**Result: PASS.** Combined Jaccard = 0.895 (well above 0.50 threshold).
The same capsules that die in single-domain training also die after
composition. Overlap coefficient = 0.986 (98.6% of single-domain dead
capsules remain dead after composition). Composition adds ~29 newly-dead
capsules per domain half (~6% of the pool) but preserves nearly all
existing death identity.

---

## What This Experiment Tests

**Q: Are the SAME capsules dead in single-domain and composed models,
or does composition create novel death patterns?**

Exp 10 (pruning_controls) showed that 87% of composed death is
training-induced -- an aggregate finding. Exp 18 (capsule_revival)
showed Jaccard = 0.669 for dead cohort stability across training TIME
points. This experiment completes the picture by comparing across
SETTINGS (single vs composed) rather than across time.

Protocol:
1. Pretrain base model on ALL data (300 steps)
2. Fine-tune MLP per domain (attention frozen, 200 steps)
3. Profile per-capsule dead/alive set in each single-domain model
4. Compose by concatenating A and B weight matrices
5. Profile per-capsule dead/alive set in composed model
6. Compare: Jaccard, overlap coefficient, decomposition by source
7. Controls: cross-domain profiling, composed-on-single-domain profiling

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> relu_router -> dead_capsule_pruning -> pruning_controls -> capsule_revival -> capsule_identity
                               (composition    (activation-based      (pre-composition       (per-capsule       (identity across
                                by concat)      dead pruning)          death rate +            identity over      single vs
                                                                       random baseline)        training time)     composed)
```

---

## Key References

**Exp 10 (pruning_controls)**: Single-domain death is 54.3% (>45%
threshold). 87% of composed death is training-induced, not
composition-specific. Our experiment confirms this at per-capsule
identity level.

**Exp 18 (capsule_revival)**: Jaccard(dead_100, dead_3200) = 0.669
for the same model across training time. Our cross-setting Jaccard
(0.895) is substantially HIGHER, meaning composition reshuffles death
identity LESS than extended training does.

**Gurbuzbalaban et al. (2024)**: >90% of revived neurons re-die. Our
finding of only ~4 capsules revived by composition (vs ~29 killed) is
consistent: the perturbation from composition is more likely to kill
borderline-alive capsules than revive dead ones.

---

## Empirical Results

### 3-Seed Aggregate (seeds 42, 123, 7)

#### Death Rates

| Setting | Death Rate | Std |
|---------|-----------|-----|
| Single-domain (a_m) | 58.5% | 4.7% |
| Single-domain (n_z) | 57.2% | 7.1% |
| Composed (joint data) | 62.9% | 4.6% |

#### Key Metric: Death Identity Overlap

| Comparison | Jaccard | Std | Overlap Coeff | Std |
|------------|---------|-----|---------------|-----|
| a_m single vs composed-half | 0.897 | 0.011 | 0.983 | 0.011 |
| n_z single vs composed-half | 0.894 | 0.062 | 0.989 | 0.005 |
| **Combined (union of single dead sets)** | **0.895** | **0.033** | **0.986** | **0.008** |
| a_m single vs composed (own data) | 0.909 | 0.028 | 0.993 | 0.009 |
| n_z single vs composed (own data) | 0.899 | 0.061 | 0.992 | 0.004 |
| a_m own-data vs cross-data (control) | 0.916 | 0.022 | 0.972 | 0.005 |
| n_z own-data vs cross-data (control) | 0.926 | 0.021 | 0.966 | 0.011 |

The combined Jaccard of 0.895 means that 89.5% of the union of dead
capsules across settings is shared. The overlap coefficient of 0.986
means 98.6% of the smaller dead set (single-domain) is contained in
the larger set (composed).

Control comparison: profiling the same single-domain model on
own-data vs cross-data gives Jaccard 0.916-0.926. Our single-vs-composed
Jaccard (0.895) is slightly lower, indicating composition adds a
small but detectable perturbation beyond what different input data
alone would cause.

#### Death Source Decomposition (per domain half, means)

| Category | a_m | n_z |
|----------|-----|-----|
| Dead in BOTH settings | 294.7 | 289.7 |
| Dead ONLY in single (revived by composition) | 5.0 | 3.3 |
| Dead ONLY in composed (killed by composition) | 28.7 | 30.7 |
| Alive in BOTH settings | 183.7 | 188.3 |

Composition kills ~29 capsules per domain half that were alive in
single-domain (6% of pool). These are borderline capsules pushed
past the ReLU boundary by the perturbation from the other domain's
capsule residuals.

Composition revives only ~4 capsules per domain -- a 7:1 asymmetry
(kills >> revivals). This is because the perturbation from uncorrelated
domain signals is more likely to push a barely-alive capsule negative
than to push a dead capsule positive.

#### Per-Layer Jaccard

| Layer | a_m Jaccard | Std | n_z Jaccard | Std |
|-------|-------------|-----|-------------|-----|
| 0 | 0.800 | 0.346 | 0.944 | 0.096 |
| 1 | 0.928 | 0.012 | 0.910 | 0.048 |
| 2 | 0.913 | 0.013 | 0.919 | 0.053 |
| 3 | 0.863 | 0.040 | 0.849 | 0.117 |

Layer 0 has high variance because it has very few dead capsules (0-6),
so single capsule changes cause large Jaccard swings. Layers 1-3
show consistently high Jaccard (0.849-0.928). Layer 3 has the lowest
mean Jaccard, consistent with Exp 18's finding that deeper layers
are more susceptible to input distribution changes.

---

## Kill Threshold Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Combined Jaccard < 0.50 | 0.895 | <0.50 | **PASS** |
| Min per-domain Jaccard (any seed) | 0.823 | | Comfortable margin |
| Max per-domain Jaccard (any seed) | 0.938 | | |

**0 of 1 kill criterion triggered.**

---

## Key Findings

### Finding 1: Death Identity Is Preserved Across Composition (J=0.895)

The same capsules that die during single-domain training remain dead after
composition. This confirms Exp 10's aggregate finding (87% training-induced)
at the per-capsule identity level. The Jaccard of 0.895 means only ~10%
of the dead-set union differs between settings.

### Finding 2: Composition Kills More Than It Revives (29 vs 4)

Composition introduces ~29 newly-dead capsules per domain half while
reviving only ~4. The 7:1 kill-to-revival asymmetry occurs because the
additive perturbation from the other domain's capsule residuals is more
likely to push borderline-alive capsules negative (killing them) than
to push dead capsules positive (reviving them). Dead capsules have
a_i^T x << 0; borderline-alive have a_i^T x ~ 0+.

### Finding 3: Cross-Setting Overlap > Cross-Time Overlap

Jaccard across settings (0.895) is substantially higher than Jaccard
across time within the same setting (0.669 from Exp 18, comparing
S=100 and S=3200). This means composition reshuffles death identity
LESS than 3100 steps of continued training does. The perturbation from
composition (adding another domain's capsule residuals) is smaller in
effect than the accumulated weight updates from extended training.

### Finding 4: Pre-Composition Profiling Is Sufficient

Since 98.6% of single-domain dead capsules remain dead after composition
(overlap coefficient), profiling can be done before composition. This
enables a more efficient pruning protocol:
1. Profile each single-domain model independently
2. Prune dead capsules from each domain's capsule pool
3. Compose the already-pruned models
4. Skip post-composition profiling (saves a calibration pass)

The only risk: ~29 capsules per domain (~6%) that are alive in single-domain
but dead in composed would be missed by pre-composition profiling. These
represent a 6% missed-pruning opportunity, not a quality risk.

### Finding 5: Layer 3 Shows Weakest Identity Preservation

Layer 3 has the lowest mean Jaccard (0.849 for domain A, 0.863 for B).
This is consistent with Exp 18's finding that layer 3 has the highest
revival rate, and with the inter-layer coupling hypothesis: deeper layers
receive inputs processed through all upstream layers, making them more
sensitive to any perturbation (including composition's additive residual).

---

## Micro-Scale Limitations

1. **N=2 domains only.** At N=5+, the perturbation from other domains'
   capsule residuals is larger (more additive terms). Jaccard may decrease
   as N increases. The N=2 result is a best-case for overlap preservation.

2. **Small model (d=64, P=128).** With larger hidden dimensions, the
   margin between dead/alive boundaries may change. Larger models may
   have wider margins (harder to perturb past boundary) or narrower
   margins (more sensitive to composition).

3. **Only 3 seeds.** The per-domain Jaccard ranges from 0.823 to 0.938,
   suggesting moderate seed sensitivity. More seeds would narrow confidence
   intervals.

4. **Binary profiling at f=0 threshold.** "Nearly dead" capsules
   (0 < f < 0.01) are classified alive. A more nuanced frequency-based
   comparison might reveal subtler composition effects on borderline capsules.

5. **Constant LR training.** Warmup+cosine (Exp 19/20) reduces death to
   ~20%. At lower death rates, the dead set may be more stable (survivors
   are more robustly alive), potentially increasing Jaccard further.

---

## What Would Kill This

### At Micro Scale (tested)

- **Low identity overlap (J < 0.50)**: NOT KILLED. J = 0.895. The same
  capsules die in both settings.

### At Macro Scale (untested)

- **N-domain scaling.** At N=5 or N=20 domains, the additive perturbation
  from all other domains' capsule residuals grows linearly. If the
  perturbation magnitude exceeds capsule margins, Jaccard could drop
  below 0.50. Test: profile N=5 composition identity.

- **Different domains.** Our two domains (names a-m vs n-z) are very
  similar (character-level name generation). Real domains (code vs prose
  vs math) may produce more different input distributions, increasing
  composition perturbation.

- **SiLU activations.** SiLU has no dead neurons (Exp 15: 0% at safe
  thresholds). The entire framework of "dead capsule identity" does not
  apply to SiLU-based models. Macro models using SiLU need alternative
  compression strategies.

---

## Implications for the Project

### Revised Pruning Protocol (Optimized)

**Previous protocol** (from Exp 9):
1. Compose domain models by concatenation
2. Profile composed model on joint data
3. Prune dead capsules
4. Calibrate router

**Revised protocol** (enabled by this finding):
1. Profile each single-domain model independently (parallelizable)
2. Prune dead capsules from each domain's pool (pre-composition)
3. Compose the already-pruned models
4. Calibrate router

Benefits:
- Profiling is parallelizable (each contributor profiles their own model)
- No joint data needed for profiling (only for calibration)
- Smaller composed model (fewer capsules to route through)
- Only ~6% missed pruning opportunity vs post-composition profiling

### Connection to Previous Experiments

| Experiment | Metric | Value | Meaning |
|------------|--------|-------|---------|
| Exp 10 (aggregate) | Training-induced death fraction | 87% | Most death is from training |
| Exp 18 (temporal) | Jaccard across time (S=100 vs S=3200) | 0.669 | Death evolves significantly over training |
| **Exp 16 (cross-setting)** | **Jaccard across composition** | **0.895** | **Death identity stable under composition** |

The narrative: death identity is more stable across composition (J=0.895)
than across training time (J=0.669). Composition is a smaller perturbation
to death identity than continued training. This is because composition
only adds an additive residual from other domains' capsules, while
training modifies the weights of every alive capsule through backpropagation.
