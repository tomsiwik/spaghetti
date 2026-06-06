# N=5 Identity Scaling: Research Digest

## Hypothesis

Dead capsule identity (which specific capsules are dead) remains preserved
when composing N=5 domain-specialized models, with Jaccard similarity
between single-domain and composed dead sets staying above 0.70.

**Falsifiable**: If combined Jaccard at N=5 drops below 0.70, pre-composition
profiling is unreliable at higher domain counts.

**Result: PASS.** Combined Jaccard at N=5 = 0.792 (above 0.70 threshold).
Identity degrades gracefully: ~0.026 Jaccard per additional domain, from
0.871 (N=2) to 0.792 (N=5). Overlap coefficient remains 0.967, meaning
96.7% of single-domain dead capsules stay dead after composition. However,
per-domain Jaccard can drop as low as 0.640 for individual domains/seeds,
indicating that some domain-seed combinations are near the safety boundary.

---

## What This Experiment Tests

**Q: Does capsule death identity preservation degrade beyond safety limits
when scaling composition from N=2 to N=5 domains?**

Exp 16 proved Jaccard = 0.895 at N=2 domains (binary split: a-m vs n-z).
The adversarial review flagged that the perturbation from composition
scales linearly with N: each additional domain adds its capsule residuals
to the hidden state. This experiment measures the actual degradation
trajectory at N=2, 3, 4, 5 using the quintary split (a-e, f-j, k-o,
p-t, u-z).

Protocol:
1. Pretrain base model on ALL data (300 steps)
2. Fine-tune MLP per domain (attention frozen, 200 steps), 5 domains
3. Profile per-capsule dead/alive set in each single-domain model
4. Compose at N=2,3,4,5 by concatenating weight matrices
5. Profile each composed model on joint validation data
6. Profile N=5 composed model per-domain
7. Compute Jaccard, overlap coefficient, decomposition
8. Repeat for 3 seeds (42, 123, 7)

---

## Lineage in the Arena

```
gpt -> relu_router -> dead_capsule_pruning -> pruning_controls -> capsule_revival -> capsule_identity -> n5_identity_scaling
                                                                                      (N=2, J=0.895)      (N=5, J=0.792)
```

---

## Key References

**Exp 16 (capsule_identity)**: Proved Jaccard = 0.895 at N=2 with combined
overlap coefficient = 0.986. Composition kills ~29 capsules/domain-half (6%)
and revives only ~4. Our experiment confirms this pattern at N=5 with larger
kill counts (~53/domain on average) and still low revival counts (~7/domain).

**Exp 10 (pruning_controls)**: 87% of composed death is training-induced.
At N=5, we find ~85% of composed death is training-induced (overlap coeff
0.967), confirming this holds at higher fan-out.

**MATH.md Section 3**: The perturbation from (N-1) other domains scales
linearly. At N=5, 4x the perturbation of N=2. The measured degradation
(0.079 Jaccard drop over 3 additional domains) is consistent with the
linear perturbation model.

---

## Empirical Results

### 3-Seed Aggregate (seeds 42, 123, 7)

#### Single-Domain Death Rates (quintary split)

| Domain | Death Rate | Std |
|--------|-----------|-----|
| a-e    | 51.4%     | 8.0% |
| f-j    | 46.2%     | 10.1% |
| k-o    | 52.0%     | 11.4% |
| p-t    | 48.4%     | 13.2% |
| u-z    | 44.1%     | 8.9% |

#### KEY METRIC: Combined Jaccard vs N

| N | Jaccard | Std   | Overlap | Std   | Death% | Std  |
|---|---------|-------|---------|-------|--------|------|
| 2 | 0.871   | 0.032 | 0.966   | 0.021 | 52.6%  | 10.6 |
| 3 | 0.853   | 0.045 | 0.977   | 0.020 | 56.1%  | 11.3 |
| 4 | 0.822   | 0.060 | 0.978   | 0.024 | 57.8%  | 11.1 |
| 5 | 0.792   | 0.054 | 0.967   | 0.034 | 57.6%  | 11.4 |

The trajectory is approximately linear: ~0.026 Jaccard per additional domain.
Extrapolating, Jaccard would reach 0.70 at approximately N=8.

#### N=5 Per-Domain Jaccard

| Domain | Jaccard | Std   | Overlap | Std   | Killed by comp | Revived |
|--------|---------|-------|---------|-------|----------------|---------|
| a-e    | 0.806   | 0.039 | 0.958   | 0.036 | 50.3           | 10.0    |
| f-j    | 0.788   | 0.025 | 0.976   | 0.029 | 56.0           | 5.0     |
| k-o    | 0.816   | 0.077 | 0.959   | 0.055 | 45.3           | 9.7     |
| p-t    | 0.776   | 0.121 | 0.978   | 0.017 | 61.7           | 4.7     |
| u-z    | 0.769   | 0.034 | 0.968   | 0.032 | 57.3           | 7.0     |

Composition at N=5 kills ~54 capsules per domain on average (10.5% of pool),
up from ~29 at N=2 (6%). Revival remains low at ~7 per domain. The
kill-to-revival ratio is ~7.7:1, similar to Exp 16's 7:1 at N=2.

The p-t domain shows the highest variance (std=0.121) and lowest minimum
Jaccard (0.640 in seed 123). This domain has intermediate training size
(5,609 names) but the worst seed-to-seed consistency.

#### N=5 Composed on Own-Domain Data

| Domain | Jaccard | Std   | Overlap | Std   |
|--------|---------|-------|---------|-------|
| a-e    | 0.799   | 0.048 | 0.972   | 0.023 |
| f-j    | 0.798   | 0.016 | 0.992   | 0.008 |
| k-o    | 0.829   | 0.057 | 0.978   | 0.032 |
| p-t    | 0.772   | 0.122 | 0.986   | 0.012 |
| u-z    | 0.768   | 0.047 | 0.972   | 0.043 |

When the composed model sees only one domain's data, the Jaccard is similar
to the joint-data profiling. This confirms that the dead set differences are
primarily from the composition perturbation, not from mixed-domain input data.

#### Per-Layer Jaccard at N=5 (mean across all domains)

| Layer | Mean J | Std   |
|-------|--------|-------|
| 0     | 0.767  | 0.400 |
| 1     | 0.774  | 0.156 |
| 2     | 0.825  | 0.063 |
| 3     | 0.756  | 0.046 |

Layer 0 has extreme variance (std=0.400) due to low absolute dead counts.
Layer 3 has the lowest mean Jaccard among stable layers, consistent with
Exp 16 and Exp 18's findings that deeper layers are more sensitive to
perturbation due to inter-layer coupling.

---

## Kill Threshold Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Combined Jaccard at N=5 | 0.792 | <0.70 | **PASS** |
| Min per-domain Jaccard (any seed/domain) | 0.640 | | Below 0.70 |
| Max per-domain Jaccard (any seed/domain) | 0.884 | | Comfortable |

**0 of 1 kill criterion triggered.** However, the per-domain minimum of
0.640 (p-t domain, seed 123) indicates that individual domain-seed
combinations can breach the 0.70 threshold.

---

## Key Findings

### Finding 1: Graceful Linear Degradation (0.026 J/domain)

Jaccard degrades approximately linearly with N: 0.871 (N=2), 0.853 (N=3),
0.822 (N=4), 0.792 (N=5). The rate is ~0.026 per additional domain.
At this rate, the combined Jaccard would cross 0.70 at approximately N=8.
Pre-composition profiling remains safe for practical domain counts (N <= 7).

### Finding 2: Overlap Coefficient Remains Very High (0.967)

While Jaccard drops from 0.966 to 0.792, the overlap coefficient barely
changes (0.966 to 0.967). This means nearly all single-domain dead capsules
remain dead after N=5 composition. The Jaccard drop is entirely from
composition KILLING new capsules (expanding the composed dead set), not
from REVIVING existing dead capsules.

This is critical for pre-composition pruning: you can safely prune dead
capsules identified in single-domain models because they will almost
certainly remain dead after composition. The only cost is missing the
~10.5% of capsules that composition kills additionally.

### Finding 3: Kill-to-Revival Asymmetry Persists at N=5 (7.7:1)

At N=5, composition kills ~54 capsules per domain and revives only ~7.
The kill:revival ratio (7.7:1) is similar to N=2 (7:1). This asymmetry
is robust because the perturbation from uncorrelated domains is more
likely to push borderline-alive capsules negative (kill) than dead
capsules positive (revive).

### Finding 4: Per-Domain Variance Is Significant

Individual domain-seed combinations range from J=0.640 to J=0.884. The
p-t domain shows the worst case (J=0.640 at seed 123) with std=0.121.
For production pre-composition profiling, a safety margin or per-domain
validation step is advisable.

### Finding 5: N=2 Under Quintary Split Matches Exp 16

Our N=2 combined Jaccard under the quintary split (0.871) is close to
Exp 16's binary split (0.895). The slight difference is expected because
the quintary domains are smaller (more data scarcity = less stable
representations = more borderline capsules).

---

## Micro-Scale Limitations

1. **Toy domains.** All 5 domains are character-level name generation,
   split only by first letter. Real domains (code vs prose vs math) have
   more distinct input distributions. This could either help (more distinct
   capsule specialization = less overlap in boundary capsules) or hurt
   (more perturbation magnitude per domain).

2. **Small model (d=64, P=128).** At macro scale (d=4096, P=11008),
   the number of borderline capsules scales differently. Larger models
   may have wider activation margins (more robust to perturbation) or
   proportionally more borderline capsules.

3. **Only 3 seeds.** Per-domain Jaccard has high variance (std up to 0.121).
   The min Jaccard (0.640) could be an outlier or representative. More
   seeds would clarify.

4. **Quintary split has unequal domain sizes.** u-z has only 2,359 names
   (7.4%) while a-e has 10,479 (32.7%). Smaller domains may produce less
   stable models.

5. **N-sweep uses first-N domains.** The N=2 composition is {a-e, f-j},
   not {a-m, n-z}. Different domain subsets might give different trajectories.

---

## What Would Kill This

### At Micro Scale (tested)

- **Combined Jaccard < 0.70 at N=5**: NOT KILLED. J = 0.792.
  However, individual per-domain Jaccard dipped to 0.640 for one
  domain-seed combination, indicating proximity to the boundary.

### At Macro Scale (untested)

- **N > 8 domains.** Extrapolating the linear degradation rate (0.026/domain),
  Jaccard would reach 0.70 at ~N=8. At N=20, it could drop to ~0.4.
  Pre-composition profiling may become unsafe at high domain counts.

- **Highly dissimilar domains.** If Python code capsules and English prose
  capsules have very different input distributions, the composition
  perturbation per domain could be larger, accelerating degradation.

- **SiLU activations.** Same as Exp 16: SiLU has no dead neurons. This
  entire analysis framework does not apply to SiLU-based models.

---

## Implications for the Project

### Pre-Composition Profiling Remains Viable at N=5

With combined Jaccard = 0.792 and overlap coefficient = 0.967, the
pre-composition profiling protocol validated in Exp 16 remains safe at N=5:
1. Profile each single-domain model independently (parallelizable)
2. Prune dead capsules pre-composition
3. Compose the already-pruned models
4. Only ~10.5% missed pruning opportunity vs post-composition profiling

### Safety Margin Recommendation

Given the per-domain minimum of 0.640, production systems should:
- Add a validation step: after composition, spot-check a few layers
  against the pre-composition dead set
- Or use conservative pruning: only prune capsules dead across multiple
  profiling runs (consensus approach from Exp 12)

### Degradation Budget for Scaling

At ~0.026 Jaccard per additional domain, a system can budget:
- N=5: J ~ 0.79 (safe)
- N=8: J ~ 0.71 (marginal)
- N=10: J ~ 0.66 (unsafe, post-composition profiling needed)

This gives concrete guidance for when to switch from pre-composition
to post-composition profiling.

### Connection to Previous Experiments

| Experiment | N | Metric | Value | Meaning |
|------------|---|--------|-------|---------|
| Exp 10 (aggregate) | 2 | Training-induced death | 87% | Most death from training |
| Exp 16 (N=2) | 2 | Combined Jaccard | 0.895 | Identity preserved at N=2 |
| **This (N=5)** | **5** | **Combined Jaccard** | **0.792** | **Identity preserved at N=5** |
| Exp 18 (temporal) | 1 | Jaccard across time | 0.669 | Death evolves more over time |

Key comparison: cross-setting Jaccard at N=5 (0.792) is still higher than
cross-time Jaccard within one model (0.669 from Exp 18). Even at N=5,
composition reshuffles death identity LESS than 3100 steps of continued
training does.
