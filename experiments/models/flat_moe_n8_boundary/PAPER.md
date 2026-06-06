# Flat MoE N=8 Boundary: Research Digest (Revised v3)

## Hypothesis

Flat MoE composition at N=8 domains (the extrapolated safe limit from N=5
identity scaling) will produce acceptable composition quality (gap <10%)
and identity preservation (combined Jaccard >=0.60) using the standard
protocol: concatenation + calibration.

**Falsifiable**: Composition gap >10% at N=8 OR combined Jaccard <0.60.

**Result: PARTIAL KILL.** Composition gap passes (+7.39%, threshold 10%).
Combined Jaccard fails (0.575, threshold 0.60). 1 of 2 kill criteria
triggered.

**Revision v3 note:** This revision reconciles all numbers with results.json
(ground truth). The central attribution claim is corrected: post-calibration
Jaccard falls BELOW 0.60 at ALL tested N values (N=2: 0.588, N=5: 0.613,
N=8: 0.575). The Jaccard kill is therefore a **universal calibration artifact**,
not an N>5-specific phenomenon. The 0.60 threshold is miscalibrated for
post-calibration measurement. Additionally, the gap pass at N=8 is marginal:
mean 7.39% but seed 42 reaches 10.26%, exceeding the 10% threshold.

---

## What This Experiment Is

This is the first experiment to combine **composition quality measurement**
(gap vs joint training) with **capsule identity tracking** (Jaccard) at
N=8 domains using the full flat MoE protocol. The revision extends the
measurement to N=2 and N=5 for proper attribution.

Prior experiments measured these separately:
- **n8_identity_boundary**: Measured Jaccard pre-calibration (J=0.800, PASS)
- **combined_n5_scaling**: Measured composition gap at N=5 (+3.32%, PASS)

This experiment measures BOTH metrics at N=2, N=5, and N=8, and critically,
measures Jaccard **post-calibration** -- the realistic scenario where
calibration modifies the capsule weights before deployment.

Protocol (applied identically at each N):
1. Pretrain base on ALL data (300 steps, d=64, P=128)
2. Fine-tune MLP per domain (attention frozen, 200 steps)
3. Joint training baseline: N*200 steps on round-robin mixed data
4. Compose all N domains by concatenating weight matrices
5. Calibrate on mixed-domain data (100/200/300 steps for N=2/5/8)
6. Profile capsule identity post-calibration
7. Evaluate composition gap vs joint training
8. 3 seeds (42, 123, 7)

---

## Lineage in the Arena

```
gpt -> relu_router -> dead_capsule_pruning -> capsule_identity -> n5_identity_scaling -> n8_identity_boundary -> flat_moe_n8_boundary
                                                 (N=2, J=0.895)     (N=5, J=0.792)       (N=8, J=0.800)         (N=8, J=0.575, gap +7.39%)
```

---

## Key References

**n8_identity_boundary**: Proved combined Jaccard = 0.800 at N=8
pre-calibration. Degradation is sublinear (~0.014/domain). Revised safe
limit to ~N=15. This experiment shows that post-calibration Jaccard is
much lower (0.575), meaning calibration itself destroys identity signal.

**combined_n5_scaling**: Measured composition gap at N=5 with parallel+
pure-linear architecture (+3.32%). Our N=8 gap (+7.39%) continues the
increasing trend.

**sequential_freeze_graft**: Confirmed flat MoE is the only viable N>2
composition strategy. Sequential grafting degrades 3.65x at N>2.

**gurbuzbalaban-neural-death**: >90% of revived neurons re-die, consistent
with calibration creating additional death that dominates the Jaccard drop.

---

## Empirical Results

All numbers below are from results.json (ground truth, revision v2_post_cal_jaccard_at_all_N).

### Post-Calibration Jaccard Across N (3 seeds each)

This is the key measurement. All three N values use the same protocol:
pretrain, fine-tune, compose, calibrate, then profile Jaccard.

| N | Cal Steps | J mean | J std | Gap mean | Gap std | Death% |
|---|-----------|--------|-------|----------|---------|--------|
| 2 | 100 | 0.588 | 0.069 | +6.13% | 0.58% | 80.8% |
| 5 | 200 | 0.613 | 0.030 | +6.04% | 1.93% | 87.4% |
| 8 | 300 | 0.575 | 0.057 | +7.39% | 2.49% | 92.5% |

**Critical observation:** Post-calibration Jaccard is below 0.60 at N=2
(0.588) and N=8 (0.575), and barely above at N=5 (0.613). The 0.60
threshold is breached at 2 of 3 tested N values. This means the Jaccard
kill is NOT N>5-specific -- it is a **universal calibration artifact**.
Calibration destroys identity tracking at all tested N values. The 0.60
threshold, which was set based on pre-calibration measurements, is
miscalibrated for the post-calibration regime.

### Comparison with Pre-Calibration Jaccard

| N | Pre-cal Jaccard | Post-cal Jaccard | Drop |
|---|----------------|-----------------|------|
| 2 | 0.895 | 0.588 | -0.307 |
| 5 | 0.792 | 0.613 | -0.179 |
| 8 | 0.800 | 0.575 | -0.225 |

Calibration drops Jaccard by 0.18-0.31 across all N, with no clear
monotonic relationship between N and the magnitude of the drop.

### Kill Threshold Analysis (N=8)

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Composition gap (calibrated) | +7.39% | >10% | **PASS (marginal)** |
| Combined Jaccard (post-cal) | 0.575 | <0.60 | **KILL** |

**1 of 2 kill criteria triggered. VERDICT: PARTIAL KILL.**

**Gap pass is marginal.** Mean gap is 7.39% (2.61% margin to threshold),
but seed 42 produces a gap of 10.26%, exceeding the 10% threshold.
With only 3 seeds and std=2.49%, the pass is not robust. A single
unlucky initialization can push the gap over threshold.

### N=8 Quality Summary (3-seed aggregate)

| Condition | Avg Val Loss | Std |
|-----------|-------------|-----|
| Joint training | 0.504 | -- |
| Calibrated composed | 0.541 | -- |
| Zero-shot composed | 1.062 | -- |
| Single-domain specialist | 0.440 | -- |

(Values from n8_aggregate in results.json: joint_mean=0.504, cal_mean=0.541,
zero_mean=1.062, spec_mean=0.440.)

### N=8 Composition Gap Per Seed

| Seed | Joint | Calibrated | Gap | Jaccard |
|------|-------|-----------|-----|---------|
| 42 | 0.503 | 0.554 | +10.26% | 0.638 |
| 123 | 0.502 | 0.531 | +5.75% | 0.527 |
| 7 | 0.507 | 0.538 | +6.17% | 0.561 |
| **Mean** | **0.504** | **0.541** | **+7.39%** | **0.575** |

Note: Seed 42 gap (10.26%) exceeds the 10% kill threshold individually.
The mean passes only because seeds 123 and 7 are well below threshold.

### N=8 Per-Domain Statistics

Min per-domain Jaccard: 0.451 (from results.json min_per_domain_jaccard).
Max per-domain Jaccard: 0.685 (from results.json max_per_domain_jaccard).
Overlap mean: 0.986 (from results.json overlap_mean).

### Death Rate

| N | Death Rate (post-cal) |
|---|----------------------|
| 2 | 80.8% |
| 5 | 87.4% |
| 8 | 92.5% |

Death rate increases with N as expected from the overcompleteness ratio
(P_comp/d = N*128/64 = 2N). Post-calibration, the model learns that most
capsules in the overcomplete pool are redundant for the mixed objective.

---

## Key Findings

### Finding 1: Calibration Universally Destroys Identity Tracking

The central finding: post-calibration Jaccard falls below the 0.60 threshold
at N=2 (0.588) and N=8 (0.575), and is only marginally above at N=5 (0.613).
The Jaccard kill is a **universal calibration artifact**, not an N-scaling
phenomenon. Any pipeline that profiles capsule identity before calibration
and expects those profiles to remain valid after calibration will face
unreliable identity tracking at ALL N values.

This is actually a more useful finding than an N>5-specific kill: it tells us
the entire pre-composition profiling pipeline needs to account for calibration's
identity cost, regardless of the number of composed domains.

### Finding 2: Composition Gap is Moderate but Marginal at N=8

The composition gap increases from ~6% at N=2,5 to ~7.4% at N=8. This is
within the 10% threshold on average, but seed 42 (10.26%) exceeds it.
With only 3 seeds and high variance (std=2.49%), the gap pass at N=8 is
marginal and should not be treated as a confident result.

### Finding 3: Zero-Shot Composition Degrades with N

Zero-shot gap increases sharply with N. At N=8, zero-shot composed loss
(1.062) is 110.8% above joint training (0.504). Calibration is mandatory
at N>=5.

### Finding 4: Profiling Distribution Mismatch

Single-domain models are profiled on their **own domain-specific data**,
while the composed model is profiled on **joint (mixed-domain) data**.
This distribution mismatch contributes to the Jaccard drop independently
of calibration weight reshuffling.

A capsule that is marginally alive on domain-specific data (where that
domain's inputs are concentrated) can appear dead on the joint mixture
(where the domain's inputs are diluted by 1/N). This means some of the
Jaccard drop reflects the change in profiling distribution, not a change
in the model's internal representations.

The overlap coefficient (0.986 at N=8) provides evidence on the direction
of the mismatch: 98.6% of capsules that are dead in single-domain
profiling remain dead after composition+calibration. The Jaccard drop is
dominated by **new deaths in the composed model** (capsules alive in
single-domain but dead in composed), not by revivals (capsules dead in
single-domain but alive in composed). This is consistent with the
distribution dilution effect creating additional apparent deaths.

---

## Micro-Scale Limitations

1. **Calibration budget is ad hoc.** The scaling 100/200/300 steps for
   N=2/5/8 is a linear heuristic. The optimal calibration budget may
   scale sublinearly or superlinearly with N. More calibration steps
   would likely reduce the composition gap but further destroy identity.

2. **High death rate is partly an artifact of P/d ratio.** At micro
   scale, P=128 and d=64 give P/d=2 per domain. At N=8 composed,
   P/d=16. At macro scale (P=11008, d=4096), P/d=2.7, so the
   overcomplete death effect would be much less severe.

3. **Toy domains.** 8 alphabetic letter ranges are far less diverse
   than 8 real programming languages or knowledge domains.

4. **Only 3 seeds.** Per-seed Jaccard at N=8 ranges from 0.527 to 0.638
   and per-seed gap ranges from 5.75% to 10.26%. More seeds would
   clarify the distributions. The gap pass is marginal with this
   sample size.

5. **Joint training baseline uses N*P capsules.** The joint model has
   N*128 capsules trained jointly, matching the composed model size.
   At macro scale, the comparison would be against a fixed-size model.

6. **Different domain splits at different N.** N=2 uses binary (a-m/n-z),
   N=5 uses quintary (5 ranges), N=8 uses octonary (8 ranges). The
   domains are not nested subsets, so the N-comparison has a confound
   from domain boundary effects. However, the consistent protocol
   across N mitigates this.

7. **Jaccard threshold may be miscalibrated.** The 0.60 threshold was
   set based on pre-calibration measurement intuitions. Post-calibration
   Jaccard consistently falls below or near this threshold at all N,
   suggesting the threshold itself needs recalibration for the
   post-calibration regime.

---

## What Would Kill This

### At Micro Scale (tested)

- **Composition gap >10%**: NOT KILLED (on average). Mean gap = +7.39%
  (2.61% margin). However, seed 42 exceeds threshold (10.26%). The pass
  is marginal with 3 seeds and std=2.49%.
- **Combined Jaccard <0.60**: KILLED. Jaccard = 0.575 (0.025 below
  threshold). But this kill applies UNIVERSALLY: N=2 Jaccard (0.588) is
  also below 0.60. The kill is a calibration artifact, not N-specific.

### At Macro Scale (untested)

- **P/d ratio at macro.** The micro-scale overcomplete death effect
  (92.5% at P/d=16) may not reproduce at macro (P/d=2.7). Post-cal
  Jaccard at macro could be higher.

- **Calibration budget at macro.** The linear scaling heuristic has
  no theoretical backing. At macro scale with real domain diversity,
  calibration dynamics could differ qualitatively.

- **Insufficient calibration at N>8.** The gap trend may not hold at
  macro scale where domain interference is stronger.

---

## Implications for the Project

### Calibration Universally Costs Identity

The key takeaway: calibration is mandatory for quality (zero-shot is
catastrophic at N>=5), but calibration destroys identity tracking at ALL
tested N values. This is a fundamental tension in the pipeline:

- Pre-composition profiling requires stable identity (Jaccard >= 0.60)
- Calibration disrupts identity (post-cal Jaccard < 0.60 at N=2 and N=8)
- Calibration is necessary for quality (gap without calibration: >100% at N=8)

The pipeline must either:
1. **Profile AFTER calibration** (post-composition profiling), losing the
   parallelization advantage
2. **Develop calibration-aware profiling** that predicts post-calibration
   dead sets from pre-calibration profiles
3. **Recalibrate the Jaccard threshold** for the post-calibration regime,
   accepting that ~0.55-0.61 is the normal range

### Gap Pass is Real but Marginal at N=8

The composition gap (+7.39% mean) passes the 10% threshold, but not by a
comfortable margin. With seed 42 at 10.26%, approximately 1 in 3
initializations may exceed the threshold. For production use, either the
calibration budget or protocol needs hardening.

### Connection to Previous Experiments

| Experiment | What it tested | Result |
|------------|---------------|--------|
| capsule_identity (N=2) | Pre-cal identity | J=0.895 (PASS) |
| n5_identity_scaling (N=5) | Pre-cal identity | J=0.792 (PASS) |
| n8_identity_boundary (N=8) | Pre-cal identity | J=0.800 (PASS) |
| prune_before_compose (N=2) | Pre-comp pruning | +0.01% (PASS) |
| combined_n5_scaling (N=5) | Composition gap | +3.32% (PASS) |
| **this experiment (N=2,5,8)** | **Gap + post-cal identity** | **Gap PASS (marginal), J KILL (universal)** |

The key insight: pre-calibration identity experiments (J=0.800 at N=8)
painted a rosier picture than reality. Once calibration is applied
(mandatory for quality at N>=5), identity tracking degrades universally,
not just at high N. The entire pre-composition profiling approach needs
to account for this cost.
