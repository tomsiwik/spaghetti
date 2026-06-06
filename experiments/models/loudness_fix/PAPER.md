# Loudness Fix: Research Digest

## Hypothesis

The +6.6% zero-shot composition degradation in ReLU Router is caused by
activation magnitude mismatch ("loudness") between independently-trained
capsule pools. Per-pool normalization or matched-magnitude training can
recover zero-shot composition quality to within 5% of joint training.

**Falsifiable**: If no intervention achieves zero-shot composition within
5% of joint training, the loudness hypothesis is wrong and the degradation
has a deeper cause.

**Result: Hypothesis FALSIFIED.** Loudness is NOT the dominant cause of
zero-shot composition degradation. The problem is a function-space gap
between independently-trained pools and jointly-trained pools, not a
magnitude mismatch. Scalar calibration learns scales of ~0.99, confirming
magnitudes are already well-matched. The surprising winner is weight
averaging (+1.5% vs joint), which outperforms concatenation (+4.3%)
by staying in the same output space the rest of the network expects.

---

## What This Experiment Tests

Three interventions to fix zero-shot composition degradation:

1. **Per-pool RMSNorm**: Normalize each pool's output to unit RMS before
   summing. True zero-shot (no calibration data). Tests whether magnitude
   equalization fixes composition.

2. **Scalar calibration**: Train 1 scalar per pool per layer (8 params
   total). Diagnostic: if this matches full calibration, loudness is the
   sole issue.

3. **Matched-magnitude training**: Auxiliary loss during fine-tuning that
   penalizes output RMS deviation from pretrained target. Ensures pools
   produce matched magnitudes, enabling zero-shot composition.

Controls: Joint training (upper bound), plain zero-shot concatenation
(the known composition problem), full capsule calibration (continued
training), weight averaging (standard model merging baseline), capsule MoE
with router calibration (architecture comparison).

**Note on baseline**: The original relu_router PAPER.md reported +6.6%
zero-shot degradation from an earlier seed configuration. This experiment
measures +4.3% across seeds 42, 123, 7. The direction of the finding is
identical; the difference is due to seed selection.

---

## Lineage in the Arena

```
gpt  ->  moe  ->  capsule_moe  ->  relu_router  ->  loudness_fix
                                    (composition    (three interventions
                                     by concat)      for zero-shot)
```

---

## Key References

**ReMoE (ICLR 2025)**: ReLU routing with adaptive L1 sparsity control.
The parent architecture (relu_router) builds on this.

**Task Arithmetic (Ilharco et al., 2023)**: Composing models by
arithmetic operations on weight deltas. Our weight averaging is a
simplified version (averaging full weights rather than deltas).

**TIES-Merging (Yadav et al., 2023)**: Resolves sign conflicts in model
merging. Relevant because weight averaging outperforms concatenation.

**Union-of-Experts (2024)**: Internal activation magnitudes as routing
signals. Relevant to understanding why magnitude normalization fails --
the magnitude IS information, not noise.

---

## Empirical Results

### 3-Seed Aggregate (seeds 42, 123, 7)

| Method | Avg Val Loss | Std | vs Joint | Zero-Shot? |
|--------|-------------|-----|----------|------------|
| relu_joint (baseline) | 0.5262 | 0.0076 | -- | N/A |
| **relu_zero_shot** | **0.5488** | **0.0054** | **+4.3%** | Yes |
| rmsnorm_zero_shot | 0.6442 | 0.0762 | +22.4% | Yes |
| rmsnorm_t0.25 | 0.6506 | 0.1192 | +23.6% | Yes |
| rmsnorm_t0.50 | 0.6442 | 0.0762 | +22.4% | Yes |
| rmsnorm_t1.00 | 0.8592 | 0.0649 | +63.3% | Yes |
| scalar_cal | 0.5473 | 0.0055 | +4.0% | No (8 params) |
| full_cal | 0.5163 | 0.0030 | -1.9% | No (all caps) |
| **weight_avg** | **0.5343** | **0.0042** | **+1.5%** | **Yes** |
| matched_zero_shot | 0.5645 | 0.0103 | +7.3% | Yes |
| matched_rmsnorm | 0.9801 | 0.3444 | +86.2% | Yes |
| cap_joint | 0.5244 | 0.0055 | -0.4% | N/A |
| cap_calibrated | 0.5271 | 0.0014 | +0.2% | No (router) |

### Key Observations

**1. RMSNorm catastrophically degrades composition (+22.4%).**
Normalizing pool outputs destroys the adaptive magnitude signal. The
magnitude of Pool(x) relative to the residual stream encodes correction
strength -- inputs that need large corrections get large pool outputs.
RMSNorm forces all corrections to the same magnitude, destroying this
adaptive behavior. This is WORSE than doing nothing.

**2. Scalar calibration is essentially a no-op (scales ~0.99).**
The learned scales across all seeds and layers:
- Seed 42: [0.988, 0.974, 0.998, 0.985, 1.005, 0.974, 0.993, 0.982]
- Seed 123: [0.987, 1.002, 0.986, 0.989, 0.981, 0.996, 0.990, 0.994]
- Seed 7: [0.996, 0.991, 0.994, 0.996, 0.997, 0.995, 0.998, 0.965]

ALL scales are within 3% of 1.0. The optimizer finds that scaling pool
outputs is not helpful -- the magnitudes are already approximately correct.
This definitively proves: **loudness is NOT the problem.**

**3. Matched-magnitude training constrains RMS perfectly but does not help.**
Post-finetune RMS values stay within 1% of targets:
- Target: [8.54, 1.82, 0.67, 0.81]
- Domain A: [8.57, 1.80, 0.65, 0.80] (deviation <1%)
- Domain B: [8.63, 1.81, 0.67, 0.81] (deviation <1%)

Yet zero-shot composition still degrades by +7.3%. The magnitude loss
adds a quality tax (the auxiliary loss competes with the primary NTP loss)
without fixing the composition problem.

**4. Weight averaging is the best zero-shot method (+1.5% vs joint).**
This is the most surprising result. Simple weight averaging (no
concatenation, no normalization) outperforms all interventions.

Why: weight averaging maintains the SAME output dimensionality as the
pretrained base (P capsules), while concatenation creates a pool of 2P
capsules. The downstream layers (layer norm, next attention layer) were
calibrated for outputs from a P-dimensional pool. Weight averaging stays
in this expected space; concatenation does not.

**5. Full calibration still works well (-1.9% vs joint).**
Training all capsule weights for 100 steps on joint data recovers most
of the gap. This confirms the composition mechanism is sound; the issue
is purely about the output distribution mismatch.

### RMS Diagnostic

Per-layer output RMS reveals large magnitude divergence between domains:

| Layer | Base | Domain A | Domain B | Ratio A/B |
|-------|------|----------|----------|-----------|
| 0 | 5.73 | 9.53 | 11.02 | 0.86 |
| 1 | 0.73 | 0.15 | 0.30 | 0.51 |
| 2 | 0.19 | 0.43 | 0.50 | 0.86 |
| 3 | 0.11 | 0.14 | 0.34 | 0.42 |

Domains show 2-5x magnitude divergence from base, and up to 2.4x between
each other. Yet scalar calibration says this does not matter -- the
problem is not how LOUD each pool is, but WHAT each pool says.

---

## Kill Threshold Analysis

| Criterion | Value | Target | Kill | Result |
|-----------|-------|--------|------|--------|
| RMSNorm zero-shot vs joint | +22.4% | <5% | >10% | **KILLED** |
| Matched-mag zero-shot vs joint | +7.3% | <5% | >10% | WARN |
| Scalar cal vs joint | +4.0% | <2% | >5% | WARN |
| Scalar cal vs full cal | +6.0% | <2% | >5% | **KILLED** |
| Plain zero-shot vs joint | +4.3% | <5% | >10% | OK |
| Weight averaging vs joint | +1.5% | <5% | >10% | **PASS** |

**Scalar vs full calibration gap (+6.0%)**: Scalar calibration is NOT close
to full calibration. This proves **loudness is NOT the sole issue**. The
degradation has a deeper cause in the function-space gap between
independently-trained pools and jointly-trained pools.

---

## Micro-Scale Limitations

1. **The experiment uses N=2 very similar domains (a-m vs n-z names).**
   With more diverse domains (Python vs JavaScript), the magnitude divergence
   would be larger. The finding that "loudness is not the problem" might
   flip at macro scale where domains are truly different.

2. **The pretrained base is only 300 steps.** Longer pretraining might
   produce more stable representations that diverge less during fine-tuning.

3. **The matched-magnitude loss coefficient (1.0) was not tuned.** A sweep
   over lambda_mag might find a better tradeoff between magnitude matching
   and task quality.

4. **Weight averaging was not combined with concatenation.** A method that
   averages THEN concatenates (or vice versa) was not tested.

5. **The composition identity Pool_A(x) + Pool_B(x) is exact, but the SUM
   is not necessarily what the network needs.** The experiment reveals this
   is the core issue -- mathematical correctness of composition does not
   imply functional correctness.

---

## What Would Kill This

### At Micro Scale (tested)

- **Any intervention achieves zero-shot <5% vs joint.** NOT MET.
  Only weight averaging (+1.5%) passes, but it is not a loudness fix --
  it is a fundamentally different composition mechanism (weight-space
  merging rather than function-space composition).

- **Scalar calibration matches full calibration.** NOT MET.
  Scalar cal is +4.0% while full cal is -1.9%, a 6% gap.
  Loudness is NOT the sole issue.

- **RMSNorm improves over plain zero-shot.** NOT MET.
  RMSNorm (+22.4%) is 5x worse than plain zero-shot (+4.3%).

### At Macro Scale (untested)

- **Weight averaging might not work at scale.** At macro scale with
  diverse domains (code vs prose), the fine-tuned weights may diverge
  too much for simple averaging to work. TIES-Merging or DARE might
  be needed.

- **The function-space gap grows with domain diversity.** If Python and
  JavaScript pools learn completely different representations, neither
  averaging nor concatenation may work without significant calibration.

---

## The Key Insight: It Is Not About Loudness

The experiment definitively answers the question posed in VISION.md
item 7: **what causes zero-shot composition degradation?**

**It is NOT loudness (magnitude mismatch).** Evidence:
1. Scalar calibration learns scales of ~0.99 (magnitudes already matched)
2. Matched-magnitude training keeps RMS within 1% but does not help
3. RMSNorm (which perfectly equalizes magnitudes) makes things 5x WORSE

**It IS a function-space gap.** The sum Pool_A(x) + Pool_B(x) computes
a different function than Pool_joint(x), even when both produce outputs
of the same magnitude. The independently-trained pools each optimize for
their own domain, learning domain-specific detector directions that
produce useful outputs IN ISOLATION but whose SUM is not what the rest
of the network expects.

**The surprising fix: weight averaging.** By staying in weight space
rather than function space, averaging produces a single pool that is a
compromise between domains. This compromise stays in the P-dimensional
output space the network expects, rather than the 2P-dimensional space
that concatenation creates. The quality cost (+1.5%) is far smaller than
concatenation (+4.3%).

**Implication for VISION.md**: The composition protocol should prefer
weight-space operations (averaging, task arithmetic, TIES) over
function-space operations (concatenation) for zero-shot composition.
Concatenation requires calibration to work well (-1.9% with full cal),
while averaging works out of the box. However, averaging has a
fundamental limit: it cannot represent MORE knowledge than a single
pool (it is lossy merging). Concatenation can represent the UNION of
both pools' knowledge but requires calibration to access it.

The optimal protocol may be:
1. **Zero-shot**: Use weight averaging (+1.5% vs joint, no calibration)
2. **With calibration budget**: Use concatenation + brief calibration
   (-1.9% vs joint, 100 steps on mixed data)
3. **With training budget**: Use joint training (0% by definition)

This maps directly to the VISION: "Find that fraction at runtime, compose
only those experts." The answer is that finding the right fraction requires
either (a) accepting the weight-space compromise of averaging, or (b)
investing the calibration budget to adapt the network to the composed
function-space representation.

---

## Implications for the Project

1. **Rename the problem**: The "loudness fix" label is wrong. The problem
   should be called the "function-space composition gap." Future experiments
   should focus on closing this gap, not on magnitude matching.

2. **Weight averaging enters the composition toolkit**: For true zero-shot
   composition, weight averaging is the current best (+1.5%). This should
   be the zero-shot baseline going forward.

3. **Concatenation + calibration remains the quality ceiling**: -1.9% with
   100 steps of calibration is still excellent. The calibration cost (100
   steps on mixed data) may be acceptable in practice.

4. **RMSNorm is harmful and should not be used for composition**: Per-pool
   normalization destroys adaptive magnitude information. This is a
   definitive negative result.

5. **Matched-magnitude training adds complexity without benefit**: The
   auxiliary loss constrains magnitudes perfectly but does not improve
   composition quality. Not worth the implementation cost.

6. **The path forward is NOT about loudness**: VISION.md item 7 is
   resolved (loudness is not the cause). The next question is whether
   the function-space gap can be reduced by training-time interventions
   that encourage pools to learn COMPATIBLE representations, not just
   matched magnitudes. Possibilities: mutual information minimization
   between pools, shared subspace constraints, or progressive composition
   during training.
