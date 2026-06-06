# Softmax Collision Quantification: Research Digest

## Hypothesis

Softmax routing collision rate (fraction of tokens with near-tied top-2 expert
scores) increases with expert count N, and temperature scaling can mitigate this
with measurable quality improvement.

**Falsifiable**: (KC1) If collision rate does not increase with N, the scaling
concern is invalidated. (KC2) If the best collision-aware mitigation improves
quality by less than 0.5% at p<0.05, the collision problem is not practically
impactful at micro scale.

---

## What This Model Is

This experiment quantifies the softmax routing "collision" phenomenon first
observed during the cuckoo routing experiment: a large fraction of tokens have
near-tied top-2 scores with standard softmax routing. The collision problem --
where softmax produces ambiguous routing decisions as expert count grows -- is
well-documented in the MoE literature (Switch Transformers load balancing,
DeepSeek-V3 dynamic bias terms, ReMoE's motivation for ReLU routing, the
softmax bottleneck literature). Our contribution is an empirical quantification
of the collision rate scaling law and a head-to-head comparison of simple
mitigations with dual-temperature measurement to separate training from
inference effects.

### How It Works

**Collision measurement**: After training, we pass validation data through
each layer's router and compute the softmax probability distribution at a
specified measurement temperature. For each token, we measure the gap between
the top-1 and top-2 probabilities. A token is "colliding" if this gap is
below threshold epsilon.

**Dual-temperature measurement** (v2): For Phase 2 models trained at
temperature T, we measure collision rates at BOTH T=1.0 (to isolate how
training dynamics affect the learned logit distribution) and at the model's
training temperature (to capture the full inference-time effect). This
separates two sources of collision reduction: (a) the training dynamics
producing inherently sharper logits, and (b) the temperature scaling at
inference time amplifying existing logit differences.

**Temperature scaling**: Replace softmax(s) with softmax(s/T). T < 1
sharpens the distribution (fewer collisions), T > 1 flattens it (more).

**Margin loss**: Add auxiliary hinge loss penalizing tokens where the
top-1/top-2 gap is below a target margin.

### Why It Exists

The cuckoo routing experiment introduced a dual-hash mechanism to resolve
collisions. That mechanism adds 2x routing parameters, -63.6% throughput,
and only +0.15% quality improvement. The underlying finding -- high collision
rates in softmax routing -- motivated this focused quantification study.

---

## Lineage in the Arena

```
gpt
`-- capsule_moe (flat softmax routing, G=8, k=2)
    `-- cuckoo_collision_free_routing (dual-hash, +0.15%)
        `-- softmax_collision_quantification (THIS: scaling law + simple fixes)
```

---

## Key References

**Switch Transformers** (Fedus et al. 2021): Load balancing auxiliary loss
for MoE routing, motivated by routing instability at scale.

**DeepSeek-V3** (2024): Auxiliary-loss-free load balancing via dynamic bias
terms. Demonstrates that auxiliary losses interfere with training -- consistent
with our margin loss finding.

**ReMoE** (Gao et al. 2024, ICLR 2025): Replaces softmax with ReLU routing
precisely because softmax produces ambiguous routing decisions. Direct prior
art for the collision problem statement.

**Softmax Bottleneck** (Yang et al. 2018): Documents how softmax capacity
limitations create indistinguishable outputs as class count grows. The
collision scaling with N is a special case.

**Cuckoo collision-free routing** (this project): Measured 57.4% collision
rate at N=8 with eps=0.05. Our work quantifies the N-scaling of this rate.

---

## Empirical Results

### Phase 1: Collision Rate Scaling with N

3 seeds (42, 123, 777), 500 training steps, k=2, d=64, 4 layers.
Total capsules held constant at ~256 (N * caps_per_group).

| N | Params | Val Loss | C(0.01) | C(0.05) | C(0.10) | Mean Gap | Median Gap |
|---|--------|----------|---------|---------|---------|----------|------------|
| 8 | 204,160 | 0.5163 | 0.209 | 0.610 | 0.736 | 0.1405 | 0.0331 |
| 16 | 206,208 | 0.5155 | 0.370 | 0.715 | 0.776 | 0.1033 | 0.0173 |
| 32 | 210,304 | 0.5154 | 0.606 | 0.780 | 0.818 | 0.0854 | 0.0063 |
| 64 | 218,496 | 0.5222 | 0.733 | 0.817 | 0.856 | 0.0455 | 0.0023 |

#### Key Observations

1. **Collision rate is monotonically increasing with N at all epsilon**.
   At eps=0.01: 20.9% -> 73.3% (3.50x). At eps=0.10: 73.6% -> 85.6% (1.16x).

2. **Median gap collapses**: From 0.033 at N=8 to 0.002 at N=64 (14x decrease).
   The typical token's routing confidence is over an order of magnitude lower
   at N=64.

3. **Quality is stable despite collisions**: Val loss only increases from
   0.5163 to 0.5222 (+1.1%) across 8x increase in N. The model compensates
   for routing ambiguity through other mechanisms (attention, residual stream).

#### Scaling Law (Empirical)

Power law fit: C(epsilon) = a * N^b

| Epsilon | a | b | r^2 | Notes |
|---------|------|-------|------|-------|
| 0.01 | 0.064 | 0.614 | 0.959 | Strong scaling |
| 0.05 | 0.471 | 0.139 | 0.935 | Moderate scaling |
| 0.10 | 0.634 | 0.073 | 0.999 | Near saturation |

**Caveat**: 4 data points for 2 parameters. The r^2 values are mechanically
high. The exponent b ~ 0.6 at tight epsilon characterizes the scaling
direction and approximate rate but should not be taken as a precise prediction.
Extrapolation to N > 64 is speculative.

**Speculative extrapolation** (eps=0.01, for context only):
N=128: ~0.96, N=256+: ~1.0 (near-universal collision). These numbers are
outside the fitted range and carry large uncertainty.

### Phase 2: Collision Mitigation at N=32 (Revised)

**5 seeds** (42, 123, 777, 2024, 314), 500 steps, N=32, k=2, d=64.
**Dual-temperature measurement**: collision rates reported at both T=1.0
(raw logit quality) and training temperature (inference behavior).

| Config | Val Loss | Std | vs Baseline | p-value | C(0.01)@T=1.0 | C(0.01)@train-T |
|--------|----------|-----|-------------|---------|----------------|-----------------|
| baseline T=1.0 | 0.5187 | 0.0079 | --- | --- | 0.589 | 0.589 |
| T=0.5 | 0.5171 | 0.0053 | -0.29% | 0.52 | 0.472 | 0.245 |
| T=2.0 | 0.5166 | 0.0079 | -0.40% | 0.48 | 0.587 | 0.738 |
| margin m=0.1 | 0.5203 | 0.0052 | +0.31% | 0.51 | 0.548 | 0.548 |

#### Dual-Temperature Decomposition

The key methodological improvement in this revision: measuring collision
rates at T=1.0 for ALL models separates training dynamics from inference
temperature effects.

| Config | C@T=1.0 | Training Effect | Inference Effect | Total |
|--------|---------|----------------|------------------|-------|
| baseline | 0.589 | (reference) | (reference) | (reference) |
| T=0.5 | 0.472 | +0.117 (20%) | +0.227 (39%) | +0.344 (58%) |
| T=2.0 | 0.587 | +0.002 (0%) | -0.151 (-26%) | -0.149 (-25%) |
| margin | 0.548 | +0.041 (7%) | +0.000 (0%) | +0.041 (7%) |

**Finding**: T=0.5's collision reduction is approximately 1/3 training
dynamics (the model learns sharper logit distributions) and 2/3 inference
sharpening (dividing by T=0.5 at measurement time). T=2.0 has essentially
zero training effect -- the learned logit quality is indistinguishable
from baseline when measured at T=1.0.

#### T=2.0 Anomaly (v1) -- Resolved

The v1 results showed seed-42 T=2.0 with lower collision rate (0.283) than
baseline (0.631) at tight epsilon, contradicting theory. This occurred because
v1 measured ALL models at T=1.0 regardless of training temperature, and the
single seed happened to learn unusually sharp logits. With 5 seeds and proper
dual-temperature measurement, T=2.0's collision rate at T=1.0 is 0.587 --
indistinguishable from baseline (0.589). The v1 anomaly was a seed-variance
artifact, not a real effect.

#### Key Findings

1. **No mitigation achieves statistically significant quality improvement**.
   The best is T=2.0 at -0.40%, but p=0.48 (far from significant). T=0.5
   at -0.29%, p=0.52. With 5 seeds, the noise floor (~0.8% std) dwarfs all
   measured effects. KC2 is KILLED.

2. **T=0.5 does reduce collision rates substantially** (0.589 -> 0.245 at
   inference temperature), but this does not translate to measurable quality
   improvement at N=32 with 500 training steps on character-level names data.

3. **Margin loss hurts quality while moderately reducing collisions at T=1.0**:
   C drops from 0.589 to 0.548, but quality degrades by +0.31%. Consistent
   with DeepSeek-V3's finding that auxiliary losses interfere with training.

4. **Training at T=2.0 does not damage learned logit quality**. The logits
   measured at T=1.0 are statistically identical to baseline. T=2.0's quality
   is also statistically identical to baseline (p=0.48). The flat routing
   during training does not prevent the router from learning useful
   discriminations -- it just doesn't help either.

### Kill Criteria Assessment

| Criterion | Threshold | Measured | Verdict |
|-----------|-----------|----------|---------|
| KC1: Collision rate increases with N | Must increase | **3.50x at eps=0.01, monotonic** | **PASSES** |
| KC2: Best mitigation >0.5% quality at p<0.05 | >0.5%, p<0.05 | T=0.5: -0.29%, p=0.52 | **KILLED** |

KC1 is well-supported. KC2 is killed: no mitigation achieves statistically
significant quality improvement. The collision problem is real (scaling
with N), but at this micro scale, the model compensates through other
pathways. The quality signal from collision reduction may emerge at larger
N or with more diverse data where routing discrimination matters more.

---

## Micro-Scale Limitations

1. **Homogeneous data**: Character-level names have weak expert specialization.
   At macro scale with diverse domains, trained routers may develop stronger
   preferences, potentially reducing collision rates. The scaling LAW (exponent)
   may differ, even if the direction holds.

2. **Small d=64**: Softmax behavior depends on logit distribution, which
   depends on router weight dimensionality. At d=896 (Qwen 0.5B), logits may
   have different variance characteristics.

3. **N-capsules confound**: Phase 1 holds N * caps_per_group ~ 256 constant,
   meaning N=8 has 32 capsules/group while N=64 has 4 capsules/group. These
   are architecturally different models: 32 capsules/group gives each expert
   a rich mixture of computations, while 4 capsules/group makes each expert
   barely expressive. The collision rate increase with N could be partly
   driven by reduced per-group expressivity (fewer capsules make the router's
   job harder), not just softmax probability compression. A proper control
   would hold capsules_per_group constant and let total params grow.

4. **Temperature is not learned**: We tested fixed temperatures. A learned
   per-layer temperature (with appropriate initialization) could outperform
   the grid search.

5. **Short training**: 500 steps may not be enough for the router to fully
   converge, especially at large N. The collision rate after full convergence
   may be lower (but the scaling trend would persist).

6. **No composition test**: Collision resolution may matter more under the
   composition protocol where experts from different contributors have
   distinct specializations.

7. **Scaling law overfit**: 4 data points for 2 parameters (a, b) in the
   power law fit. The exponent b has no confidence interval. Extrapolation
   beyond N=64 is speculative.

---

## What Would Kill This

### At Micro Scale

- **Collision rate does not increase with N**: SURVIVED. Monotonic increase
  at all three epsilon levels with power law fit (r^2 > 0.93).

- **Quality improvement from mitigation at p<0.05**: KILLED. No mitigation
  achieves statistically significant quality improvement with 5 seeds.

### At Macro Scale (untested)

- **Trained routers at macro scale have clear expert preferences**: If domain
  diversity creates strong routing signals, collision rates may plateau at
  lower values. The scaling exponent b may decrease.

- **Temperature interacts with load balance**: T=0.5 sharpens routing, which
  could concentrate tokens on fewer experts. At macro scale with real load
  balancing constraints, this could cause severe token dropping.

- **Collision rate is an artifact of initialization, not a fundamental limit**:
  If collision rates decrease substantially with longer training, the scaling
  law describes early training, not the converged regime.

- **ReLU routing eliminates the problem entirely**: ReMoE (ICLR 2025) shows
  that ReLU routing avoids softmax's probability compression altogether.
  Temperature tuning may be the wrong approach if the architecture is
  the real bottleneck.

---

## Summary

Softmax routing collision rates **increase monotonically with expert count N**,
following an empirical power law C(eps=0.01) ~ 0.064 * N^0.614 (r^2=0.959,
4 points). At N=64, 73.3% of tokens have top-1 vs top-2 probability gap below
0.01, compared to 20.9% at N=8.

**Temperature scaling T=0.5 reduces collision rates substantially** (0.589 ->
0.245 at inference, decomposed as 1/3 training dynamics + 2/3 inference
sharpening) **but does not translate to statistically significant quality
improvement** at micro scale (p > 0.5 with 5 seeds). KC2 is killed.

The scaling law predicts near-universal collision at N=256+ (at tight epsilon,
with large uncertainty in the extrapolation), suggesting that either
(a) sharp temperature becomes important at scale where routing discrimination
matters more, (b) alternative routing mechanisms (hash-based, ReLU-based)
are needed, or (c) the problem is self-correcting at scale because domain
diversity creates naturally discriminable experts.

**Contribution**: An empirical quantification of collision rate scaling with
N and a dual-temperature decomposition showing that T=0.5's collision
reduction comes primarily from inference-time sharpening (2/3) rather than
learned logit quality (1/3). The collision problem in softmax routing is
well-documented in the literature; our contribution is the specific scaling
measurement and the decomposition methodology.

**Recommended next steps**:
1. Validate collision scaling at macro scale (d=896, real LoRA experts)
2. Compare collision rate under composition protocol (diverse domain experts)
3. Evaluate ReLU routing (ReMoE) as a structural alternative to temperature tuning
