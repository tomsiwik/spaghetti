# Warmup Fraction Sensitivity: Mathematical Foundations

## 1. Problem Statement

Experiment 19 established that 10% warmup (320/3200 steps) eliminates 74%
of the ReLU death spike: constant LR spikes to 51.6% dead at S=50, while
warmup reaches only 13.2%. This yielded a revised macro prediction of ~20%
equilibrium dead neurons under warmup+cosine schedules.

However, 10% warmup is atypically long. Standard LLM training recipes use
much shorter warmup fractions:

| Model | Total steps | Warmup steps | Warmup fraction |
|-------|------------|-------------|-----------------|
| GPT-3 (Brown 2020) | 300K | 375 | 0.125% |
| Chinchilla (Hoffmann 2022) | 1.5M | 5K | 0.33% |
| Llama-2 (Touvron 2023) | 2T tokens | 2K steps | ~0.1% |
| Qwen-2.5 (2024) | ~18T tokens | unknown | likely 0.5-2% |
| Typical LoRA fine-tune | 1K-10K | 10-100 | 1-10% |

The critical question: **What is the minimum warmup fraction that prevents
the ReLU death spike?**

If the answer is >5%, then Exp 19's revised macro prediction (~20% dead)
only holds for LoRA fine-tuning (which uses longer warmup fractions), not
for full pre-training (which uses <1%). If the answer is <2%, then the
prediction holds broadly.

---

## 2. Notation

All notation follows lr_schedule_death/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
S         -- number of fine-tuning steps (varied: 0 to 3200)
S_total   -- total training steps for schedule computation (3200)

eta(s)    -- learning rate at step s (schedule-dependent)
eta_peak  -- peak learning rate (3e-3 in all schedules)

f_w       -- warmup fraction in (0, 1); warmup steps = floor(f_w * S_total)
S_w       -- warmup steps = floor(f_w * S_total)

delta(S)  -- death rate after S steps (fraction of capsules with f_i = 0)

T_spike   -- timescale of death spike (empirically ~50 steps from Exp 17)
```

### 2.1 Warmup Schedule Definition

For warmup fraction f_w, the warmup+cosine schedule is:

```
S_w = floor(f_w * S_total)

eta(s) = eta_peak * (s / S_w)                                   if s <= S_w
       = (eta_peak / 2) * (1 + cos(pi * (s - S_w) / (S_total - S_w)))   if s > S_w
```

The five experimental conditions:

| f_w | S_w | eta(50) | eta(50)/eta_peak | Regime at S=50 |
|-----|-----|---------|-----------------|----------------|
| 0.01 | 32 | 3.0e-3 | 1.00 | PAST warmup |
| 0.02 | 64 | 2.3e-3 | 0.78 | Mid warmup |
| 0.05 | 160 | 9.4e-4 | 0.31 | Early warmup |
| 0.10 | 320 | 4.7e-4 | 0.16 | Early warmup |
| 0.20 | 640 | 2.3e-4 | 0.08 | Early warmup |

---

## 3. Theory: Warmup Fraction and the Death Spike

### 3.1 The Critical Ratio: S_w vs T_spike

From Exp 17, the death spike occurs in the first ~50 steps. This defines
a characteristic timescale T_spike ~ 50 steps. The key insight from Exp 19
is that warmup suppresses the spike by keeping eta(s) << eta_peak during
the critical window s in [0, T_spike].

The suppression depends on the ratio S_w / T_spike:

```
R = S_w / T_spike = f_w * S_total / T_spike
```

For our setup (S_total = 3200, T_spike ~ 50):

| f_w | S_w | R = S_w/T_spike | Expected behavior |
|-----|-----|-----------------|-------------------|
| 0.01 | 32 | 0.64 | Warmup ends DURING spike -> partial suppression |
| 0.02 | 64 | 1.28 | Warmup covers spike -> moderate suppression |
| 0.05 | 160 | 3.2 | Warmup >> spike -> strong suppression |
| 0.10 | 320 | 6.4 | Warmup >> spike -> strong suppression (Exp 19) |
| 0.20 | 640 | 12.8 | Warmup >> spike -> strong suppression |

**Prediction**: There is a phase transition around R ~ 1 (f_w ~ 1.5%).
For R < 1, warmup ends before the spike completes, providing minimal
protection. For R > 1, warmup covers the entire spike window, providing
substantial protection.

### 3.2 Quantitative Death Rate Prediction at S=50

From Exp 19 MATH.md Section 3.1, the death probability per step scales
approximately linearly with eta(s):

```
P(alive -> dead at step s) ~ eta(s) * C
```

where C depends on gradient norms and margins (assumed constant across
schedules for the same architecture).

The cumulative death at S=50 is approximately:

```
delta(50) - delta(0) ~ C * sum_{s=1}^{50} eta(s)
```

For warmup+cosine with fraction f_w, the LR at step s (during warmup) is:

```
eta(s) = eta_peak * s / S_w = eta_peak * s / (f_w * S_total)
```

The cumulative LR integral from s=1 to s=min(50, S_w) is:

```
If S_w >= 50 (warmup covers entire spike window):
  sum_{s=1}^{50} eta(s) = eta_peak / S_w * sum_{s=1}^{50} s
                        = eta_peak / S_w * (50 * 51 / 2)
                        = eta_peak * 1275 / S_w

If S_w < 50 (warmup ends during spike):
  sum = eta_peak / S_w * (S_w * (S_w + 1) / 2)  +  eta_peak * (50 - S_w)
                                                     [warmup portion]         [post-warmup portion]
  sum = eta_peak * ((S_w + 1) / 2 + 50 - S_w)
      = eta_peak * (50 - (S_w - 1) / 2)
```

Normalizing by the constant-LR integral (eta_peak * 50):

```
Relative suppression factor (S_w >= 50):
  F = 1275 / (50 * S_w) = 25.5 / S_w

Relative suppression factor (S_w < 50):
  F = (50 - (S_w - 1) / 2) / 50 = 1 - (S_w - 1) / 100
```

Numerical predictions:

| f_w | S_w | F | Predicted delta(50) |
|-----|-----|---|---------------------|
| 0.00 | 0 | 1.00 | 51.6% (constant baseline) |
| 0.01 | 32 | 0.69 | ~39% |
| 0.02 | 64 | 0.40 | ~28% |
| 0.05 | 160 | 0.16 | ~19% |
| 0.10 | 320 | 0.08 | ~16% (Exp 19: 13.2%) |
| 0.20 | 640 | 0.04 | ~14% |

The model predicts:
- **1% warmup provides only partial protection** (39% vs 51.6% constant)
- **2% warmup provides substantial protection** (28% vs 51.6%)
- **5% warmup provides most of the benefit** (19% vs 13.2% at 10%)
- **Diminishing returns above 5%** (19% -> 14% from 5% to 20%)

### 3.3 Equilibrium Death Rate (S=3200)

The equilibrium death rate depends on both the spike suppression (fewer
neurons killed early) and the revival dynamics (cosine decay phase). We
predict that equilibrium death under warmup+cosine is primarily determined
by two factors:

1. **How many neurons survive the spike**: Lower spike -> more survivors
2. **Revival during cosine decay**: Similar across warmup fractions (same
   cosine phase after warmup ends)

Since the cosine decay phase is similar for all warmup fractions (the
post-warmup LR trajectory converges), the equilibrium difference should
be smaller than the spike difference. But neurons killed during the spike
are harder to revive than neurons that never died, so we expect:

```
delta_eq(f_w=0.01) > delta_eq(f_w=0.02) > ... > delta_eq(f_w=0.20)
```

with the largest gap between f_w=0.01 and f_w=0.02 (crossing R=1).

---

## 4. Experimental Design

### 4.1 Warmup Fractions

Five warmup fractions with cosine decay after warmup:

| Condition | f_w | S_w | R = S_w/T_spike |
|-----------|-----|-----|-----------------|
| wc_01 | 0.01 | 32 | 0.64 |
| wc_02 | 0.02 | 64 | 1.28 |
| wc_05 | 0.05 | 160 | 3.2 |
| wc_10 | 0.10 | 320 | 6.4 |
| wc_20 | 0.20 | 640 | 12.8 |

Plus two controls:
- **constant**: eta(s) = 3e-3 (no warmup, no cosine) -- from Exp 19
- **cosine_only**: cosine decay from 3e-3 (no warmup) -- from Exp 19

### 4.2 Checkpoint Sweep

Same as Exp 19: S in {0, 50, 100, 200, 400, 800, 1600, 3200}

The S=0 baseline is shared across all conditions.

### 4.3 Training Details

Identical to Exp 19:
- Pretraining: 300 steps on all data, constant LR
- Fine-tuning: attention frozen, MLP only
- Profiling: 20 batches x 32 samples on domain validation data
- Seeds: 42, 123, 7

---

## 5. Kill Criteria

| # | Criterion | Threshold | What it means |
|---|-----------|-----------|---------------|
| 1 | All fractions >= 1% within 5pp at S=50 | max - min < 5pp | Warmup fraction does not matter |
| 2 | Critical threshold < 1% (S_w < 32) | f_w=0.01 provides >90% of f_w=0.10 benefit | Question moot for LLM recipes |
| 3 | Non-monotonic: some f_w shows MORE death than smaller f_w | Any inversion | Linear death-vs-LR model wrong |

### Kill criterion 2 formalization

Define spike suppression as:
```
suppression(f_w) = delta_constant(50) - delta_wc(50, f_w)
```

Kill 2 triggers if:
```
suppression(0.01) > 0.90 * suppression(0.10)
```

This means 1% warmup captures >90% of the 10% warmup benefit.

---

## 6. Worked Numerical Example

At d=4, P=4, S_total=100, f_w=0.02 (S_w=2), T_spike~5 steps:

### S=0 (pretrained base)
```
delta(0) = 25% (1/4 capsules dead)
```

### S=5 (during spike window)
```
Warmup covers s in [0, 2]. At s=1: eta = 3e-3 * 1/2 = 1.5e-3
At s=2: eta = 3e-3. At s=3,4,5: cosine from 3e-3.

Cumulative LR = 1.5e-3 + 3e-3 + ~3e-3 + ~3e-3 + ~3e-3 = 13.5e-3
Constant LR = 5 * 3e-3 = 15e-3

Ratio = 0.90 -> warmup barely helps (S_w=2 << T_spike=5)
Expected: delta(5) ~ 50% (similar to constant)
```

### Compare f_w=0.10 (S_w=10):
```
At s=5: eta = 3e-3 * 5/10 = 1.5e-3 (mid-warmup)

Cumulative LR = sum_{s=1}^{5} 3e-3 * s/10 = 3e-3/10 * 15 = 4.5e-3
Constant LR = 15e-3

Ratio = 0.30 -> warmup strongly suppresses spike
Expected: delta(5) ~ 30%
```

This illustrates the phase transition: when S_w < T_spike, warmup
provides negligible protection. When S_w > T_spike, protection is strong.

---

## 7. Assumptions

1. **Same base model and training protocol as Exp 19.** Only the warmup
   fraction varies. All conditions use warmup+cosine schedule.

2. **Death probability is approximately proportional to cumulative LR
   integral.** This is the same linearization used in Exp 19 MATH.md.
   It predicts direction correctly but may be inaccurate for very short
   warmup (f_w=0.01) where the transition from warmup to cosine is abrupt.

3. **T_spike ~ 50 steps is consistent across seeds.** Exp 17 showed the
   spike peaks at S=50 across all 3 seeds. This timescale is architecture-
   dependent (d, P, L) and may differ at macro scale.

4. **Revival during cosine decay is independent of warmup fraction.** We
   assume the cosine phase (which is similar across conditions) drives
   revival similarly. This may be wrong if the number of neurons killed
   during the spike affects the revival dynamics (e.g., more dead neurons
   = more potential for revival).

5. **Frozen attention during fine-tuning.** Same as all prior experiments.
   Full fine-tuning would create larger distribution shifts during warmup,
   potentially amplifying the warmup benefit.
