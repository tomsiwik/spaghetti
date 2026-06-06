# LR Schedule Impact on Death Trajectory: Mathematical Foundations

## 1. Problem Statement

Experiment 17 established that ReLU capsule death follows a "spike and slow
decay" pattern under constant learning rate: 18.8% at init, spike to 55% at
~50 steps, gradual recovery to 47% by 3200 steps. This used constant LR = 3e-3.

Macro-scale training universally uses warmup + cosine decay. The open question:

**Q: Does the LR schedule qualitatively change the death trajectory?**

Three specific predictions to test:
1. **Warmup softens the spike** (smaller initial gradients = fewer deaths)
2. **Cosine decay boosts revival** (Gurbuzbalaban et al. 2024)
3. **Warmup+cosine yields lower equilibrium death** (combined effect)

---

## 2. Notation

All notation follows training_duration/MATH.md and pruning_controls/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
S         -- number of fine-tuning steps (varied: 0 to 3200)
S_total   -- total training steps for schedule computation (3200)

a_i(S) in R^d  -- detector vector for capsule i after S fine-tuning steps
b_i(S) in R^d  -- expansion vector for capsule i after S fine-tuning steps

eta(s)    -- learning rate at step s (schedule-dependent)
eta_peak  -- peak learning rate (3e-3 in all schedules)

D = {x_1, ..., x_M}  -- profiling hidden states (validation data)

f_i(S) = (1/M) * sum_{x in D} 1{a_i(S)^T x > 0}
       -- activation frequency of capsule i after S steps

delta(S) = (1/P) * sum_{i=1}^{P} 1{f_i(S) = 0}
         -- death rate after S steps
```

### 2.1 LR Schedule Definitions

**Constant:**
```
eta(s) = eta_peak                         for all s
```

**Warmup only** (linear 0 -> eta_peak over warmup phase, then constant):
```
S_w = floor(0.10 * S_total)              -- warmup steps (10% of total)

eta(s) = eta_peak * (s / S_w)            if s <= S_w
       = eta_peak                         if s > S_w
```

**Cosine only** (cosine decay from eta_peak -> 0):
```
eta(s) = (eta_peak / 2) * (1 + cos(pi * s / S_total))
```

**Warmup + cosine** (linear warmup, then cosine decay):
```
eta(s) = eta_peak * (s / S_w)                                if s <= S_w
       = (eta_peak / 2) * (1 + cos(pi * (s - S_w) / (S_total - S_w)))   if s > S_w
```

---

## 3. Theory: LR Schedule Effects on Neuron Death

### 3.1 Death Probability per Step

From training_duration/MATH.md Section 3.4, the probability of killing an
alive neuron in one step is approximately:

```
P(alive -> dead at step s) ~ eta(s) * ||g_i(s)|| / ||a_i(s)|| * phi(margin_i(s))
```

where margin_i(s) = min_x (a_i(s)^T x) is the distance to the death boundary.

This is proportional to eta(s). Therefore:
- Smaller eta(s) during warmup -> fewer deaths per step
- Larger eta(s) at peak -> more deaths per step
- Decreasing eta(s) during cosine decay -> fewer NEW deaths per step

### 3.2 Prediction 1: Warmup Reduces the Initial Spike

At S=50 with warmup fraction 0.10 of S_total=3200:
```
S_w = 320 warmup steps
```

At S=50, the warmup is only 15.6% complete:
```
eta(50) = eta_peak * (50 / 320) = eta_peak * 0.156 = 4.69e-4
```

Compared to constant LR:
```
eta_constant(50) = 3e-3
```

The effective LR at S=50 under warmup is 6.4x smaller than constant. Since
death probability scales with LR, we predict:

```
delta_warmup(50) << delta_constant(50)
```

More precisely, under a linear death-rate-vs-LR model:
```
Expected: delta_warmup(50) - delta_0 ~ (0.156) * (delta_constant(50) - delta_0)
```

Numerical estimate (using Exp 17's delta_0 ~ 19%, delta_constant(50) ~ 55%):
```
delta_warmup(50) ~ 19% + 0.156 * (55% - 19%) = 19% + 5.6% = 24.6%
```

### 3.3 Prediction 2: Cosine Decay Boosts Revival

From Exp 17 and Exp 18, revival occurs through inter-layer coupling: weight
updates in layers 0..l-1 shift x_l, potentially pushing dead neurons' inputs
above zero.

Cosine decay reduces eta(s) smoothly as s increases. This has two effects:

1. **Fewer new deaths**: As eta decreases, P(alive->dead) decreases
2. **Continued but gentler weight shifts**: Even with smaller eta, weight
   updates still shift input distributions, enabling revival

The net effect: fewer new deaths + continued revival = more net revival.

Gurbuzbalaban et al. (2024) reported that >90% of neurons that revive during
training eventually die again, but "neural revival mostly happens when the
learning rate is decayed." This suggests that LR decay creates a favorable
regime for revival by reducing the probability of re-death.

### 3.4 Prediction 3: Combined Effect (Warmup + Cosine)

Warmup + cosine has the lowest LR during the critical spike phase (S < 320)
AND decreasing LR during the revival phase (S > 320). We predict the lowest
equilibrium death rate across all schedules:

```
delta_warmup_cosine(3200) < delta_constant(3200)
```

### 3.5 A Note on Why Cosine-Only Does NOT Reduce the Spike

Cosine decay starts at eta_peak (3e-3) and decays. At S=50:
```
eta_cosine(50) = (3e-3 / 2) * (1 + cos(pi * 50 / 3200))
               = (3e-3 / 2) * (1 + cos(0.049))
               = (3e-3 / 2) * (1 + 0.9988)
               = 2.998e-3
```

This is essentially identical to constant LR at S=50. Cosine decay starts
slowly and only becomes noticeable after hundreds of steps. Therefore:

```
delta_cosine(50) ~ delta_constant(50)
```

The cosine schedule differs from constant mainly after S > 400 (where the
LR has dropped to ~90% of peak) and dramatically after S > 1600 (LR < 50%
of peak).

---

## 4. Experimental Design

### 4.1 Schedules

Four LR schedules, all with peak LR = 3e-3:
1. **Constant**: eta(s) = 3e-3 (Exp 17 baseline)
2. **Warmup**: linear 0 -> 3e-3 over first 320 steps, then constant
3. **Cosine**: 3e-3 -> 0 cosine decay over 3200 steps
4. **Warmup + cosine**: linear warmup (320 steps) then cosine decay

### 4.2 Checkpoint Sweep

Same as Exp 17: S in {0, 50, 100, 200, 400, 800, 1600, 3200}

The S=0 baseline is shared across all schedules (profiling the pretrained
base before any fine-tuning).

### 4.3 Training Details

- Pretraining: 300 steps on all data, constant LR (identical to Exp 17)
- Fine-tuning: attention frozen, MLP only (identical to Exp 17)
- Profiling: 20 batches x 32 samples on domain validation data
- Seeds: 42, 123, 7

### 4.4 Key Design Decision: Schedule Based on S_total, Not S

Each checkpoint trains for S steps using a schedule defined over S_total = 3200.
This means:
- The S=50 checkpoint of warmup+cosine sees LR = 4.59e-4 (mid-warmup)
- The S=800 checkpoint of cosine sees LR = 2.56e-3 (early decay)
- The S=3200 checkpoint of cosine sees LR ~ 0 (end of decay)

This is correct because it simulates "what does the model look like at step
S of a full 3200-step training run?" rather than compressing each schedule
into exactly S steps.

---

## 5. Kill Criteria

| Criterion | Threshold | What it means |
|-----------|-----------|---------------|
| Warmup effect on S=50 spike | max(abs(warmup_50 - const_50), abs(wc_50 - const_50)) < 3pp | Warmup does not affect spike |
| Equilibrium death range at S=3200 | max - min across schedules < 3pp | Schedule does not affect equilibrium |
| Cosine revival boost | max(cosine_revival, wc_revival) <= const_revival | LR decay does not boost revival |

Revival = death decrease from S=200 to S=3200 (same metric as Exp 17).

---

## 6. Worked Numerical Example

At d=4, P=4, using warmup+cosine schedule with S_total = 100, S_w = 10:

### S=0 (pretrained base, before fine-tuning)
```
LR: N/A (no fine-tuning yet)
Capsule 0: a_0 fires 60% -> alive
Capsule 1: a_1 fires 45% -> alive
Capsule 2: a_2 fires  0% -> DEAD
Capsule 3: a_3 fires 30% -> alive
delta(0) = 1/4 = 25%
```

### S=5 (mid-warmup, LR = peak * 5/10 = 1.5e-3)
```
LR: 1.5e-3 (half of peak)
Capsule 0: fires 58% -> alive (small shift)
Capsule 1: fires 42% -> alive (small shift)
Capsule 2: fires  0% -> DEAD (still dead)
Capsule 3: fires 28% -> alive (small shift)
delta(5) = 1/4 = 25%  (no new deaths from gentle updates)
```

### S=50 (mid-training, cosine LR ~ 1.5e-3)
```
LR: ~1.5e-3 (cosine at midpoint)
Capsule 0: fires 50% -> alive
Capsule 1: fires  0% -> DEAD (died at step 35 when LR peaked)
Capsule 2: fires  0% -> DEAD
Capsule 3: fires 35% -> alive
delta(50) = 2/4 = 50%
```

### S=100 (end, cosine LR ~ 0)
```
LR: ~0 (end of cosine)
Capsule 0: fires 48% -> alive
Capsule 1: fires  3% -> alive (REVIVED! Inter-layer coupling + low LR)
Capsule 2: fires  0% -> DEAD
Capsule 3: fires 33% -> alive
delta(100) = 1/4 = 25%  (revival brought death back to init level)
```

This example illustrates:
- Warmup delays death (S=5 still at 25%)
- Peak LR still causes some death (S=50 at 50%)
- Cosine decay enables revival (S=100 back to 25%)

Under constant LR at the same S=100, we'd expect delta ~ 50% (no revival).

---

## 7. Assumptions

1. **Same base model as Exp 17.** All schedules share the same pretrained
   base (300 steps, constant LR). The only variable is the fine-tuning LR
   schedule.

2. **Death probability is proportional to LR.** This is a linearization
   of the actual relationship. At very low LR, neurons may still die
   (from accumulated small updates), and at very high LR, death may
   saturate (all marginal neurons already dead). The proportionality
   is an approximation that predicts the direction but not exact magnitude.

3. **Revival is enabled by weight updates regardless of LR magnitude.**
   Inter-layer coupling requires non-zero weight updates in earlier layers.
   Even small LR produces some shift. However, the magnitude of shift
   (and therefore revival probability) may be LR-dependent.

4. **Schedule is based on S_total, not S.** Each checkpoint is a snapshot
   of a full 3200-step training run at step S. This is the physically
   meaningful quantity for macro predictions.

5. **Frozen attention during fine-tuning.** Same as Exp 17. Full fine-tuning
   would create additional input distribution shifts (from attention changes)
   that could interact with LR schedules differently.

6. **Profiling protocol unchanged.** 20 batches x 32 samples. Exp 12
   validated this protocol: same-checkpoint disagreement 2.6-3.8%.
