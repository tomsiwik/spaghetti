# Freeze-Then-Prune Protocol: Mathematical Foundations

## 1. Problem Statement

Experiments 17-20 established that:
- 47% of ReLU capsules are dead after full training under constant LR (Exp 17)
- 28.1% of capsules dead at S=100 revive by S=3200 (Exp 18)
- Inter-layer coupling drives 79-94% of this revival (Exp 20)
- Revival is strictly feed-forward: training layer k revives downstream layers only

**Q: When should pruning occur -- during or after training?**

Mid-training pruning risks removing capsules that would have revived via
inter-layer coupling. Post-training (freeze-then-prune) removes only
capsules that are permanently dead because all upstream weight updates
have ceased.

Falsifiable predictions:
1. Post-training profiling identifies >= 5pp more dead capsules than
   mid-training profiling at any checkpoint (because revival inflates
   the alive count during training)
2. Post-training pruned quality is within 3% of mid-training pruned
   quality (pruning permanently dead capsules is safe)

---

## 2. Notation

Follows death_recovery_mechanism/MATH.md and capsule_revival/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
S         -- number of fine-tuning steps
S_total   -- total training steps (3200 at micro scale)
S_mid     -- mid-training pruning checkpoint (100, 400, 800, 1600)

D^l_S     -- set of dead capsule indices in layer l at step S
A^l_S     -- set of alive capsule indices in layer l at step S
R^l(S1,S2) -- revival set: capsules dead at S1, alive at S2
             = D^l_{S1} intersect A^l_{S2}

delta(S)  -- death rate at step S = |D_S| / P_total
             where D_S = union over l of D^l_S

prune(M, D) -- model M with capsules in set D removed
              (rows from A, columns from B)
```

---

## 3. The Pruning Timing Problem

### 3.1 Why Mid-Training Pruning Is Suboptimal

At step S_mid < S_total, the death profile includes:

```
D^l_{S_mid} = D^l_permanent union D^l_transient(S_mid)
```

where:
- D^l_permanent: capsules that will remain dead through S_total
- D^l_transient(S_mid): capsules that APPEAR dead at S_mid but will
  revive by S_total via inter-layer coupling

Pruning at S_mid removes both sets. The transient set contains capsules
that would have contributed to model quality if allowed to revive.

From Exp 18: 28.1% of S=100 dead cohort revives by S=3200.
From Exp 20: 79-94% of this revival comes from upstream weight changes.

**Expected false positive rate** (pruning capsules that would revive):

```
FP(S_mid) = |D^l_transient(S_mid)| / |D^l_{S_mid}|
```

At S_mid=100 (peak death): FP ~ 28.1% (from Exp 18 directly)
At S_mid=3200 (end): FP ~ 0% (no future training, no revival possible)

### 3.2 Post-Training Profiling Is Ground Truth

After S_total steps with all layers trained:

```
D^l_{S_total} = D^l_permanent   (exactly, because no future training)
```

Freezing all weights after S_total ensures:
- No upstream weight changes -> no inter-layer coupling -> no revival
- The profiled dead set IS the permanent dead set
- Pruning this set has exactly zero quality impact

### 3.3 The Yield Advantage

The freeze-then-prune protocol should show FEWER dead capsules than
mid-training profiling at early checkpoints, because revival reduces
the dead set over time:

```
|D_{S_total}| < |D_{S_mid}|   for S_mid near the death peak (~100 steps)
```

But this seems to CONTRADICT kill criterion 1 (freeze-then-prune yields
MORE dead capsules). The resolution:

The key insight is that mid-training pruning REMOVES dead capsules at
S_mid. After pruning, the continued training creates NEW dead capsules
from the remaining alive set. The comparison should be:

```
Yield_A = |D_{S_total}|                    (death rate at end)
Yield_B = |D_{S_mid}|                      (death rate at prune point)
         + new deaths in remaining steps   (minus, some alive capsules die)
         - revival in remaining steps      (these were already pruned)
```

At early S_mid (e.g., 100), Yield_B is high because the death rate peaks
at ~55% (Exp 17). But many of these capsules would have revived.

At S_total, Yield_A captures the equilibrium death rate (~47% constant
LR). The question is whether this is >5pp higher than the EFFECTIVE
compression from mid-training pruning.

Actually, let me reframe. The kill criteria compare:
1. Death rate at Protocol A's profiling point vs Protocol B's profiling point
2. Quality after pruning

For (1), the death spike at S=100 is ~55%, while equilibrium at S=3200
is ~47%. So mid-training pruning at the death peak actually identifies
MORE dead capsules than post-training pruning. The advantage of post-
training pruning is not yield but ACCURACY: every capsule profiled dead
at S_total IS permanently dead. Mid-training pruning has false positives
(capsules that would revive) but also captures the higher transient death.

Reinterpreting kill criterion 1: if post-freeze death < mid-training
death + 5pp, the hypothesis that freeze-then-prune has a yield advantage
is killed. But the hypothesis may STILL hold on criterion 2 (quality):
freeze-then-prune may produce better quality because it only removes
truly dead capsules.

### 3.4 Quality Prediction

Protocol A (freeze-then-prune) quality:
```
L(prune(M_{S_total}, D_{S_total})) = L(M_{S_total})   (exact, zero loss)
```

Protocol B (mid-training-prune) quality:
```
L(prune(M_{S_mid}, D_{S_mid}), after continued training)
```

The continued training after pruning recovers some quality by adapting
remaining capsules. But the pruned-away transient dead capsules cannot
be recovered. The quality gap depends on:
1. How many transient-dead capsules were pruned (false positives)
2. How important those capsules would have been after revival
3. How well remaining capsules compensate during continued training

---

## 4. Experimental Design

### 4.1 Protocols

| Protocol | Steps | Profile Point | Prune | Continue | Total Steps |
|----------|-------|---------------|-------|----------|-------------|
| Control  | 3200  | 3200          | No    | No       | 3200        |
| A (freeze-then-prune) | 3200 | 3200 | Yes | No | 3200 |
| B (mid-prune S=100) | 100 | 100 | Yes | +3100 | 3200 |
| B (mid-prune S=400) | 400 | 400 | Yes | +2800 | 3200 |
| B (mid-prune S=800) | 800 | 800 | Yes | +2400 | 3200 |
| B (mid-prune S=1600) | 1600 | 1600 | Yes | +1600 | 3200 |

### 4.2 Measurements

For each protocol:
- Death rate at profiling point (% of capsules with f_i = 0)
- Per-layer death rates
- Val loss before pruning
- Val loss after pruning (Protocol A) or after continued training (Protocol B)
- Quality change vs control (% degradation)

### 4.3 Kill Criteria

| Criterion | Condition | Interpretation |
|-----------|-----------|----------------|
| Kill 1 | Proto A death rate < max(Proto B death rates) + 5pp | Freeze-then-prune does NOT yield more dead capsules |
| Kill 2 | Proto A quality > best Proto B quality + 3% | Freeze-then-prune hurts quality |

Note: Kill 1 is likely to trigger because mid-training death peaks at
~55% (S=100) while equilibrium is ~47% (S=3200). The experiment's value
is in criterion 2: does removing only permanently dead capsules produce
better quality than removing a mix of permanent + transient dead?

---

## 5. Computational Cost

```
Protocol A: 300 + 3200 + profile + prune = ~3520 steps equivalent
Protocol B: 300 + S_mid + profile + prune + (3200 - S_mid) = ~3520 steps equivalent
Control:    300 + 3200 = 3500 steps

Per seed: 1 control + 1 Proto A + 4 Proto B = 6 runs x 3500 steps = 21,000 steps
3 seeds: 63,000 total steps
At ~1000 steps/sec (micro): ~63 seconds total
```

---

## 6. Worked Numerical Example

At d=4, P=4, L=2:

### Control (full training, no prune)

```
S=100:  L0 dead = {}, L1 dead = {cap0, cap1, cap2}  (75% dead)
S=3200: L0 dead = {}, L1 dead = {cap0, cap2}          (50% dead)
        cap1 revived via inter-layer coupling from L0
Val loss = 0.50
```

### Protocol A (freeze-then-prune)

```
Train 3200 steps -> profile -> L1 dead = {cap0, cap2} -> prune
Prune removes 2/4 capsules from L1 = 50% pruned
Val loss after prune = 0.50 (exact, both were truly dead)
```

### Protocol B (mid-prune at S=100)

```
Train 100 steps -> profile -> L1 dead = {cap0, cap1, cap2} -> prune
Prune removes 3/4 capsules from L1 = 75% pruned
  cap1 was transiently dead (would have revived)
Continue training 3100 more steps with only cap3 remaining in L1
Val loss final = 0.55 (worse: lost cap1's future contribution)
```

Result:
- Kill 1: Proto A death = 50%, Proto B death = 75%. A < B. Kill 1 triggers.
- Kill 2: Proto A quality = 0.50, Proto B quality = 0.55. A is 9% BETTER.
  Kill 2 does NOT trigger.

Interpretation: Mid-training pruning yields more dead capsules (higher
compression) but worse quality (false positive pruning). Freeze-then-prune
yields fewer dead capsules but only truly dead ones, preserving quality.
The experiment value is in criterion 2.

---

## 7. Assumptions

1. **Constant learning rate.** All experiments use constant LR (3e-3).
   Under cosine decay, the death trajectory differs (Exp 19: 19.6%
   equilibrium vs 47.3%). The relative advantage of freeze-then-prune
   may differ with LR schedules.

2. **Single domain.** Pruning is on domain a_m fine-tuned capsules.
   Multi-domain composed models may show different dynamics.

3. **Binary dead threshold (f=0).** Capsules firing on 0.01% of inputs
   are classified as alive. Nearly-dead capsules are not pruned.

4. **Same optimizer state.** Protocol B continues with fresh Adam state
   after pruning (the pruned capsules' optimizer state is lost). This
   could slightly disadvantage Protocol B.

5. **Profiling protocol.** 20 batches x 32 samples, consistent with
   Exp 12's validation (<4% noise). The profiling seed is shared
   across protocols for comparability.
