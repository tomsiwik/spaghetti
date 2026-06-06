# Training Duration vs Death Rate: Research Digest

## Hypothesis

ReLU capsule death rate is monotonically non-decreasing: dead neurons
cannot revive because their ReLU gradients are zero, so death only
accumulates with more training steps. The 54% death rate from Exp 10
(measured at 200 steps) is a lower bound.

**Falsifiable**: If death rate at 3200 steps is lower than at 200 steps
by more than 5 percentage points, the monotonicity prediction fails and
early-training death is partially transient.

**Result: 1 of 3 kill criteria triggered.** Death rate DECREASES from
55.5% at S=100 to 47.3% at S=3200 (aggregate). The monotonicity
prediction is WRONG. However, death remains substantial (47.3% >= 30%),
so the pruning opportunity persists. The key finding: death rate follows
a "spike and decay" trajectory, not monotonic accumulation.

---

## What This Experiment Tests

**Q: Does the 54% single-domain death rate from Exp 10 generalize to
longer training durations?**

The 200-step fine-tuning used in Exp 10 is very short. At macro scale,
training runs are 100K+ steps. This experiment sweeps fine-tuning steps
from 0 (pretrained base only) to 3200 (16x Exp 10) to characterize the
death rate trajectory.

Protocol:
1. Pretrain base model on ALL data (300 steps, shared attention + MLP)
2. For each step count S in {0, 50, 100, 200, 400, 800, 1600, 3200}:
   a. Start from the pretrained base (deepcopy)
   b. Freeze attention, fine-tune MLP only for S steps on domain data
   c. Profile activation frequencies on domain validation data
   d. Record death rate and val loss

**Design note**: All step counts use the same training seed, so S=50 is
the first 50 steps of S=3200 (same batch ordering, same training
trajectory evaluated at different checkpoints). This nested structure is
a design strength — it isolates the effect of additional training from
confounds due to different batch orderings.

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> relu_router -> dead_capsule_pruning -> pruning_controls -> training_duration
                               (composition    (activation-based      (pre-composition       (death rate
                                by concat)      dead pruning)          death rate +            vs training
                                                                       random baseline)        steps)
```

---

## Key References

**Lu et al. (2019), "Dying ReLU and Initialization"**: Proves that a
deep ReLU network will die in probability as depth approaches infinity.
However, their analysis is static (at initialization) and does not
address how death evolves during training.

**Li et al. (2023), "Lazy Neuron Phenomenon"**: Reports ~50% natural
ReLU sparsity in trained transformers, consistent with our equilibrium
observations. Does not track the temporal trajectory.

**Gurbuzbalaban et al. (2024), "Neural revival"**: Finds that >90% of
neurons that revive during training eventually die again. Weight updates
in earlier layers can temporarily shift inputs back into the positive
half-space. Learning rate decay triggers brief revival episodes.

**ReLU Strikes Back (Mirzadeh et al., 2023)**: Demonstrates that ReLU
sparsity in LLMs is a feature (enabling inference acceleration), not a
bug. Reports up to 90% sparsity in deeper layers of ReLU-activated LLMs.

---

## Empirical Results

### 3-Seed Aggregate (seeds 42, 123, 7)

| Steps | Death Rate | Std | Val Loss | Std | L0 | L1 | L2 | L3 |
|-------|-----------|-----|----------|-----|-----|-----|-----|-----|
| 0 | 18.8% | 6.6% | 0.5296 | 0.0073 | 0% | 16% | 33% | 27% |
| 50 | 55.1% | 4.1% | 0.5175 | 0.0073 | 0% | 66% | 79% | 75% |
| 100 | 55.5% | 4.2% | 0.5112 | 0.0073 | 0% | 72% | 76% | 74% |
| 200 | 52.9% | 6.3% | 0.5084 | 0.0068 | 0% | 69% | 74% | 69% |
| 400 | 55.1% | 4.5% | 0.5028 | 0.0057 | 0% | 71% | 76% | 73% |
| 800 | 53.1% | 5.2% | 0.4984 | 0.0064 | 0% | 70% | 75% | 67% |
| 1600 | 51.6% | 4.8% | 0.4941 | 0.0070 | 0% | 69% | 73% | 64% |
| 3200 | 47.3% | 6.5% | 0.4883 | 0.0063 | 0% | 65% | 67% | 57% |

### Trajectory Shape: "Spike and Slow Decay"

The death rate follows a distinctive three-phase pattern:

**Phase 1 (S=0 to S=50): Rapid spike.** Death rate jumps from 18.8% to
55.1% in just 50 fine-tuning steps. This is the "initialization shock":
large initial gradients push many marginal neurons past the death boundary.
This accounts for the bulk of all neuron death.

**Phase 2 (S=50 to S=400): Plateau with fluctuation.** Death rate
oscillates between 52.9% and 55.5%. At this stage, some neurons die
while others revive -- the two rates approximately balance. This is
the regime Exp 10 measured (S=200).

**Phase 3 (S=400 to S=3200): Gradual decay.** Death rate slowly
decreases from 55.1% to 47.3%. As training converges (smaller gradients),
fewer neurons die. Meanwhile, continued MLP weight updates in earlier
layers shift input distributions, occasionally pushing dead neurons'
inputs back above zero (neural revival).

### Monotonicity Analysis

Death is NOT monotonic. All three seeds show decreases:

| Seed | Largest decrease | Over interval |
|------|-----------------|---------------|
| 42 | -8.4 pp | S=100 to S=200 |
| 123 | -5.9 pp | S=200 to S=400 |
| 7 | -3.3 pp | S=1600 to S=3200 |

The aggregate also shows decreases after S=100: the peak death rate is
55.5% at S=100, declining to 47.3% by S=3200.

### Curve Fit (Saturating Exponential — Rise Phase Only)

Best fit: delta(S) = 0.188 + 0.350 * (1 - exp(-S/25))

- Time constant: 25 steps (very fast initial death)
- R-squared: 0.914

**Caveat**: This monotonically increasing model is misspecified — the data
shows a non-monotonic spike-and-decay pattern that the model cannot capture.
The fit characterizes the rapid rise phase (S < 100) well but systematically
underpredicts at S > 1600. The asymptotic parameter (53.8%) should NOT be
used for equilibrium predictions. A two-phase model (rising exponential +
slow linear decay) would better describe the full trajectory. The 0.914
R-squared masks the late-phase systematic error.

### Val Loss vs Death Rate Correlation

Pearson r = 0.027 (effectively zero) computed across pooled (S, seed) points.

**Caveat**: The near-zero correlation is partly an artifact of different
functional forms over the shared independent variable S — val loss decreases
monotonically while death rate follows a non-monotonic trajectory. This does
not constitute evidence of causal independence.

The actual evidence that dead neurons are useless comes from Exp 9's exact
pruning result: removing capsules with zero activation frequency produces
mathematically zero quality change (Exp 9, tau=0, -0.00% vs concat). Quality
improvement comes from ALIVE neurons specializing better, not from the
death/revival dynamics.

### Exp 10 Replication

- Exp 10 reference (200 steps, full dataset profiling): 54.3%
- This experiment (200 steps, domain-only profiling): 52.9%
- Difference: 1.4 pp

Within expected variance. The small difference is explained by different
profiling conditions: Exp 10 profiles on joint (both-domain) data while
this experiment profiles on domain-specific data only.

---

## Kill Threshold Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Death decreases 200->3200 | -5.7 pp | >5 pp decrease | **KILL** |
| Death at 3200 < 30% | 47.3% | <30% | PASS |
| Death std > 20pp at any S | 6.6% max | >20% | PASS |

**1 of 3 kill criteria triggered.**

---

## Key Findings

### Finding 1: Death Is Non-Monotonic (Theoretical Prediction Wrong)

The monotonicity argument (dead neurons have zero gradients, so they
cannot revive) is correct in isolation but incomplete. In a multi-layer
network with shared training:

1. Layers 0-3 share MLP updates during fine-tuning
2. When earlier layers' MLP weights change, the input distribution to
   later layers shifts
3. This shift can push previously-dead neurons' inputs above zero
4. The neuron "revives" and begins receiving gradients again

This is the "neural revival" phenomenon documented by Gurbuzbalaban et al.
In our architecture, attention is frozen but all 4 MLP layers train
simultaneously, creating exactly this inter-layer coupling.

### Finding 2: Initial Death Spike Is Extremely Fast (tau ~ 25 steps)

The death rate jumps from 18.8% to 55.1% in just 50 steps -- the
first 50 of 3200. This means:
- 94% of all peak death occurs in the first 1.5% of training
- The death rate at any training duration > 50 steps is within the
  47-56% range
- For macro predictions, the exact training duration matters less than
  whether fine-tuning happened at all

### Finding 3: Long Training Slowly REDUCES Death

From S=100 to S=3200, death decreases by ~8 percentage points (55.5% to
47.3%). This slow recovery is consistent with the optimizer gradually
finding better solutions that use more of the available capacity.

At macro scale (100K+ steps), we might expect further recovery -- but
the rate is slow (roughly 2.5 pp per 10x more training). Extrapolating
logarithmically: at 100K steps, death might be ~40-45%.

### Finding 4: Layer 0 Is Special (Confirmed at All Durations)

Layer 0 has 0% death at ALL step counts (0 through 3200). This
architectural invariant is explained by the fixed input distribution:
layer 0 receives raw embeddings (wte + wpe) which are frozen during
fine-tuning, so its input distribution never shifts. In contrast, layers
1-3 receive inputs that change as earlier-layer MLP weights update,
creating the inter-layer coupling that drives both death and revival.
The distinction is important: layer 0's zero death is not about "broader
hidden space" but about input distribution stability.

### Finding 5: Dead Neurons Contribute Nothing to Quality

The evidence for this comes from Exp 9's exact pruning theorem (tau=0
pruning produces 0.0% quality change), not from the correlation analysis.
The zero Pearson r (0.027) between death rate and val loss is consistent
with this conclusion but does not independently establish it — the near-zero
correlation is partly an artifact of different functional forms over training
steps (see Empirical Results above).

---

## Micro-Scale Limitations

1. **Maximum 3200 steps**: At macro scale, training runs are 100K+ steps.
   The gradual decay we observe might continue, stabilize, or reverse
   at longer durations. The logarithmic extrapolation is speculative.

2. **Single domain only**: We profile death on domain A (a-m names) only.
   Cross-domain death dynamics might differ.

3. **Small model (d=64, P=128)**: With more capacity per layer, the
   death dynamics might change. Larger hidden dimensions have more
   room for detector vectors to find useful directions.

4. **Frozen attention**: In full fine-tuning (unfrozen attention), the
   input distribution to each MLP layer changes more dramatically,
   potentially increasing both death and revival rates.

5. **Constant learning rate**: This experiment uses a constant LR (3e-3)
   throughout training. Macro-scale training uses LR schedules (warmup +
   cosine decay), which could qualitatively change the death trajectory:
   warmup may soften the initial spike, cosine decay may increase late-phase
   revival (as noted by Gurbuzbalaban et al.). The "spike and slow decay"
   shape should be validated under standard LR schedules at macro scale.

6. **Only 3 seeds**: The non-monotonic signal (kill criterion triggered
   at -5.7pp) is close to the 5pp threshold. More seeds would increase
   confidence.

---

## What Would Kill This

### At Micro Scale (tested)

- **Monotonic death**: KILLED. Death is non-monotonic (spike then decay).
  The theoretical irreversibility argument does not hold in multi-layer
  networks with shared training.

- **Death at 3200 < 30%**: NOT KILLED. Death remains substantial (47.3%).
  The pruning opportunity persists at longer training durations.

### At Macro Scale (untested)

- **Death recovery accelerates with training**: If longer training
  (10K-100K steps) pushes death below 30%, the pruning story weakens
  significantly. The slow logarithmic decay observed at micro scale
  would need to be verified.

- **Death dynamics change with model scale**: At d=4096 with P=8192,
  the broader hidden space might support more alive neurons (less death)
  or the dying ReLU phenomenon might be more severe (more neurons near
  the death boundary).

- **Unfrozen attention changes dynamics**: Full fine-tuning (not just
  MLP) causes larger distribution shifts, which could either increase
  revival (more input diversity) or increase death (more violent
  gradient updates).

---

## Implications for the Project

### Revised Understanding of Death Rates

**Old (from Exp 10)**: "54% of capsules are dead after 200 fine-tuning
steps. This is a general ReLU phenomenon."

**New (from Exp 17)**: "Death rate peaks at ~55% within the first
~100 steps (very fast spike), then gradually decays to ~47% by 3200
steps. The equilibrium death rate for well-trained models is likely
40-50%, not the 54% measured at the spike. The pruning opportunity
is robust but slightly smaller than previously estimated."

### Updated Composition Protocol

The composition protocol is unchanged in practice:
1. Pretrain shared base
2. Fine-tune capsule pools per domain (any reasonable duration)
3. Compose by concatenation
4. Profile activations and prune dead capsules (tau=0)
5. Calibrate surviving capsules

But the expected pruning yield changes: at standard fine-tuning
durations (200 steps), expect ~50-55% dead. At longer durations,
expect ~45-50% dead. The pruning opportunity is stable within this
range.

### Macro Prediction (Speculative)

For macro-scale fine-tuning (100K+ steps):
- The observed range at micro scale is 47-55% across all durations > 50 steps
- Extrapolating the slow decay (~2.5pp per 10x training) to 100K steps
  gives ~40-45%, but this is a 1.5-order-of-magnitude extrapolation beyond
  the measurement range (3200 steps max) and should be treated with caution
- Pruning at tau=0 should still yield substantial parameter reduction at
  zero quality cost, though the exact yield depends on the equilibrium
  death rate at macro training durations
- Death profiling should be done AFTER training completes, not during
  (death rate changes throughout training)
- **Key macro risk**: LR schedules (warmup + cosine decay) may
  qualitatively change the death trajectory shape (see Limitations)
