# Per-Capsule Revival Tracking: Research Digest

## Hypothesis

The aggregate death decrease observed in Exp 17 (55% at S=100 to 47% at
S=3200) is dominated by population turnover (different capsules cycling
through dead/alive states), not true revival of the same dead capsules.

**Falsifiable**: If Jaccard(dead_100, dead_3200) > 0.85, death is sticky
and revival negligible. If the S=100 dead cohort shows <5% revival by
S=3200, the inter-layer coupling mechanism is too weak to matter.

**Result: 0 of 3 kill criteria triggered.** The hypothesis is WRONG.
True revival of the SAME capsules is the dominant mechanism. 28.1% of
capsules dead at S=100 revive by S=3200. The inter-layer coupling
revival mechanism (hypothesized in Exp 17) is confirmed at the
per-capsule identity level.

---

## What This Experiment Tests

**Q: Does the aggregate death decrease reflect the SAME capsules reviving,
or population turnover?**

Exp 17 showed aggregate death decreases from 55% to 47% over training
but tracked only aggregate rates, not per-capsule identity. This
experiment records the binary dead/alive mask for every capsule at
every training checkpoint and computes:

1. **Transition matrices**: D->D, D->A, A->D, A->A counts between
   consecutive checkpoints
2. **Cohort tracking**: the set of capsules dead at S=100, tracked
   through all later checkpoints
3. **Jaccard similarity**: overlap between dead sets at different
   checkpoints

Protocol (identical to Exp 17 except for per-capsule tracking):
1. Pretrain base model on ALL data (300 steps, shared attention + MLP)
2. For each step count S in {0, 50, 100, 200, 400, 800, 1600, 3200}:
   a. Start from pretrained base (deepcopy)
   b. Freeze attention, fine-tune MLP only for S steps
   c. Profile per-capsule activation frequencies (20 batches x 32)
   d. Record binary dead/alive mask
3. Compute transitions, cohort survival, Jaccard

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> relu_router -> dead_capsule_pruning -> pruning_controls -> capsule_revival
                               (composition    (activation-based      (pre-composition       (per-capsule
                                by concat)      dead pruning)          death rate +            identity
                                                                       random baseline)        tracking)
```

---

## Key References

**Gurbuzbalaban et al. (2024), "Neural revival"**: Reports that >90% of
neurons that revive during training eventually die again. Our results
are consistent with this: revival is real but ongoing (not one-shot).
The 28.1% cohort revival we observe is cumulative across 3100 training
steps; individual revival events may be transient.

**Li et al. (2023), "Lazy Neuron Phenomenon"**: Reports ~50% natural
ReLU sparsity in trained transformers. Our equilibrium death rate
(44-56%) is consistent.

**Lu et al. (2019), "Dying ReLU and Initialization"**: Proves ReLU
networks die in probability as depth increases. Our finding that Layer 0
has near-zero death while layers 1-3 have 50-80% death is consistent
with depth-dependent death.

---

## Empirical Results

### 3-Seed Aggregate (seeds 42, 123, 7)

#### Death Rates (matches Exp 17)

| Steps | Death Rate | Std |
|-------|-----------|-----|
| 0 | 13.3% | 5.3% |
| 50 | 52.0% | 4.9% |
| 100 | 56.1% | 6.6% |
| 200 | 53.9% | 5.3% |
| 400 | 52.8% | 7.5% |
| 800 | 50.8% | 5.2% |
| 1600 | 49.2% | 4.4% |
| 3200 | 44.1% | 4.2% |

#### Transition Analysis (consecutive checkpoints)

| Interval | D->D | D->A | A->D | A->A | Revival% | NewDeath% |
|----------|------|------|------|------|----------|-----------|
| 0->50 | 62 | 6 | 204 | 240 | 9.3% | 46.0% |
| 50->100 | 251 | 15 | 37 | 209 | 5.8% | 14.9% |
| 100->200 | 259 | 29 | 17 | 207 | 10.0% | 7.7% |
| 200->400 | 249 | 27 | 21 | 215 | 9.8% | 9.0% |
| 400->800 | 239 | 32 | 22 | 220 | 11.7% | 9.0% |
| 800->1600 | 232 | 28 | 20 | 232 | 10.9% | 7.8% |
| 1600->3200 | 212 | 40 | 14 | 246 | 15.9% | 5.5% |

Key pattern: revival rate INCREASES over training (5.8% at S=50->100 to
15.9% at S=1600->3200) while new death rate DECREASES (14.9% to 5.5%).
The aggregate death decrease is driven by the growing gap between revival
and new death rates.

#### Cohort Tracking (capsules dead at S=100)

| Steps | Still Dead | Std | Revived | Std |
|-------|-----------|-----|---------|-----|
| 100 | 100.0% | 0.0% | 0.0% | 0.0% |
| 200 | 90.0% | 1.2% | 10.0% | 1.2% |
| 400 | 86.4% | 5.2% | 13.6% | 5.2% |
| 800 | 83.8% | 5.3% | 16.2% | 5.3% |
| 1600 | 80.6% | 1.9% | 19.4% | 1.9% |
| 3200 | 71.9% | 5.8% | 28.1% | 5.8% |

The S=100 dead cohort shows steady, ongoing revival: by S=3200, more
than one quarter of them have revived. This is not a burst event but
continuous recovery at roughly 5 percentage points per 2x training.

#### Jaccard Similarity of Dead Sets

| Pair | Jaccard | Std |
|------|---------|-----|
| 100->200 | 0.847 | 0.028 |
| 100->400 | 0.805 | 0.065 |
| 100->800 | 0.781 | 0.030 |
| 100->1600 | 0.752 | 0.030 |
| 100->3200 | 0.669 | 0.027 |
| 50->3200 | 0.634 | 0.030 |
| 200->3200 | 0.680 | 0.029 |

Jaccard decays steadily with training distance. At 100->3200 (3100
steps apart), Jaccard = 0.669 -- well below the null random model
prediction of 0.340 (Section 4.2 of MATH.md), confirming death IS
sticky, but well below 0.85, confirming significant identity change.

The dead set evolves continuously: roughly one third of its members
change over the full training trajectory.

#### Per-Layer Revival Rates (S=100 onward)

| Layer | D->A transitions | Dead at S=100 |
|-------|-----------------|---------------|
| 0 | 2 | 3 |
| 1 | 51 | 95 |
| 2 | 43 | 99 |
| 3 | 60 | 90 |

Layer 0 has almost no dead capsules (processes frozen embeddings), so
revival is trivially small. Layers 1-3 show substantial revival
activity: 43-60 D->A transitions from 90-99 dead capsules. Layer 3
shows the highest revival rate (60/90 = 67% of dead capsules underwent
at least one revival transition), consistent with it receiving the most
shifted inputs from upstream layer updates.

Note: these are cumulative transition counts, not unique capsules. A
capsule that revives at S=200 and dies again at S=400 and revives at
S=800 contributes 2 to the D->A count.

### Decomposition of Aggregate Decrease

```
Aggregate death decrease (S=100 -> S=3200):     12.0 pp
S=100 dead cohort size:                          287 / 512
Of S=100 dead cohort, revived by S=3200:         28.1%
Revival contribution to aggregate decrease:      15.8 pp
New deaths avoided (from gradient shrinkage):    -3.8 pp
```

Revival contribution (15.8 pp) EXCEEDS the aggregate decrease (12.0 pp)
because new deaths partially offset the revival. The aggregate decrease
is a net effect: many capsules revive (15.8 pp contribution) while some
alive capsules die (3.8 pp offset).

---

## Kill Threshold Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Jaccard(dead_100, dead_3200) > 0.85 | 0.669 | >0.85 | **PASS** |
| Max revival rate < 5% | 15.9% | <5% | **PASS** |
| Total turnover < 10/seed | 505+ | <10 | **PASS** |

**0 of 3 kill criteria triggered.**

---

## Key Findings

### Finding 1: True Revival Dominates Over Population Turnover

28.1% of capsules dead at S=100 revive by S=3200. Revival contribution
to the aggregate death decrease (15.8 pp) exceeds the aggregate decrease
itself (12.0 pp), with new deaths partially offsetting revival. The
inter-layer coupling mechanism hypothesized in Exp 17 is confirmed at
the per-capsule identity level.

### Finding 2: Revival Accelerates With Training

Revival rate per interval increases from 5.8% (S=50->100) to 15.9%
(S=1600->3200), while new death rate decreases from 14.9% to 5.5%.
As gradients shrink (later training), fewer capsules are pushed past
the death boundary, while input distribution shifts from accumulating
weight updates continue to revive dead capsules.

This creates a positive feedback: as alive-neuron weights stabilize,
fewer new deaths occur, while inter-layer coupling continues to revive
dead neurons through accumulated input distribution drift.

### Finding 3: Death Is Sticky But Not Permanent

Jaccard(dead_100, dead_3200) = 0.669: substantially above the random
null (0.340), confirming death is sticky (most dead capsules at S=100
are still dead at S=3200). But well below 1.0: one third of the dead
set identity changes over 3100 training steps.

This is consistent with Gurbuzbalaban et al.'s finding that >90% of
revived neurons eventually die again. The dead set is not static but
slowly evolving.

### Finding 4: Layer 3 Shows Highest Revival Activity

Layer 3 has 60 D->A transitions from 90 dead capsules, the highest
revival rate. This is consistent with inter-layer coupling: layer 3
receives inputs that pass through layers 0, 1, and 2, so weight
updates in any upstream layer can shift its input distribution. Deeper
layers accumulate more distribution shift, enabling more revival.

### Finding 5: Pruning Decisions Should Be Checkpoint-Aware

Because 28% of dead capsules revive over 3100 steps, pruning decisions
made at checkpoint S may not hold at checkpoint S + 3000. At macro
scale:
- Profile death AFTER training completes (end of training)
- Do not prune during training (dead capsules may revive later)
- Or re-profile periodically if pruning during training

---

## Micro-Scale Limitations

1. **Maximum 3200 steps**: Revival rate is still increasing at S=3200.
   At macro scale (100K+ steps), revival may continue, potentially
   recovering a larger fraction of dead capsules.

2. **Binary classification**: We classify as dead/alive at f=0 threshold.
   "Nearly dead" capsules (0 < f < 0.01) may represent a borderline
   population that flickers between states.

3. **Same-seed nested checkpoints**: S=100 and S=3200 share the first
   100 steps of the training trajectory. This is a feature (isolates
   temporal evolution) but means the transition statistics include
   within-trajectory correlations.

4. **Single domain only**: Revival dynamics may differ in composed models
   where domain-specific capsules see cross-domain inputs.

5. **Small model (d=64, P=128)**: With larger hidden dimensions, the
   "margin" of each capsule's dead/alive boundary may change, affecting
   revival probability.

6. **Only 3 seeds**: The cohort revival rate std is 5.8 pp (28.1% mean),
   so the finding is directional but would benefit from more seeds.

---

## What Would Kill This

### At Micro Scale (tested)

- **Death is sticky (Jaccard > 0.85)**: NOT KILLED. Jaccard = 0.669.
  Significant identity change.

- **Revival too weak (<5%)**: NOT KILLED. Max revival rate = 15.9%.
  Revival is substantial and meaningful.

- **Dynamics too sparse**: NOT KILLED. 505+ turnover events per seed.
  Rich dynamics at micro scale.

### At Macro Scale (untested)

- **Revival diminishes at scale**: If larger models have wider margins
  between dead/alive boundaries, revival through input distribution
  shift may be insufficient to cross those margins.

- **LR schedule changes dynamics**: Warmup may reduce the initial spike
  (fewer capsules to revive). Cosine decay may alter late-phase revival
  rates. See Exp 19.

- **Revival is transient**: If revived capsules die again within a few
  hundred steps (consistent with Gurbuzbalaban et al.), the practical
  benefit of waiting for revival before pruning may be minimal.

---

## Implications for the Project

### Revised Understanding of Death Dynamics

**Old (from Exp 17)**: "Death rate decreases over training. Dead neurons
CAN revive through inter-layer coupling."

**New (from Exp 18)**: "Revival is real and substantial: 28% of the
S=100 dead cohort revives by S=3200. Revival accelerates as training
progresses (revival rate increases from 6% to 16% per interval). The
dead set is not static but slowly evolving, with roughly one third of
its identity changing over the full training trajectory."

### Updated Pruning Protocol

The composition protocol is unchanged but with a timing recommendation:

1. Pretrain shared base
2. Fine-tune capsule pools per domain (any reasonable duration)
3. Compose by concatenation
4. **Profile activations AFTER training completes** (not during)
5. Prune dead capsules (tau=0)
6. Calibrate surviving capsules

The timing matters because capsules dead mid-training may revive by
training's end. Pruning at an intermediate checkpoint would incorrectly
remove capsules that would later contribute.

### Connection to Future Experiments

- **Exp 16 (Capsule identity across composition)**: Does composition
  reshuffle which capsules are dead, or does it preserve the single-domain
  dead set? The per-capsule tracking infrastructure from this experiment
  transfers directly.

- **Exp 19 (LR schedule)**: Warmup may soften the initial spike (fewer
  deaths, less to revive). Cosine decay may alter the revival rate in
  Phase 3.

- **Exp 20 (Layer freezing)**: Freezing layer l during training should
  reduce revival in layer l+1, directly testing the inter-layer coupling
  mechanism. Finding 4 (layer 3 has highest revival) predicts that
  freezing layers 0-2 would suppress revival in layer 3 most dramatically.
