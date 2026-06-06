# Freeze-Then-Prune Protocol: Research Digest

## Hypothesis

Pruning dead capsules after training completes (freeze-then-prune) yields
higher pruning yield and equal quality compared to pruning during training,
because inter-layer coupling (79-94% of revival, Exp 20) means neurons
pruned mid-training may have revived.

**Result: KILL (criterion 1).** Freeze-then-prune yields 7.7pp FEWER dead
capsules than mid-training pruning (47.1% vs 54.9%), not more. Kill
criterion 2 passes: quality is equivalent (+0.10%). The experiment reveals
that mid-training pruning is superior to post-training pruning on yield,
and quality-equivalent, making freeze-then-prune unnecessary.

---

## What This Experiment Tests

Exp 20 proved that inter-layer coupling drives 79-94% of neural revival.
This raised a question: if dead neurons can revive during training, should
we prune DURING training (risking false positives from transient death) or
AFTER training (when death is permanent)?

Two protocols compared:
- **Protocol A (freeze-then-prune)**: Train fully (3200 steps), then
  profile and prune. Dead capsules are permanently dead.
- **Protocol B (mid-training prune)**: Profile and prune at intermediate
  checkpoints (S=100, 400, 800, 1600), then continue training. The model
  can compensate for pruned capsules.

Control: full training with no pruning.

---

## Lineage in the Arena

```
gpt -> ... -> relu_router -> dead_capsule_pruning -> pruning_controls
                                                         |
                                    +--------------------+--------------------+
                                    |                    |                    |
                             training_duration    capsule_revival    death_recovery_mechanism
                               (Exp 17)            (Exp 18)           (Exp 20)
                                                                         |
                                                                freeze_then_prune
                                                                  (THIS experiment)
```

---

## Key References

- **Exp 20 (death_recovery_mechanism)**: Inter-layer coupling drives
  79-94% of revival. The direct motivation for this experiment.
- **Exp 17 (training_duration)**: Death follows "spike and slow decay":
  18.8% init -> 55.1% peak at S~50 -> 47.3% at S=3200.
- **Exp 18 (capsule_revival)**: 28.1% of S=100 dead cohort revives by
  S=3200. Layer 3 shows highest revival.
- **Gurbuzbalaban et al. (2024)**: >90% of revived neurons re-die.
  Cosine decay promotes revival.

---

## Empirical Results

### Summary Table (3 seeds)

| Protocol | Death Rate at Profile | Val Loss (Final) | vs Control | vs Proto A |
|----------|----------------------|-------------------|------------|------------|
| Control (no prune) | 47.1% | 0.4874 | baseline | -- |
| A: freeze-then-prune | 47.1% | 0.4874 | -0.01% | baseline |
| B: mid-prune S=100 | 54.4% | 0.4898 | +0.48% | +0.48% |
| B: mid-prune S=400 | 54.3% | 0.4907 | +0.67% | +0.67% |
| B: mid-prune S=800 | 54.9% | 0.4869 | -0.11% | -0.10% |
| B: mid-prune S=1600 | 51.6% | 0.4884 | +0.20% | +0.20% |

### Kill Criteria

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| 1: Proto A yields >=5pp more dead than Proto B | >=5pp | -7.7pp (A has FEWER) | **KILL** |
| 2: Proto A quality >3% worse than best Proto B | >3% | +0.10% | PASS |

### Key Findings

**1. Mid-training profiling captures MORE dead capsules than post-training.**

This was the opposite of the hypothesis. The death rate peaks at ~55% during
early training (S=100-800) and declines to ~47% by S=3200 due to inter-layer
coupling revival. Profiling at the peak captures the full dead set including
transiently dead capsules, yielding 7-8pp more prunable capsules.

**2. Mid-training pruning followed by continued training WORKS.**

The model compensates for pruned capsules during continued training. All four
mid-training prune points produce final quality within 0.67% of control.
The best (S=800) actually outperforms control by 0.11%.

**3. Post-prune death rate drops dramatically.**

After mid-training pruning, the post-prune death rate drops from 47-55% to
13-19%. The model redistributes computation across remaining capsules,
achieving much higher utilization:

| S_mid | Death at Prune | Post-Prune Final Death |
|-------|----------------|------------------------|
| 100 | 54.4% | 18.9% |
| 400 | 54.3% | 19.0% |
| 800 | 54.9% | 13.4% |
| 1600 | 51.6% | 13.9% |

This suggests pruning + continued training creates a more efficient model:
similar quality with 55% fewer capsules AND 80-87% utilization of remaining
capsules (vs 53% without pruning).

**4. Freeze-then-prune is perfectly safe but unnecessary.**

Protocol A (freeze-then-prune) produces exactly zero quality change
(-0.01%). This confirms that profiling at training end identifies truly
dead capsules. But it prunes fewer capsules than mid-training pruning
for the same quality outcome.

### Per-Layer Death Rates (3-seed mean)

| Protocol | L0 | L1 | L2 | L3 |
|----------|------|------|------|------|
| Control | 0.8% | 69.5% | 65.9% | 52.3% |
| A: freeze-then-prune | 0.8% | 69.5% | 65.9% | 52.3% |
| B: S=100 | 0.3% | 76.3% | 78.1% | 62.8% |
| B: S=800 | 0.3% | 72.1% | 77.1% | 70.1% |

The death rate gradient across layers (L0 lowest, mid-layers highest) is
consistent across protocols. Layer 0 has near-zero death in all conditions.

---

## Micro-Scale Limitations

1. **Constant learning rate only.** Under cosine decay (which promotes
   revival per Exp 19), the death rate at training end would be ~20%
   instead of 47%. The yield advantage of mid-training pruning may differ
   with cosine decay because fewer capsules die overall.

2. **Single domain.** The experiment tests domain-specific fine-tuning
   pruning. Composed model pruning dynamics may differ (Exp 16 showed
   composition adds ~6% extra deaths).

3. **Adam optimizer state.** After mid-training pruning, Protocol B uses
   a fresh Adam optimizer for continued training (seed offset). The
   pruned capsules' optimizer state is lost, which may slightly
   disadvantage Protocol B. Despite this, Protocol B matches or beats
   Protocol A.

4. **Small capsule count (128/layer, 512 total).** At macro scale with
   thousands of capsules, the percentage dynamics should be similar but
   the absolute compression savings are larger.

5. **Binary threshold (f=0).** Only truly dead capsules (zero activation)
   are pruned. A threshold sweep might find a different optimal timing.

6. **n=3 seeds.** Effect sizes are small (0.1-0.7% quality differences)
   and may not be statistically significant. The yield difference (7.7pp)
   is large and consistent.

---

## What Would Kill This

**At micro scale:**
- If cosine decay LR schedule changes the ordering (freeze-then-prune
  yields MORE dead capsules with cosine decay due to enhanced revival).
- If using a threshold > 0 (nearly-dead) instead of binary dead changes
  the yield comparison.

**At macro scale:**
- If the model's ability to compensate after mid-training pruning
  degrades with model size (more interdependence between capsules).
- If SiLU/SwiGLU models (with ~0% truly dead neurons) make this entire
  mechanism irrelevant.
- If the mid-training prune + continue paradigm causes training
  instability at larger scale due to optimizer state disruption.

---

## Implications for the Composition Protocol

1. **Pruning timing is flexible, not critical.** Both protocols produce
   equivalent quality. The original recommendation (prune after training
   per Exp 18) is safe but not necessary.

2. **Mid-training pruning offers HIGHER compression.** Profiling at the
   death peak (S~100-800) captures 7-8pp more dead capsules. With
   continued training, the model compensates and reaches equivalent quality.
   For maximum compression: prune early, then continue training.

3. **Prune-then-continue creates more efficient models.** Post-prune
   death rate drops to 13-19%, meaning the model learns to use its
   remaining capsules more efficiently. This "forced efficiency" is a
   potentially valuable regularization effect.

4. **For the contributor workflow**, freeze-then-prune remains the
   SIMPLER protocol (no need to interrupt training). The quality and
   yield differences are small enough that protocol simplicity may
   outweigh the minor yield advantage of mid-training pruning.

---

## Relationship to Exp 20 (Death Recovery Mechanism)

Exp 20 proved inter-layer coupling drives revival. This experiment tested
the PRACTICAL IMPLICATION: does revival invalidate mid-training pruning?

Answer: No. Revival means mid-training profiling identifies capsules that
would LATER revive, but this is an ADVANTAGE for compression (more capsules
to prune), not a disadvantage. The model compensates for their loss during
continued training. The 79-94% coupling-driven revival rate is real but
does not make mid-training pruning harmful.

The original Exp 18 recommendation ("prune after training completes") was
CORRECT for safety but CONSERVATIVE for compression. Mid-training pruning
is equally safe and more aggressive.
