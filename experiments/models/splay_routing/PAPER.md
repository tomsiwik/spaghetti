# Splay-Tree Adaptive Routing: Research Digest

## Hypothesis

Adding splay-tree-inspired runtime bias correction to binary tree gates
will enable faster adaptation to distribution shift (domain change) than
static tree routing, measured as lower val_loss on a new domain with fewer
training steps.

**Falsifiable**: If splay routing does not reduce routing cost on non-
stationary data (KC1) or splay overhead exceeds routing savings (KC2),
the approach is dead.

---

## What This Model Is

`SplayTreeGPT` extends `HierarchicalTreeGPT` (the proven parent) with a
runtime splay bias mechanism on the binary tree gates. The tree topology,
leaf capsule groups, and learned gate weights are identical. The only
addition is:

1. **EMA frequency tracking**: After each forward pass, track per-leaf
   selection frequency via exponential moving average (decay=0.95).

2. **Log-odds bias correction**: For each internal gate, compute the
   ratio of left vs right subtree frequencies and add alpha * log(ratio)
   as an additive bias to the gate logit.

3. **Domain switch reset**: On domain change, reset all frequencies to
   uniform and all biases to zero, allowing the tree to re-adapt from
   scratch.

This implements the splay tree's "move to root" property in soft form:
frequently selected leaves get boosted gate probabilities along their
paths, without restructuring the tree topology (which would destroy
learned weights).

The splay biases are NOT learned by gradient descent. They are a pure
runtime optimization, adding zero trainable parameters.

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> hierarchical_tree -> splay_routing
                                                    (adds runtime splay
                                                     bias to tree gates)
```

---

## Key References

**Splay Trees** (Sleator & Tarjan, 1985): Self-adjusting binary search
trees with amortized O(log n) access and optimal working-set performance.
Our mechanism adapts the working-set property to MoE routing via soft
bias correction rather than tree rotations.

**Hierarchical Mixtures of Experts** (Jordan & Jacobs, 1994): Original
HME with gating networks at each tree node. Our work adds adaptive
runtime bias to the HME framework.

**Fast Feedforward Networks** (Belcak & Wattenhofer, ICML 2024): Binary
tree routing for conditional computation. No adaptive mechanism.

---

## Empirical Results

### Domain Shift Experiment (3 seeds)

Protocol: Train 300 steps on domain A (a-m names), switch to domain B
(n-z names) and train 200 steps. Compare val_loss on domain B.

| Model | Domain A val (mean) | Domain B val (mean) | Entropy after |
|-------|-------|-------|-------|
| Static tree | 0.5193 | **0.5085** | 0.633 |
| Splay tree (alpha=1.0) | 0.5195 | 0.5114 | 0.619 |

**Delta: +0.57% (splay WORSE than static on domain B).**

| Metric | Static | Splay | Verdict |
|--------|--------|-------|---------|
| Domain A quality | 0.5193 | 0.5195 | Tied |
| Domain B quality | **0.5085** | 0.5114 | Static better |
| Entropy reduction | -0.138 | -0.149 | Splay slightly sharper |
| Early convergence (step 50) | **0.5488** | 0.5534 | Static better |
| Wall-clock time | **11.9s** | 18.0s | Static much faster |

### Alpha Sweep (single seed=42)

| alpha | Domain A val | Domain B val |
|-------|-------------|-------------|
| 0.0 | 0.5285 | 0.5061 |
| 0.1 | 0.5214 | 0.5020 |
| **0.5** | 0.5225 | **0.4970** |
| 1.0 | 0.5192 | 0.5077 |
| 2.0 | 0.5266 | 0.5065 |
| 5.0 | 0.5274 | 0.4990 |

Moderate alpha (0.5) shows the best domain-B quality in single-seed sweep,
but this is 1 seed and the improvement (0.4970 vs 0.5061 at alpha=0) is
not reliable. High alpha (5.0) hurts domain-A quality by interfering with
gradient-based learning.

### Kill Criteria Assessment

**KC1: Splay reduces routing cost on non-stationary data.**
Result: KILL. Splay domain-B val_loss is +0.57% worse than static (3 seeds).
The splay bias correction provides no measurable adaptation advantage over
standard gradient-based gate recalibration at this scale.

**KC2: Splay overhead does not exceed routing savings.**
Result: KILL. Wall-clock overhead is +51.5%. This is entirely Python-level
overhead (EMA updates, `.tolist()` calls forcing MLX graph sync), not
algorithmic complexity. The actual FLOPs overhead is <0.1%, but the
implementation cost dominates at micro scale.

**Overall: KILL.**

---

## Why It Failed

Three factors explain the failure:

1. **Small tree, fast gradient adaptation.** With only L=8 leaves and
   I=7 gates, gradient descent can recalibrate the entire routing tree
   in ~50-100 steps. The splay mechanism's ~13-step head start (EMA
   half-life) is negligible within a 200-step adaptation budget. At
   L=256 with 255 gates, the gradient adaptation would be much slower,
   and splay's constant-time adaptation might show an advantage.

2. **Similar domains.** a-m and n-z names share most character frequency
   distributions. The routing restructuring needed for domain shift is
   minimal. With truly distinct domains (code vs natural language), the
   routing patterns would differ more, making the splay advantage larger.

3. **Splay-gradient interaction.** The splay bias changes the loss
   landscape that gradient descent optimizes. This can cause gradient
   steps to partially undo the splay correction, or vice versa. The
   two adaptation channels may conflict rather than complement.

The routing entropy comparison is the most interesting signal: splay
produces slightly sharper routing (0.619 vs 0.633 normalized entropy),
suggesting the mechanism IS influencing gate behavior, just not enough
to overcome the noise.

---

## What Was Learned

1. **Runtime bias correction on tree gates is mechanically sound**: the
   splay biases integrate cleanly with sigmoid gates, leaf probabilities
   still sum to 1, and parameter count is identical to the parent.

2. **EMA frequency tracking works**: leaf frequencies diverge from uniform
   and correctly track which leaves are being selected.

3. **The overhead problem is implementation, not algorithmic**: at <0.1%
   FLOPs, the mechanism is cheap; the +51.5% wall-clock overhead comes
   from Python-level synchronization, not from the computation itself.

4. **Moderate alpha (0.5) is better than strong alpha (1.0+)**: the
   alpha sweep suggests weak splay correction helps, while strong
   correction interferes with gradient-based learning.

5. **Small trees do not need adaptive routing**: at L=8, gradient descent
   is already fast enough. Adaptive routing would matter more at L=64+
   where gradient recalibration of 63+ gates is slow.

---

## Micro-Scale Limitations

1. **L=8 is too small for structural routing advantages.** The splay
   mechanism's O(log L) advantage over O(L) gradient recalibration is
   meaningless at L=8. At L=256 (production MoE scale), the gradient must
   recalibrate 255 gates vs splay's immediate bias correction.

2. **a-m vs n-z is a weak domain shift.** Character-level names in these
   two groups share most statistical properties. Real domain shifts (code
   -> math -> conversation) would require much more routing restructuring.

3. **Python implementation overhead masks algorithmic cost.** The +51.5%
   wall-clock overhead is an artifact of `.tolist()` synchronization in
   MLX. A C++ kernel would reduce this to <1%.

4. **Single EMA half-life tested.** The experiment uses gamma=0.95
   (half-life ~13 steps). Different half-lives might work better for
   different shift speeds. Not swept in the main experiment.

5. **Reset-on-switch is unrealistic.** The experiment resets splay state
   on domain switch. In production, domain shifts are gradual and
   unannounced. The splay mechanism should handle this naturally (old
   domain frequencies decay via EMA), but this is not tested.

---

## What Would Kill This

### At Micro Scale (tested, killed)

- **Splay does not reduce routing cost on non-stationary data.** KILLED.
  Domain-B val_loss is +0.57% worse than static (3 seeds). The splay bias
  provides no measurable adaptation advantage at L=8.

- **Splay overhead exceeds routing savings.** KILLED. Wall-clock overhead
  is +51.5%, entirely from Python-level synchronization.

### At Macro Scale (untested, but informative failure)

- **Gradient recalibration is fast at any scale.** If Adam with learning
  rate warmup can recalibrate even 255 gates in ~100 steps, splay's
  statistical adaptation offers no speed advantage at any scale.

- **Splay-gradient conflict at scale.** If the splay bias systematically
  opposes gradient updates (by shifting the loss landscape), the mechanism
  could actively hurt convergence at larger models where optimization is
  more delicate.

- **Working-set property irrelevant for MoE.** If MoE routing distributions
  do not exhibit the working-set pattern (recent accesses predict future
  accesses), the splay tree's theoretical optimality does not apply.

---

## Salvageable Direction

The alpha sweep hints that **very weak splay (alpha=0.1-0.5)** may help
as a soft regularizer rather than a hard routing optimizer. This reframes
the mechanism from "adaptive routing" to "frequency-aware routing
regularization" -- encouraging the tree to develop paths proportional to
leaf usage frequency. This is closer to Huffman shaping (already proven
at micro scale) than to splay tree adaptation.

If pursued, the next experiment should:
- Use alpha=0.5 (best from sweep)
- Test at L=32+ (larger tree where gradient recalibration is slower)
- Test with truly distinct domains (not just a-m vs n-z)
- Implement frequency tracking in C/MLX kernel to eliminate Python overhead

---

## Summary

The splay-tree adaptive routing mechanism is mechanically sound but
empirically dead at micro scale. Both kill criteria trigger: splay does
not improve domain-shift adaptation (+0.57% worse than static, 3 seeds)
and the implementation overhead is +51.5%. The fundamental problem is that
gradient descent recalibrates a small tree (L=8, 7 gates) faster than
splay's statistical EMA can accumulate useful frequency information. The
mechanism might matter at L=256+ where gradient recalibration is expensive,
but this is speculation, not evidence.
