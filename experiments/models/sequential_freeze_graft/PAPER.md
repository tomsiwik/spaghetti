# Sequential Freeze-Graft-Calibrate: Research Digest

## Hypothesis

Sequential freeze-graft-calibrate at N>2 domains maintains cumulative
degradation growth within 2x between N=2 and N=4 grafts, with calibration
cost per graft growing at most linearly with N.

**Falsifiable**:
- KC1 KILL: cumulative degradation grows >2x between N=2 and N=4
- KC2 KILL: calibration cost per graft grows superlinearly with N

---

## What This Model Is

`SequentialFreezeGraftGPT` uses the same architecture as `HierarchicalTreeGPT`
(depth-3 binary tree, 8 leaf capsule groups, beam=2). The model tests
the sequential contribution protocol: can multiple contributors add
domain expertise one at a time, each freezing their work for the next
contributor, without cumulative quality collapse?

Protocol:
```
Step 0: Train base tree on domain A (all 8 leaves, 400 steps)
Graft 1: Freeze A's subtree (leaves 0-3). Graft B into leaves 4-7. Calibrate.
Graft 2: Freeze half of B (leaves 4-5). Graft C into leaves 6-7. Calibrate.
Graft 3: Freeze half of C (leaf 6). Graft D into leaf 7. Calibrate.
```

Two calibration strategies tested:
- **All-unfrozen**: calibrate all unfrozen gates + leaves on mixed data
- **Selective**: calibrate only root + graft-point gates (from minimal_graft_recal)

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> hierarchical_tree -> split_freeze_protocol -> sequential_freeze_graft
                              (tree routing)       (N=2 protocol)           (N=2,3,4 protocol)
                                     |
                                     +-> subtree_grafting -> minimal_graft_recal
                                         (tree composition)  (selective calibration)
```

---

## Key References

**Split-Freeze Protocol (this project)**: Validated single freeze-graft (N=2)
with right-tree calibration (+0.09%). Established that grafted leaves must
be trainable during calibration.

**Minimal Graft Recalibration (this project)**: Showed root+graft-point
gates (3/7) are sufficient for single graft. +0.19% vs all-gates.

**Progressive Neural Networks** (Rusu et al., 2016): Add new columns while
freezing old ones. Our sequential protocol is the tree-structured analog
with the same fundamental problem: interference accumulates.

**PackNet** (Mallya & Lazebnik, 2018): Per-neuron freezing for continual
learning. Reports increasing interference with task count.

---

## Empirical Results

### Main Experiment (3 seeds, fixed 200-step calibration)

#### All-Unfrozen Calibration

| Graft | N | Mean Max Degradation | Cal Params | Mean Cal Time |
|-------|---|---------------------|------------|---------------|
| 1     | 2 | +3.72%              | 66,576     | 3.3s          |
| 2     | 3 | +6.73%              | 33,548     | 2.9s          |
| 3     | 4 | +13.58%             | 17,164     | 2.9s          |

**KC1: KILLED.** Ratio |deg(N=4)| / |deg(N=2)| = 3.65x (threshold: 2.0x).
Cumulative degradation grows superlinearly.

**KC2: PASSES.** Calibration cost per domain decreases with N (ratio 0.13x).
More tree is frozen, so fewer params to calibrate.

#### Selective Calibration (root + graft-point gates only)

| Graft | N | Mean Max Degradation | Cal Params |
|-------|---|---------------------|------------|
| 1     | 2 | +24.24%             | 520        |
| 2     | 3 | +35.34%             | 520        |
| 3     | 4 | +32.99%             | 520        |

Selective calibration is catastrophically insufficient for sequential
grafting. At N=2, split_freeze_protocol showed that gates-only calibration
was already borderline (+2.5%); at N>2, the routing complexity overwhelms
the capacity of a few gate parameters.

#### Per-Seed Detail (all-unfrozen)

| Seed | N=2 Max Deg | N=3 Max Deg | N=4 Max Deg | Ratio N=4/N=2 |
|------|-------------|-------------|-------------|----------------|
| 42   | +5.11%      | +6.83%      | +19.89%     | 3.89x          |
| 123  | +4.01%      | +8.40%      | +14.63%     | 3.65x          |
| 777  | +2.03%      | +4.95%      | +6.22%      | 3.07x          |

All three seeds show superlinear growth. Domain A (frozen first)
consistently suffers the worst degradation.

### Extended Calibration (can more steps help?)

| Schedule         | Cal Steps     | N=4 Max Deg | Ratio N=4/N=2 | Verdict |
|------------------|---------------|-------------|----------------|---------|
| fixed_200        | [200,200,200] | +10.48%     | 3.92x          | KILL    |
| scaled_1.5x      | [200,300,400] | +13.09%     | 4.90x          | KILL    |
| scaled_2x        | [200,400,600] | +9.70%      | 3.63x          | KILL    |

More calibration helps marginally (10.48% -> 9.70% at 3x budget) but
does not change the verdict. The degradation is structural, not due to
insufficient calibration convergence.

---

## Key Findings

### 1. Cumulative degradation is superlinear

Each graft compounds the routing drift on previously frozen domains.
Domain A, frozen first, degrades from +3.7% (N=2) to +13.6% (N=4).
The acceleration between grafts is consistent across seeds.

### 2. First-frozen domains suffer most

Domain A is always the worst case because it endures all subsequent
grafts. The degradation hierarchy follows chronological freeze order:
```
Seed 42, N=4: a_f=+19.89%, g_m=+17.25%, n_s=+8.08%
```

### 3. Calibration cost is sublinear (not the problem)

The number of calibration parameters decreases with N because more of
the tree is frozen. KC2 passes easily. The bottleneck is not cost but
quality: the remaining unfrozen parameters cannot compensate for the
accumulated routing drift.

### 4. Selective calibration is dead for sequential grafting

The minimal graft recalibration approach (root + graft-point gates, 520
params) that worked at N=2 catastrophically fails at N>2 (24-35%
degradation). The routing problem at N>2 exceeds the capacity of gate-only
calibration.

### 5. More calibration steps do not rescue the protocol

Tripling the calibration budget from 200 to 600 steps reduces degradation
only marginally (10.5% -> 9.7% at N=4). The problem is structural: the
binary tree topology cannot cleanly separate N>2 domains with a cascade
of binary decisions when most of the tree is frozen.

---

## Protocol Specification

Based on these results, the sequential freeze-graft protocol has a
practical limit at N=2 with the tested tree depth:

```
Sequential Freeze-Graft Protocol:
  N=2:  VIABLE. Max degradation +3.7% with 200-step all-unfrozen calibration.
  N=3:  MARGINAL. Max degradation +6.7%. Usable if 7% degradation acceptable.
  N=4:  NOT VIABLE. Max degradation +13.6%, growing superlinearly.

For N>2, alternatives to sequential grafting:
  - Weight averaging (zero-shot, +1.5% at N=2, scales better)
  - Full joint recalibration (unfreeze all, recalibrate on mixed data)
  - Deeper trees with more leaves per domain
  - Flat MoE composition instead of tree-structured grafting
```

---

## Micro-Scale Limitations

1. **Fixed tree depth.** A depth-3 tree with 8 leaves constrains the
   allocation: domain D_3 gets only 1 leaf (vs D_0's 4 leaves). A
   depth-5 tree (32 leaves) would allow more balanced allocation and
   might reduce degradation, but would also increase routing complexity.

2. **Unequal domain capacity.** The progressive halving allocation gives
   the first domain 4x the capacity of the last. An alternative protocol
   that pre-allocates equal subtrees for each domain (requiring N to be
   known in advance) might perform better but contradicts the "sequential
   contribution" premise.

3. **Character-level data.** The 4 quaternary domains (a-f, g-m, n-s,
   t-z) share most character distributions. With truly distinct domains,
   the routing problem might be easier (cleaner separation) or harder
   (larger representation shift).

4. **Attention is frozen throughout.** The shared attention layers never
   adapt to the multi-domain context. At macro scale, fine-tuning
   attention during calibration might help but would increase cost and
   complicate the "frozen contribution" model.

5. **Quaternary split is unbalanced.** Domain sizes range from 3,667
   (t_z) to 11,629 (g_m), a 3.2x ratio. This introduces a confound
   between domain difficulty and domain size.

---

## What Would Kill This

### At Micro Scale (tested, killed)

- **KC1: Degradation ratio N=4/N=2 > 2.0x.** KILLED at 3.65x (3 seeds).
  Cumulative degradation is superlinear. The sequential freeze-graft
  protocol does not scale beyond N=2.

- **KC2: Calibration cost superlinear.** SURVIVED at 0.13x (strongly
  sublinear). Cost is not the issue; quality is.

### What Would Rescue This

- **Deeper trees (D=5+) with equal allocation.** More leaves per domain
  could reduce the routing pressure. But this contradicts the "one
  contributor at a time" sequential model.

- **Full recalibration instead of selective.** Unfreezing ALL parameters
  including previously frozen domains during calibration. But this
  violates the freeze guarantee (domain A's knowledge is no longer
  structurally preserved).

- **Non-tree composition.** Flat MoE with concatenation + pruning +
  calibration (the validated composition protocol from VISION.md)
  already handles N=5 at +1.6%. The tree structure adds overhead
  without benefit for sequential grafting.

---

## Summary

The sequential freeze-graft protocol fails KC1: cumulative degradation
grows 3.65x between N=2 and N=4, exceeding the 2.0x threshold. The
failure is structural, not addressable by more calibration (tested at
3x budget with no meaningful improvement). KC2 passes: calibration cost
per graft is sublinear. Selective calibration (minimal_graft_recal) is
catastrophically insufficient at N>2.

The practical implication: for multi-domain contribution, use the flat
MoE composition protocol (concatenation + pruning + calibration from
VISION.md) which handles N=5 at +1.6%. Tree-structured sequential
grafting is viable only at N=2 where it was already validated by
split_freeze_protocol. The progressive halving allocation creates an
irrecoverable capacity imbalance that compounds routing drift.

The positive finding: calibration cost is not the bottleneck. The
parameter count for calibration decreases as N grows (more tree frozen).
If the routing quality problem is solved (e.g., via deeper trees or
alternative composition), the cost profile is favorable.
