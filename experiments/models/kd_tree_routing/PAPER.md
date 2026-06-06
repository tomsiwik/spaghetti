# KD-Tree Feature-Space Partitioning for Expert Routing: Research Digest

## Hypothesis

KD-tree-style axis-aligned feature-space partitioning at each tree node will
produce sharper, non-overlapping expert territories than unconstrained sigmoid
gates, improving routing quality and composition at micro scale.

**Falsifiable**: (1) KD-tree routing quality worse than softmax/sigmoid routing
at same active params. (2) Learned split dimensions don't produce semantically
meaningful partitions.

---

## What This Model Is

`KDTreeRoutingGPT` replaces each transformer block's MLP with a KD-tree of
capsule groups. Each internal tree node is a **KD-tree split node** that learns:

1. A projection vector v in R^d (the split direction)
2. A threshold tau (the split point)
3. Uses a shared temperature T that controls soft/hard boundary

The routing decision at node i: `p_left(x) = sigmoid(T * (v_i^T x - tau_i))`.

**Temperature annealing**: T starts at 1.0 (soft splits, smooth gradients) and
linearly anneals to 10.0 (near-hard partitioning). First 20% of training is
warm-up at T=1.0. This forces sharp routing by construction, independent of
whether the model learns to route sharply.

**Split diversity loss**: An auxiliary loss penalizes high cosine similarity
between sibling split directions, encouraging diverse partitions.

### How It Differs from hierarchical_tree

The parent model (`hierarchical_tree`) uses `sigmoid(w^T x + b)` at each gate.
This is mathematically equivalent to the KD-tree split at T=1. The key
differences are:

1. **Temperature parameter**: KD-tree has T that anneals from 1 to 10, forcing
   sharp routing. The hierarchical_tree relies on the model to learn sharpness
   via the entropy loss.
2. **Split diversity regularization**: Encourages orthogonal split directions
   at each tree level.
3. **Explicit threshold**: tau is a separate parameter from the projection
   (cosmetic -- mathematically equivalent to bias).

### Why It Exists

The hierarchical_tree experiment found routing entropy of 0.745 (nearly uniform
out of a maximum of 1.0). This near-uniform routing is a persistent problem at
micro scale -- routers never learn sharp specialization on character-level names
data. The KD-tree approach addresses this by **forcing** sharp partitioning via
temperature annealing rather than hoping the model learns it.

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> hierarchical_tree -> kd_tree_routing
                                                   (adds temperature annealing
                                                    and split diversity loss)
```

---

## Key References

**Fast Feedforward Networks** (Belcak & Wattenhofer, ICML 2024): Binary tree
of feedforward layers with learned conditional execution. 220x inference
speedup. The most directly related prior work -- applies tree-structured
binary routing to FFN layers.

**COMET** (KDD 2023): Hierarchical expert routing achieving +13% over hash
routing. Validates hierarchical partitioning for MoE routing.

**Hierarchical Mixtures of Experts** (Jordan & Jacobs, 1994): Original HME
with gating networks at each tree node. This work is a modern instantiation
with temperature-controlled gating.

---

## Empirical Results

### Single-Domain Quality (500 steps, 3 seeds)

| Model | Params | Val Loss (mean) | Per-seed |
|-------|--------|-----------------|----------|
| hierarchical_tree (parent) | 203,932 | 0.5168 | 0.5182, 0.5150, 0.5174 |
| **kd_tree_routing** | **203,932** | **0.5165** | **0.5161, 0.5181, 0.5153** |

**Delta: -0.06% (KD-tree negligibly better).** Within noise at 3 seeds.
Both models are essentially tied on single-domain quality.

Kill criterion 1 (KD-tree quality worse than softmax): **PASSES** (not worse).

### Routing Sharpness (the key diagnostic)

| Metric | KD-tree (T=10) | hierarchical_tree |
|--------|---------------|-------------------|
| Normalized entropy (L0) | 0.048 | 0.745 |
| Normalized entropy (L2) | 0.016 | 0.745 |
| Mean max leaf prob (L0) | 0.953 | 0.358-0.386 |
| Mean max leaf prob (L2) | 0.974 | 0.358-0.386 |

**The temperature annealing works as intended.** KD-tree achieves near-
deterministic routing (entropy 0.016-0.060) vs the hierarchical_tree's
near-uniform routing (0.745). The mean max leaf probability is 0.95+ (the
router is >95% confident in its top leaf), compared to 0.36-0.39 for the
parent model.

This demonstrates that **forced sharp routing via temperature annealing does
not degrade quality**. The model can partition tokens into exclusive expert
territories without sacrificing predictive performance.

### Composition Quality (300+200+100 steps, 3 seeds)

| Model | Mean Joint | Mean Composed | Mean Gap |
|-------|-----------|---------------|----------|
| hierarchical_tree | 0.5144 | 0.5398 | +4.93% |
| **kd_tree_routing** | **0.5179** | **0.5504** | **+6.27%** |

**KD-tree composition gap (+6.27%) is worse than hierarchical_tree (+4.93%).**

This is the key negative result. The sharp partitioning that helps single-domain
routing **hurts composition**. When domain-specific trees are weight-averaged, the
hard partition boundaries from temperature annealing create conflicts: domain A's
boundary at `v^T x = 0.3` and domain B's boundary at `v^T x = -0.1` average to
a boundary at `v^T x = 0.1`, which is meaningful for neither domain.

The hierarchical_tree's soft routing (entropy 0.745) provides more overlap between
domains, making weight averaging more forgiving.

### Split Direction Analysis

| Metric | Observed | Axis-aligned | Uniform |
|--------|----------|-------------|---------|
| Concentration | 0.04-0.07 | 1.0 | 0.016 |

Kill criterion 2 (semantically meaningful partitions): **MARGINAL.**
The learned projections are slightly more concentrated than uniform but far
from axis-aligned. At d=64, the splits use distributed representations across
many dimensions. The "KD-tree" framing (axis-aligned splits) is not what the
model learns -- it learns arbitrary hyperplane splits, just like the
hierarchical_tree's sigmoid gates.

The difference in top dimensions across layers (44, 24, 9, 13, 4, 48, 63, 46, ...)
shows diversity, but the low concentration (max 0.084) means no single dimension
dominates any split.

---

## Parameter Comparison

| Component | hierarchical_tree | kd_tree_routing | Difference |
|-----------|------------------|-----------------|------------|
| Routing params/layer | 455 | 455 | 0% |
| Capsule params/layer | 32,768 | 32,768 | 0% |
| Total params | 203,932 | 203,932 | 0% |
| Active capsules/token | 2/8 = 25% | 2/8 = 25% | Same |

The models have identical parameter counts and identical sparsity. The only
difference is the temperature-controlled sharpness mechanism.

---

## Micro-Scale Limitations

1. **Temperature annealing is the active ingredient, not KD-tree structure.**
   The KD-tree split node and the hierarchical_tree's sigmoid gate are
   mathematically equivalent at T=1. The temperature parameter could be added
   to the parent model without any structural change. This experiment tests
   temperature annealing more than KD-tree partitioning per se.

2. **Axis-alignment is not learned at d=64.** The "KD-tree" framing implies
   axis-aligned splits, but the model learns distributed projections. A true
   KD-tree would constrain each split to a single coordinate axis. We did not
   impose this constraint because it would severely limit expressiveness at
   d=64.

3. **Composition protocol may be suboptimal for sharp routing.** Weight
   averaging is known to work best when models operate in similar function
   spaces. Sharp partitioning pushes different domains into different regions,
   making averaging less effective. A composition protocol that recalibrates
   the split thresholds (not the directions) might recover the gap.

4. **Small tree on simple data.** As with the parent experiment, D=3 with
   8 leaves on character-level names is a minimal test. The partition structure
   may show more benefit with diverse data where tokens genuinely belong to
   different semantic regions.

---

## What Would Kill This

### At Micro Scale (tested)

- **KD-tree quality worse than softmax at same params.** SURVIVED (barely).
  KD-tree is -0.06% better than parent, within noise. Not killed but not
  a meaningful improvement either.

- **Composition gap >5% worse than parent.** BORDERLINE. KD-tree gap is
  +6.27% vs parent's +4.93%. The gap is 1.34pp worse, and one seed (seed=42)
  hit +9.35%. If composition is the primary use case, this is a concern.

### At Macro Scale (untested)

- **Temperature annealing hurts convergence.** The linear temperature ramp
  may cause optimization instability at larger scales where the loss landscape
  is more complex.

- **Sharp routing prevents cross-expert knowledge sharing.** At macro scale
  with diverse data, tokens may benefit from soft routing to multiple experts.
  Forcing hard partitioning could reduce model capacity.

- **Split directions collapse.** The diversity loss may be insufficient at
  deeper trees (D=8+). If split directions at different depths converge to
  similar projections, the tree degenerates into a chain.

---

## Key Insight

**Temperature annealing on binary tree routing produces near-deterministic
expert assignment (entropy 0.016-0.060) without degrading single-domain
quality, but at the cost of worse composition (+6.27% vs +4.93% gap).**

The result separates two concerns:
1. **Sharp routing is achievable** at micro scale via extrinsic temperature control.
   The persistent near-uniform routing finding (0.745 entropy) is a property of
   unconstrained sigmoid gates, not a fundamental limitation.
2. **Sharp routing hurts composition** because weight averaging requires soft
   boundaries to interpolate between domains gracefully.

**Implication for the project**: If the goal is composition quality, soft routing
(hierarchical_tree or flat softmax) is strictly better. If the goal is inference
efficiency (zero routing ambiguity, O(D) hard splits), the KD-tree approach works
but requires a different composition protocol (e.g., recalibrate thresholds only,
or use concatenation instead of weight averaging).

---

## Summary

The KD-tree routing experiment validates that temperature-annealed binary splits
can force sharp expert assignment (normalized routing entropy 0.016-0.060 vs
the parent's 0.745) without degrading single-domain quality. However, this
sharp routing worsens composition quality (+6.27% gap vs +4.93% for the parent).

The experiment reveals that near-uniform routing at micro scale is not a
fundamental limitation but a design choice: soft routing enables better
composition, while hard routing enables better inference efficiency. The
tradeoff is explicit and quantified.

The "KD-tree" framing adds minimal structural benefit beyond temperature
annealing -- the learned projections are not axis-aligned, and the split
node is mathematically equivalent to the parent's sigmoid gate at T=1.
The active ingredient is temperature control, not spatial partitioning.
