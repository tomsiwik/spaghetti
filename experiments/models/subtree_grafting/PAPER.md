# Subtree Grafting Composition: Research Digest

## Hypothesis

Composing domain experts by grafting trained subtrees onto a shared root
(preserving domain routing decisions intact) will match or beat weight
averaging composition, because grafting avoids the function-space gap from
parameter blending.

**Falsifiable**: If subtree grafting composition is >3% worse than weight
averaging composition, or grafting produces >5% degradation on the donor
subtree's original domain, the approach is dead.

---

## What This Model Is

`SubtreeGraftingGPT` uses the same architecture as `HierarchicalTreeGPT`
(depth-3 binary tree, 8 leaf capsule groups, beam=2). The difference is
purely in the **composition protocol**:

Instead of weight-averaging all domain-specific tree parameters (which
blends routing decisions from different domains into meaningless midpoints),
subtree grafting **assigns each domain its own subtree** and composes by
combining subtrees:

1. **Pretrain** base model on all data (shared initialization)
2. **Assign**: domain A owns the left subtree (leaves 0-3, gates 1,3,4),
   domain B owns the right subtree (leaves 4-7, gates 2,5,6)
3. **Fine-tune** each domain's assigned subtree only (other subtree frozen)
4. **Graft**: combine left subtree from A's model + right subtree from B's model
5. **Calibrate** all gates on mixed data (100 steps) -- the root gate
   learns to route between domain subtrees

### Why It Exists

Weight averaging is the current best composition method for the hierarchical
tree (+0.17% gap vs joint, from the hierarchical_tree experiment). But weight
averaging a sigmoid gate between two domain-specific versions produces a
routing function that neither domain learned. The nonlinearity means:

```
sigma((w_A + w_B)/2 * x)  !=  (sigma(w_A * x) + sigma(w_B * x)) / 2
```

Subtree grafting preserves each domain's routing decisions exactly. The root
gate becomes a domain router, and all within-subtree routing is preserved.

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> hierarchical_tree -> subtree_grafting
                              (tree routing)       (tree composition)
```

---

## Key References

**Hierarchical Mixtures of Experts** (Jordan & Jacobs, 1994): The original
HME with gating networks at each tree node. Our grafting extends HME
composition by assigning subtrees to domains.

**TIES Merging** (Yadav et al., 2023): Resolves sign conflicts in delta
merging. Grafting sidesteps the sign conflict problem entirely by not
blending parameters.

**DARE Merging** (Yu et al., 2023): Random drop + rescale before merging.
Like TIES, addresses the parameter blending problem that grafting avoids.

**Fast Feedforward Networks** (Belcak & Wattenhofer, ICML 2024): Binary
tree of FFN layers. Tree decomposition for routing, same structural prior.

---

## Empirical Results

### Experiment v1 (unfair calibration: root-only 50 steps vs all-gates 100 steps)

| Method | Val Loss (mean, 3 seeds) | Gap vs Joint | Gap vs Wt Avg |
|--------|-------------------------|-------------|--------------|
| Joint training | 0.5189 | -- | -- |
| Weight averaging | 0.5198 | +0.19% | -- |
| Subtree grafting (root-only, 50 cal) | 0.5384 | +3.76% | +3.57% |

Kill criterion 1 TRIGGERED (+3.57% > 3%). But this comparison was unfair:
grafting used root-only calibration (260 params, 50 steps) vs weight
averaging's all-gates calibration (1,820 params, 100 steps).

### Diagnostic: Calibration Budget Sweep (seed=42)

| Configuration | Val Loss | Gap vs Wt Avg |
|---------------|----------|--------------|
| Weight avg (100 cal) | 0.5137 | -- |
| Graft root-only 50 | 0.5272 | +2.63% |
| Graft root-only 100 | 0.5262 | +2.42% |
| Graft root-only 200 | 0.5239 | +1.98% |
| Graft all-gates 50 | 0.5240 | +1.99% |
| Graft all-gates 100 | 0.5225 | +1.70% |
| Graft all-gates 200 | 0.5206 | +1.34% |

All-gates calibration at matched budget (100 steps) reduces the gap from
+3.57% to +1.70%. The gap continues to shrink with more calibration.

### Experiment v2 (fair: all-gates 100 steps for both methods)

| Method | Val Loss (mean, 3 seeds) | Per-seed | Gap vs Joint | Gap vs Wt Avg |
|--------|-------------------------|----------|-------------|--------------|
| Joint training | 0.5208 | 0.5123, 0.5213, 0.5289 | -- | -- |
| Weight averaging | 0.5222 | 0.5134, 0.5284, 0.5248 | +0.27% | -- |
| **Subtree grafting** | **0.5257** | **0.5205, 0.5284, 0.5282** | **+0.94%** | **+0.67%** |

**Kill criterion 1: PASSES** (+0.67%, well within 3% threshold).

### Per-Domain Preservation (kill criterion 2)

| Domain | Joint | Wt Avg | Graft | Graft Degradation |
|--------|-------|--------|-------|-------------------|
| a_m | 0.5204 | 0.5275 (+1.38%) | 0.5273 (+1.34%) | PASS |
| n_z | 0.5346 | 0.5284 (-1.15%) | 0.5346 (+0.00%) | PASS |

**Kill criterion 2: PASSES** (no domain degradation >5%).

---

## Key Findings

### 1. Grafting works but does not beat weight averaging

Subtree grafting is +0.67% worse than weight averaging (well within the
3% kill threshold). The hypothesis that preserving routing decisions would
be superior to blending is directionally unsupported at micro scale -- both
methods produce similar results, with weight averaging slightly better.

### 2. The calibration budget matters enormously

The v1 experiment appeared to kill the hypothesis at +3.57%. The diagnostic
revealed this was largely a calibration budget artifact. With matched
calibration (all-gates, 100 steps), the gap drops to +0.67%. Root-only
calibration (260 params) is insufficient; the internal gates need to
re-coordinate after grafting.

### 3. Grafting is cheaper during fine-tuning

Each domain fine-tunes only half the tree (66K vs 133K trainable params).
This 2x reduction in per-domain fine-tuning cost is a practical advantage,
especially at scale. The cost trade-off:
- Fine-tuning: grafting ~2x cheaper
- Calibration: similar cost (both need all-gates calibration)
- Total: grafting marginally cheaper overall

### 4. Domain preservation is excellent

Grafting preserves donor domain quality within +1.34% (comparable to weight
averaging at +1.38%). Neither method significantly degrades individual
domain quality.

---

## Micro-Scale Limitations

1. **Binary split only.** With N=2 domains and a binary tree, the subtree
   assignment is natural (left/right). For N>2, the tree does not decompose
   cleanly unless N is a power of 2. Variable-depth splits or a different
   tree structure would be needed.

2. **Similar domains.** a-m vs n-z character names share most character
   distributions. With truly distinct domains (code vs prose), the routing
   preservation advantage of grafting might be larger or smaller.

3. **Small tree.** At D=3 with 8 leaves, each domain gets only 4 leaves.
   At D=5 with 32 leaves, each domain would get 16 leaves -- more room
   for within-subtree specialization. The grafting advantage may scale
   with tree depth.

4. **All-gates recalibration blurs the distinction.** When we calibrate
   ALL gates (not just root) after grafting, the internal gates can drift
   from their fine-tuned values. This partially undermines the "preserve
   routing decisions" argument. However, root-only calibration is clearly
   insufficient (+2.42%), suggesting the gates DO need some re-coordination
   after grafting.

---

## What Would Kill This

### At Micro Scale (tested, survived)

- **Grafting >3% worse than weight averaging.** SURVIVED at +0.67% with
  matched calibration. The v1 result (+3.57%) was a calibration artifact.

- **Grafting >5% domain degradation.** SURVIVED. Max degradation +1.34%
  (comparable to weight averaging's +1.38%).

### At Macro Scale (untested)

- **N>2 domains.** The binary subtree assignment does not generalize cleanly.
  If N=5 domains each need a subtree, the tree must be depth-5+ (32 leaves)
  with unequal subtree sizes. This is where Huffman tree shaping becomes
  important.

- **Diverse domains.** With truly distinct domains (code, math, prose),
  the within-subtree routing may be much more domain-specific. Grafting
  should benefit more from preservation in this case. Or it could hurt
  if the base tree's structure (learned on mixed data) is incompatible
  with domain-specific subtree specialization.

- **Scale.** At d=4096 with L=64, each domain gets 32 leaves. The
  within-subtree routing has much more capacity. The calibration budget
  finding (all-gates calibration needed) suggests the function-space
  reconciliation problem does not disappear with grafting -- it just moves
  from "blended weights" to "reconnected gates."

---

## Summary

Subtree grafting is a viable composition method for tree-structured experts
(+0.67% vs weight averaging, both kill criteria pass). It does not
demonstrate the hoped-for superiority of preserving routing decisions --
at micro scale, the simpler weight averaging approach is slightly better.

The main finding is nuanced: **grafting works, but the gates still need
recalibration after grafting** (root-only calibration is insufficient).
This means the "preserved routing decisions" argument is partially
undermined -- the gates drift during calibration, reducing the distinction
between grafting and averaging.

The practical value of grafting is in **reduced fine-tuning cost** (each
domain updates only half the tree) rather than in composition quality.
At macro scale with larger trees and more diverse domains, the routing
preservation advantage may emerge more clearly.
