# Skip-List Routing under Composition: Research Digest

## Hypothesis

Skip-list multi-resolution routing (proven at -0.93% vs flat single-domain)
survives the shared-base composition protocol without degradation, and
confidence gates maintain non-uniform level-weight concentration under
composition.

**Falsifiable**: If skip-list composition gap exceeds flat composition gap
by >3pp, or if level-weight distribution collapses to uniform (weight above
Level 0 drops below 10%), the approach is killed.

---

## What This Model Is

This experiment tests whether `SkipListRoutingGPT` (the proven skip-list
multi-resolution router) works under the shared-base composition protocol.
The architecture is identical to the parent -- N=8 leaf experts organized
into 4 levels (Level 0: 8 experts, Level 1: 4, Level 2: 2, Level 3: 1),
with learned confidence gates controlling level assignment via stick-breaking.

The composition protocol is:
1. Pretrain base model on all domains (300 steps)
2. Fine-tune expert modules per domain, attention frozen (200 steps)
3. Compose by weight-averaging all expert module parameters
4. Calibrate routers + confidence gates on mixed data (100 steps)
5. Evaluate against jointly-trained baseline

The key question: when leaf experts are weight-averaged from different
domains, do the coarse experts (which average the already-averaged leaves)
become too diluted? Would this push confidence gates to collapse weight
to Level 0 (fine-grained), eliminating the adaptive depth advantage?

---

## Lineage in the Arena

```
gpt
`-- capsule_moe (flat softmax routing, G=8, k=2)
    `-- hierarchical_tree (binary tree, depth-3, beam=2)
        `-- skip_list_routing (multi-level adaptive depth)
            `-- skip_list_composition_test (composition protocol test)
```

---

## Key References

- **skip_list_routing (this project)**: Parent experiment. -0.93% vs flat
  single-domain, 60.6% level-weight concentration above Level 0.
- **hierarchical_tree (this project)**: Tree composition gap was +0.17%
  (vs flat's +0.26%). Proved hierarchical routing survives composition.
- **LoRA Soups (COLING 2025)**: Concatenation + calibration protocol.
  Our composition protocol follows this pattern.

---

## Empirical Results

### Setup
- d=64, N=8 experts, 32 capsules/expert, top_k=2
- 3 seeds (42, 123, 777)
- Binary domain split (a-m vs n-z)
- 300 pretrain + 200 finetune + 100 calibration steps

### Composition Quality

| Model | Total Params | Joint Val Loss | Composed Val Loss | Composition Gap |
|-------|-------------|----------------|-------------------|-----------------|
| flat (G=8, k=2) | 204,160 | 0.5205 | 0.5259 | +1.04% |
| tree (D=3, B=2, prior exp) | 203,932 | 0.5186 | 0.5195 | +0.17% |
| **skip (N=8, k=2)** | **206,732** | **0.5140** | **0.5131** | **-0.17%** |

Per-seed values:
- Flat joint:     [0.5138, 0.5153, 0.5324]
- Flat composed:  [0.5281, 0.5177, 0.5318]
- Skip joint:     [0.5093, 0.5081, 0.5246]
- Skip composed:  [0.5157, 0.5087, 0.5151]

**The skip-list composed model matches or slightly beats its own joint
baseline.** The composition gap is -0.17% (negative = composed is better).
This is the first routing variant to achieve a negative composition gap.

### Level-Weight Concentration

| Condition | Weight Above Level 0 | Avg Depth |
|-----------|---------------------|-----------|
| Single-domain (parent exp) | 60.6% | 1.576 |
| Joint training (this exp) | 91.6% | -- |
| **After composition** | **97.2%** | -- |

Composition INCREASES level-weight concentration. The confidence gates push
even more weight to coarse levels after composition + calibration.

### Per-Seed Level Distribution (Composed, averaged across layers)

| Seed | L3 (coarsest) | L2 | L1 | L0 (finest) | Above L0 |
|------|--------------|-----|-----|-------------|----------|
| 42   | 80.0% | 13.1% | 6.3% | 0.8% | 99.2% |
| 123  | 90.9% | 7.2% | 1.1% | 0.9% | 99.1% |
| 777  | 35.2% | 7.6% | 50.6% | 6.7% | 93.4% |

Seed 777 shows a different pattern: Level 1 dominates over Level 3 in some
layers. But even this seed maintains 93.4% above Level 0 -- far from
collapse.

### Kill Criteria Assessment

| Criterion | Threshold | Measured | Verdict (as-formulated) | Behavioral verdict |
|-----------|-----------|----------|-------------------------|--------------------|
| KC1: skip composition gap vs flat | >3pp worse | -1.20pp (better) | PASSES | **FAIL — mechanism-level** |
| KC2: level-weight collapse | <10% above L0 | 97.2% above L0 | PASSES | **FAIL — mechanism-level** |

### Verdict: KILLED (preempt — behavioral-mechanism-failure)

Both pre-registered KC pass as numerically formulated, but REVIEW-adversarial.md
identified that these passes occur via **degenerate routing collapse**, not via
the hypothesized adaptive multi-resolution routing. Per-seed L3 (coarsest)
weight: 80.0% (seed 42), 90.9% (seed 123), 35.2% (seed 777). In 2/3 seeds the
model routes 80-91% of token-weight to a single coarsest "expert" that is the
uniform mean of all averaged leaf experts. This is uniform averaging
masquerading as routing — the proposed "adaptive depth" mechanism is inactive.

**Antipattern match (researcher hat Step 5 pre-flight #6): "KC measures wrong
object."** KC2 was designed to detect L0-collapse (fine-grained dominance),
but the actual failure mode is L3-collapse (coarsest dominance). KC2 as
formulated is vacuously satisfied by the exact degeneracy it should catch.

**Cascade to F#664 (fixed-algebraic-blend family, iter-44, 2026-04-19):** when
the routing collapses to a single coarsest level whose weights are the
data-agnostic uniform mean of domain-averaged leaf experts, the composition
reduces to an identically-weighted linear blend — the F#664 kill family.
Under F#157 (equal-hierarchical: -7.29%, unequal: -7.05% at rank-matched
budget), uniform-averaging compositions are killed at ≥5pp degradation.
The -1.20pp "better than flat" number reflects that `mean(all averaged
experts)` happens to be a competent generalist on a character-level binary
domain split (a-m vs n-z are similar), not that skip-list multi-resolution
routing works under composition.

**Guardrail 1006 (behavioral outcomes over metrics):** the metric (composition
gap) passed, but the behavioral claim (multi-resolution adaptive routing
survives composition) is empirically falsified by the observed L3-collapse.

Both K#477 and K#478 are marked `fail` at the mechanism level despite passing
numerically, per the antipattern-match + behavioral-falsification rule.

---

## Analysis

### Why Composition Improves Skip-List Routing

The surprising result -- negative composition gap -- has a plausible
explanation:

1. **Weight averaging creates better coarse experts.** When leaf experts
   from different domains are averaged, the result is a generalist. This
   generalist IS what coarse experts are designed to be. Weight averaging
   pre-computes the "ensemble of domain experts" that skip-list coarse
   levels approximate via recursive child averaging.

2. **Calibration refines what matters.** With 100 calibration steps, the
   routers and confidence gates (1155 params/layer, 4620 total = 2.2% of
   model) re-learn level assignment. The gates discover that the averaged
   coarse experts are now BETTER generalists than before domain finetuning,
   and push even more weight to coarse levels.

3. **Flat routing lacks this benefit.** Flat softmax has no coarse/fine
   distinction. It must discriminate among 8 averaged experts using only
   one routing layer. Skip-list can fall back to "use the average of
   everything" (Level 3) for easy tokens, which is exactly what weight
   averaging produces.

### Comparison to Tree Composition

The hierarchical tree achieved +0.17% composition gap -- already good.
Skip-list achieves -0.17%. The difference (0.34pp) is within noise at
3 seeds, but the trend is clear: hierarchical routing structures are
composition-friendly.

The skip-list's advantage over the tree likely comes from the confidence
gates' ability to shift weight distribution after calibration. The tree
always traverses full depth; the skip-list adapts.

### Joint vs Single-Domain Level Weights

The parent experiment measured 60.6% above Level 0 for single-domain
training. This experiment's joint training baseline shows 91.6% above
Level 0. The difference (31pp) reflects the composition training protocol:
300 pretrain + 200 finetune vs 500 single-domain steps. The pretrain phase
on mixed data may encourage coarser routing because mixed-domain tokens
benefit more from generalist experts.

---

## Micro-Scale Limitations

1. **Binary domain split only.** Two domains (a-m vs n-z) is the simplest
   composition scenario. At N=5+ domains, the weight-averaged experts
   become more diluted, potentially reducing coarse expert quality.

2. **Small N=8.** With 8 leaf experts and only 3 coarse levels, the
   skip-list structure is minimal. At N=64+ (8 levels), the dilution
   from recursive coarse averaging could be more severe.

3. **Character-level data.** The a-m vs n-z split produces similar
   domains. With truly distinct domains (code vs prose vs math), the
   averaging effect might differ.

4. **3 seeds, no confidence intervals.** The -0.17% gap could be noise.
   The key finding is not the sign of the gap but that it is well within
   the +3% kill threshold.

5. **Training FLOPs not measured.** Skip-list soft training evaluates
   every leaf expert L+1 times per token (16x cost vs flat top-k=2).
   The quality improvement must be weighed against this cost.

6. **Joint training comparison is imperfect.** Joint training uses
   different random initialization per model type (via mx.random.seed).
   The skip-list joint baseline (0.5140) is better than flat (0.5205),
   which inflates the flat composition gap and deflates the skip gap.
   A fairer comparison would use delta from each model's own joint
   baseline, which is what the gap metric does.

---

## What Would Kill This

### At micro scale
- Skip-list composition gap >3% worse than flat: **REFUTED** (-1.20pp better)
- Level-weight collapse to uniform under composition: **REFUTED** (97.2% above L0)

### At macro scale (untested)
- **N=5+ domain composition**: More domains = more dilution in weight-averaged
  experts. If coarse experts degrade, level-weight concentration could drop.
  The current binary split (M=2) is the easiest case.
- **Diverse domains**: Code vs prose vs math produce more orthogonal experts.
  Weight averaging might produce poor generalists at coarse levels.
- **Training cost**: 16x expert evaluation cost during soft training makes
  skip-list routing expensive. At macro scale, hard routing during training
  may be necessary.
- **Calibration budget scaling**: 100 steps of calibration worked for M=2.
  At M=20+, the gates may need proportionally more steps.

---

## Next Steps

1. **Hard inference routing**: Implement confidence-threshold early stopping
   at inference time to measure actual FLOP savings. The 97.2% concentration
   above Level 0 suggests dramatic savings (most tokens stop at Level 3).

2. **N=5 domain composition**: Test with quintary domain split to see if
   level-weight concentration holds with more domains.

3. **Level-weight dynamics during calibration**: Track how confidence gates
   evolve during the 100 calibration steps. Do they start collapsed and
   recover, or maintain concentration throughout?
