# Gumbel-Sigmoid Routing Ablation: Research Digest

## Hypothesis

The default Gumbel-sigmoid router configuration (temperature anneal 2.0->0.5, top-2, no load balancing, 3000 steps) is suboptimal, and ablating temperature, top-k, gate type, load-balancing, and straight-through estimation will find a configuration that improves routing accuracy by >5% absolute while fixing zero-accuracy domains.

## What This Experiment Is

A systematic ablation study of the Gumbel-sigmoid router used for selecting top-k ternary LoRA adapters in the N=50 composition pipeline. The router takes mean-pooled hidden states from BitNet-b1.58-2B-4T and outputs per-adapter gate logits. We test 22 configurations across 5 axes while reusing the cached hidden states from all 49 domains, making the sweep very cheap (~41s total for all configs).

## Key References

- L2R (Learning to Route): Gumbel-sigmoid non-competing multi-adapter routing
- Gumbel-Softmax (Jang et al., 2017): differentiable discrete sampling
- Note: The auxiliary loss is NOT Switch Transformer load-balancing. It is L1 gate regularization (see MATH.md).

## Empirical Results

### Baseline

- In-experiment baseline (3000 steps, no LB): Top-1 71.84%, Top-2 85.10%
- N=50 original result: 86.33% top-2 (canonical K1 reference)
- Zero-accuracy domains at baseline: 2 (wikitext, dialogue)

### Full Ablation Leaderboard (top-2 accuracy, 22 configs)

| Rank | Config | Top-2 | Top-1 | Zero-Acc | Key Finding |
|------|--------|-------|-------|----------|-------------|
| 1 | sigmoid k=4 (baseline anneal) | 90.82% | 71.84% | 2 | Wider net, same router |
| 2 | **6000 steps, no LB (control)** | **90.41%** | **83.67%** | **2** | Training length is the main driver |
| 3 | sigmoid LB=0.1 + 6000 steps | 90.00% | 79.39% | 1 | LB adds wikitext recovery |
| 4 | sigmoid k=3 | 89.18% | 71.84% | 2 | Diminishing returns past k=3 |
| 5 | softmax anneal 2->0.5 | 86.12% | 74.29% | 1 | Competing gates viable |
| 6 | sigmoid fixed tau=2.0 | 85.71% | 74.69% | 2 | Fixed temp works well |
| 7 | baseline (anneal 2->0.5 k=2) | 85.10% | 71.84% | 2 | Reference (3000 steps) |
| 8 | sigmoid fixed tau=1.0 | 84.90% | 74.90% | 3 | Slightly worse |
| 9 | sigmoid h=128 | 84.69% | 71.02% | 3 | Narrower hurts slightly |
| 10 | sigmoid fixed tau=0.5 | 84.49% | 72.65% | 2 | OK |
| 11 | sigmoid lb=0.01 | 84.49% | 72.04% | 3 | Too little regularization |
| 12 | sigmoid anneal 5->0.5 | 84.08% | 70.20% | 4 | Too-high start temp |
| 13 | sigmoid lb=0.1 (3000 steps) | 83.88% | 70.61% | 4 | LB at 3000 steps hurts |
| 14 | sigmoid fixed tau=5.0 | 83.67% | 70.20% | 3 | Temperature too high |
| 15 | sigmoid lr=3e-3 | 82.24% | 73.47% | 4 | Overshoot |
| 16 | softmax anneal 1->0.1 | 82.04% | 68.98% | 3 | Low-temp anneal hurts |
| 17 | sigmoid anneal 1->0.1 | 78.37% | 66.53% | 4 | Annealing to 0.1 too aggressive |
| 18 | sigmoid k=1 | 71.84% | 71.84% | 7 | Top-1 only, very fragile |
| 19 | sigmoid lb=0.1 + anneal 1->0.1 | 64.90% | 53.27% | 12 | Bad combination |
| 20 | sigmoid lb=0.5 | 52.45% | 37.76% | 17 | Over-regularized |
| 21 | sigmoid straight-through | 34.49% | 24.90% | 28 | Hard forward kills learning |
| 22 | sigmoid fixed tau=0.1 | 4.08% | 4.08% | 47 | Catastrophic -- Gumbel swamps signal |

### K1 Assessment

**K1 FAIL**: Against the canonical N=50 baseline (86.33%), the best same-k config is the 6000-step no-LB control at 90.41%, an improvement of **4.08%**. This falls below the 5% threshold. The k=4 config reaches 90.82% (4.49% improvement), still below threshold.

The original run incorrectly assessed K1 as PASS by comparing against the in-experiment baseline (85.10%) instead of the N=50 original (86.33%). With the corrected baseline, K1 fails.

However, the more meaningful findings are (a) the training length control experiment and (b) zero-accuracy domain recovery, neither of which K1 was designed to measure.

### Training Length vs Load-Balancing (New Control)

The addition of a 6000-step no-LB control reveals that **training length is the primary driver of improvement**, not load-balancing:

| Config | Steps | LB | Top-2 | Top-1 | Zero-Acc |
|--------|-------|----|-------|-------|----------|
| Baseline | 3000 | 0 | 85.10% | 71.84% | 2 |
| LB only | 3000 | 0.1 | 83.88% | 70.61% | 4 |
| **More training only** | **6000** | **0** | **90.41%** | **83.67%** | **2** |
| LB + more training | 6000 | 0.1 | 90.00% | 79.39% | 1 |

Key observations:
- 6000 steps alone: +5.31pp top-2, +11.83pp top-1 over 3000-step baseline
- LB at 3000 steps: HURTS (-1.22pp top-2, -1.23pp top-1), adds 2 more zero-acc domains
- LB at 6000 steps: marginal effect on accuracy (-0.41pp top-2, -4.28pp top-1)

**LB's real value is zero-accuracy domain recovery, not aggregate accuracy.**

### Zero-Accuracy Domain Analysis

| Domain | Baseline (3k) | 6000 no-LB | LB+6000 | Softmax | Root Cause |
|--------|---------------|------------|---------|---------|------------|
| chemistry | 0% | 80% | **100%** | 20% | Confused with science_qa (cos=0.992) |
| wikitext | 0% | 0% | **40%** | 0% | Confused with history (cos=0.996) |
| dialogue | 0% | 0% | 0% | 20% | High variance (4.375, 13x typical) |
| debate | 0% | **70%** | 50% | 40% | Confused with legal/reviews (cos=0.96) |

More training alone fixes chemistry (0% -> 80%) and debate (0% -> 70%).
Adding LB further improves chemistry (80% -> 100%) and uniquely recovers wikitext (0% -> 40%).
Neither approach fixes dialogue.

### Hidden State Similarity Analysis

The zero-accuracy domains have the highest cosine similarity to confusing neighbors:
- chemistry vs science_qa: 0.992
- wikitext vs history: 0.996
- debate vs legal: 0.961, vs reviews: 0.961
- dialogue: low similarity to ALL domains (max 0.633), but extreme internal variance

### Intra-Domain Variance

| Domain | Variance | Routing Accuracy (baseline) |
|--------|----------|---------------------------|
| reasoning | 0.332 | 100% |
| cooking | 0.350 | 100% |
| wikitext | 0.637 | 0% |
| health | 0.656 | 100% |
| sql | 0.668 | 100% |
| debate | 0.840 | 0% |
| javascript | 0.852 | 100% |
| chemistry | 0.981 | 0% |
| dialogue | 4.375 | 0% |

Variance alone does not predict failure (javascript has high variance but routes perfectly due to unique content). The combination of high variance AND high centroid similarity to neighbors is the killer.

## Auxiliary Loss Description

The "load-balancing" loss is more accurately described as L1 gate activation regularization:

L_aux = alpha * sum_i mean_batch(g_i)

This penalizes total gate mass, pushing non-target gates toward 0. Combined with the BCE target loss (which pushes the target gate toward 1), the net effect is sharpened routing with suppressed off-target activations. See MATH.md for full derivation.

This is NOT the Switch Transformer loss (which computes f_i * p_i products for capacity balancing). The mechanism is different: L1 regularization vs load-fraction balancing.

## Key Findings

1. **Training length (3000 -> 6000 steps) is the single most impactful change** -- +5.3pp top-2, +11.8pp top-1. The router at 3000 steps is undertrained for 49 experts.
2. **L1 gate regularization (LB) has a specific niche**: it recovers zero-accuracy domains that more training alone cannot fix (wikitext: 0% -> 40%), but its aggregate accuracy effect is negligible to slightly negative.
3. **Temperature is forgiving in the range [0.5, 2.0]** -- fixed tau=1.0 or tau=2.0 work nearly as well as annealing. But tau=0.1 is catastrophic (Gumbel noise magnitude ~ -log(-log(U)) can reach 5-10, completely swamping logits at low temperature).
4. **Softmax (competing) gates are viable** -- 86.12% top-2 with only 1 zero-acc domain. The competition naturally prevents collapse into a few experts.
5. **Straight-through estimation destroys learning** -- 34.49% accuracy. The hard forward pass creates high-variance gradient estimates that prevent convergence at 49-class scale.
6. **Increasing top-k is a free lunch** -- k=3 (89.18%), k=4 (90.82%) with no downside except slightly more dilution in composition.
7. **The zero-accuracy problem has two distinct causes**:
   - **Expert collapse** (chemistry, debate, wikitext): high centroid similarity causes the router to consistently prefer a similar-looking neighbor. More training + L1 regularization fixes this.
   - **Unroutable domains** (dialogue): internal heterogeneity is too high for mean-pooled features. Requires per-token or attention-based routing.

## Recommended Configuration

For the N=50 composition pipeline:
```
gate_type: sigmoid
temperature: anneal 2.0 -> 0.5
top_k: 2 (for composition quality) or 3 (for routing accuracy)
load_balance_alpha: 0.1
n_steps: 6000
lr: 1e-3
hidden_dim: 256
```

This gives 90.00% top-2 accuracy with only 1 unroutable domain (dialogue). The 6000-step no-LB control achieves 90.41% top-2 with slightly better top-1 (83.67%), but retains 2 zero-accuracy domains. Choose based on whether zero-domain recovery or peak accuracy matters more.

## Limitations

1. **Micro scale only**: 49 domains, 20 train / 10 val samples per domain, 2B parameter base model
2. **Hidden states cached from base model** (no LoRA applied during extraction) -- slight discrepancy from online extraction
3. **Single seed**: all configs use seed=42. Variance across seeds unknown
4. **Composition PPL not measured**: we test routing accuracy only, not downstream composition quality
5. **Dialogue remains unroutable**: mean-pooled features fundamentally cannot separate this domain. Would need per-token routing, attention-weighted pooling, or domain-specific data augmentation
6. **K1 FAILS against canonical baseline**: the best improvement (4.49% with k=4) falls short of the 5% threshold when measured against the correct N=50 baseline of 86.33%

## What Would Kill This

- **Cross-seed variance > 5%**: If the 90% accuracy from 6000 steps is not reproducible, the improvement claim fails
- **Composition PPL worse**: If better routing accuracy does not translate to better gamma_routed, the accuracy metric is misleading
- **Scale dependence**: If the optimal configuration changes at N=100 or N=200, the ablation results are not transferable
