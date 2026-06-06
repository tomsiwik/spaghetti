# Expert Pruning from Composition: Research Digest

## Hypothesis

Pruning low-quality experts from a SOLE composition of N=50 LoRA adapters
improves aggregate model quality (PPL and/or MMLU accuracy) by >1%, and the
optimal subset can be identified via O(N log N) ranking rather than O(N^2)
greedy search.

## What This Experiment Is

A macro-scale investigation of the quality-quantity tradeoff in SOLE
composition. The experiment answers three questions:

1. **Does pruning help?** Remove the bottom-20% experts (by individual domain
   PPL) and measure whether the composed model improves.

2. **Is quality ranking stable?** Test whether domain PPL ranking, LOO
   contribution ranking, and MMLU ranking agree (Kendall tau >= 0.6).

3. **Is ranking-based selection scalable?** Compare greedy forward selection
   against rank-ordered selection. If they agree, O(N log N) ranking is
   sufficient for N=500+.

## Key References

| Paper | Relevance |
|-------|-----------|
| TIES-Merging (Yadav 2023) | Parameter-level sign conflict resolution; may be redundant under orthogonality |
| DARE (Yu 2023) | Random parameter dropout for merging; complementary to expert-level pruning |
| REAP (Cerebras 2025) | Router-weighted expert pruning for MoE; 50% lossless compression |
| EASY-EP | Output-aware importance scoring; pruned model sometimes exceeds full model |
| Shapley-MoE | Monte Carlo Shapley for expert importance; collapses to LOO under orthogonality |
| LoRA Soups/CAT (2024) | CAT weights converge to ~1.0; cannot identify weak experts |
| exp_leave_one_out_expert_ranking | LOO methodology for contribution scoring |
| exp_composition_dropout_robustness | Bootstrap robustness of random 80% subsets |
| exp_pilot50_composition_quality | K1 FAIL: 127% degradation at equal weight; K3 PASS: all domains beat base |

## Lineage

```
exp_distillation_pilot_50 (supported, 98% win rate)
    |
    +---> exp_pilot50_composition_quality (K1 FAIL, K3 PASS)
    |         |
    |         +---> exp_individual_expert_held_out (diagnosis: distillation vs composition)
    |         |
    |         +---> exp_selective_composition_mmlu (top-k by domain relevance)
    |
    +---> exp_composition_weight_sensitivity (CAT weights -> ~1.0)
              |
              +---> exp_expert_pruning_from_composition (THIS)
```

## Empirical Results

TO BE FILLED from `/workspace/llm/results/expert_pruning_composition/results.json`

### Reference Quality

| Metric | All-50 Composed | Base Model |
|--------|----------------|------------|
| PPL (Set A) | TO BE FILLED | TO BE FILLED |
| PPL (Set B) | TO BE FILLED | TO BE FILLED |
| MMLU Accuracy | TO BE FILLED | TO BE FILLED |

### Bottom-20% Removal

| Metric | All-50 | Top-40 (pruned) | Delta |
|--------|--------|-----------------|-------|
| PPL (Set A) | TO BE FILLED | TO BE FILLED | TO BE FILLED |
| MMLU | TO BE FILLED | TO BE FILLED | TO BE FILLED |

### Accumulation Curve (PPL)

| k | PPL (Set A) | Delta from Base (%) |
|---|-------------|-------------------|
| 1 | TO BE FILLED | TO BE FILLED |
| 5 | TO BE FILLED | TO BE FILLED |
| 10 | TO BE FILLED | TO BE FILLED |
| 20 | TO BE FILLED | TO BE FILLED |
| 30 | TO BE FILLED | TO BE FILLED |
| 40 | TO BE FILLED | TO BE FILLED |
| 50 | TO BE FILLED | TO BE FILLED |

### Accumulation Curve (MMLU)

| k | MMLU Accuracy | Delta from Base (pp) |
|---|--------------|---------------------|
| 5 | TO BE FILLED | TO BE FILLED |
| 10 | TO BE FILLED | TO BE FILLED |
| 20 | TO BE FILLED | TO BE FILLED |
| 30 | TO BE FILLED | TO BE FILLED |
| 40 | TO BE FILLED | TO BE FILLED |
| 50 | TO BE FILLED | TO BE FILLED |

### Ranking Stability

| Pair | Kendall Tau | p-value |
|------|------------|---------|
| Domain vs LOO | TO BE FILLED | TO BE FILLED |
| Domain vs MMLU | TO BE FILLED | TO BE FILLED |
| Set A vs Set B | TO BE FILLED | TO BE FILLED |

### Greedy vs Ranking

| k | Greedy PPL | Ranked PPL | Discrepancy (%) |
|---|-----------|-----------|----------------|
| 1 | TO BE FILLED | TO BE FILLED | TO BE FILLED |
| 5 | TO BE FILLED | TO BE FILLED | TO BE FILLED |
| 10 | TO BE FILLED | TO BE FILLED | TO BE FILLED |

### Kill Criteria

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| K1: Bottom-20% removal improves PPL >1% | >1% improvement | TO BE FILLED | PENDING |
| K2: Ranking stability (Kendall tau) | >= 0.6 | TO BE FILLED | PENDING |
| K3: Ranking matches greedy (scalable) | <0.5% discrepancy | TO BE FILLED | PENDING |

**Verdict: PENDING**

## Interpretation

### If K1 KILLED (pruning does not help >1%)

This is actually a **strong positive result** for SOLE. It means:
- Under structural orthogonality, every expert contributes non-negatively
- "More experts is always better" holds empirically at N=50
- No quality ceiling from composition -- safe to scale to N=500+
- Expert evolution (clone-and-compete) can focus on IMPROVING experts rather
  than REMOVING them
- The -3.71pp MMLU regression is NOT caused by bad experts but by distribution
  shift (all experts shift logits away from base calibration)

### If K1 PASSES (pruning helps >1%)

This reveals a pruning opportunity:
- Some experts actively harm the composition
- Within-cluster interference or low-quality distillation creates harm
- SOLE's Evolve phase should include quality-based culling
- Optimal composition size k* < N exists and must be determined
- Scaling to N=500 requires periodic quality audits

### If K2 KILLED (rankings unstable)

Rankings are metric-dependent -- "quality" means different things on different
evaluations. This is problematic for automated pruning:
- Cannot trust any single metric for pruning decisions
- Need multi-metric ensemble or domain-specific pruning
- Complicates the Evolve phase (which metric drives tournament?)

### If K3 KILLED (greedy outperforms ranking)

Expert interactions are significant despite low cosine similarity:
- Ranking-based pruning at O(N log N) is insufficient
- Need O(N^2) greedy or Shapley-based approaches
- Does not scale to N=500 without approximation
- Suggests orthogonality is insufficient for independent composition

## Limitations

1. **N=50 only.** Results may not extrapolate to N=500. At larger N, more
   experts means more potential for within-cluster interference, but also
   more redundancy (easier to prune safely).

2. **Domain PPL ranking from contaminated eval.** The pilot50 benchmark used
   the last 100 training examples for eval (see PAPER.md caveats). The ranking
   reflects training data fit, not held-out generalization. LOO and MMLU
   rankings are uncontaminated.

3. **Calibration text composition may bias results.** If calibration texts
   over-represent some domains, experts in those domains will rank higher.
   The A/B split tests this (K2 stability criterion).

4. **4-bit quantization.** Both composed and base models use NF4. Absolute
   values may differ from FP16, but rankings should be preserved.

5. **Equal-weight composition only.** Does not test weighted composition
   (1/sqrt(N), PPL-probe weights, etc.). Pruning under weighted composition
   may behave differently.

6. **MMLU is a noisy metric for individual adapters.** Most adapters have
   no direct MMLU counterpart. MMLU measures "does the adapter not break
   general knowledge?" rather than "does the adapter help on its domain?"

## What Would Kill This

### At This Scale (N=50)
- K1: Removing bottom-20% does not improve PPL >1%. (Actually a positive
  finding -- see interpretation above.)
- K2: Rankings unstable (tau < 0.6). Would mean quality is metric-dependent.
- K3: Greedy materially outperforms ranking. Would mean expert interactions
  are significant and O(N log N) is insufficient.

### At Production Scale (N=500)
- The PPL accumulation curve saturates before N=50 (diminishing returns)
- Within-cluster interference grows superlinearly with cluster size
- Quality ranking correlation with downstream tasks is < 0.3
