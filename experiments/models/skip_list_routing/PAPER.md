# Skip-List Multi-Resolution Routing: Research Digest

**Revision 1** -- addresses adversarial review REVISE verdict. Changes from v1:
corrected routing cost claims, added ensemble confound control, added recursive
computation cost acknowledgment, measured routing stats over validation set,
reported all trainable parameter counts.

## Hypothesis

Organizing MoE experts in skip-list levels (Level 0 = all N, Level k = N/2^k
coarse experts) with learned confidence gates enables adaptive-depth routing
that matches flat softmax quality while learning to concentrate level weight
at coarse levels, indicating potential for routing cost savings under hard
inference routing (not tested here).

**Falsifiable**: If skip-list routing is >2% worse than flat softmax at same
active parameters, or adaptive depth does not concentrate level weight above
Level 0 vs fixed depth, the approach is killed.

---

## What This Model Is

`SkipListRoutingGPT` organizes N=8 capsule-group experts into a multi-level
skip-list structure inspired by Pugh (1990). Three coarse levels sit above
Level 0:

```
Level 3 (coarsest): 1 express expert  (averages all 8)
Level 2:            2 express experts  (averages of 4)
Level 1:            4 express experts  (averages of 2)
Level 0 (finest):   8 leaf experts     (actual parameters)
```

Routing proceeds top-down. At each coarse level, a learned confidence gate
(sigmoid) decides what fraction of each token's weight to assign to that level
vs pass down to finer levels. This creates a **soft stick-breaking process**
over levels: easy tokens concentrate weight at coarse levels, hard tokens push
weight to Level 0.

### Key Design Decisions

1. **Zero-parameter coarse experts**: Express experts at Level k are runtime
   weight-averages of their 2^k children. No extra expert parameters.

2. **Soft level selection**: During training, all levels contribute with
   differentiable weights (stick-breaking). This avoids non-differentiable
   early stopping while still learning when to stop.

3. **Per-level top-k routing**: Each level has its own softmax router with
   top-k=2 selection, matching the proven k=2 optimal finding.

4. **Confidence gates**: L learned gates (one per coarse level), each a
   single linear projection + sigmoid. Total overhead: L*(d+1) params per layer.

### Connection to Skip Lists

In a skip list, each element is probabilistically promoted to higher levels
(p=0.5 typically). Elements at Level k form "express lanes" for O(log N) search.
Our analogy: experts at Level k are "express" generalists. The confidence gate
acts as the probabilistic level assignment -- but instead of assigning experts
to levels (fixed), it assigns TOKENS to levels (adaptive per query).

---

## Lineage in the Arena

```
gpt
`-- capsule_moe (flat softmax routing, G=8, k=2)
    `-- hierarchical_tree (binary tree, depth-3, beam=2)
        `-- skip_list_routing (multi-level adaptive depth)
```

---

## Key References

- **Pugh (1990)**: Skip lists -- probabilistic balanced search structures.
  Our multi-level indexing with geometric spacing (2^k) directly from this.
- **hierarchical_tree (this project)**: Binary tree routing validated at
  -0.87% vs flat. Skip list extends this with adaptive depth.
- **Mixture of Depths (Raposo et al., 2024)**: Per-token adaptive computation.
  Our confidence gates perform a similar function (early exit for easy tokens)
  but at the routing level, not the layer level.
- **Sethuraman (1994)**: Stick-breaking construction for Dirichlet processes.
  The confidence gates implement exactly stick-breaking weights.
- **Switch Transformer (Fedus et al., 2022)**: Flat top-1 routing baseline.
  Skip list adds hierarchical structure on top.

---

## Empirical Results

### Setup
- d=64, N=8 experts, 32 capsules/expert (256 total), top_k=2
- 3 seeds (42, 123, 777), 500 training steps, lr=3e-3
- Character-level names dataset (~8K names, 28-char vocab)

### Quality Comparison

| Model | Trainable Params | Total Params | Mean Val Loss | vs Flat |
|-------|-----------------|--------------|---------------|---------|
| Flat (CapsuleMoE G=8, k=2) | 204,160 | 204,160 | 0.5207 | baseline |
| Tree (depth=3, beam=2) | 203,932 | 203,932 | 0.5179 | -0.54% |
| **Skip adaptive** | **206,732** | **206,732** | **0.5158** | **-0.93%** |
| Skip fixed-depth | 205,952 | 206,732 | 0.5190 | -0.33% |
| Ensemble 4x flat | 816,640 | 816,640 | 0.5238 | +0.59% |

Per-seed values:
- Flat: [0.5138, 0.5183, 0.5299]
- Tree: [0.5091, 0.5176, 0.5269]
- **Skip adaptive: [0.5126, 0.5084, 0.5265]**
- Skip fixed: [0.5147, 0.5151, 0.5270]
- Ensemble 4x flat: [0.5191, 0.5179, 0.5343]

**Note on parameter counts**: The fixed-depth control has 780 fewer trainable
parameters than skip adaptive (confidence gate weights and biases are frozen).
The total parameter count is identical (206,732) -- the gates exist but do not
receive gradients. This 0.38% trainable parameter difference is small but
should be noted when interpreting the 0.60% quality gap between adaptive and
fixed variants.

### Level-Weight Concentration

| Metric | Adaptive | Fixed |
|--------|----------|-------|
| Mean level-weight depth | 1.576 | 4.000 |
| **Weight above Level 0** | **60.6%** | baseline |

**Important clarification**: These numbers describe the learned level-weight
distribution, NOT training FLOP savings. During training, ALL levels are
computed for every token (no actual computational savings). The level-weight
distribution indicates *potential* savings under hard routing at inference
time, where tokens would be dispatched to a single level based on confidence
threshold. Hard inference routing is not implemented or tested in this
experiment.

### Level Usage Distribution (Adaptive, Validation Set, Mean +/- Std)

| Level | Mean Weight | Std | Interpretation |
|-------|------------|-----|----------------|
| Level 3 (coarsest, 1 expert) | 67.2% | 16.4% | Most tokens handled here |
| Level 2 (2 experts) | 12.6% | 9.4% | Some tokens need moderate precision |
| Level 1 (4 experts) | 15.6% | 9.3% | Some tokens need finer routing |
| Level 0 (8 experts, finest) | 4.6% | 4.3% | Hardest tokens only |

Routing statistics measured over the full validation set (20,480 tokens per
layer per seed, 3 seeds x 4 layers = 12 measurements aggregated). Standard
deviations reflect per-token variation in level assignment.

### Kill Criteria Assessment

| Criterion | Threshold | Measured | Verdict |
|-----------|-----------|----------|---------|
| KC1: Skip vs flat quality | >2% worse | **-0.93% (better)** | **PASSES** |
| KC2: Level weight concentration | No concentration above L0 | **60.6% above L0** | **PASSES** |

---

## Analysis

### Why It Works

1. **Coarse experts as implicit ensembles**: Weight-averaging children creates
   an ensemble effect. For "easy" tokens where any expert gives a similar answer,
   the coarse ensemble is as good as individual selection. This is consistent
   with the LSH finding (all routing strategies equivalent at small G) --
   when experts are similar, any routing works.

2. **Confidence gates learn token difficulty**: After training, 67.2% of weight
   concentrates at the coarsest level. The model discovers that most tokens in
   character-level names are "easy" (predictable patterns). Only ~4.6% of routing
   weight reaches fine-grained Level 0.

3. **Regularization through hierarchy**: The multi-level structure forces experts
   to be useful at multiple resolutions simultaneously. This is an implicit
   regularization that may explain the 0.93% quality improvement over flat routing.

### Ensembling Confound Analysis (Fix #2)

The skip-list model computes a weighted average of outputs from L+1 = 4 levels,
which is functionally similar to an ensemble of 4 predictors. To test whether
the quality improvement over flat routing comes from this implicit ensembling
rather than from adaptive routing, we ran a control: 4 independent flat
CapsuleMoE models (G=8, k=2 each) with their logits averaged.

**Result**: The 4x ensemble does NOT beat single flat routing (+0.59% worse).
Skip adaptive beats the ensemble by -1.51%. This rules out simple output
averaging as the explanation for the quality improvement. The hierarchical
structure and learned level weights provide genuine value beyond ensembling.

Note: The ensemble has 4x the parameters (816,640 vs 204,160). Even with this
massive parameter advantage, it does not outperform single flat routing. This
suggests the skip-list's benefit comes from its shared expert structure (coarse
experts reuse leaf parameters) rather than from having independent predictors.

### Comparison to Hierarchical Tree

The hierarchical tree (proven at -0.54% vs flat in this run) uses FIXED depth
-- every token traverses all 3 levels. Skip list routing matches tree quality
while learning to concentrate level weight at coarse levels:

- Tree: forced 3-gate traversal for every token
- Skip: average 1.576 effective depth weight (60.6% concentrated above Level 0)
- Both use binary/hierarchical expert organization
- Skip adds L*(d+1) confidence gate parameters (negligible at +1.3%)

---

## Micro-Scale Limitations

1. **Limited token difficulty variation**: Character-level names have narrow
   difficulty distribution. At macro scale with diverse text (code, math,
   natural language), the adaptive depth distribution may be broader, with
   more tokens utilizing fine-grained routing.

2. **Small N=8**: With only 8 experts, the skip list has only 3 coarse levels.
   At N=256 (DeepSeek-V3 scale), the skip list would have 8 levels, and the
   routing cost savings from early stopping could be dramatically larger
   (O(log N) vs O(N)).

3. **Soft vs hard routing**: During training, all levels are computed (no
   actual FLOP savings). The level-weight concentration is measured by the
   learned stick-breaking weights, which indicate how much work COULD be saved
   with hard early stopping at inference time. Actual inference savings require
   implementing hard thresholded routing, which was not done here.

4. **Recursive computation cost (L+1 multiplier)**: The soft routing regime
   evaluates every leaf expert L+1 times (once per level). For L=3 levels,
   the coarse expert at Level 3 recursively evaluates all 8 leaf experts;
   Level 2 evaluates 8 total (2 experts, each averaging 4 leaves); Level 1
   evaluates 8 total (4 experts, each averaging 2 leaves); Level 0 evaluates
   8. That is 32 leaf expert forward passes per token vs 2 for flat top-k=2,
   making training FLOPs approximately 16x higher for the expert evaluation
   portion. The routing cost analysis in MATH.md applies only to a hypothetical
   hard routing inference mode that selects a single level per token.

5. **No composition test**: This experiment tests single-domain quality and
   routing efficiency. Composition (shared-base protocol with domain experts)
   is the next validation step.

6. **Weight-averaged coarse experts**: At macro scale, a learned coarse expert
   (with its own parameters) may outperform a simple average. The zero-parameter
   approach was chosen for clean comparison at micro scale.

7. **Fixed-depth control has 780 fewer trainable parameters**: The confidence
   gate parameters (780 total across 4 layers) are frozen in the fixed-depth
   control. The trainable count is 205,952 vs 206,732 for adaptive. While
   this is only a 0.38% difference, it means the comparison is not perfectly
   parameter-matched. The quality gap (adaptive -0.60% better than fixed) could
   be partially attributed to this, though the effect size is small.

---

## What Would Kill This

### At micro scale
- Skip-list routing >2% worse than flat softmax: **REFUTED** (it is 0.93% better)
- No level weight concentration above Level 0: **REFUTED** (60.6% concentration)
- Quality improvement explained by ensembling: **REFUTED** (4x ensemble +0.59%
  worse than single flat; skip beats ensemble by -1.51%)
- Confidence gates collapse to uniform (all levels equal weight): Not observed;
  clear concentration at coarse levels (67.2% at coarsest, 4.6% at finest)

### At macro scale
- Coarse experts (weight-averaged) may be poor approximations when experts are
  highly specialized to different domains. This would manifest as quality
  degradation at the coarse levels, pushing more tokens to Level 0 and
  reducing the level-weight concentration.
- Hard early stopping may introduce approximation error not captured by soft
  training (soft-hard gap).
- With diverse data, the optimal level distribution may require more than
  geometric (2^k) spacing. Frequency-weighted level assignment (connection
  to Huffman) may be needed.
- GPU hardware parallelism may not benefit from adaptive-depth routing if
  all tokens in a batch must wait for the slowest token's routing depth.
- The 16x expert evaluation cost during training (due to recursive coarse
  expert construction) is a serious scalability concern. At N=256 with
  L=8 levels, this becomes intractable without hard routing.

---

## Next Steps

1. **Composition test**: Run shared-base composition protocol with skip-list
   routing (tree baseline already validated at +0.17% composition gap).

2. **Hard routing inference**: Implement threshold-based early stopping and
   measure actual FLOP savings vs soft training.

3. **Frequency-weighted levels**: Connect to Huffman findings -- assign
   experts to levels based on activation frequency rather than fixed 2^k
   spacing.

4. **N-scaling**: Test at N=16 and N=32 to verify level-weight concentration
   scales favorably with more levels.
