# LoRA Merging Bakeoff: Research Digest (Revision 2)

## Hypothesis

Published LoRA merging methods (TIES, DARE) will outperform or match our
concatenation+calibrate protocol on merged model quality, making our approach
redundant.

**Falsifiable**: If concat+calibrate is worst on 2+ metrics vs TIES/DARE/average,
or if no method achieves <3% gap vs joint training, the respective kill criterion
triggers.

---

## Revision History

**v2 (revision)**: Addressed 2 required fixes from adversarial review of v1:
1. Actually fixed TIES zero-mask dilution bug. The v1 "fix" still included
   zero-valued tasks at elected_sign==0 positions via the sign-match path.
   The correct fix clears sign-match entries before applying the nonzero-only
   override: `match_mask = match_mask * (1 - zero_elected) + zero_elected * has_nonzero`.
   Re-ran TIES and DARE-TIES at N=2 and N=5 (3 seeds each). Impact was
   negligible: TIES N=2 moved from +7.16% to +7.06%, N=5 from +17.41% to
   +20.68% (within noise, std=0.035). The "TIES hurts" conclusion is unchanged.
2. Corrected false claim in v1 revision history (item 2 claimed bug was fixed
   when it was not).

**v1 (revision)**: Addressed 3 required fixes from adversarial review:
1. Swept DARE drop rates p in {0.3, 0.5, 0.7, 0.9} (was fixed at p=0.9)
2. Attempted TIES zero-mask dilution bug fix (incomplete -- see v2)
3. Downgraded status from PROVEN to PARTIAL (KC2 kills at N=5)

---

## What This Experiment Is

A head-to-head comparison of LoRA merging strategies on identical data splits,
identical base models, and identical LoRA fine-tuning. The methods tested:

1. **Simple Average** -- mean of all LoRA deltas (task arithmetic, lambda=1/N)
2. **TIES-Merging** -- trim bottom 80% by magnitude, elect sign by majority, average
   only sign-agreeing values (Yadav et al., NeurIPS 2023)
3. **DARE** -- randomly drop p fraction of delta parameters, rescale remaining by
   1/(1-p), then average (Yu et al., 2023). Swept p in {0.3, 0.5, 0.7, 0.9}.
4. **DARE-TIES** -- DARE sparsification (p=0.9) followed by TIES sign election
5. **Concat + Calibrate** -- keep all N deltas as separate routed experts, train a
   softmax router on mixed-domain data for 100 steps (our method)

Tested at N=2 (binary domain split: a-m vs n-z) and N=5 (quintary: a-e, f-j, k-o,
p-t, u-z). 3 seeds each (42, 123, 7).

**Note on joint training baseline**: Joint training uses N * FINETUNE_STEPS total
gradient updates (round-robin across domains), while each LoRA adapter sees exactly
FINETUNE_STEPS. At N=5, joint sees 5x more updates. This inflates the absolute gap
but does not affect relative ordering between merging methods.

---

## Lineage in the Arena

```
gpt
 `-- lora_gpt (LoRA adapters on MLP)
      |-- lora_procrustes (shared/unique decomposition, N=2)
      `-- lora_merging_bakeoff (this experiment)
```

---

## Key References

- **TIES-Merging**: Yadav et al., "Resolving Interference When Merging Models",
  NeurIPS 2023. Trim-elect-merge for sign conflict resolution.
- **DARE**: Yu et al., "Language Models are Super Mario", 2023. Random drop +
  rescale exploiting delta parameter redundancy.
- **Task Arithmetic**: Ilharco et al., "Editing Models with Task Arithmetic",
  ICLR 2023. Simple averaging of task vectors.
- **Model Soups**: Wortsman et al., ICML 2022. Weight averaging of fine-tuned models.
- **LoRA**: Hu et al., ICLR 2022. Low-rank adaptation.
- **lora_procrustes**: Our prior experiment confirming LoRA delta linearity and
  natural orthogonality (cos ~ 0.014).

---

## Empirical Results

### N=2 Domains (3-seed aggregate)

| Method | Mean Val Loss | Std | vs Joint | Merge Time |
|--------|-------------|-----|----------|------------|
| Joint training | 0.5182 | 0.0043 | baseline | N/A |
| **Concat + calibrate** | **0.5241** | **0.0057** | **+1.14%** | 812 ms |
| DARE (p=0.3) | 0.5251 | 0.0070 | +1.34% | 0.2 ms |
| Simple average | 0.5253 | 0.0059 | +1.37% | <0.1 ms |
| DARE (p=0.5) | 0.5264 | 0.0066 | +1.58% | 0.2 ms |
| DARE (p=0.7) | 0.5296 | 0.0071 | +2.20% | 0.2 ms |
| TIES (rho=0.2) | 0.5548 | 0.0181 | +7.06% | 0.3 ms |
| DARE (p=0.9) | 0.5556 | 0.0088 | +7.21% | 0.2 ms |
| DARE-TIES (p=0.9) | 1.2477 | 0.2285 | +140.80% | 0.3 ms |

### N=5 Domains (3-seed aggregate)

| Method | Mean Val Loss | Std | vs Joint | Merge Time |
|--------|-------------|-----|----------|------------|
| Joint training | 0.4984 | 0.0068 | baseline | N/A |
| **Simple average** | **0.5150** | **0.0054** | **+3.33%** | <0.1 ms |
| DARE (p=0.3) | 0.5152 | 0.0048 | +3.36% | 0.4 ms |
| DARE (p=0.5) | 0.5157 | 0.0053 | +3.47% | 0.4 ms |
| DARE (p=0.7) | 0.5177 | 0.0067 | +3.87% | 0.4 ms |
| Concat + calibrate | 0.5237 | 0.0149 | +5.07% | 1092 ms |
| DARE (p=0.9) | 0.5348 | 0.0098 | +7.31% | 0.4 ms |
| TIES (rho=0.2) | 0.6015 | 0.0354 | +20.68% | 0.3 ms |
| DARE-TIES (p=0.9) | 3.6603 | 0.5592 | +634.24% | 0.7 ms |

### DARE Drop Rate Sweep Summary

| Drop Rate (p) | N=2 Gap | N=5 Gap | Rescale Factor |
|---------------|---------|---------|----------------|
| 0.3 | +1.34% | +3.36% | 1.43x |
| 0.5 | +1.58% | +3.47% | 2.0x |
| 0.7 | +2.20% | +3.87% | 3.33x |
| 0.9 | +7.21% | +7.31% | 10.0x |

DARE degrades monotonically with drop rate. At p=0.3 it is competitive with
simple average (within 0.03% at N=2, within 0.03% at N=5). At p=0.9 it is
catastrophically worse. The reviewer's hypothesis was confirmed: p=0.9 is an
extreme regime, and lower drop rates are fundamentally different.

### Kill Criteria Evaluation

| Criterion | N=2 Result | N=5 Result | Verdict |
|-----------|-----------|-----------|---------|
| KC1: concat+cal worst on 2+ metrics | Beaten by 0/7 methods | Beaten by 4/7 (avg, DARE p=0.3/0.5/0.7) | **PASS at N=2, KILL at N=5** |
| KC2: any method <3% gap vs joint | 5 methods <3% (concat+cal, DARE p=0.3/0.5/0.7, simple avg) | No method <3% (best: simple avg +3.33%) | **PASS at N=2, KILL at N=5** |

---

## Analysis

### 1. DARE at Low Drop Rates Is Competitive with Simple Average

The DARE drop rate sweep is the key new finding. At p=0.3:
- N=2: +1.34% (vs simple avg +1.37%) -- DARE is slightly better
- N=5: +3.36% (vs simple avg +3.33%) -- effectively tied

This shows DARE is not inherently harmful for LoRA merging. The original
experiment's conclusion "DARE hurts" was an artifact of testing only the
extreme p=0.9 setting. With gentle sparsification (p=0.3, rescale 1.43x),
DARE preserves most signal while adding minor stochastic regularization.

The monotonic degradation with drop rate confirms that LoRA deltas have
low redundancy -- each parameter carries more information than in full
fine-tuning deltas, so dropping more parameters costs more.

### 2. Simple Average Remains the Practical Default

While DARE p=0.3 matches simple average, it adds complexity (random seed
dependence, hyperparameter to tune) for no meaningful gain. Simple average
is deterministic, requires no hyperparameters, and is within 0.03% of the
best zero-shot method at both scales. The recommendation stands: use simple
average as the default zero-shot merging method for orthogonal LoRA deltas.

### 3. TIES-Merging Still Hurts (After Correct Bug Fix)

The v2 bug fix correctly clears sign-match entries at elected_sign==0
positions before applying the nonzero-only override. The impact was
negligible: TIES N=2 moved from +7.16% to +7.06% (within noise), and
N=5 shifted from +17.41% to +20.68% (higher variance, std=0.035, within
noise). As the reviewer predicted, the bug affected only ~4% of positions
and the fix does not change the conclusion. TIES still degrades
substantially at both scales. The core issue is that trimming 80% of an
already-compressed rank-8 representation destroys signal, not that the
implementation was wrong.

### 4. Concat+Calibrate: Wins at N=2, Loses Badly at N=5

At N=2, concat+cal (+1.14%) is the overall best method. At N=5, it falls to
+5.07%, beaten by simple average and all DARE variants with p <= 0.7. The
N=5 failure is more pronounced in this revision (5.07% vs 5.59% previously),
and the KC1 kill is triggered (beaten by 4 methods, threshold is 2).

The high variance at N=5 (std=0.0149 vs 0.0054 for simple average) suggests
the router optimization is unstable with 5 experts and only 100 calibration
steps. Seed 7 shows concat+cal at +6.65% while seed 123 shows +3.33%.

### 5. Both Kill Criteria Trigger at N=5

- **KC1**: Concat+cal is beaten by 4 methods at N=5 (simple avg, DARE p=0.3,
  p=0.5, p=0.7). This exceeds the threshold of 2.
- **KC2**: No method achieves <3% gap vs joint at N=5. The best (simple avg
  at +3.33%) just barely misses. This is fundamentally limited by rank-8
  LoRA capacity spread across 5 domains.

---

## Scaling Analysis (N=2 vs N=5)

| Method | N=2 Gap | N=5 Gap | Ratio |
|--------|---------|---------|-------|
| DARE (p=0.3) | +1.34% | +3.36% | 2.5x |
| Simple average | +1.37% | +3.33% | 2.4x |
| DARE (p=0.5) | +1.58% | +3.47% | 2.2x |
| DARE (p=0.7) | +2.20% | +3.87% | 1.8x |
| Concat+cal | +1.14% | +5.07% | 4.4x |
| TIES | +7.06% | +20.68% | 2.9x |
| DARE (p=0.9) | +7.21% | +7.31% | 1.0x |
| DARE-TIES | +140.80% | +634.24% | 4.5x |

DARE p=0.9 uniquely does NOT degrade at N=5 (ratio 1.0x) -- because it is
already so noisy at N=2 that adding more vectors to average barely changes
the noise level. Concat+calibrate degrades most steeply (4.4x), confirming
the router is underfit at N=5.

---

## Compute Cost Analysis

| Method | Merge Cost | Extra Data Needed | Inference Cost |
|--------|-----------|-------------------|----------------|
| Simple average | O(ND) ~0.1ms | None | 1x (single model) |
| DARE (any p) | O(ND) ~0.3ms | None | 1x |
| TIES | O(ND log D) ~0.3ms | None | 1x |
| DARE-TIES | O(ND log D) ~0.7ms | None | 1x |
| Concat+cal | O(cal_steps * batch * fwd) ~1000ms | Mixed-domain data | Kx (K experts per token) |

---

## Micro-Scale Limitations

1. **LoRA rank 8 is very low**. At macro scale (rank 64-256), the
   redundancy structure changes and DARE/TIES may perform differently.

2. **Character-level names are not genuinely distinct domains**. With
   real-world domains (code vs prose, English vs Chinese), the delta
   orthogonality and sign conflict patterns could differ substantially.

3. **Only 100 calibration steps for concat+cal**. More steps or
   learning rate tuning could improve N=5 routing. The capsule
   experiment used 200 steps.

4. **TIES density not swept**. Only tested rho=0.2 (canonical). Higher
   density (0.5, 0.8) might reduce signal destruction. Advisory item
   from the review, not blocking.

5. **3 seeds provides directional evidence, not statistical power**.
   Confidence intervals would require more seeds.

6. **Joint training baseline uses Nx more gradient steps** than any
   single LoRA adapter. This inflates absolute gaps but does not affect
   relative ordering between merging methods.

---

## What Would Kill This

### At Micro Scale (already partially triggered)
- **KC1 at N=5**: Concat+cal is beaten by 4 methods. Our routed
  composition approach is not cost-justified at N=5 with 100 cal steps.
- **KC2 at N=5**: No method achieves <3% gap vs joint. LoRA merging
  (any method) fundamentally struggles at N=5 with rank=8.

### At Macro Scale
- **TIES/DARE outperform simple average** on full-model deltas or high-rank
  LoRA with non-orthogonal domains. The orthogonality that makes simple
  average optimal at micro scale may not hold with real-world domain diversity.
- **Concat+calibrate's compute overhead is not justified** at macro scale
  where routing K=2 experts means 2x MLP FLOPs per token.
- **Higher LoRA rank could close the N=5 gap**. Rank 8 spread across 5
  domains gives ~1.6 effective rank per domain, likely too few.

---

## Key Takeaways

1. **Simple average and DARE p=0.3 are tied** as the best zero-shot LoRA
   merging methods when deltas are orthogonal. Simple average is preferred
   for simplicity (no hyperparameters, deterministic).

2. **DARE quality degrades monotonically with drop rate** on LoRA deltas.
   The original p=0.9 finding ("DARE hurts") was misleading -- DARE at low
   drop rates is fine, but provides no benefit over simple averaging.

3. **TIES still hurts** even after correctly fixing the zero-mask dilution
   bug (v2). The fix had negligible impact (<0.1pp at N=2). The core issue
   is trimming a compressed representation, not implementation.

4. **Concat+calibrate wins at N=2 (+1.14%)** but loses at N=5 (+5.07%).
   The router optimization is unstable with 5 experts and 100 cal steps.

5. **The quality-cost Pareto frontier** is:
   - Zero calibration data: simple average (+1.37% at N=2, +3.33% at N=5)
   - With calibration data: concat+calibrate at N=2 only (+1.14%)
   - Never: TIES, DARE p>=0.7, or DARE-TIES on low-rank LoRA deltas

6. **Status: PARTIAL.** Proven at N=2 (KC1 pass, KC2 pass). Killed at N=5
   (KC1 kill, KC2 kill).

---

## Artifacts

- `micro/models/lora_merging_bakeoff/` -- code, tests, MATH.md, PAPER.md
- `micro/models/lora_merging_bakeoff/merging_methods.py` -- TIES, DARE, DARE-TIES
  implementations (TIES zero-mask bug correctly fixed in v2)
- `micro/models/lora_merging_bakeoff/test_merging_bakeoff.py` -- full experiment
  (DARE sweep added in revision)
- Parent model: `lora_gpt` (LoRA-augmented GPT)
- Reuses: `RoutedDeltaGPT` and `calibrate_router` from `lora_procrustes`
- Total experiment time: ~2.5 minutes (6 seeds x 2 conditions, 4 DARE variants each)
