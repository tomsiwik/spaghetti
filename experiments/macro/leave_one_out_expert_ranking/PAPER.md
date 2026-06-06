# Leave-One-Out Expert Ranking: Research Digest

## Hypothesis

Leave-one-out PPL ranking identifies harmful vs helpful experts in SOLE
composition without task labels: removing each expert from the N=50 composition
and measuring perplexity change on generic calibration text produces a stable,
meaningful ranking of expert contributions.

## What This Experiment Is

A composition diagnostic for the SOLE architecture. For each of N=50 pilot LoRA
adapters, compose the remaining N-1 via weight-space merge and measure perplexity
on generic calibration text. Experts whose removal IMPROVES PPL are harmful;
experts whose removal HURTS PPL are helpful.

**Why PPL and not KL divergence:** The KL health experiment (exp_composition_health_kl)
was KILLED because KL(composed || base) anti-correlates with quality (rho=-0.7).
KL measures distributional DISTANCE from base, which captures expert STRENGTH
not QUALITY. PPL measures absolute model quality on the calibration text. A
strong expert that shifts distributions far from base has high KL but may have
low PPL (good quality). This is the fundamental distinction.

**Why LOO approximates Shapley under orthogonality:** Shapley values measure the
average marginal contribution across all possible coalitions. LOO measures the
marginal contribution when removing from the full coalition only. The gap between
them comes from interaction effects (I_ij), which are bounded by cosine similarity
between expert deltas. At SOLE production cosines (~0.0002), interaction effects
are O(10^-4) of main effects, making LOO a valid approximation.

## Key References

| Paper | Relevance |
|-------|-----------|
| Shapley-MoE (2025) | Monte Carlo Shapley for MoE pruning; M=20 samples, 112min for Qwen2-57B; 25% expert reduction with +0.92 PPL |
| MoE-I2 (NeurIPS 2025) | LOO loss-increase for per-layer pruning budget allocation |
| EvoESAP (2026) | Expected Speculative Acceptance Proxy for fast pruning eval |
| ShapLoRA (2025) | Shapley sensitivity for LoRA rank allocation |
| exp_composition_health_kl (KILLED) | KL divergence anti-correlates with quality; PPL avoids this failure mode |
| exp_composition_weight_sensitivity (micro) | LOO at micro scale vacuous (zero specialization); macro needed |

## Empirical Results

TO BE FILLED from `/workspace/llm/results/leave_one_out_expert_ranking/results.json`

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-7B (4-bit NF4) |
| N experts | TO BE FILLED |
| Calibration texts | 30 per set x 2 sets = 60 total |
| Max seq len | 512 tokens |
| Composition | Weight-space merge (naive addition) |

### Reference PPL (All N Composed)

| Set | PPL |
|-----|-----|
| Set A | TO BE FILLED |
| Set B | TO BE FILLED |

### LOO Delta Distribution

| Metric | Set A | Set B |
|--------|-------|-------|
| Mean delta (%) | TO BE FILLED | TO BE FILLED |
| Std delta (%) | TO BE FILLED | TO BE FILLED |
| Min delta (%) | TO BE FILLED | TO BE FILLED |
| Max delta (%) | TO BE FILLED | TO BE FILLED |
| N harmful (delta < 0) | TO BE FILLED | TO BE FILLED |
| N helpful (delta > 0) | TO BE FILLED | TO BE FILLED |

### Top-5 Most Helpful Experts

| Expert | Delta_A (%) | Delta_B (%) |
|--------|-------------|-------------|
| TO BE FILLED | | |

### Top-5 Most Harmful Experts

| Expert | Delta_A (%) | Delta_B (%) |
|--------|-------------|-------------|
| TO BE FILLED | | |

### Kill Criteria

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| K1: PPL delta std | >= 0.1% | TO BE FILLED | PENDING |
| K2: Total runtime | <= 4 hrs | TO BE FILLED | PENDING |
| K3: Kendall tau | >= 0.5 | TO BE FILLED | PENDING |

**Verdict: PENDING**

### Bonus: Correlation with Individual Quality

| Metric | Value |
|--------|-------|
| Spearman rho (LOO vs benchmark) | TO BE FILLED |
| p-value | TO BE FILLED |

## Limitations

1. **Only 50 experts tested.** At N=500+, the per-expert LOO delta may drop
   below the PPL noise floor. The experiment tests feasibility at pilot scale,
   not production scale.

2. **Generic calibration text only.** LOO-PPL detects composition-level harm
   (expert hurts general text quality) but not domain-level harm (expert hurts
   only its own domain). An expert that is harmful only on medical text but
   neutral on general text would not be detected.

3. **No causal validation.** LOO identifies experts correlated with PPL changes,
   not necessarily experts that CAUSE the changes. At near-zero cosine, this
   distinction should be negligible, but it's not proven.

4. **Calibration set composition may bias rankings.** If Set A has more code
   tokens than Set B (despite stratification), code experts may rank differently.
   K3 (Kendall tau) directly tests this risk.

5. **4-bit quantization.** NF4 quantization may affect PPL absolute values.
   Rankings should be preserved (both composed and LOO models use same quant).

## What Would Kill This

### At Macro Scale (this experiment)
- K1 KILL: All 50 experts contribute identically (std < 0.1%). Would mean
  composition is insensitive to individual expert quality at N=50.
- K2 KILL: Runtime > 4 hours. Would make LOO impractical for iterative use.
  (Very unlikely given subtraction approach.)
- K3 KILL: Rankings unstable (tau < 0.5). Would mean LOO captures noise not
  signal, or interaction effects dominate despite low cosines.

### At Production Scale (future)
- LOO delta drops below noise floor at N > 200 (each expert contributes < 0.02%)
- Ranking changes when composition method changes (e.g., weighted vs naive addition)
- Harmful experts identified by LOO are actually domain-beneficial on downstream tasks
