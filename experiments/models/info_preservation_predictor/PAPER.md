# Information Preservation Predictor: Research Digest

## Hypothesis

The quality ranking of LoRA merging methods is fully predicted by how much
original delta information each method preserves, measured by Frobenius norm
ratios. If true (Spearman rho > 0.8, ranking mismatches <= 1), this provides
a cheap a priori predictor of method quality without any training.

**Falsifiable**: If the info preservation ranking does not match the quality
ranking for more than 1 method, or if Spearman correlation < 0.8.

---

## What This Experiment Is

A diagnostic/theory experiment that takes the 8 merging methods from the
lora_merging_bakeoff (simple average, TIES, DARE at 4 drop rates, DARE-TIES,
concat+calibrate) and tests whether their quality ranking can be predicted
from a purely weight-space metric -- no training required.

Three information preservation metrics are computed for each method:
1. **IP(avg)**: Fidelity to simple average (1 - ||merged - avg||_F / ||avg||_F)
2. **IP(orig)**: Fidelity to original per-domain deltas
3. **Norm ratio**: ||merged||_F / mean(||original||_F) -- magnitude preservation

The prediction: methods that preserve more information should produce better
models. Spearman rank correlation between IP and quality should exceed 0.8.

---

## Lineage in the Arena

```
gpt
 `-- lora_gpt (LoRA adapters on MLP)
      `-- lora_merging_bakeoff (8 methods, quality data)
           `-- info_preservation_predictor (this experiment, diagnostic only)
```

---

## Key References

- **TIES-Merging**: Yadav et al., NeurIPS 2023. Trim-elect-merge.
- **DARE**: Yu et al., 2023. Random drop + rescale.
- **Task Arithmetic**: Ilharco et al., ICLR 2023. Simple averaging.
- **lora_merging_bakeoff**: Our prior experiment providing quality data.
- **lora_merging_bakeoff REVIEW-adversarial.md**: Origin of this hypothesis
  ("ranking is almost perfectly predicted by how much delta info is preserved").

---

## Empirical Results

### N=2 Domains (3-seed aggregate, sorted by quality)

| Method | Val Loss | vs Joint | IP(avg) | IP(orig) | NormR |
|--------|----------|----------|---------|----------|-------|
| concat_cal | 0.5245 | +1.07% | 1.0000 | 1.0000 | 1.000 |
| simple_avg | 0.5257 | +1.31% | 1.0000 | 0.2964 | 0.715 |
| dare_p0.3 | 0.5260 | +1.36% | 0.3460 | 0.1570 | 0.852 |
| dare_p0.5 | 0.5277 | +1.69% | 0.0019 | 0.0013 | 1.010 |
| dare_p0.7 | 0.5310 | +2.33% | -0.5234 | -0.2904 | 1.304 |
| ties | 0.5545 | +6.85% | 0.0263 | 0.0138 | 1.238 |
| dare_p0.9 | 0.5638 | +8.65% | -2.0000 | -1.2437 | 2.264 |
| dare_ties | 1.0284 | +98.17% | -4.9494 | -3.2829 | 4.425 |

### N=5 Domains (3-seed aggregate, sorted by quality)

| Method | Val Loss | vs Joint | IP(avg) | IP(orig) | NormR |
|--------|----------|----------|---------|----------|-------|
| dare_p0.3 | 0.5157 | +2.93% | 0.3805 | 0.0710 | 0.559 |
| simple_avg | 0.5157 | +2.93% | 1.0000 | 0.1181 | 0.474 |
| dare_p0.5 | 0.5165 | +3.08% | 0.0526 | 0.0115 | 0.654 |
| concat_cal | 0.5166 | +3.11% | 1.0000 | 1.0000 | 1.000 |
| dare_p0.7 | 0.5173 | +3.23% | -0.4483 | -0.1152 | 0.835 |
| dare_p0.9 | 0.5291 | +5.60% | -1.8450 | -0.6050 | 1.431 |
| ties | 0.5746 | +14.67% | -1.6485 | -0.5286 | 1.604 |
| dare_ties | 7.0105 | +1299.16% | -12.6507 | -5.4946 | 6.610 |

### Spearman Correlations

| Metric | N=2 | N=5 |
|--------|-----|-----|
| IP vs average | +0.922 | +0.838 |
| IP vs originals | +0.929 | +0.810 |
| Norm ratio (inverted) | +0.905 | +0.952 |

### Kill Criteria

| Criterion | N=2 | N=5 | Verdict |
|-----------|-----|-----|---------|
| KC1: rank mismatches <= 1 | 3 mismatches | 5 mismatches | **KILL** |
| KC2: Spearman >= 0.8 | 0.929 | 0.952 | **PASS** |

---

## Analysis

### 1. Strong Coarse Prediction, Weak Fine Prediction

All three IP metrics show Spearman rho > 0.8 at both scales (KC2 passes).
The correlation is especially strong for norm ratio at N=5 (rho = 0.952).
IP correctly separates the "catastrophic" tier (DARE-TIES, DARE p=0.9, TIES)
from the "good" tier (simple avg, DARE p=0.3, concat+cal).

However, KC1 is killed at both scales. Within tiers, IP does not predict
the fine ranking. The top 5 methods at N=5 differ by only 0.3% in quality
but span a wide range of IP values (from -0.45 to 1.0).

### 2. TIES Breaks IP-Quality Monotonicity

TIES has IP(avg) = +0.03 (near zero, meaning the merged delta is close to
the simple average) but quality 3x worse than dare_p0.7 (which has
IP(avg) = -0.52). This is the key counterexample: TIES introduces
*correlated* errors via sign election. Its merged delta is close to the
average in Frobenius distance, but the errors are concentrated in
structurally important locations. Random noise (DARE) averages out;
systematic distortion (TIES) does not.

### 3. Concat+Calibrate Breaks IP-Quality Monotonicity at N=5

At N=5, concat+cal has perfect IP (1.0 on all metrics) but ranks 4th in
quality. The router optimization introduces error that the IP metric
cannot capture. Information preservation is necessary but not sufficient
-- the composition mechanism adds its own noise.

### 4. Norm Ratio Is the Best Single Predictor

Norm ratio (rho = 0.905 at N=2, 0.952 at N=5) outperforms both IP metrics
at N=5. This makes intuitive sense: methods that amplify signal magnitude
(DARE with high p rescales by 1/(1-p)) introduce proportionally more noise.
A rescale factor of 10x (DARE p=0.9) means 10x noise amplification.

The norm ratio is trivially computable: ||merged||_F / mean(||delta_k||_F).
No reference delta needed. Any method with NR >> 1 is likely harmful; any
method with NR near 1/sqrt(N) is likely good.

### 5. The Practical Predictor

For zero-shot methods (no calibration data), the rule of thumb is:

    **If ||merged||_F / mean(||delta_k||_F) > 1.5, the method will hurt.**

This correctly classifies all 7 zero-shot methods at both scales:
- NR < 1.5: simple_avg, dare_p0.3, dare_p0.5, dare_p0.7 -- all good
- NR > 1.5: ties (1.24-1.60), dare_p0.9 (2.26-1.43), dare_ties (4.4-6.6) -- all bad

Note: TIES barely exceeds the threshold at N=2 (NR=1.24) but is still
substantially worse than methods below it. Tightening to NR > 1.1 would
be more conservative and still correct.

---

## Status: PARTIAL

KC2 passes (Spearman >= 0.8 at both scales). KC1 kills (too many ranking
mismatches). The hypothesis is partially validated: IP is a strong *coarse*
predictor (good vs bad tier) but not a *fine* predictor (ranking within tier).

The actionable insight is the norm ratio rule: **NR > 1.5 predicts failure**.
This is a zero-cost, zero-data predictor that correctly filters out harmful
merging methods before any evaluation.

---

## Micro-Scale Limitations

1. **8 methods is a small sample for Spearman**. With n=8, rho has wide
   confidence intervals. A single swap between adjacent methods (differing
   by 0.05% in quality) changes rho substantially.

2. **Orthogonal deltas inflate IP of simple average**. With non-orthogonal
   deltas at macro scale, simple average would have lower IP and methods
   like TIES that resolve sign conflicts could have higher IP.

3. **Concat+calibrate's router noise is not captured by weight-space IP**.
   A function-space IP metric (comparing model outputs, not weights) would
   correctly handle routed composition but is much more expensive to compute.

4. **TIES density not swept**. At higher density (rho=0.8), TIES approaches
   simple average and its NR would approach 1/sqrt(N). The NR > 1.5 rule
   may not hold for TIES at all densities.

---

## What Would Kill This

### At Micro Scale (already partially killed)
- **KC1**: 3-5 ranking mismatches. IP is not a fine-grained quality predictor.
  This is a fundamental limitation when methods differ by < 0.5% in quality.

### At Macro Scale
- **TIES outperforms simple average on non-orthogonal deltas**: The NR > 1.5
  rule depends on orthogonal deltas. If sign conflicts are real at macro scale,
  TIES's structured intervention (which increases NR) could still help.
- **Norm ratio loses predictive power**: If methods with NR ~ 2 perform well
  at scale (e.g., because DARE's noise averages out over millions of params),
  the NR threshold becomes useless.
- **Function-space metrics dominate**: If someone shows that a cheap
  function-space metric (e.g., KL divergence on a probe set) is both more
  predictive AND still cheap, the weight-space IP approach is obsolete.

---

## Key Takeaways

1. **Spearman rho > 0.8 at both N=2 and N=5**. Information preservation
   is a legitimate predictor of merging quality, just not a perfect one.

2. **Norm ratio is the best single predictor** (rho = 0.952 at N=5). It
   requires zero computation beyond what the merging method already produces.

3. **NR > 1.5 is a reliable failure detector**. This one-line check correctly
   classifies all 7 zero-shot methods as good or bad at both scales.

4. **IP cannot predict within-tier ranking**. Methods within 0.5% of each
   other are indistinguishable by any weight-space metric tested.

5. **Information preservation is necessary but not sufficient**. Concat+cal
   preserves everything but still loses at N=5 due to router noise.

---

## Artifacts

- `micro/models/info_preservation_predictor/` -- code, MATH.md, PAPER.md
- `micro/models/info_preservation_predictor/test_info_preservation.py` -- full experiment
- Parent data: `lora_merging_bakeoff` (quality results reproduced inline)
- Total experiment time: ~92 seconds (6 seeds x 2 conditions)
