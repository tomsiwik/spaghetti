# Delta Rank Scaling v2: Mathematical Foundations

## Revision Notes

v2 addresses 5 fixes from adversarial review. Key mathematical changes:
- Convergence control (Section 4.1)
- FFN+Attn-only primary metric (Section 4.2)
- Bootstrap CI on power law exponent (Section 4.3)
- Multi-checkpoint rho analysis (Section 4.5)

## 1. Setup and Notation

| Symbol | Definition | Values |
|--------|-----------|--------|
| d | Model embedding dimension | {64, 128, 256} (micro) |
| d_ff | FFN intermediate dimension | 4d |
| L | Number of transformer layers | 4 |
| W_p | Pretrained weight matrix | R^{d_out x d_in} |
| W_s | Skeleton (random init) weight matrix | R^{d_out x d_in} |
| Delta | Base delta: W_p - W_s | R^{d_out x d_in} |
| r_eff(M) | Effective rank of matrix M (Roy & Vetterli) | exp(H(p)) |
| sigma_i | i-th singular value of Delta | sigma_1 >= sigma_2 >= ... |
| p_i | Normalized singular value: sigma_i / sum(sigma_j) | |
| rho(d) | Effective rank ratio: r_eff(Delta) / min(d_out, d_in) | [0, 1] |
| rho_FA(d) | FFN+Attention mean ratio (primary metric, v2) | [0, 1] |

## 2. Effective Rank Definition

The effective rank (Roy & Vetterli, 2007) of a matrix M in R^{m x n} is:

    r_eff(M) = exp(H(p))

where:
    p_i = sigma_i / sum_j(sigma_j)
    H(p) = -sum_i p_i * log(p_i)

Properties:
- r_eff = 1 for rank-1 matrices
- r_eff = min(m, n) for identity/uniform-spectrum matrices
- Continuous, differentiable

## 3. Why Embeddings Are Excluded (Fix #2)

Embedding matrices have shape (V x d) where V = 27 (character vocab).
The min_dim = min(V, d) = V = 27 for all d >= 27. Therefore:

    rho_emb(d) = r_eff(Delta_emb) / 27

This ratio is NOT a function of d -- it reflects vocabulary structure,
not model dimension scaling. Including it in the aggregate contaminates
the d-scaling signal:

| d | FFN+Attn rho | Emb rho | All-weights rho |
|---|-------------|---------|----------------|
| 64 | 0.650 | 0.774 | 0.664 |
| 128 | 0.590 | 0.804 | 0.614 |
| 256 | 0.501 | 0.833 | 0.538 |

Embeddings bias the aggregate upward at large d (0.833 vs 0.501),
weakening the apparent scaling. The primary metric rho_FA(d) is
the mean ratio across FFN and attention weight matrices only.

## 4. Empirical Results (v2 Revised)

### 4.1 Convergence Control

v1 used a linear step heuristic (1000/2000/3000 for d=64/128/256).
v2 trains all dimensions to the same target validation loss, established
by training d=64 with the default 1000 steps.

| d | Target val loss | Achieved | Steps needed | v1 steps |
|---|----------------|----------|-------------|----------|
| 64 | 0.500 | 0.500 | 1000 | 1000 |
| 128 | 0.500 | 0.497 | 1267 | 2000 |
| 256 | 0.500 | 0.496 | 2100 | 3000 |

v1 over-trained d=128 (2000 vs 1267 needed) and d=256 (3000 vs 2100
needed). This means v1 rho values for d=128 and d=256 were INFLATED
by extra training, not deflated by under-training. The convergence
confound, if anything, weakened v1's observed decline -- the true
decline is at least as steep as v1 reported.

### 4.2 FFN+Attention Mean Ratio (Primary Metric)

3-seed aggregate with convergence control:

| d | rho_FA (mean) | std | r99 ratio | r95 ratio |
|---|--------------|-----|-----------|-----------|
| 64 | 0.6503 | 0.003 | 0.642 | 0.438 |
| 128 | 0.5897 | 0.004 | 0.580 | 0.366 |
| 256 | 0.5010 | 0.003 | 0.487 | 0.273 |

Effect sizes (Cohen's d for rho_FA):
- d=64 vs d=128: (0.650 - 0.590) / 0.004 = 15.0 (extremely large)
- d=128 vs d=256: (0.590 - 0.501) / 0.004 = 22.3 (extremely large)
- d=64 vs d=256: (0.650 - 0.501) / 0.003 = 49.7 (extremely large)

### 4.3 Power Law Fit with Bootstrap CI (Fix #3)

Fitting rho_FA(d) = a * d^b in log-log space:

    rho_FA(d) = 1.438 * d^(-0.188)
    R-squared = 0.980

Bootstrap CI (10,000 resamples from 3-seed per-d ratios):

    b = -0.188, 95% CI: [-0.190, -0.185]
    a = 1.438, 95% CI: [1.427, 1.448]

**Interpretation of narrow CI**: The bootstrap CI reflects per-seed
sampling variance only. With 3 seeds and std < 0.004, the mean at each
d is well-determined, making the log-log fit stable. This does NOT
mean the power law form is correct -- with 3 points and 2 parameters,
any monotonic function would fit well. The R^2 = 0.980 has 1 degree
of freedom and is not a meaningful goodness-of-fit test.

### 4.4 Extrapolations with Uncertainty

Using rho_FA(d) = 1.438 * d^(-0.188):

| d | Predicted rho | 95% CI (bootstrap) | Predicted rank | 95% CI |
|---|--------------|-------------------|----------------|--------|
| 512 | 0.445 | [0.433, 0.459] | 228 | [221, 235] |
| 896 | 0.400 | [0.389, 0.414] | 359 | [348, 371] |
| 3584 | 0.308 | [0.299, 0.320] | 1105 | [1071, 1147] |
| 4096 | 0.301 | [0.291, 0.312] | 1231 | [1193, 1279] |
| 8192 | 0.264 | [0.255, 0.275] | 2162 | [2091, 2251] |

**These CIs are misleadingly narrow.** They capture only seed variance,
not the systematic uncertainty of extrapolating 32x beyond the data.
True uncertainty at d=4096 is much wider. Do not use these CIs for
planning -- use them only to confirm that the per-seed measurements
are consistent.

### 4.5 Multi-Checkpoint Rho Trajectory (Fix #5)

FFN+Attn rho measured at 25%, 50%, 75%, 100% of convergence-controlled
training, averaged across 3 seeds:

| d | 25% | 50% | 75% | 100% | 75%->100% delta |
|---|-----|-----|-----|------|----------------|
| 64 | 0.556 | 0.606 | 0.633 | 0.650 | +0.017 |
| 128 | 0.493 | 0.548 | 0.575 | 0.590 | +0.015 |
| 256 | 0.444 | 0.484 | 0.497 | 0.501 | +0.004 |

Key observations:
1. Rho is monotonically increasing with training at all d.
2. The rate of increase slows (diminishing returns).
3. d=256 nearly plateaus (75%->100% delta = 0.004), while d=64 is
   still climbing (+0.017).
4. **The inter-d ordering is preserved at every checkpoint**: d=64 > d=128 > d=256
   at 25%, 50%, 75%, and 100%. This is strong evidence that the scaling
   trend is real, not a convergence artifact.
5. If d=64 were trained further, its rho would increase, potentially
   WIDENING the gap with d=256. The convergence control is therefore
   conservative (it underestimates the true scaling effect).

## 5. Kill Criteria Assessment

### K1: Shannon rho > 0.5 at d=128 AND d=256

**KILLED (ACCEPTED).** Using all-weights Shannon rho (the pre-registered metric):
- d=128: rho = 0.614 > 0.5
- d=256: rho = 0.538 > 0.5

This holds under convergence control and across all 3 seeds. The kill
is accepted without retroactive reinterpretation.

Note: FFN+Attn rho at d=256 = 0.501, barely above 0.5. The r_95 ratio
at d=256 = 0.273, well below 0.5. These are informative but do not
change the K1 verdict.

### K2: Ratio increases with d

**SURVIVES.** FFN+Attn ratio monotonically decreases:
0.650 -> 0.590 -> 0.501. Cohen's d = 49.7 for the d=64 vs d=256
comparison. Unanimous across 3 seeds.

## 6. By Weight Type Analysis

| d | FFN ratio | Attn ratio | Emb ratio (excluded) |
|---|----------|-----------|---------------------|
| 64 | 0.766 | 0.592 | 0.774 |
| 128 | 0.700 | 0.535 | 0.804 |
| 256 | 0.632 | 0.436 | 0.833 |

Attention weights show the steepest decline and lowest absolute ratio.
FFN weights decline more slowly. Embeddings increase (fixed V=27).

## 7. Implications for Macro Scale

### 7.1 Base Adapter Feasibility

At d=4096 (Qwen 7B), the power law predicts:
- FFN+Attn Shannon effective rank: ~1,231 (30% of d) [CI: 1193-1279]
- Practical rank (r_95): extrapolating from 0.438 -> 0.366 -> 0.273,
  the r_95 ratio at d=4096 would be roughly 0.10-0.15, giving rank ~500.

### 7.2 ASVD and BitDelta

These compression techniques could reduce practical rank further.
Combined with layer-adaptive rank allocation (attention needs less
rank than FFN), the base adapter could be much smaller.

## 8. Assumptions and Limitations

1. **3 data points**: Cannot distinguish power law from other monotonic
   forms. The exponent b = -0.188 should be treated as "the ratio
   decreases" not as a precise scaling law.

2. **Bootstrap CI is narrow but misleading**: Reflects seed variance
   only, not model/extrapolation uncertainty.

3. **Convergence is approximate**: Multi-checkpoint analysis shows rho
   is still increasing at 100% for d=64 (less so for d=256). Extended
   training would likely narrow the inter-d gap somewhat but not
   eliminate it (the ordering is preserved at all checkpoints).

4. **Toy data**: V=27, 32K names. Real models have V=150K and
   internet-scale training data.

5. **Shannon effective rank is tail-sensitive**: Practical rank (r_95)
   gives different and arguably more useful numbers.

## 9. Worked Example (v2)

For FFN fc1 at d=256 with convergence control (2100 steps, val=0.496):
- Delta = W_trained - W_init, shape (1024, 256)
- min_dim = 256
- Effective rank: ~162 (ratio 0.632)
- Rank-95: ~70 (ratio 0.273)
- Rank-99: ~125 (ratio 0.487)

FFN+Attn aggregate at d=256: rho_FA = 0.501

At macro scale fc1 at d=4096: W in R^{18944 x 4096}
- Predicted FFN+Attn effective rank ratio: ~0.301
- Predicted rank needed (Shannon): ~1231
- Predicted rank needed (r_95, informal extrapolation): ~500
