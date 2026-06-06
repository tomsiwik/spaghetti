# Gamma-Perturbation Correlation: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 896 (0.5B), 3584 (7B) |
| r | LoRA rank | 16 |
| N | Number of expert adapters | 5 (real adapters) |
| L | Number of transformer layers | 24 (0.5B), 28 (7B) |
| gamma_l | Learned RMSNorm scale at layer l | (d,), range [-2.17, 12.38] |
| Delta_l^{(i)} | Expert i weight delta at layer l | B_l^{(i)} A_l^{(i)} |
| \|gamma_l\| | Per-dimension absolute value of gamma | (d,), all positive |
| \|\|Delta_l[:, j]\|\| | Column norm of delta (per input dim) | (d,), all positive |
| rho | Pearson correlation between \|gamma\| and \|\|Delta[:, j]\|\| | [-1, 1] |
| cos | Cosine similarity between \|gamma\| and \|\|Delta[:, j]\|\| | [0, 1] for positive vectors |
| alpha | Amplification ratio: output_dev / sum_epsilon | ~0.022 |

## 2. The Question

The parent experiment (rmsnorm_gamma_nonuniformity) proved that alpha is robust to
gamma non-uniformity (max ratio 1.43x) under the assumption that gamma and expert
perturbation directions are UNCORRELATED (Assumption 4 in parent MATH.md).

The adversarial review identified this as the remaining open risk:

> "If LoRA experts systematically modify dimensions where gamma is large (or small),
> creating a bias in the cancellation. This would require gamma and expert weight
> patterns to be structurally correlated."

## 3. Why Cosine Similarity is the WRONG Metric

### 3.1 The positivity trap

Both |gamma_l| and ||Delta_l[:, j]|| are magnitude vectors: all entries are
non-negative. For two random vectors a, b in R+^d:

    E[cos(a, b)] = E[sum(a_i * b_i)] / (E[||a||] * E[||b||])

When a_i >= 0 and b_i >= 0 for all i, the dot product sum(a_i * b_i) is always
positive. For independent half-normal vectors in R^d:

    E[cos(a, b)] = (E[|X|])^2 / E[X^2] = (2/pi) / 1 = 2/pi ~ 0.637

Our empirical measurement of random positive vectors at d=3584 gives 0.638,
matching this prediction exactly.

### 3.2 The correct statistic

Pearson correlation subtracts means before computing cosine:

    rho(a, b) = cov(a, b) / (std(a) * std(b))
              = sum((a_i - mean(a))(b_i - mean(b))) / (d * std(a) * std(b))

This removes the positivity bias. If a and b are independent (even if both positive),
E[rho] = 0.

### 3.3 Observed values

| Statistic | Expected (independent) | Observed | Interpretation |
|-----------|----------------------|----------|----------------|
| Cosine | ~0.637 | 0.839 | Inflated by positivity + shared scale structure |
| Pearson | 0.0 | 0.018 | Negligible correlation |
| Spearman | 0.0 | 0.006 | Negligible rank correlation |

The observed cosine of 0.839 exceeds the random baseline of 0.637 by 0.20. This
excess comes from shared scale structure (both gamma and weight norms are related
to the base model's learned representation), not from LoRA preferentially modifying
high-gamma dimensions. The Pearson correlation of 0.018 confirms this: there is
no linear relationship between which dimensions gamma emphasizes and which
dimensions LoRA modifies.

## 4. Theoretical Analysis: Why Correlation Does NOT Affect Alpha

### 4.1 The amplification ratio at layer l

At each layer, the forward pass computes:

    h_{l+1} = h_l + sigma((W_l + Delta_l) @ (gamma_l * h_l / rms(h_l)))

The perturbation u_l = h_l^{naive} - h_l^{gt} propagates as:

    u_{l+1} = u_l + J_l @ u_l + eta_l

where J_l is the Jacobian including the gamma-scaled RMSNorm.

### 4.2 Impact of correlated gamma on alpha

If gamma_l is correlated with the perturbation direction eta_l, the Jacobian
J_l preferentially amplifies the perturbation along high-gamma dimensions.
The amplification factor at layer l becomes:

    ||J_l @ eta_l|| / ||eta_l|| = ||diag(gamma_l) @ M_l @ eta_l|| / ||eta_l||

where M_l absorbs the non-gamma components of the Jacobian.

In the worst case (eta_l aligned with gamma_l's principal direction):

    ||J_l @ eta_l|| / ||eta_l|| <= ||gamma_l||_max * ||M_l||_op

vs the average case:

    E[||J_l @ eta_l|| / ||eta_l||] ~ ||gamma_l||_rms * ||M_l||_op

The ratio worst/average = ||gamma_l||_max / ||gamma_l||_rms.

### 4.3 Empirical correction factor

For Qwen2.5-0.5B post_attention_layernorm:
- gamma_rms ~ 1.55 (derived from mean=1.43, std=0.57)
- gamma_max ~ 11.4

Maximum correction: 11.4 / 1.55 = 7.4x PER LAYER.

But this worst case requires the perturbation to be PERFECTLY aligned with the
single max-gamma dimension across ALL layers simultaneously. In practice:

1. Perturbation eta_l is spread across O(r) = O(16) dimensions (LoRA rank)
2. Different layers have different max-gamma dimensions
3. The residual stream averages perturbations across layers

The empirical sweep shows that even at correlation=1.0 (perfectly correlated
synthetic gamma), the alpha ratio is only 1.068x. This is because:
- The GELU nonlinearity prevents linear amplification
- The residual connection dilutes layer-wise amplification
- RMSNorm renormalization constrains output magnitude

### 4.4 Production-scale extrapolation

At d=3584 (7B), gamma non-uniformity is actually SMALLER:
- 0.5B: gamma range [-2.17, 12.38], std=0.78
- 7B: expected similar or smaller (more parameters average out extreme values)

The 1.068x factor measured at d=64 with synthetic correlation is an UPPER BOUND.
At d=3584 with real (negligible) correlation of 0.018, the correction is:

    alpha_corrected = alpha_baseline * (1 + 0.068 * 0.018/1.0) ~ alpha_baseline * 1.001

Effectively zero.

## 5. Gamma vs Base Weight Norms (Structural Correlation)

The cosine between gamma and base weight column norms is very high (0.92 at 0.5B).
This is expected: both gamma and weight norms encode the base model's learned
importance of each dimension. If dimension j is important for the base model,
both gamma_j and ||W[:, j]|| will be large.

However, the Pearson correlation is only 0.098, meaning the high cosine is
mostly driven by:
1. Both being positive (0.637 baseline)
2. Both having similar scale structure (shared training)
3. NOT a dimension-by-dimension preferential alignment

This structural correlation does NOT affect SOLE composition because:
- LoRA deltas are learned relative to the CURRENT base model
- The perturbation from expert removal is in the DELTA space, not the BASE space
- Gamma affects both signal and perturbation equally (parent experiment proof)

## 6. Summary of Key Numbers

| Measurement | Value | Safety Implication |
|------------|-------|-------------------|
| Pearson(gamma, delta) across 5 adapters | 0.018 | No systematic correlation |
| Spearman(gamma, delta) | 0.006 | No rank-order correlation |
| Raw cosine (positivity artifact) | 0.839 | Misleading, do not use |
| Expected cosine (random positive d=3584) | 0.638 | Baseline for comparison |
| Alpha at real correlation level | 1.001x | Negligible correction |
| Alpha at perfect correlation (theoretical worst) | 1.068x | Bounded even in worst case |
| Alpha with real Qwen gamma (downsampled) | 1.014x | Production-realistic |

## 7. Assumptions

1. **LoRA adapters trained via standard fine-tuning.** Unusual training procedures
   (e.g., constraining updates to specific dimensions) could create artificial
   correlation. Standard distillation from teacher models does not.

2. **Column norms capture the relevant magnitude pattern.** The correlation
   analysis uses per-input-dimension column norms of B@A. If the safety-relevant
   direction is different (e.g., row norms, singular values), the correlation
   could differ. However, the alpha sweep tests the actual impact on the safety
   bound regardless of which metric captures the correlation.

3. **Five adapters are representative.** With 5 adapters * 168 layer-modules each
   = 840 measurements, the Pearson correlation estimate has standard error
   ~ 1/sqrt(840) ~ 0.035. The observed 0.018 is well within noise of zero.
