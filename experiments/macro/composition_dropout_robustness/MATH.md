# Composition Dropout Robustness: Mathematical Foundations

## 1. Setup

### 1.1 Notation

| Symbol | Definition | Shape |
|--------|-----------|-------|
| W | Frozen base model weights | R^{d_out x d_in} per layer |
| delta_i = B_i @ A_i | Expert i LoRA weight delta | R^{d_out x d_in} |
| N | Total number of experts | scalar (=50 for pilot) |
| k | Number of experts retained (= floor(N * p)) | scalar (=40 at p=0.8) |
| p | Retention fraction | scalar (=0.8) |
| S_b | Random subset b, |S_b| = k, drawn uniformly without replacement from {1..N} | set |
| B | Number of bootstrap subsets | scalar (=20) |
| PPL(M, D) | Perplexity of model M on dataset D | scalar |
| PPL_ref | PPL of model composed with all N experts | scalar |
| PPL_b | PPL of model composed with subset S_b | scalar |
| CV | Coefficient of variation = std(PPL) / mean(PPL) | scalar |
| r | LoRA rank | scalar (=16) |
| d | Model hidden dimension | scalar (=3584 for Qwen2.5-7B) |

### 1.2 Composition Model

The composed model with all N experts uses additive (sum) pre-merge:

    W_composed = W + sum_{i=1}^{N} delta_i

The composed model with subset S_b:

    W_b = W + sum_{i in S_b} delta_i

**Note:** We use *sum* composition (not averaged). Each expert contributes its
full delta regardless of how many experts are in the composition. This is the
SOLE default, consistent with all macro experiments.

## 2. Dropout Analysis

### 2.1 Perturbation from Dropping Experts

When we drop a set D_b = {1..N} \ S_b of (N - k) experts, the weight
perturbation relative to the full composition is:

    W_b - W_composed = -sum_{i in D_b} delta_i

The Frobenius norm of this perturbation:

    ||W_b - W_composed||_F = ||sum_{i in D_b} delta_i||_F

Under structural orthogonality (cos(delta_i, delta_j) ~ 0 for i != j):

    ||sum_{i in D_b} delta_i||_F^2 ~= sum_{i in D_b} ||delta_i||_F^2

If experts have approximately equal norm ||delta_i||_F ~ sigma:

    ||W_b - W_composed||_F ~= sigma * sqrt(N - k) = sigma * sqrt(N * (1-p))

For N=50, p=0.8: sqrt(10) ~ 3.16 * sigma.

### 2.2 Relative Perturbation

The relative perturbation (fraction of total expert signal removed) is:

    ||W_b - W_composed||_F / ||W_composed - W||_F
    = ||sum_{D_b} delta_i||_F / ||sum_{all} delta_i||_F
    ~= sqrt(N - k) / sqrt(N)     (under orthogonality, equal norms)
    = sqrt(1 - p)

For p=0.8: sqrt(0.2) = 0.447, i.e., dropping 20% of experts removes ~45% of
the Frobenius norm signal.

**Key insight:** Under orthogonality with sum composition, dropout fraction p
removes sqrt(1-p) of the weight signal, NOT (1-p). This is because orthogonal
contributions add in quadrature. The PPL impact depends on how the loss function
responds to this norm reduction.

### 2.3 Expected PPL Response

For small perturbations, the loss change from dropping experts is approximately:

    L(W_b) - L(W_composed) ~= -<grad_L, sum_{D_b} delta_i> + O(||perturbation||^2)

Under orthogonality, the dropped experts' gradients are approximately independent,
so the expected loss change is:

    E[L(W_b) - L(W_composed)] ~= -(N-k)/N * <grad_L, sum_{all} delta_i>

This is a first-order approximation. The actual PPL change depends on:
1. Whether the dropped experts were *helpful* (negative gradient alignment)
2. Whether higher-order interactions exist between experts
3. Whether some experts are redundant (overlapping specializations)

### 2.4 Bootstrap Variance Under Orthogonality

For B bootstrap subsets, each of size k drawn uniformly from N, the
variance of PPL across subsets measures how much individual expert
identity matters versus aggregate count.

**Null hypothesis (experts are interchangeable):**
If all experts contribute equally, then PPL depends only on k (the
count), not on which k experts are selected. Variance across subsets
would be zero.

**Alternative (experts are specialized):**
If expert i uniquely contributes to domain D_i, then a subset missing
the expert specialized on the calibration domain will have higher PPL.
The variance reflects this specialization.

The coefficient of variation:

    CV = std(PPL_b, b=1..B) / mean(PPL_b, b=1..B) * 100%

Under orthogonality with equal-quality experts: CV ~ 0%.
Under heterogeneous quality: CV scales with the Gini coefficient of
expert quality contributions.

### 2.5 Combinatorial Argument for Robustness

The number of possible 80% subsets of 50 experts is:

    C(50, 40) = C(50, 10) = 10,272,278,170

We sample B=20 from this space. With 50 experts and 40 selected per
subset, each expert appears in approximately:

    E[appearances of expert i] = B * k/N = 20 * 40/50 = 16 subsets

The probability that expert i is dropped from a specific subset:

    P(i not in S_b) = (1-p) = 0.2

So each expert is absent from roughly 4 of 20 subsets. This gives
sufficient coverage: every expert's absence is tested multiple times,
and the variance captures the impact of any single expert's removal.

## 3. Kill Criteria Formalization

### K1: CV > 5% (composition is fragile)

    CV = std({PPL_b : b=1..B}) / mean({PPL_b : b=1..B}) * 100

    KILL if CV > 5.0

**Interpretation:** CV > 5% means the standard deviation of PPL across
random subsets exceeds 5% of the mean PPL. At the pilot-50 scale with
Qwen2.5-7B (typical PPL ~10-50 on calibration data), this would mean
std(PPL) > 0.5-2.5 PPL points. If true, composition is brittle: which
experts you include matters a lot.

**Expected value given micro results:** Micro experiments showed CV ~ 0%
(zero degradation). Macro may differ because experts have real specialization.
Prediction: CV < 2% (between micro's 0% and the 5% threshold).

### K2: Best 80% subset improves > 10% over all-50

    KILL if min_b (PPL_b - PPL_ref) / PPL_ref * 100 < -10

**Interpretation:** If removing some experts significantly improves quality,
then the removed experts were *harmful*. This means composition is not
plug-and-play: you must curate which experts to include. At -10%, the quality
gain from pruning is operationally significant.

This tests whether the composition contains "dead weight" or harmful experts
that drag down quality. Under perfect orthogonality, each expert contributes
independently, so removing any should only *hurt* (since all experts were
independently useful). K2 failing would suggest experts interfere constructively
in unexpected ways, or some pilot experts are low-quality and contaminate the sum.

### K3: Worst 80% subset degrades > 15% from all-50

    KILL if max_b (PPL_b - PPL_ref) / PPL_ref * 100 > 15

**Interpretation:** If removing 20% of experts causes >15% PPL degradation
in the worst case, then random expert dropout is dangerous. This means the
system is not robust to expert failures (e.g., NVMe cache misses, version
upgrades, expert pruning).

Under orthogonality, the worst case occurs when the 10 dropped experts
happen to be the highest-quality ones. The degradation depends on the
distribution of expert quality. With 50 roughly-equal experts, losing 10
should cause ~sqrt(10/50) = 45% norm reduction but much less PPL impact
(due to the loss landscape's local smoothness).

**Expected range:** If K3 threshold is 15% and micro showed 0% degradation,
the real value likely falls in 2-8%. K3 passes unless pilot adapters have
extreme quality variance.

## 4. Worked Example (Micro-Scale Projection)

### Micro: d=32, r=8, N=50

From cross_domain_composition (d=32, N=50):
- Multi-expert composition: -1.0% degradation vs single-expert (within noise)
- All 50 experts active, sum composition
- Cross-domain types: 10 different combinations

Projecting to dropout:
- Remove 10/50 experts (20% dropout)
- Under orthogonality, perturbation = sqrt(10/50) * total_norm = 0.447 * total_norm
- At micro scale with zero specialization: PPL change ~ 0% (no expert matters)
- CV ~ 0%, K1 trivially passes

### Macro: d=3584, r=16, N=50

Expected differences from micro:
1. Experts have real domain specialization (trained on distinct topic data)
2. Expert quality varies (42.2% avg PPL improvement, but range likely wide)
3. Calibration data comes from expert domains (not synthetic toy tasks)

Conservative estimates:
- CV: 1-3% (some variance from losing domain-specific experts)
- Best delta: -2 to -5% (removing worst experts helps slightly)
- Worst delta: +3 to +8% (removing best experts hurts noticeably)

All within kill thresholds (5%, 10%, 15%).

## 5. Statistical Power Analysis

### Bootstrap Sample Size

With B=20 subsets, the standard error of the estimated CV is:

    SE(CV) ~= CV / sqrt(2 * (B - 1)) ~= CV / sqrt(38)

If true CV = 2%, estimated CV has SE ~= 0.32%, so 95% CI is [1.4%, 2.6%].
This is well-separated from the K1 threshold of 5%.

If true CV = 4%, estimated CV has SE ~= 0.65%, so 95% CI is [2.7%, 5.3%].
The upper bound touches the threshold -- marginal case requires interpretation.

B=20 is sufficient to distinguish "clearly robust" (CV < 3%) from "clearly
fragile" (CV > 7%) but not to precisely resolve boundary cases (CV ~ 4-6%).

### Best/Worst Subset Extremes

With B=20 samples from C(50,40) possible subsets, the observed min and max
are conservative estimates of the true extremes. The expected range depends on
the underlying PPL distribution:

- If PPL ~ Normal(mu, sigma), the expected range of 20 samples is ~3.7 * sigma
- This means our observed worst-case underestimates the true worst-case
- K3 at 15% provides margin for this: if observed worst is 8%, true worst
  is likely < 12% (within threshold)

## 6. Assumptions

1. **Sum composition (not averaged).** Each expert contributes its full delta
   weight. This is the SOLE production configuration. Averaged composition
   would show even less sensitivity to dropout.

2. **Structural orthogonality holds at d=3584, r=16.** Proven: cos~0.0002 at
   d=896. At d=3584, orthogonality is even stronger (cos ~ d^{-0.673}).
   Cross-terms in the perturbation norm are negligible.

3. **Expert quality is heterogeneous.** The 50 pilot adapters were trained on
   different domains with varying quality (42.2% avg PPL improvement, likely
   with substantial variance). This heterogeneity drives bootstrap variance.

4. **Calibration data is representative.** PPL is measured on texts from the
   pilot adapter domains, providing a balanced quality signal across experts.

5. **Perplexity is a smooth function of weight perturbation.** Required for
   the first-order Taylor analysis. Valid at SOLE production cosines where
   perturbations are small relative to base weights.

6. **20 bootstrap samples provide sufficient coverage.** Each expert is
   absent from ~4 subsets, enough to estimate its marginal contribution
   through bootstrap variance.

## 7. Computational Complexity

| Operation | Cost | Time Estimate |
|-----------|------|--------------|
| Load base model (4-bit) | 1x | ~30s |
| Compose k=40 adapters (CPU merge) | O(k * L * d^2) | ~15s |
| Load composed model + forward pass | O(T * seq_len * d^2) | ~10s |
| Total per subset (compose + load + eval) | | ~25s |
| Full experiment (1 ref + 1 base + 20 subsets) | 22x above | ~550s (~9 min) |

Total runtime budget: 15 minutes. Expected: ~10 minutes.
Smoke test (3 subsets of 5 adapters): < 60 seconds.

## 8. Connection to Other Experiments

- **composition_health_kl_divergence:** Measures KL divergence as a composition
  health score. Dropout robustness complements by testing aggregate sensitivity
  rather than per-expert contribution.
- **leave_one_out_expert_ranking:** LOO tests individual expert value (which
  one matters most). Dropout robustness tests aggregate stability (does
  any particular combination matter).
- **composition_weight_sensitivity (micro):** Tested N=2..100 with zero
  specialization. Showed 0% degradation, confirming orthogonality prediction.
  Macro test needed because real specialization may break this.
- **cross_domain_composition (micro):** Tested N=50 composition at d=32.
  Showed -1% degradation on cross-domain queries, within noise. Macro provides
  higher-dimensional confirmation with real domain specialization.
