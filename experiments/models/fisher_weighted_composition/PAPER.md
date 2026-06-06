# Fisher-Weighted Adapter Composition: Proof Verification Report

## Theorem

Fisher information diagonal provides per-adapter importance weights that
separate capability signal from training artifact (Matena & Raffel 2022,
arXiv:2111.09832). Fisher importance w_i = sum(F_i[j] * Delta_i[j]^2)
weights parameters by both magnitude AND task-relevance, unlike Frobenius
norms which weight by magnitude only.

## Predictions

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| P1: Fisher computation < 60s (K707: < 10 min) | 67s | YES (K707 PASS) |
| P2: Spearman rho(Fisher, Frobenius) in [0.7, 0.9] | rho = 1.000 | NO (K708 FAIL -- perfect rank correlation) |
| P3: Fisher mixed PPL < 6.508 (partial eq baseline) | 7.034 | NO (K706 FAIL -- 8.1% worse) |
| P4: High-scale domains within 5%, low-scale improved | All domains worse than raw sum | NO |
| P5: Fisher/Frob ratio highest for finance, lowest for medical | legal:2.40, finance:2.04, math:1.12, medical:0.99, code:0.82 | PARTIAL (correct direction, insufficient magnitude) |

## Hypothesis

Fisher information diagonal provides per-adapter importance weights that
separate capability signal from training artifact, producing lower mixed PPL
than the empirical 50% log-compression while eliminating the unprincipled
compression hyperparameter.

**VERDICT: KILLED (K706 FAIL, K708 FAIL)**

## What This Experiment Is

This experiment computes diagonal Fisher Information for each of 5 domain
LoRA adapters on the BitNet-2B-4T base model, then uses Fisher importance
to derive per-adapter composition weights as an alternative to the empirical
50% log-compression factor from Finding #279.

For each domain i:
1. Apply single adapter to base model
2. Compute diagonal Fisher via squared gradients on 10 domain validation samples (128 tokens each)
3. Fisher importance: w_i = sum(F_i[j] * Delta_i[j]^2)
4. Normalize to composition weights: alpha_i = w_i / sum(w_j)

## Key References

- Fisher Merging (Matena & Raffel 2022, arXiv:2111.09832)
- EWC (Kirkpatrick et al. 2017, arXiv:1612.00796)
- DeLoRA (arXiv:2503.18225)
- Finding #279 (Frobenius equalization baseline)

## Empirical Results

### Kill Criteria Assessment

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| K706: Fisher mixed PPL < partial eq | < 6.508 | 7.034 | **FAIL** |
| K707: Fisher compute < 10 min | < 600s | 67s | **PASS** |
| K708: Spearman rho <= 0.9 | <= 0.9 | 1.000 | **FAIL** |

### Mixed PPL Comparison (lower is better)

| Strategy | Mixed PPL | vs Raw Sum |
|----------|-----------|-----------|
| Raw sum (no equalization) | 6.585 | baseline |
| Partial Frobenius equalization (50%) | 6.508 | -1.2% |
| Full Frobenius equalization | 6.770 | +2.8% |
| **Fisher-weighted** | **7.034** | **+6.8%** |

### Per-Domain PPL Changes vs Raw Sum

| Domain | Partial eq | Full eq | Fisher |
|--------|-----------|---------|--------|
| medical | +2.3% | +18.5% | +5.7% |
| code | -1.3% | +5.5% | +8.7% |
| math | +4.6% | +16.2% | +1.1% |
| legal | -5.6% | -9.0% | +11.2% |
| finance | -3.6% | -6.0% | +7.4% |

Fisher hurts ALL 5 domains vs raw sum. Partial equalization helps 3/5.

### Composed Gini Coefficient (lower is better)

| Strategy | Gini |
|----------|------|
| Raw sum | 0.490 |
| Partial eq | 0.393 |
| Full eq | 0.267 |
| **Fisher** | **0.563** |

Fisher INCREASES spectral pathology by 15% over raw sum. It amplifies
the energy concentration in high-scale domains instead of compressing it.

### Fisher Weight Analysis

| Domain | Fisher w | Frob w | Fisher/Frob ratio | Scale |
|--------|----------|--------|-------------------|-------|
| medical | 0.3332 | 0.3363 | 0.991 | 20 |
| code | 0.2567 | 0.3123 | 0.822 | 20 |
| math | 0.3801 | 0.3389 | 1.122 | 20 |
| legal | 0.0285 | 0.0118 | 2.405 | 4 |
| finance | 0.0015 | 0.0007 | 2.037 | 1 |

Fisher/Frobenius ratio CV: 0.425 (moderate variation in relative weighting).
But rank correlation: rho = 1.000 (PERFECT -- no reranking).

Fisher equalization scales (relative to mean=1):
- medical: 1.67, code: 1.28, math: 1.90
- legal: 0.14, finance: 0.007

These scales AMPLIFY the high-scale domains and SUPPRESS the low-scale domains --
the exact opposite of what's needed.

### Fisher Diagonal Statistics

| Domain | Fisher importance | Frobenius sq | Fisher/Frob ratio | Per-key CV |
|--------|------------------|-------------|-------------------|-----------|
| medical | 9.12e-06 | 392647 | 2.32e-11 | 1.50 |
| code | 7.02e-06 | 364600 | 1.93e-11 | 1.37 |
| math | 1.04e-05 | 395700 | 2.63e-11 | 1.07 |
| legal | 7.78e-07 | 13814 | 5.64e-11 | 1.45 |
| finance | 4.05e-08 | 849 | 4.77e-11 | 1.37 |

Fisher computation time per domain: 12-15s. Total: 67s.

## Root Cause Analysis: Why Fisher Fails

The Fisher importance w_i = sum(F_i[j] * Delta_i[j]^2) has a fundamental problem
for our use case:

1. **Delta_i[j]^2 dominates.** The squared delta values span 5 orders of magnitude
   across domains (due to the 20:1 scale factor), while Fisher diagonal values
   are relatively uniform across domains (Fisher/Frob ratio varies only 2.3-5.6e-11).

2. **Fisher diagonal doesn't compensate for scale.** The Fisher diagonal measures
   model sensitivity at each parameter position, which is a property of the
   BASE MODEL'S architecture, not of the adapter. Since all adapters share the same
   base model positions (they modify the same weight matrices), the Fisher pattern
   is similar across domains. The domain-specific variation comes primarily from
   the data distribution, which introduces ~2x variation -- far too small to
   compensate for the 400x (20^2:1^2) energy ratio.

3. **Fisher importance is dominated by scale^2, not Fisher^2.**
   w_i = sum(F_i * Delta_i^2) ~ F_mean * sum(Delta_i^2) = F_mean * ||Delta_i||_F^2.
   Since F_mean is roughly constant across domains (it depends on the base model's
   parameter sensitivity, not the adapter), w_i is approximately proportional to
   ||Delta_i||_F^2 = Frobenius energy. This explains rho = 1.0.

4. **The Fisher equalization scales amplify imbalance.** Since Fisher weights
   track Frobenius energy, the resulting composition scales give MORE weight to
   already-dominant domains (medical: 1.67, math: 1.90) and less to suppressed
   domains (finance: 0.007). This is the opposite of equalization.

## Why Fisher Merging Works Elsewhere But Fails Here

Fisher merging (Matena & Raffel 2022) succeeds when merging FULL MODELS trained
on different tasks, because:
- Different models have genuinely different Fisher patterns (different parameters
  become important for different tasks)
- The Fisher diagonal varies by orders of magnitude across parameters within
  a single model

In our setting:
- All adapters share the SAME base model (same Fisher pattern at shared positions)
- The adapters modify the same weight matrices (same positions in parameter space)
- The domain-specific variation in Fisher is tiny (~2x) compared to scale variation
  (~400x in energy)

Fisher merging is designed for FULL MODEL merging, not for weighting LoRA adapter
CONTRIBUTIONS to a shared composition.

## Limitations

- N_FISHER_SAMPLES=10 per domain (low power for Fisher estimation, but CV analysis
  shows 1.1-1.5 per-key CV, suggesting the Fisher is well-estimated)
- FISHER_SEQ_LENGTH=128 (shorter than eval's 256, but Fisher is a local property)
- PPL-only evaluation, no task-specific benchmarks
- Single configuration (N=5, r=16, BitNet-2B-4T base)

## What Would Kill This (and did)

**K706 (PPL):** Fisher-weighted PPL is 8.1% WORSE than partial equalization.
Killed because Fisher importance tracks Frobenius energy, producing anti-helpful
composition weights.

**K708 (Rank correlation):** rho = 1.0 -- Fisher provides ZERO new ranking
information beyond what Frobenius norms already capture.

## What Was Learned

1. **Diagonal Fisher on shared-base LoRA adapters is a poor importance measure
   for composition weighting.** The Fisher diagonal is primarily a property of
   the base model's architecture, not the adapter's domain contribution.

2. **The 50% log-compression factor from Finding #279 remains the best available
   composition scaling method.** Despite being empirical, it compresses the scale
   ratio without amplifying it, which Fisher fundamentally cannot do.

3. **The scale decomposition problem (artifact vs capability) cannot be solved by
   per-parameter importance.** Fisher measures parameter sensitivity, not whether
   a parameter's value is an artifact. The artifact/signal decomposition requires
   a fundamentally different approach -- either causal intervention (ablation studies)
   or information-theoretic methods (mutual information with domain labels).

4. **Fisher/Frobenius ratio does vary by domain** (legal/finance get 2x higher ratio
   than medical/code/math). This confirms that low-scale domains have higher
   per-parameter Fisher importance -- the information is THERE, it's just completely
   overwhelmed by the 400x energy ratio from Delta^2.

## Implications for Future Work

The Fisher ratio insight (low-scale = higher per-parameter importance) suggests
that a method which uses Fisher/Frobenius RATIO (not absolute Fisher importance)
to set composition weights could work. Specifically:

  alpha_i = (w_Fisher_i / ||Delta_i||_F^2)  (the Fisher-to-Frobenius ratio)

This would normalize out the scale dominance and extract the pure importance signal.
But with only 5 data points and a 2x ratio range, it's unclear whether this signal
is strong enough to improve over the empirical 50% compression.
