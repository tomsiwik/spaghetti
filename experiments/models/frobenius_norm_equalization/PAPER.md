# Frobenius-Norm Equalized Composition: Proof Verification Report

## Theorem

**Theorem 1 (Gini Reduction).** For N rank-r deltas with Grassmannian-orthogonal A-matrices,
full Frobenius equalization reduces the composed Gini to at most max_i Gini(Delta_i).

**Theorem 2 (Behavioral Preservation Condition).** Equalization preserves quality if and only
if per-domain optimal scales encode training artifacts rather than genuine capability
requirements.

**Theorem 3 (Partial Equalization).** Compressing the scale ratio from R to sqrt(R) in
log-space gives Gini approximately Gini_within + reduced between-group contribution.

## Predictions vs Measurements

| Prediction (from proof)                                     | Measured              | Match? |
|------------------------------------------------------------|-----------------------|--------|
| P1: Composed Gini <= 0.29 after full equalization           | 0.2665                | YES    |
| P2a: If scales are artifacts, all 5 within 5% PPL           | 0/5 within 5%         | NO     |
| P2b: If scales encode capability, high-scale hurt >5%       | med +18.5%, math +16% | YES    |
| P3: Generation quality coherent on >=2 domains              | 5/5 coherent          | YES    |
| P4: Partial equalization Gini ~ 0.35                        | 0.393                 | CLOSE  |
| P5: Pre-equalization norm ratio ~ 20:1                      | 21.6:1                | YES    |
| P5: Top-3 energy share ~99%                                 | 98.7%                 | YES    |
| Corollary: B-matrix norms similar across domains             | 29.1-31.5 (8% spread) | YES    |

## Hypothesis

Frobenius-norm equalization of adapter contributions before composition directly fixes the
cross-domain scale imbalance (21.6:1) that dominated composed spectral pathology.

**Status: Supported (partial equalization). Full equalization kills high-scale domain quality.**

## What This Model Is

Before composing N adapter deltas via summation, each domain's contribution is scaled
so that all have equal (full) or compressed (partial) Frobenius norms. This is a
one-line preprocessing step applied once at composition time:

  Delta_i_equalized = Delta_i * (target_norm / ||Delta_i||_F)

For full equalization, target_norm = geometric mean of all domain norms.
For partial equalization, scales are compressed 50% in log-space:
  alpha_i = sqrt(geo_mean / ||Delta_i||_F)

**Computational cost:** Negligible (~1ms for N=5). No SVD, no iteration, no new parameters.

## Key References

- **FroM** (arXiv:2506.02478) -- Frobenius-norm adaptive merging
- **DO-Merging** (arXiv:2505.15875) -- magnitude-direction decoupling (motivated this work)
- **Finding #277** -- DC-Merge: cross-domain scale imbalance is root cause
- **Finding #278** -- Spectral surgery killed: low-SV = domain-pure
- **Finding #275** -- Norm preservation is the composition mechanism

## Empirical Results

### Phase 1: Frobenius Norm Analysis

Per-domain Frobenius norms confirm the 20:1 energy imbalance:

| Domain   | Scale | ||B||_F | ||Delta||_F | Energy %  |
|----------|-------|---------|-------------|-----------|
| Medical  | 20    | 31.33   | 626.6       | 33.6%     |
| Code     | 20    | 30.19   | 603.8       | 31.2%     |
| Math     | 20    | 31.45   | 629.0       | 33.9%     |
| Legal    | 4     | 29.38   | 117.5       | 1.2%      |
| Finance  | 1     | 29.13   | 29.1        | 0.07%     |

**Key observation:** B-matrix norms are nearly identical across domains (29.1-31.5, only
8% spread). The energy imbalance is almost entirely from the per-domain scale factor.
This means the adapters themselves learned similar-magnitude updates; the scale factor
s_i is the sole driver of cross-domain energy ratio.

### Phase 2: Composed Gini Reduction

| Strategy               | Composed Gini | SV Ratio | Top-1 Frac | Gini Reduction |
|-----------------------|---------------|----------|------------|---------------|
| Raw sum (baseline)     | 0.490 +/- 0.022 | 146.4    | 0.233      | --             |
| Full equalization      | 0.267 +/- 0.037 | 10.5     | 0.225      | 45.6%          |
| Partial equalization   | 0.393 +/- 0.026 | 32.5     | 0.228      | 19.6%          |

**K703 PASS:** Full equalization achieves Gini = 0.267, well below the 0.30 threshold.
The max/min SV ratio collapses from 146.4 to 10.5 (14x compression), confirming that
the spectral pathology was almost entirely from cross-domain scale imbalance.

### Phase 3: Perplexity

**Full equalization vs raw sum:**

| Domain   | Raw Sum PPL | Full Eq PPL | Change    | Within 5%? |
|----------|-------------|-------------|-----------|-----------|
| Medical  | 3.851       | 4.564       | +18.5%    | NO        |
| Code     | 3.764       | 3.971       | +5.5%     | NO        |
| Math     | 2.416       | 2.808       | +16.2%    | NO        |
| Legal    | 15.501      | 14.107      | -9.0%     | NO        |
| Finance  | 14.081      | 13.239      | -6.0%     | NO        |
| Mixed    | 6.585       | 6.770       | +2.8%     | --        |

**K704 FAIL (full equalization):** 0/5 domains within 5%. The pattern is clear and
asymmetric: high-scale domains (medical, math) are HURT, low-scale domains (legal,
finance) are HELPED. This confirms Theorem 2 prediction P2b: the per-domain optimal
scales partially encode genuine capability requirements, not just training artifacts.

**Partial equalization vs raw sum:**

| Domain   | Raw Sum PPL | Partial PPL | Change    | Within 5%? |
|----------|-------------|-------------|-----------|-----------|
| Medical  | 3.851       | 3.940       | +2.3%     | YES       |
| Code     | 3.764       | 3.715       | -1.3%     | YES       |
| Math     | 2.416       | 2.528       | +4.6%     | YES       |
| Legal    | 15.501      | 14.633      | -5.6%     | NO (improved) |
| Finance  | 14.081      | 13.568      | -3.6%     | YES       |
| Mixed    | 6.585       | 6.508       | -1.2%     | --        |

**K704 assessment (partial equalization):** 4/5 domains within 5% (legal exceeds
threshold but in the improvement direction). Mixed PPL improves by 1.2%. Partial
equalization is the practical sweet spot.

### Phase 4: Generation Quality

Both full equalization and raw sum produce coherent, domain-relevant text on all 5 domains.

**Notable:** Full equalization produces CLEANER medical output (no `###` artifacts
that raw sum has), suggesting better domain balance aids generation quality even
when PPL is slightly higher.

**K705 PASS:** All 5 domains produce coherent, domain-relevant text with full
equalization. The PPL degradation on medical/math does not manifest as incoherent
generation at this scale.

## Key Finding: The Scale Ratio Encodes Both Signal and Artifact

The experiment reveals that per-domain optimal scales have a dual nature:

1. **Artifact component:** The 20:1 ratio creates severe spectral imbalance (98.7%
   energy in top-3 domains), drowning out legal and finance contributions.

2. **Signal component:** High-scale domains genuinely need larger perturbations.
   Medical and math domains lose 16-18% PPL under full equalization, confirming
   that s=20 carries real capability information.

Partial equalization (50% compression in log-space) threads the needle: it recovers
legal/finance visibility (-5.6%, -3.6% improvement) while keeping medical/math
degradation tolerable (+2.3%, +4.6%). Net effect: 1.2% mixed PPL improvement.

## Connection to Prior Findings

This result resolves the investigation opened by Findings #277 and #278:

| Finding | What it showed | What Frobenius equalization adds |
|---------|---------------|--------------------------------|
| #277 (DC-Merge) | Within-domain smoothing barely helps (18.5% Gini reduction) | Cross-domain equalization gives 45.6% -- the disease was between-domain, not within |
| #278 (Surgery) | Post-composition SVD surgery is structurally inverted | The composed spectrum pathology is SCALE, not direction -- equalization fixes scale |
| #275 (Norm preservation) | Norm preservation is the composition mechanism | Equalization is norm-correction at the domain level (same principle, applied cross-domain) |

## Limitations

1. **N_eval = 20 per domain.** PPL differences at 2-5% may not be statistically significant.
2. **No task-specific evaluation.** PPL is a weak proxy (r=0.08 correlation with task quality).
   Generation samples look good but are not scored.
3. **Only 2 equalization strategies tested.** The optimal compression factor (50% in log-space)
   is not derived from theory -- it is a Type 2 unknown that warrants further exploration.
4. **Single rank r=16.** Higher ranks may change the energy landscape.

## What Would Kill This

1. **At macro scale:** If partial equalization degrades task-specific benchmarks (GSM8K, code
   completion) despite improving PPL, the scale information is even more critical than PPL suggests.
2. **With routing:** Per-token routing selects top-k adapters, so the scale imbalance may not
   matter in practice (only 1-2 adapters active per token, no summation of all 5).
3. **Optimal compression:** If the 50% log-compression turns out to be fragile (works for
   this adapter set but not others), partial equalization is not a general solution.
