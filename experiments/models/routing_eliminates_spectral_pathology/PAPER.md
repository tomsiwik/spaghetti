# Routing vs Spectral Optimization: Proof Verification Report

## Theorem (from MATH.md)

**Theorem 1:** Under top-k routing with Grassmannian-orthogonal A matrices, the
composed perturbation has rank at most k*r, and the Gini coefficient depends only
on the within-adapter SV distributions of the k selected adapters and their
relative scales.

**Corollary (claimed):** Under oracle top-2 routing, the between-domain Gini
component (~45% of N=5 uniform Gini) vanishes.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? | Notes |
|------------------------|----------|--------|-------|
| P1: Top-2 oracle Gini < 0.15 | Mean 0.329, Max 0.508 | **NO** | Individual adapter Gini already 0.23-0.25; legal+finance pair 0.44 |
| P2: Top-2 PPL within 5% of single (on >=3/5) | 4/5 within 5% | **YES** | Only medical exceeds (+17.6%); math actually improves (-0.7%) |
| P3: Top-2 behavioral within 15% (on >=2/3) | 2/3 within 15% | **YES** | Math 0% degradation, medical 3.8%, but code 40.7% |
| P4: Top-2 beats uniform+eq on majority | 2/5 domains | **NO** | Top-2 wins math+code but loses medical, legal, finance |
| P5: Individual adapter Gini < 0.15 | 0.23-0.25 | **NO** | Within-adapter SV spread creates baseline Gini of 0.23-0.25 |

## Hypothesis

Top-k routing with k=2 eliminates spectral composition pathology, making the
entire spectral optimization track production-irrelevant.

**Verdict: PARTIALLY SUPPORTED with important corrections.**

## What This Experiment Tests

Whether top-k routing eliminates spectral pathology (high Gini from mixing
N=5 adapters at different scales) and whether the 5-experiment spectral arc
(Findings #277-282) is production-irrelevant.

## Kill Criteria Assessment

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| K712: Max oracle Gini | < 0.15 | 0.508 | **KILL** |
| K713: PPL >5% worse on >=3/5 | <3 domains | 1 domain | **PASS** |
| K714: Behavioral >15% worse on >=2/3 | <2 domains | 1 domain | **PASS** |

**K712 KILLS the Gini prediction but NOT the practical conclusion.**

## Empirical Results

### Phase 1: Spectral Analysis (Gini)

**Individual adapter Gini (baseline within-adapter spread):**

| Domain | Mean Gini | Max Gini |
|--------|-----------|----------|
| medical | 0.233 | 0.333 |
| code | 0.251 | 0.346 |
| math | 0.245 | 0.382 |
| legal | 0.247 | 0.355 |
| finance | 0.239 | 0.377 |

Key finding: Individual adapters already have Gini 0.23-0.25 from within-adapter
SV spread. The 0.15 threshold was **wrong** -- it assumed singular values within a
single rank-16 adapter would be near-uniform, but B-matrix training creates
SV concentration.

**Top-2 pair Gini (all C(5,2) = 10 pairs):**

| Pair | Scale Ratio | Mean Gini | Max Gini |
|------|-------------|-----------|----------|
| medical+code | 1:1 | 0.251 | 0.331 |
| medical+math | 1:1 | 0.248 | 0.348 |
| code+math | 1:1 | 0.257 | 0.357 |
| medical+legal | 5:1 | 0.468 | 0.533 |
| code+legal | 5:1 | 0.470 | 0.532 |
| math+legal | 5:1 | 0.473 | 0.551 |
| legal+finance | 4:1 | 0.441 | 0.508 |
| medical+finance | 20:1 | 0.575 | 0.630 |
| code+finance | 20:1 | 0.581 | 0.634 |
| math+finance | 20:1 | 0.580 | 0.652 |

**Critical structural result:** Gini is completely determined by scale ratio:
- **1:1 scale pairs: Gini 0.25** (matches individual adapter Gini -- no between-domain addition)
- **4:1-5:1 scale pairs: Gini 0.44-0.47** (scale imbalance dominates)
- **20:1 scale pairs: Gini 0.58** (worst case, but these pairs are unlikely under routing)

Equal-scale pairs have Gini approximately equal to individual adapter Gini.
The between-domain contribution truly IS eliminated when scales match.

**Comparison across strategies:**

| Strategy | Mean Gini | Max Gini | Reduction vs Uniform |
|----------|-----------|----------|---------------------|
| Uniform N=5 | 0.490 | 0.543 | (baseline) |
| Uniform N=5 + 50% eq | 0.393 | 0.451 | 19.6% |
| Oracle top-2 (mean across domains) | 0.329 | 0.508 | 32.8% |
| Equal-scale top-2 pairs only | 0.252 | 0.357 | 48.5% |

### Phase 2: Perplexity Comparison

| Domain | Base | Single-adapter | Top-2 | Uniform N=5 | Uniform+Eq |
|--------|------|---------------|-------|-------------|------------|
| medical | 6.734 | 3.415 | 4.014 (+17.6%) | 3.851 (+12.8%) | 3.940 (+15.4%) |
| code | 5.693 | 3.513 | 3.618 (+3.0%) | 3.764 (+7.2%) | 3.715 (+5.8%) |
| math | 3.791 | 2.345 | 2.328 (-0.7%) | 2.416 (+3.0%) | 2.528 (+7.8%) |
| legal | 20.979 | 17.260 | 16.745 (-3.0%) | 15.501 (-10.2%) | 14.633 (-15.2%) |
| finance | 18.358 | 17.887 | 15.366 (-14.1%) | 14.081 (-21.3%) | 13.569 (-24.1%) |

Percentages are vs single-adapter oracle.

**Key PPL findings:**
- Top-2 is BEST for **math** (only -0.7% vs single) and **code** (+3.0%)
- Top-2 HURTS **medical** (+17.6% vs single) -- adding math adapter to medical is counterproductive at high scale
- Uniform composition is BETTER for **legal** and **finance** -- these low-scale domains benefit from the implicit regularization of averaging many high-scale adapters
- The uniform vs top-2 tradeoff is domain-dependent, not universally one-directional

### Phase 3: Behavioral Evaluation (math, code, medical)

| Domain | Metric | Single-adapter | Top-2 | Degradation |
|--------|--------|---------------|-------|-------------|
| math | Answer correctness | 70% (7/10) | 70% (7/10) | 0% |
| code | Syntax validity | 90% (9/10) | 50% (5/10) | 40.7% |
| medical | Factual recall | 28.7% | 27.6% | 3.8% |

**Code degradation is severe (40.7%)** despite modest PPL degradation (+3.0%).
Adding the math adapter to the code adapter makes generated code less syntactically
valid, even though math reasoning is related. This is a behavioral outcome that
PPL completely fails to predict.

## Limitations

1. **Oracle routing only.** Production routing could select worse pairs (e.g.,
   medical+finance at 20:1 scale ratio). Softmax router quality determines actual Gini.
2. **10 prompts per domain** for behavioral eval. Low statistical power.
3. **Oracle top-2 selection was heuristic.** We selected domain-similar pairs, but
   the actual optimal second adapter is unknown. Medical+math may be a bad choice
   (17.6% PPL degradation suggests it is).
4. **Single seed.** No variance estimates.
5. **Gini threshold 0.15 was wrong.** Should have been calibrated to individual
   adapter Gini (~0.25) rather than assumed from theoretical uniformity.

## What Would Kill This

At micro scale: K712 already killed the strict Gini prediction. The experiment
is PARTIALLY supported because K713 and K714 pass.

At macro scale: If a learned softmax router selects scale-mismatched pairs
frequently (e.g., >30% of tokens get 20:1 pairs), the spectral pathology would
persist under routing. The Finding #28 result (softmax matches oracle) suggests
this won't happen, but it needs verification.

## The Real Finding: Spectral Gini Is The Wrong Metric

The most important result is the **disconnect between spectral metrics and
practical quality**:

1. Equal-scale top-2 pairs achieve Gini 0.25 (same as individual adapters) yet
   PPL degrades by up to 17.6% (medical). Spectral health does not guarantee quality.

2. Uniform N=5 has the WORST Gini (0.49) but gives the BEST PPL for legal (-10.2%
   vs single) and finance (-21.3% vs single). Bad Gini can mean BETTER quality.

3. Code shows 40.7% behavioral degradation at only +3.0% PPL change and Gini 0.257.
   Neither PPL nor Gini predicted this behavioral failure.

**Conclusion: The entire spectral Gini metric track (Findings #277-282) was
optimizing a proxy that does not predict the behavioral outcomes that matter.**
This is consistent with Finding #238 (metric-behavioral gap) and the project's
r=0.08 PPL-quality correlation finding.

The correct intervention is not routing OR spectral optimization -- it is
**direct behavioral evaluation** at composition time. Spectral Gini, PPL delta,
and even MMLU are all unreliable proxies for what users actually care about.

## Spectral Arc Resolution

| Finding | Method | Gini Effect | PPL Effect | Behavioral Effect | Verdict |
|---------|--------|-------------|------------|-------------------|---------|
| #277 (DC-Merge) | Within-domain smoothing | -18.5% | +0.99% | Untested | Wrong variable |
| #278 (Surgery) | Post-composition SVD | Inverted | Degraded | Untested | Structurally wrong |
| #279 (Frobenius eq) | Cross-domain equalization | -45.6% | Mixed | Untested | Practical ceiling |
| #281 (Fisher) | Fisher weighting | = Frobenius | = Frobenius | Untested | Reduces to #279 |
| #282 (Norm-bounded) | Training-time constraints | Worse | Worse | Untested | Worse than post-hoc |
| **This (routing)** | **Top-2 routing** | **-32.8%** | **4/5 pass** | **2/3 pass** | **Partially supported** |

**Routing reduces Gini by 32.8% (vs 19.6% for equalization) and passes PPL/behavioral
criteria, but does NOT eliminate spectral pathology. More importantly, spectral
pathology does not predict practical quality -- the entire 6-experiment spectral
track optimized the wrong thing.**
