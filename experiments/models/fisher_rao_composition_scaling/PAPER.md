# Fisher-Rao Manifold Composition: Proof Verification Report (Revised)

## Theorem

**Theorem 1 (Norm Preservation).** Fisher-Rao spherical proxy merge (Karcher mean on S^(d-1) + mean norm rescaling) preserves ||merged|| / mean(||source_i||) = 1.0 exactly, while Euclidean averaging shrinks norms as 1/sqrt(N) for orthogonal adapter deltas.

**Conjectures 2-3** (downgraded from Theorems): Activation variance and effective rank predictions from the original linear response model were wrong in direction. See "Honest Assessment" below.

## Predictions vs Measurements

### Theorem 1 (proven): Norm Shrinkage

| Prediction (from proof) | Measured | Match? |
|-------------------------|----------|--------|
| FR norm shrinkage = 1.0 at all N | 1.0000 at N=1,3,5,10,15 | YES (tautological by construction) |
| NRE norm shrinkage = 1.0 at all N | 1.0000 at N=1,3,5,10,15 | YES (by construction) |
| Euc norm shrinkage = 1/sqrt(3) = 0.577 at N=3 | 0.5750 | YES |
| Euc norm shrinkage = 1/sqrt(5) = 0.447 at N=5 | 0.4482 | YES |
| Euc shrinkage continues to 1/sqrt(10) at N=10 | 0.4476 (plateau at ~0.447) | NO -- synthetic adapters are correlated |
| Norm-preserved PPL < raw Euclidean PPL at N>=3 | FR 9.00 vs Euc 9.60 at N=3 | YES |
| Norm-preserved PPL < raw Euclidean PPL at N=5 | FR 9.20, NRE 9.17 vs Euc 10.44 | YES |

### Original Conjectures 2-3 (unproven, predictions wrong in direction)

| Prediction (from proof sketch) | Measured | Match? |
|-------------------------------|----------|--------|
| Euc act. var. should DECREASE with N (scale as 1/N) | INCREASES: 0.000257 -> 0.000296 | **WRONG SIGN** |
| FR act. var. should be stable (~1.0 ratio) | Ratio = 1.074 (within 10%) | DIRECTION OK, wrong mechanism |
| Eff. rank should DEGRADE with N | IMPROVES: 4.55 -> 5.11 (FR) | **WRONG SIGN** |

### Critical New Finding: Norm-Rescaled Euclidean Matches Fisher-Rao

| N | Euclidean PPL | Norm-Rescaled Euc PPL | Fisher-Rao PPL | NRE vs FR |
|---|--------------|----------------------|----------------|-----------|
| 1 | 8.98 | 8.98 | 8.97 | identical |
| 3 | 9.60 | 9.00 | 9.00 | identical |
| 5 | 10.44 | 9.17 | 9.20 | NRE slightly better |
| 10 | 10.44 | 9.17 | 9.19 | NRE slightly better |
| 15 | 10.45 | 9.17 | 9.20 | NRE slightly better |

**The Karcher mean provides no measurable benefit over simple norm rescaling.** The entire PPL advantage of Fisher-Rao over Euclidean comes from preserving norms, not from superior directional averaging.

## Hypothesis

Norm preservation is the single mechanism that prevents composition degradation at scale. The Riemannian manifold structure (Karcher mean) is mathematically elegant but unnecessary in practice -- a one-line norm rescaling after Euclidean averaging achieves identical results.

## What This Model Is

A comparison of three composition methods for adapter B-matrices:

1. **Euclidean averaging:** B_merged = (1/N) sum B_i. Known to shrink norms as 1/sqrt(N).
2. **Norm-rescaled Euclidean:** Same as above, then rescale result to have mean source norm. One line of code.
3. **Fisher-Rao Karcher mean:** Normalize to unit sphere, compute Riemannian mean via iterative optimization, rescale by mean source norm.

Methods 2 and 3 produce equivalent results on all metrics. Method 1 degrades progressively with N.

## Key References

- arXiv:2603.04972 -- "Fisher-Rao Manifold Merging" (Wang, Ye, Yin 2025). Source of the Karcher mean approach.
- Karcher (1977) -- Existence and uniqueness of Frechet mean on complete Riemannian manifolds.
- Jang et al. (2024) -- Norm shrinkage analysis of Euclidean model averaging.

## Empirical Results

### Norm Shrinkage (Theorem 1 -- verified exactly)

| N | Euclidean | Norm-Rescaled Euc | Fisher-Rao | Theory (Euc) |
|---|-----------|-------------------|------------|--------------|
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 0.5750 | 1.0000 | 1.0000 | 0.5774 |
| 5 | 0.4482 | 1.0000 | 1.0000 | 0.4472 |
| 10 | 0.4476 | 1.0000 | 1.0000 | 0.3162* |
| 15 | 0.4472 | 1.0000 | 1.0000 | 0.2582* |

*N=10,15 use synthetic adapters (noisy copies of 5 real). Euclidean shrinkage plateaus at 1/sqrt(5) = 0.447, confirming only 5 independent directions exist.

### Perplexity

| N | Euclidean | Norm-Rescaled Euc | Fisher-Rao | NRE advantage vs Euc |
|---|-----------|-------------------|------------|---------------------|
| 1 | 8.98 | 8.98 | 8.97 | 0% |
| 3 | 9.60 | 9.00 | 9.00 | -6.3% |
| 5 | 10.44 | 9.17 | 9.20 | -12.2% |
| 10 | 10.44 | 9.17 | 9.19 | -12.2% |
| 15 | 10.45 | 9.17 | 9.20 | -12.2% |

Both norm-preserving methods produce nearly identical PPL. The ~12% advantage over raw Euclidean at N=5 comes entirely from norm preservation.

### Activation Variance (fixed N=1 scale confound)

All N values use scale=12.8 (mean of optimal scales). N=1 uses only the medical adapter B-matrix but at the same scale as N>1 for fair comparison.

| N | Euclidean | Norm-Rescaled Euc | Fisher-Rao | FR ratio vs N=1 |
|---|-----------|-------------------|------------|-----------------|
| 1 | 0.000257 | 0.000257 | 0.000257 | 1.000 |
| 3 | 0.000284 | 0.000273 | 0.000273 | 1.063 |
| 5 | 0.000296 | 0.000277 | 0.000277 | 1.078 |
| 10 | 0.000296 | 0.000277 | 0.000277 | 1.074 |
| 15 | 0.000296 | 0.000277 | 0.000277 | 1.074 |

K690 PASSES with the scale confound fixed: FR ratio 1.074, within 10% threshold. Note that variance INCREASES with N (contradicting the original Theorem 2 prediction of decrease). This is because multi-domain composition introduces diverse hidden state trajectories.

### Effective Rank

| N | Euclidean | Norm-Rescaled Euc | Fisher-Rao |
|---|-----------|-------------------|------------|
| 1 | 4.55 | 4.55 | 4.55 |
| 3 | 5.23 | 4.91 | 4.91 |
| 5 | 5.48 | 5.11 | 5.12 |
| 10 | 5.47 | 5.10 | 5.11 |
| 15 | 5.48 | 5.11 | 5.11 |

K691 PASSES: effective rank improves (not degrades) with N. Multi-domain composition diversifies the activation space. Both norm-preserving methods produce identical effective rank.

### Computational Cost

| N | Euclidean (s) | Norm-Rescaled Euc (s) | Fisher-Rao (s) |
|---|---------------|----------------------|----------------|
| 1 | 0.04 | 0.24 | 0.12 |
| 5 | 0.08 | 0.13 | 2.01 |
| 10 | 0.02 | 0.17 | 2.04 |
| 15 | 0.02 | 0.24 | 2.68 |

Norm-rescaled Euclidean is ~10x faster than Fisher-Rao while producing equivalent or slightly better results.

## Kill Criteria Results

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K690: Act. var. within 10% at N=10 vs N=1 | **PASS** | FR ratio 1.074 (7.4% above N=1). Fixed by using consistent scale=12.8 for all N. |
| K691: Eff. rank degradation < 5% | **PASS** | No degradation; rank improves by 12.5% from N=1 to N=10 (FR). |
| K692: FR outperforms Euclidean at N>5 | **PASS** | FR PPL 9.19 vs Euc 10.44 at N=10 (-12.0%). |
| K692b: FR outperforms Norm-Rescaled Euc | **FAIL** | FR PPL 9.19 vs NRE 9.17. NRE is slightly better AND 10x faster. |

**Overall: SUPPORTED with major caveat.** Theorem 1 (norm preservation) is verified exactly. But the Karcher mean adds no value beyond what a trivial norm rescaling provides. The practical recommendation is to use norm-rescaled Euclidean averaging.

## Honest Assessment of Original Predictions

### What the original experiment got right:
- Theorem 1 (norm shrinkage ratio) is exactly verified for N=1 through 5
- Norm preservation does improve PPL vs raw Euclidean averaging
- K692 passes: Fisher-Rao outperforms Euclidean

### What the original experiment got wrong:
1. **Theorem 2 predicted Euclidean variance would DECREASE.** It INCREASED. The linear response model predicts the wrong sign because it ignores the domain-diversity effect of multi-adapter composition.
2. **Theorem 3 predicted effective rank would DEGRADE.** It IMPROVED. Multi-domain composition diversifies activations.
3. **Claims of "scaling to N=15" were not supported.** All metrics plateau at N=5 because synthetic adapters for N>5 are noisy copies of the same 5 real adapters.
4. **The N=1 reference was confounded.** Original experiment used scale=20.0 for N=1 (medical only) vs scale=12.8 for N>1, conflating scale and composition effects. Fixed in this revision.
5. **Fisher-Rao's advantage is entirely from norm preservation.** A one-line norm rescaling achieves the same result 10x faster.

## Limitations

1. **N<=5 effective ceiling.** Only 5 real domain adapters exist. N=10,15 use synthetic variants that are correlated with the 5 real ones. All metrics plateau at N=5. Scaling claims are restricted to N=5.

2. **B-matrix level merging.** Fisher-Rao is applied to rank-16 B-matrix vectors, not full weight deltas. This is justified (A matrices are frozen) but may miss full-delta interactions.

3. **Single A-matrix slot.** Multi-domain composition uses a single A matrix from domain 0's skeleton. Production would use per-domain A matrices with routing.

## What Would Kill This

1. **Norm-rescaled Euclidean significantly outperforming FR at scale.** Would confirm FR adds no directional value and is pure overhead.

2. **At scale with N>5 truly independent adapters, norm preservation showing no benefit.** Would indicate the mechanism is specific to our correlated adapter regime.

3. **Discovery that moderate norm shrinkage is actually beneficial (implicit regularization).** Would invalidate the premise that norm preservation is desirable.

## Practical Recommendation

**Use norm-rescaled Euclidean averaging** for adapter composition:
```python
result = euclidean_mean * (mean_source_norm / euclidean_mean_norm)
```
This is one line of code, produces equivalent results to Fisher-Rao Karcher mean, and is 10x faster. The Riemannian manifold machinery is mathematically correct but practically unnecessary at this scale.
