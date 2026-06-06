# Persistence-Guided Bridge Extraction: Sparse Low-Rank Correction

## STATUS: KILLED (K628 FAIL, K629 FAIL, K630 FAIL)

The bridge matrix reduces H1 bottleneck distance by up to 71.8% in individual
modules but falls short on all three kill criteria. More critically, bridge
correction HURTS PPL (-1.0% average), revealing a fundamental flaw in the
approach: the perturbation's top SVD components carry useful adapter learning
signal, not topological damage. Removing them degrades the model.

## Theorem

**Theorem 1 (Bridge Restoration Bound).** For weight matrix W with point cloud P,
composed W' = W + Delta, and bridge-corrected W'' = W' + R_k (where R_k is the
rank-k truncated SVD of -Delta):

  d_B(Dgm(P), Dgm(P'')) <= max_i ||(R - R_k)_i||_2

where (R - R_k)_i is the i-th row of the residual.

**Theorem is correct but the approach is misguided.** The bridge correction is
mathematically optimal for minimizing topological distortion, but topological
distortion is NOT damage -- it is useful adaptation.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? | Information Content |
|------------------------|----------|--------|---------------------|
| P1: H1 features lost in >= 10/35 modules | 0/35 modules with net H1 loss | **NO** | H1 features are not destroyed by composition at current scale, despite being in the vulnerability window. |
| P2: Bridge at rank 16 reduces d_B by >= 50% (K628) | Mean reduction = 47.9%, median = 52.2% | **BORDERLINE FAIL** | Mean misses by 2.1pp. Median passes. 10/19 modules individually pass. |
| P3: Bridge rank < 16 suffices (K629) | No rank < 16 achieves >= 50% mean reduction | **FAIL** | Rank-8 achieves 37.2% mean, rank-2 only 19.9%. Bridge is NOT cheap. |
| P4: PPL improvement >= 5% (K630) | PPL worsens by 1.0% average | **FAIL** | Bridge correction HURTS quality. Every domain gets worse. |

## Hypothesis

High-persistence H1 features destroyed by adapter composition can be restored via
a sparse low-rank bridge matrix targeting the perturbation's top singular vectors.

**Verdict: KILLED.** The hypothesis fails on multiple levels:
1. H1 features are NOT destroyed (0 net loss across 35 modules)
2. Even reducing bottleneck distance does NOT help -- it hurts PPL
3. The perturbation's SVD captures adapter learning, not topological damage

## What This Experiment Is

**Weight-space topological bridge extraction:** For each weight matrix module,
compute the composed perturbation Delta = (scale/N) * sum(A_i @ B_i), take its
truncated SVD at various ranks, construct a bridge matrix B = -Delta_k that
partially reverses the perturbation, and measure whether this restores the
persistence diagram and improves PPL.

35 modules analyzed across 5 layers (0, 7, 15, 22, 29) and 7 projection types.
300 rows subsampled per module. Bridge correction tested at ranks 2, 8, 16.
PPL evaluated with rank-8 bridge on all 30 layers.

## Key References

- Cohen-Steiner, Edelsbrunner, Harer (2007): Algebraic Stability Theorem
- Eckart-Young-Mirsky (1936): Optimal low-rank approximation
- Williams & Beer (2010), arXiv:2411.07483: PID for knowledge decomposition
- Garin & Tauzin (2020), arXiv:2312.10702: PH for model compression

## Empirical Results

### H1 Vulnerability Analysis

| Metric | Value |
|--------|-------|
| Modules analyzed | 35 |
| Modules with H1 vulnerability (vuln > median_h1) | 19/35 (54%) |
| Modules with net H1 feature loss | 0/35 (0%) |
| Total H1 features (base) | 11,962 |
| Total H1 features lost | 0 |

**Critical finding:** Despite 19 modules being theoretically H1-vulnerable
(vulnerability bound exceeds median H1 persistence), ZERO net H1 features are
destroyed. The stability theorem's vulnerability window is a sufficient but not
necessary condition for feature loss -- features CAN survive even when the bound
says they COULD be destroyed.

### Bridge Restoration by Rank

| Bridge Rank | Mean d_B Reduction | Median d_B Reduction | Max d_B Reduction | Modules >= 50% |
|-------------|-------------------|---------------------|-------------------|----------------|
| 2 | 19.9% | 22.7% | 48.3% | 0/19 |
| 8 | 37.2% | 36.8% | 60.8% | 5/19 |
| 16 | 47.9% | 52.2% | 71.8% | 10/19 |

Rank-16 nearly meets the 50% mean threshold (47.9% vs 50%). The relationship
between rank and d_B reduction is sublinear -- doubling rank from 8 to 16
only improves mean reduction from 37.2% to 47.9% (+10.7pp).

### SVD Energy Analysis

Mean rank to capture 90% of perturbation energy: 21.0. This means the adapter
perturbation is NOT low-rank -- it requires ~21 dimensions to capture 90% of
variance, exceeding the per-adapter rank of 16. The 5 combined adapters create
a perturbation that fills more dimensions than any individual adapter.

### PPL Impact

| Domain | Composed PPL | Bridge PPL (rank-8) | Change |
|--------|-------------|--------------------:|-------:|
| Medical | 3.94 | 3.95 | -0.4% |
| Code | 4.20 | 4.21 | -0.1% |
| Math | 2.83 | 2.85 | -0.7% |
| Legal | 15.09 | 15.45 | -2.4% |
| Finance | 14.24 | 14.48 | -1.7% |
| **Mean** | **8.06** | **8.19** | **-1.0%** |

Bridge correction worsens PPL across ALL 5 domains. Legal and finance show
the largest degradation (-2.4%, -1.7%), consistent with these being the domains
where adapters provide the most benefit (legal and finance have the highest
composed PPL, indicating weakest adaptation).

### Best-Performing Modules

The bridge works best on:
- layer_22 self_attn.o_proj: 71.8% d_B reduction at rank-16
- layer_22 self_attn.q_proj: 60.8% reduction at rank-8
- layer_15 self_attn.q_proj: 59.2% reduction at rank-16

These are attention output/query projections in middle-late layers.

## Kill Criteria Assessment

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K628: d_B reduction >= 50% | **FAIL** | Mean = 47.9% at rank-16 (2.1pp below threshold) |
| K629: Bridge rank < r=16 | **FAIL** | No rank < 16 achieves >= 50% mean reduction |
| K630: PPL improvement >= 5% | **FAIL** | PPL worsens by 1.0% on average |

## Key Findings

### Finding 1: H1 Features Survive Despite Being in the Vulnerability Window
Zero net H1 features lost across all 35 modules, even though 19 modules have
vulnerability bounds exceeding median H1 persistence. The stability theorem's
vulnerability window is conservative -- it overpredicts feature loss. At current
adapter scale (rank-16, 5 domains, scale=20), composition preserves ALL
topological features in BOTH H0 and H1.

### Finding 2: Topological Restoration Hurts Model Quality
The bridge matrix reduces bottleneck distance but INCREASES PPL. This is the
critical insight: the perturbation's top singular vectors are not "topological
damage" -- they are the useful adapter learning signal. Partially undoing the
perturbation removes useful adaptation.

This reveals a fundamental mismatch between topological preservation and model
quality. Good adaptation SHOULD change the topology (creating new features,
restructuring weight geometry). Trying to preserve the original topology works
against the purpose of adaptation.

### Finding 3: Adapter Perturbation is Not Low-Rank
Mean rank for 90% energy capture is 21 -- exceeding per-adapter rank of 16.
The composed perturbation of 5 rank-16 adapters spans ~21 effective dimensions,
meaning the adapters are NOT operating in redundant subspaces. Each adapter
contributes genuinely different directions.

## Implications for the Architecture

1. **Bridge matrices are counterproductive.** The entire "bridge extraction"
   approach is misguided at current scale. There is no topological damage to
   repair, and attempting repair degrades quality.

2. **The pathway preservation line of research should be closed.** Between
   exp_persistence_diagram_diff (no H0 features lost) and this experiment
   (no H1 features lost, bridges hurt quality), the conclusion is clear:
   composition at rank-16 with 5 domains preserves all topological structure.

3. **Scaling threshold remains unknown.** At what adapter rank, domain count,
   or scale does topological damage become real? The current experiments cannot
   answer this because no damage occurs at the tested scale.

4. **Feature CREATION, not preservation, is the topological signature of
   composition.** The dependency experiment showed +242 H0 and +401 H1 features
   created. This experiment confirms the created features are part of useful
   adaptation, not artifacts. Preserving original topology (via bridges) removes
   these useful new features.

## Limitations

1. **300-row subsample.** Reduced from 500 in the dependency experiment for speed.
   May miss some H1 features, though the direction of results is clear.
2. **Bridge correction applied to all 30 layers for PPL.** The topological
   analysis covers only 5 layers, but PPL evaluation needed consistent treatment.
3. **Only rank-8 bridge tested for PPL.** Different ranks might produce different
   PPL outcomes, though the trend is clear (higher rank = more undoing = worse PPL).
4. **Single model (BitNet-2B-4T).** Results may differ for other architectures.

## What Would Kill This (What Already Killed It)

- K630 FAIL is definitive: bridge correction hurts quality. Even if K628 marginally
  passed (median = 52.2%), the PPL degradation means topological restoration is
  counterproductive.
- The fundamental assumption (that topological change from composition is "damage")
  is falsified. Topological change is the mechanism of useful adaptation.

## Runtime

Total: 517s (~8.6 min) on Apple M5 Pro 48GB. Breakdown:
- Weight loading: ~1s
- PH computation + bridge analysis (35 modules): ~6 min
- PPL evaluation (2 model loads, 5 domains each): ~2 min
