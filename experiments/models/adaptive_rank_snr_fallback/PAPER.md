# SNR-Aware Rank Predictor with Fallback: Research Digest

## Hypothesis

A compound heuristic that uses energy_rank_99 by default but falls back to
energy_rank_95 when r_99/r_95 > 2.0 (indicating noise-dominated spectra)
will beat r_99 at low SNR without regressing at high SNR.

## What This Model Is

An extension of the adaptive rank selection experiment that fixes the one
failure mode identified by the adversarial review: energy_rank_99 systematically
overpredicts at low SNR because noise inflates the spectral tail.

The compound heuristic uses the ratio R = r_99/r_95 as a noise diagnostic.
When R > 2.0, the energy gap between 95% and 99% is spread across many noise
dimensions, and r_95 (which is robust to noise) is substituted as the predictor.
When R <= 2.0, the standard r_99 predictor is used.

This produces a single universal rank predictor that works across all tested
SNR levels (5 to 100), all tested dimensions (64 to 256), and both domain
types (exact-rank and spectral-decay).

## Lineage in the Arena

```
exp_adaptive_rank_selection (proven, v2)
  |
  +-> exp_adaptive_rank_snr_fallback (this experiment)
```

## Key References

- **Parent experiment** (exp_adaptive_rank_selection): Proved energy_rank_99
  correlates with Kneedle-optimal rank at rho=0.94-0.99. Adversarial review
  identified SNR=5 failure.
- **Weyl's perturbation theorem**: Bounds how noise shifts singular values,
  explaining why r_99 inflates under noise.
- **Satopaa et al. (2011)**: Kneedle algorithm for optimal rank ground truth.

## Empirical Results

### Kill Criteria

| Criterion | Verdict | Evidence |
|-----------|---------|----------|
| K1: improvement over r_99 at SNR<=10 | **PASS** | Mean +23.3pp, max +60.0pp at SNR=5 |
| K2: no condition worse than null | **PASS** | 0/12 conditions below null, min margin +13.3pp |

**Overall: PROVEN**

### Within-2x Accuracy Across All 12 Conditions

| Condition | null_16 | r_99 | r_95 | compound_r95_t2.0 |
|-----------|---------|------|------|-------------------|
| d=64, SNR=5 | 66.7% | 53.3% | 80.0% | **80.0%** |
| d=64, SNR=10 | 60.0% | 100.0% | 93.3% | **100.0%** |
| d=64, SNR=20 | 60.0% | 100.0% | 86.7% | **100.0%** |
| d=64, SNR=100 | 60.0% | 100.0% | 86.7% | **100.0%** |
| d=128, SNR=5 | 53.3% | 33.3% | 86.7% | **86.7%** |
| d=128, SNR=10 | 53.3% | 100.0% | 80.0% | **100.0%** |
| d=128, SNR=20 | 53.3% | 100.0% | 80.0% | **100.0%** |
| d=128, SNR=100 | 53.3% | 93.3% | 73.3% | **93.3%** |
| d=256, SNR=5 | 60.0% | 26.7% | 86.7% | **86.7%** |
| d=256, SNR=10 | 53.3% | 100.0% | 80.0% | **100.0%** |
| d=256, SNR=20 | 53.3% | 100.0% | 80.0% | **100.0%** |
| d=256, SNR=100 | 60.0% | 93.3% | 73.3% | **93.3%** |

**Mean across all conditions: 95.0% (compound) vs 83.3% (r_99) vs 82.2% (r_95) vs 57.2% (null)**

### Key Finding: The Fallback Never Hurts

The compound heuristic at T=2.0 achieves the best or tied-best accuracy in
every single condition. It never regresses below r_99 at any SNR level:

- **SNR=5**: compound matches r_95 (+26.7 to +60.0pp vs r_99)
- **SNR=10**: compound matches r_99 exactly (0.0pp delta, fallback never triggers)
- **SNR>=20**: compound matches r_99 exactly (0.0pp delta)

This is because the r_99/r_95 ratio cleanly separates the two regimes:
- SNR=5: 53-93% of domains have R > 2.0 (fallback triggers)
- SNR>=10: 0% of domains have R > 2.0 (fallback never triggers)

### Threshold Sensitivity

| Threshold T | Mean within_2x | Notes |
|-------------|---------------|-------|
| 1.5 | 91.1% | Too aggressive -- triggers at SNR=10, regresses |
| **2.0** | **95.0%** | Optimal -- separates SNR=5 from SNR>=10 cleanly |
| 2.5 | 95.0% | Identical to T=2.0 (wide gap between regimes) |
| 3.0 | 93.9% | Slight degradation at d=64 SNR=5 |

T=2.0 and T=2.5 are both optimal. The wide gap between R at SNR=5 (mean 3.8-13.0)
and R at SNR=10 (mean 1.3-1.5) means any threshold in [2.0, 2.5] works equally well.

### Alternative Fallback: Effective Rank

compound_eff_t2.0 (falling back to effective_rank instead of r_95) achieves
87.8% mean accuracy -- worse than compound_r95 (95.0%). The effective rank is
a continuous entropy-based measure that is less calibrated as a direct rank
predictor. r_95 is the better fallback because it is also an energy threshold,
just more conservative.

### Best Predictor Ranking (All 12 Conditions)

| Predictor | Mean within_2x |
|-----------|---------------|
| **compound_r95_t2.0** | **95.0%** |
| compound_r95_t2.5 | 95.0% |
| compound_r95_t3.0 | 93.9% |
| compound_eff_t3.0 | 93.3% |
| compound_eff_t2.5 | 92.8% |
| compound_r95_t1.5 | 91.1% |
| compound_eff_t2.0 | 87.8% |
| r_99 (unconditional) | 83.3% |
| r_95 (unconditional) | 82.2% |
| effective_rank | 75.0% |
| null (rank 16) | 57.2% |

## Practical Heuristic for SOLE (Updated)

The recommended automatic rank selection is now:

1. Train a pilot LoRA adapter at rank r_init (e.g., 16)
2. Compute SVD of the learned delta: Delta = A @ B
3. Measure r_95 and r_99 from the singular values
4. **Decision rule:**
   - If r_99 / r_95 <= 2.0: use r_99 (normal case)
   - If r_99 / r_95 > 2.0: use r_95 (noise-dominated case)
5. Set production rank = snap_to_nearest_available(r_compound)

This replaces the parent's unconditional r_99 recommendation. The overhead
is one additional cumulative sum scan (O(d)), which is negligible.

## Micro-Scale Limitations

1. **All limitations from parent experiment apply.** Simulated, not trained.
   Single-matrix domains. Synthetic spectral structure.

2. **SNR mapping to real training is unknown.** The SNR=5 regime where the
   fallback helps corresponds to very noisy fine-tuning (signal is only 5x
   stronger than noise). Whether real LoRA training produces such low
   effective SNR depends on learning rate, data quality, and training duration.
   If real training always produces SNR >= 10, the fallback is unnecessary
   (though harmless).

3. **The threshold T=2.0 is validated at 3 dimensions and 4 SNR levels.**
   The clean separation between regimes suggests robustness, but the
   threshold may need adjustment for d >> 256 or for real LoRA spectra
   that do not follow exact-rank or geometric-decay models.

4. **No test of adaptive threshold.** T=2.0 is fixed. A data-driven threshold
   (e.g., based on the spectral gap distribution) might be more robust but
   adds complexity.

## What Would Kill This

**At micro scale (tested and survived):**
- K1 KILLED if compound does not improve over r_99 at SNR<=10: DID NOT HAPPEN.
  Mean +23.3pp improvement.
- K2 KILLED if any condition where compound is worse than null: DID NOT HAPPEN.
  Zero failures, minimum margin +13.3pp.

**At macro scale (needs validation):**
- Real LoRA training always has effective SNR >= 10, making the fallback
  unnecessary (but harmless). The heuristic would be correct but vacuous.
- Real LoRA spectra have r_99/r_95 ratios in the 1.5-2.0 transition zone
  even at high SNR, causing false fallback triggers. This would cause
  underprediction at high SNR.
- Per-layer variation in effective SNR means a single per-domain threshold
  is insufficient -- some layers may be noise-dominated while others are clean.

## Configuration

- Dimensions: d in {64, 128, 256}
- SNR values: {5, 10, 20, 100}
- Ratio thresholds swept: {1.5, 2.0, 2.5, 3.0}
- Seeds per domain: 5
- Domain types: 8 exact-rank + 7 spectral-decay = 15 per condition
  (adjusted for d: fewer exact-rank domains at d=64)
- Total conditions: 12 (3d x 4 SNR)
- Null baseline: always predict rank 16
- Runtime: 6.3 seconds (all 12 conditions)
- Architecture: Pure numpy/scipy, CPU-only
