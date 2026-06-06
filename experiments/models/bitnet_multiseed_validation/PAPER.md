# BitNet-2B Multi-Seed Validation: Research Digest

## Hypothesis

Ternary QAT+STE LoRA composition on BitNet-b1.58-2B-4T is reproducible across
random seeds: composition ratio CV < 50% and no seed shows catastrophic
composition (ratio > 10x).

## What This Experiment Is

A reproducibility study that trains 5 domain-specific ternary LoRA adapters
(QAT with Straight-Through Estimator) on BitNet-b1.58-2B-4T across 3 seeds
(42, 137, 314), then composes them via averaged-factor composition and measures variance of
all key metrics. This resolves the single-seed limitation of
exp_bitnet_ternary_convergence.

## Key References

- Prior single-seed result: micro/models/bitnet_ternary_convergence/ (ratio 3.45x, |cos| 0.0019)
- BitNet-2B proven pipeline: micro/models/bitnet_2b_real_composition/
- MoTE (2506.14435): frozen shared base + ternary routed experts
- LoTA-QAF: lossless ternary merge on integer grid

## Empirical Results

### Kill Criteria

| Criterion | Threshold | Measured | Verdict |
|-----------|-----------|----------|---------|
| K1: CV(composition ratio) | < 50% | **0.5%** | **PASS** (100x margin) |
| K2: max ratio across seeds | < 10x | **3.45x** | **PASS** (2.9x margin) |

### Per-Seed Composition Ratio

| Seed | Composition Ratio | Mean |cos| | Best Individual PPL |
|------|-------------------|-----------|---------------------|
| 42 | 3.454x | 0.00202 | 2.966 |
| 137 | 3.442x | 0.00184 | 2.967 |
| 314 | 3.423x | 0.00230 | 2.966 |
| **Mean +/- Std** | **3.440 +/- 0.016** | **0.00205 +/- 0.00023** | **2.966 +/- 0.001** |
| **CV** | **0.5%** | **11.3%** | **0.02%** |

### Per-Domain Individual PPL (Mean +/- Std across 3 seeds)

| Domain | Base PPL | Individual PPL | Improvement | Std |
|--------|----------|---------------|-------------|-----|
| Medical | 18.98 | 9.085 +/- 0.017 | +52.1% | 0.19% |
| Code | 3.78 | 2.966 +/- 0.001 | +21.5% | 0.03% |
| Math | 4.54 | 3.079 +/- 0.002 | +32.2% | 0.08% |
| Legal | 26.93 | 18.911 +/- 0.029 | +29.8% | 0.16% |
| Creative | 3.51 | 3.121 +/- 0.002 | +11.2% | 0.07% |

### Per-Domain Composed PPL (Averaged-Factor, Mean +/- Std across 3 seeds)

| Domain | Composed PPL | Std | CV |
|--------|-------------|-----|-----|
| Medical | 15.433 +/- 0.094 | 0.61% | 0.61% |
| Code | 3.450 +/- 0.012 | 0.36% | 0.36% |
| Math | 4.155 +/- 0.023 | 0.54% | 0.54% |
| Legal | 24.690 +/- 0.088 | 0.36% | 0.36% |
| Creative | 3.289 +/- 0.017 | 0.52% | 0.52% |

### Training Convergence (consistent across seeds)

| Domain | Converged (all 3 seeds) | Loss Pattern |
|--------|------------------------|--------------|
| Medical | Yes (3/3) | 2.93 -> 1.71 |
| Code | No (0/3) | 1.07 -> 1.12 (train loss increases, but val PPL improves) |
| Math | Yes (3/3) | 1.35 -> 1.27 |
| Legal | Yes (3/3) | 3.15 -> 2.89 |
| Creative | No (0/3) | 1.24 -> 1.64 (train loss increases, but val PPL improves) |

Note: "code" and "creative" domains show increasing train loss with batch_size=1
(noisy gradient), but validation PPL consistently improves (2.97 and 3.12 vs base
3.78 and 3.51). The convergence metric (last_50 < first_50 * 0.95) is overly
strict for STE training with batch_size=1.

## Interpretation

The CV of 0.5% for composition ratio is striking -- it means the ternary STE
training process is essentially deterministic in its composition behavior despite
different random initializations. This is likely because:

1. **Ternary quantization constrains the solution space.** The STE forces weights
   to {-1, 0, 1} * alpha, so different random starts converge to similar ternary
   configurations.

2. **High dimensionality guarantees orthogonality.** At d=2560 with r=16, the
   structural bound sqrt(r/d) = 0.079 provides enormous headroom. Measured
   |cos| ~ 0.002 is 40x below this bound.

3. **Averaged-factor composition is stable.** The composition merges A and B
   factor matrices separately with 1/N scaling each, yielding an effective
   1/N^2 scaling on each adapter's diagonal contribution plus small cross-terms
   (small because adapters are nearly orthogonal). The composition ratio of
   ~3.4x reflects this 1/N^2 dilution and is stable because orthogonality
   keeps cross-terms negligible. This is the same composition method used in
   all prior experiments (bitnet_2b_real_composition, bitnet_ternary_convergence),
   so all results are self-consistent.

The per-domain PPL CV is even lower (0.02-0.61%), confirming that both individual
adapter quality and composed quality are reproducible.

## Comparison with Prior Single-Seed Result

| Metric | Prior (seed 42, ternary_convergence) | This (seed 42) | Match? |
|--------|--------------------------------------|----------------|--------|
| Composition ratio | 3.447 | 3.454 | Yes (0.2% diff) |
| Mean |cos| | 0.00186 | 0.00202 | Yes (8.6% diff, within noise) |
| Medical individual PPL | 9.041 | 9.075 | Yes (0.4% diff) |
| Code individual PPL | 2.763 | 2.966 | No (7.3% diff) |

The code domain shows a 7.3% difference between this run and the prior
ternary_convergence result. This is explained by different LoRA initialization
schemes: this experiment uses a domain-dependent seeding formula
`seed * 1000 + hash(domain_name) % 10000` for each adapter's LoRA initialization
(see run_experiment.py line 526), while the prior ternary_convergence experiment
used a different seeding scheme. So "seed 42" produces different LoRA weight
initializations in the two experiments, and the code domain (with its small
absolute PPL range) is most sensitive to this difference. Crucially, the
composition ratio still matches within 0.2% despite the per-domain weight
differences, which actually strengthens the reproducibility claim: the
macro-level composition behavior is robust to the specific adapter weights.

## Limitations

1. **3 seeds is the minimum for variance estimation.** The CV confidence interval
   is wide; 10+ seeds would give tighter bounds.
2. **Same data across seeds.** Only LoRA initialization and training data order
   vary. Data sampling variance is not tested.
3. **400 steps, not 1000.** Apple Silicon runtime constraint. Convergence
   patterns are consistent but longer training may introduce divergence.
4. **5 domains only.** Reproducibility at N=15+ untested.
5. **FP16 latent composition.** compose_adapters merges FP16 latent weights,
   not quantized ternary. True ternary-native composition (LoTA-QAF) is
   untested for reproducibility.
6. **Thin validation set.** ~3,200 tokens per domain per eval. Statistical
   power for individual domain PPL is limited.

## What Would Kill This

- **At micro:** CV > 50% with more seeds (10+), or a pathological seed that
  produces ratio > 10x.
- **At macro:** Different model scale (BitNet-30B when available) shows higher
  variance, or longer training (1000+ steps) causes seeds to diverge.
- **Architecturally:** If ternary-native composition (quantize-then-compose)
  shows higher variance than FP16-latent composition (compose-then-quantize).

## Verdict: SUPPORTED

Both kill criteria pass with enormous margins (K1: 0.5% vs 50% threshold,
K2: 3.45x vs 10x threshold). Ternary QAT+STE LoRA composition on BitNet-2B
is highly reproducible across seeds. This validates the single-seed results
from exp_bitnet_ternary_convergence and unblocks scaling experiments
(exp_bitnet_scale_n15).

## Cost

- Runtime: 54 minutes on Apple Silicon (M-series)
- Compute cost: $0 (local)
- Data cost: $0 (HuggingFace public datasets, reused from prior experiment)
