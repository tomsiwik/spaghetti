# SiLU Gamma Correction: Research Digest

## Hypothesis

SiLU activation (used by Qwen2.5, Llama, Mistral) produces a similar gamma correction
factor as GELU (~1.43x), validating that safety bounds derived under GELU transfer to
production SiLU-based architectures.

**Falsifiable:**
- K1: SiLU worst-case correction factor exceeds 2.0x (vs GELU 1.43x)
- K2: SiLU correction diverges from GELU by >50% at matched gamma profiles

---

## What This Experiment Tests

The parent experiment (rmsnorm_gamma_nonuniformity) PROVED that non-uniform RMSNorm
gamma has negligible effect on the amplification ratio alpha under GELU activation, with
a worst-case correction factor of 1.43x (single layer gamma=10). The adversarial review
flagged a critical gap:

> "Qwen2.5 and Llama use SiLU, not GELU. The 1.43x factor needs revalidation with the
> actual production activation."

This experiment runs the IDENTICAL gamma sweep with both GELU and SiLU side-by-side,
using the same random seeds, same gamma profiles, and same network architecture.
The only variable is the activation function.

---

## Key References

- Parent: rmsnorm_gamma_nonuniformity (PROVEN, 1.43x GELU worst-case)
- Grandparent: alpha_residual_scaling_ablation (PROVEN, uniform scaling invariant)
- Elfwing et al. (2018): SiLU/Swish activation analysis
- Qwen2.5 architecture: SiLU in SwiGLU FFN, RMSNorm, pre-norm

---

## Key Finding: SiLU Correction is Slightly Better Than GELU

### Analytical prediction confirmed empirically

SiLU has lower curvature than GELU (max |sigma''|: 0.500 vs 0.798, ratio 0.627).
Since the gamma correction factor is a second-order effect driven by activation
curvature, SiLU should produce equal or smaller correction factors. This is confirmed.

### Baselines (d=64, N=8, L=24, uniform gamma=1)

| Activation | Baseline Alpha | Max Correction Ratio |
|-----------|---------------|---------------------|
| GELU | 0.0217 | 1.43x |
| SiLU | 0.0240 | 1.41x |

SiLU baseline alpha is 10% higher (different forward-pass dynamics), but the
gamma correction ratio is 1.4% lower (lower curvature).

### Test 1: Log-Normal Gamma Variance Sweep (d=64, N=8, L=24)

| Sigma | Gamma Range | GELU Ratio | SiLU Ratio | Divergence |
|-------|------------|-----------|-----------|------------|
| 0.0 (uniform) | [1, 1] | 1.00x | 1.00x | 0.0% |
| 0.5 (realistic) | [0.2, 4.7] | 1.02x | 0.99x | 2.2% |
| 1.0 (extreme) | [0.04, 22] | 1.06x | 0.99x | 6.6% |
| 3.0 (pathological) | [0.01, 100] | 1.13x | 1.03x | 9.3% |

SiLU is MORE robust to gamma variance: only +3% at sigma=3.0 vs GELU's +13%.

### Test 2: Layer-Wise Gamma Profiles (worst-case analysis)

| Profile | GELU Ratio | SiLU Ratio | Divergence |
|---------|-----------|-----------|------------|
| alternating_0.5/2.0 | 1.20x | 1.20x | 0.6% |
| early_high_5.0 | 1.31x | 1.23x | 6.0% |
| late_high_5.0 | 1.28x | 1.23x | 4.4% |
| **single_spike_10.0** | **1.43x** | **1.41x** | **1.1%** |

The worst-case (single layer gamma=10) is near-identical for both activations.
Max divergence across all profiles is 9.3%, well below the 50% kill threshold.

### Test 3: Production Scale (d=256, N=50)

| Gamma Profile | GELU D% | SiLU D% | Margin Below 5% |
|---------------|---------|---------|-----------------|
| uniform | 0.099 | 0.101 | 50x |
| lognormal 0.5 | 0.101 | 0.103 | 49x |
| bimodal 0.2/5.0 | 0.098 | 0.098 | 51x |

At target scale, both activations produce deviations ~0.1%, with 42-51x margin
below the 5% safety threshold.

---

## Kill Criteria Assessment

**K1: SiLU worst-case correction factor < 2.0x**
- SiLU max correction ratio: **1.41x**
- PASS (29% below threshold)

**K2: SiLU correction diverges from GELU by <50%**
- Max divergence across all 16 gamma profiles: **9.3%**
- PASS (5.4x below threshold)

---

## Verdict: PROVEN

SiLU produces a worst-case gamma correction factor of 1.41x (vs GELU's 1.43x), with
maximum divergence of only 9.3% across all tested profiles. The safety bounds derived
under GELU are directly applicable to SiLU-based architectures (Qwen2.5, Llama, Mistral).
SiLU is marginally safer than GELU due to its lower activation curvature.

This closes the last activation-function gap flagged by the adversarial review of the
gamma nonuniformity experiment.

---

## Limitations

1. **Micro scale (d=64-256).** The curvature argument is dimension-independent, but
   specific ratios may shift slightly at d=896+. The 42-51x margin at d=256 provides
   substantial buffer.

2. **SiLU, not SwiGLU.** Production models use SwiGLU (gated SiLU with two branches).
   The gate mechanism is a fixed function of the same RMSNorm'd input, so the
   cancellation argument still applies, but this was not directly tested.

3. **Random base weights.** Structured pre-trained weights may create correlations
   between perturbation direction and gamma. The micro-scale test cannot capture this.

4. **3 seeds.** Sufficient for the clear signal (max divergence 9.3% with threshold 50%),
   but more seeds would tighten confidence intervals.

---

## What Would Kill This

- SiLU correction factor > 2.0x at any gamma profile (K1)
- SiLU/GELU divergence > 50% at matched profiles (K2)
- SwiGLU gating mechanism creating gamma-dependent asymmetry between signal and
  perturbation paths (would require separate SwiGLU experiment)

---

## Platform

- Apple Silicon (M-series), CPU-only
- numpy + scipy, no GPU required
- Runtime: ~714s (12 min) -- includes side-by-side GELU+SiLU comparison at all scales
- 3 seeds, 16 gamma profiles, 2 activations, 3 scale configs
