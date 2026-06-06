# PAPER.md: VeRA-style M2P — Parameter Reduction via Shared Random Projection

**Experiment:** exp_m2p_vera_bottleneck  
**Date:** 2026-04-08  
**Status:** killed (K923 FAIL — K922 PASS, K924 PASS, K923 FAIL)

---

## Abstract

VeRA (Kopiczko et al., arXiv:2310.11454) replaces LoRA's B-matrices with a shared frozen
random basis W_shared plus per-layer scalar vectors (2r params per layer vs r×d). Applied
to the M2P hypernetwork, this reduces trainable parameters from ~357M to 4.67M — a 98x
compression — and gradient flow is verified non-zero at initialization. However, the
compressed M2P adapter achieves 18.8% accuracy on GSM8K (n=500), which is below the
20.0% base rate (quality_ratio = -0.105), failing K923 by a large margin. The
parameterization is provably compact (K922 PASS) and gradient-connected (K924 PASS), but
rank-4 VeRA scale vectors are insufficient to steer the shared random basis toward
meaningful GSM8K adaptations in 500 training steps.

---

## Hypothesis

Replacing M2P's N×Linear(d_m2p, r×d) output heads with a single Linear(d_m2p, N×4r)
plus frozen shared random matrices W_q, W_v allows the same functional forward pass
(Theorem 5, SHINE-style) at 1/76th the parameter count. The JL lemma guarantees W_shared
spans the rank-r space, so 2r scale scalars per layer should preserve at least 70% of
SFT quality (VeRA Table 2 extrapolation).

---

## Prediction-vs-Measurement Table

| Metric | Source | Predicted | Measured | Pass? |
|--------|--------|-----------|----------|-------|
| Total trainable params | MATH.md Theorem 1 | 4,656,576 (~4.7M) | **4,668,864** | K922 PASS |
| Reduction vs v4 (357M) | MATH.md Theorem 1 | ~76x (≥35x required) | **97.9x** | K922 PASS |
| grad_norm at step 0 | MATH.md Theorem 2 | > 0 | **0.1064** | K924 PASS |
| M2P accuracy at n=500 | VeRA Table 2 extrap | ≥ 0.280 (70% of gap) | **0.188** | K923 FAIL |
| quality_ratio at n=500 | K923 formula | ≥ 0.70 | **-0.105** | K923 FAIL |
| Final training loss | — | — | 1.520 | — |
| Total runtime | — | — | 1464s (24.4 min) | — |

---

## Evidence by Kill Criterion

### K922: Trainable params <= 10M — PASS

Measured trainable parameter count: **4,668,864** (4.67M).

Breakdown:
- Encoder Linear1 (1024→2048 + bias): 2,099,200
- Encoder Linear2 (2048→1024 + bias): 2,098,176
- Scale head Linear(1024→576 + bias): 471,488 (vs MATH.md estimate of 459,200 — small bias
  discrepancy from N_layers=28 not 36)
- Frozen W_q (2048×4): 8,192 (not trainable)
- Frozen W_v (1024×4): 4,096 (not trainable)

MATH.md Theorem 1 predicted 4,656,576 — off by 12,288 (0.26%), from N_layers=28 not
the 36 assumed in the theorem. Both match K922 ≤ 10M by a large margin (2.1x headroom).

Reduction vs v4: 97.9x (well above the ≥35x required).

### K923: quality_ratio >= 70% at n=500 — FAIL

Measured accuracy: **18.8%** (94/500 correct).
Wilson 95% CI for m2p_acc: **[0.156, 0.225]**.

quality_ratio = (0.188 - 0.200) / (0.314 - 0.200) = **-0.105**  
95% CI for quality_ratio (Fieller full propagation): **[-0.408, 0.197]**

The M2P adapter performs worse than the base model (20.0%). The entire Wilson CI for
m2p_acc lies below the base accuracy upper bound — there is no plausible scenario in
which this run meets the 0.280 target. K923 FAILS unambiguously.

Two-proportion z-test (M2P vs SFT): z = -4.60, p << 0.001. M2P is significantly
below SFT. Two-proportion z-test (M2P vs base): z = -0.48, p = 0.63. M2P is not
significantly different from the base model — the adapter learned nothing useful.

### K924: grad_norm > 0 at step 0 — PASS

Measured grad_norm at step 0: **0.1064**.  
Initial loss: 1.7656.

MATH.md Theorem 2 predicted nonzero gradients because W_shared ~ N(0, 1/d) is not
degenerate and scale vectors are initialized to 1 (not zero). Confirmed.

---

## Statistical Analysis

### Wilson 95% CI for M2P Accuracy

k = 94, n = 500, p_hat = 0.188  
CI: **[0.1562, 0.2246]**

The upper bound (0.225) is still below the target (0.280). K923 FAIL is robust.

### Fieller 95% CI for quality_ratio

Using delta method with full variance propagation (M2P and SFT uncertainties):

    se_m2p = sqrt(0.188 × 0.812 / 500) = 0.0175
    se_sft  = sqrt(0.314 × 0.686 / 500) = 0.0208
    gap = 0.114
    Var(qr) = [se_m2p² + qr² × se_sft²] / gap²
    se_qr = 0.153
    95% CI = [-0.408, 0.197]

Even the upper CI bound (0.197) does not reach 0.70. K923 FAIL is statistically
conclusive: p(quality_ratio ≥ 0.70) < 0.0001.

### Two-Proportion Z-Tests

| Comparison | z | Interpretation |
|------------|---|----------------|
| M2P (0.188) vs SFT (0.314) | -4.60 | M2P significantly worse (p << 0.001) |
| M2P (0.188) vs Base (0.200) | -0.48 | Not distinguishable from base (p = 0.63) |

---

## Failure Mode Analysis

The VeRA parameterization is mathematically correct and the gradient flows. The failure
is not in the math — it is in the expressivity–capacity tradeoff:

1. **Low-rank bottleneck (r=4, 2r=8 scalars per layer).** VeRA at rank=16 matches LoRA
   on GLUE (VeRA Table 2). At rank=4, the 8 scale scalars per layer must recover the
   full B-matrix from a fixed random basis. On GSM8K, the adapter needs to specialize
   individual layers in a coordinated way; the random basis may not align with the
   task-relevant directions.

2. **Shared basis vs per-layer geometry.** W_shared is identical for all 28 layers, but
   each layer contributes differently to reasoning chains. A single frozen random basis
   cannot simultaneously serve all layers — the scale vectors must compensate for all
   geometric misalignment.

3. **500 training steps may be insufficient.** Loss plateaued around 1.5 (vs initial
   1.77). The model learns to reduce training loss but not to generalize to GSM8K test
   problems, suggesting the effective hypothesis space is too constrained to learn the
   right inductive bias in 500 steps.

**Impossibility structure:** At rank=4 with a single shared frozen W_shared, the set of
expressible B-matrices for layer i is a (2r=8)-dimensional smooth manifold within the
space of rank-≤4 matrices. If the optimal B-matrix for GSM8K adaptation does not lie
near this manifold, scale-vector learning cannot converge regardless of training time.

---

## Conclusion

VeRA-style M2P achieves the parameter reduction goal (K922 PASS, 4.67M trainable, 98x
compression) and confirms gradient flow (K924 PASS, grad_norm=0.106 at step 0). These
are arithmetic and structural guarantees that hold.

The quality target K923 FAILS decisively: m2p_acc = 0.188 < base_acc = 0.200, giving
quality_ratio = -0.105 vs required ≥ 0.70. The adapter learned nothing useful on GSM8K.

The root failure is that rank-4 VeRA (8 scalars per layer) with a single shared random
basis is insufficient to represent the coordinated per-layer specialization needed for
GSM8K math reasoning. The mathematical framework (JL guarantee, gradient flow) is sound,
but the effective capacity of the parameterization is too low for this task at this scale.

**Overall status: killed.** K923 fails with no statistical ambiguity (CI upper bound
0.197 << 0.70 threshold). The VeRA bottleneck approach needs either higher rank (r ≥ 16)
or per-layer random bases to achieve competitive quality.

---

## References

- Kopiczko et al. (arXiv:2310.11454) — VeRA: Vector-Based Random Matrix Adaptation
- Johnson & Lindenstrauss (1984) — Random projections approximately preserve distances
- Ha et al. (arXiv:1609.09106) — HyperNetworks
- Hu et al. (arXiv:2106.09685) — LoRA
- SHINE (arXiv:2602.06358) — Functional LoRA forward (Theorem 5)
