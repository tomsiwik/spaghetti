# LoRA Scale Ablation: Theoretical Framework

## A. Failure Mode Identification (The Disease)

Every adapter experiment since `falcon_e3b_composition` used `lora_scale=20` -- a raw
multiplier on the LoRA update. MLX's `LoRALinear.__call__` computes:

```
output = W*x + scale * (x @ A) @ B
```

where `A` is initialized as `Uniform(-1/sqrt(d_in), 1/sqrt(d_in))` with shape
`(d_in, r)` and `B` is initialized as zeros with shape `(r, d_out)`. The fused
weight delta is:

```
Delta_W = scale * B^T @ A^T     (shape: d_out x d_in)
```

**Standard LoRA** (Hu et al. 2022, arXiv:2106.09685): The scale should be `alpha/r`.
For rank `r=16` with `alpha=16`, this gives `scale = 1.0`. The standard range is
`alpha/r in [0.5, 2.0]`. A scale of 20 is **10-20x outside the standard range**.

**The disease:** At `scale=20`, the LoRA perturbation magnitude can dominate the base
weight contribution, effectively replacing the base model's learned representations
rather than adapting them. On a ternary base (weights in {-1, 0, 1} scaled by
per-channel factors), this problem is acute because the base weight norm is constrained.

This is not merely "too large a learning rate" -- it is a structural mismatch between
the perturbation magnitude and the base weight scale that makes ALL downstream
measurements (SFT vs NTP comparisons, routing accuracy, composition quality)
uninterpretable.

## B. The Right Question (Reframe)

**Wrong question:** "What is the best lora_scale value?"

**Right question:** "At what scale does the LoRA perturbation exceed the base weight's
contribution to the output, making the adapter a replacement rather than an adaptation?"

The answer to this question defines a **critical threshold** above which the model
effectively ignores its pretraining. Below this threshold, the adapter modulates the
base; above it, the adapter overwrites the base.

## C. Prior Mathematical Foundations

### C.1 LoRA Scaling Convention (Hu et al. 2022)

Hu et al. introduce scaling `alpha/r` so that changing rank does not require
re-tuning the learning rate. At initialization (B=0), the update is zero regardless
of scale. After training, the update magnitude is:

```
||Delta_W||_F = scale * ||B^T @ A^T||_F
```

The implicit contract is that `scale * ||B^T @ A^T||_F << ||W||_F`, i.e., the
perturbation is a small fraction of the base weight.

### C.2 Ternary Weight Norms

For a ternary BitLinear layer with weight matrix `W_ternary` of shape `(d_out, d_in)`
where entries are in `{-1, 0, 1}` and a per-channel scale `s`:

```
W = s * W_ternary
```

The expected Frobenius norm, assuming each entry is independently `{-1, 0, 1}` with
sparsity ratio `p_0` (fraction of zeros):

```
E[||W_ternary||_F^2] = d_out * d_in * (1 - p_0)
||W||_F = ||s|| * sqrt(d_out * d_in * (1 - p_0))
```

For Falcon-E-3B with `d_model = 3072`, typical attention projections have
`d_in = d_out = 3072`. With typical ternary sparsity `p_0 ~ 0.33`:

```
||W_ternary||_F ~ sqrt(3072 * 3072 * 0.67) ~ sqrt(6.32e6) ~ 2514
```

The scale factor `s` is typically `O(1/sqrt(d_in))` for weight-normalized networks,
giving `||W||_F ~ 2514 / sqrt(3072) ~ 45.3`.

### C.3 LoRA Update Norm After Training

After `T` steps of Adam with learning rate `eta`, the LoRA matrices evolve from
initialization. B starts at zero and grows. A starts at `O(1/sqrt(d_in))` and shifts.

For a trained rank-16 LoRA on attention projections:
- `A` has shape `(3072, 16)`, entries `O(1/sqrt(3072)) ~ 0.018`
- `B` has shape `(16, 3072)`, entries grow during training

The update `B^T @ A^T` has shape `(3072, 3072)` but rank at most 16. Its Frobenius
norm depends on training but empirically:

```
||B^T @ A^T||_F ~ O(1)  to  O(10)
```

for 300 steps of Adam at `lr=1e-4` (the update is modest because B starts at zero and
the gradient signal through the low-rank bottleneck is limited).

## D. Proof of the Critical Scale Threshold

**Theorem 1** (Perturbation-to-Base Ratio).

Let `W` be a ternary weight matrix with Frobenius norm `||W||_F`, and let
`Delta = scale * B^T @ A^T` be the LoRA perturbation. Define the
perturbation-to-base ratio:

```
rho = ||Delta||_F / ||W||_F = scale * ||B^T @ A^T||_F / ||W||_F
```

*Then:*

(i) For `rho < 1`, the base weight dominates and the adapter is a perturbation.
    The output `(W + Delta)x` has cosine similarity > `1/sqrt(2)` with `Wx` for
    typical inputs (by Cauchy-Schwarz, when the perturbation is smaller than the
    base, the angle between `Wx` and `(W + Delta)x` is at most 45 degrees).

(ii) For `rho >> 1`, the adapter dominates. The output is approximately `Delta * x`,
     meaning the pretrained knowledge in `W` is effectively overwritten.

(iii) The critical threshold is `rho = 1`, i.e.:

```
scale_critical = ||W||_F / ||B^T @ A^T||_F
```

*Proof.*

Write the adapted output as:
```
y = (W + Delta)x = Wx + Delta * x
```

The ratio of perturbation energy to base energy in the output is:
```
E[||Delta * x||^2] / E[||Wx||^2] = ||Delta||_F^2 / ||W||_F^2 = rho^2
```

(using the fact that for isotropic input `x`, `E[||Mx||^2] = ||M||_F^2 * E[||x||^2/d]`
for any matrix `M`, so the ratio depends only on the Frobenius norms).

When `rho < 1`: `||Delta * x|| < ||Wx||` in expectation, so the base output dominates.
When `rho > 1`: `||Delta * x|| > ||Wx||` in expectation, so the perturbation dominates.
When `rho >> 1`: `(W + Delta)x ~ Delta * x`, the base is negligible.

The cosine similarity between `Wx` and `(W + Delta)x` satisfies:
```
cos(theta) = <Wx, (W + Delta)x> / (||Wx|| * ||(W + Delta)x||)
           >= ||Wx||^2 / (||Wx|| * (||Wx|| + ||Delta * x||))
           = 1 / (1 + rho)
```

For `rho < 1`: `cos(theta) > 0.5`, the adapted output is within 60 degrees of base.
For `rho = 1`: `cos(theta) > 0.5` (borderline).
For `rho = 19` (scale=20 vs scale_critical~1): `cos(theta) > 0.05`, nearly orthogonal.

QED.

### D.1 Quantitative Predictions

For Falcon-E-3B with `d_in = d_out = 3072`, `r = 16`, ternary weights:

Assume `||W||_F ~ 45` (ternary with per-channel scale ~1/55) and
`||B^T @ A^T||_F ~ 5` after 300 training steps (empirical estimate, to be measured):

| Scale | `||Delta||_F` | `rho` | Regime | Prediction |
|-------|--------------|-------|--------|------------|
| 1.0   | 5            | 0.11  | Perturbation | Adapter modulates base; base capabilities preserved |
| 2.0   | 10           | 0.22  | Perturbation | Slightly stronger modulation; still safe |
| 4.0   | 20           | 0.44  | Perturbation | Noticeable but base still dominates |
| 8.0   | 40           | 0.89  | Borderline | Near critical threshold; some degradation expected |
| 20.0  | 100          | 2.22  | Overwrite | Adapter dominates; pretrained knowledge destroyed |

**Prediction P1:** At `scale <= 2`, individual adapters should degrade at most 1/6
benchmarks vs base (the perturbation is small enough to preserve base capabilities).

**Prediction P2:** At `scale = 20`, individual adapters degrade most benchmarks
(perturbation overwrites base representations).

**Prediction P3:** At `scale = 20` with `N=5` uniform composition, effective scale per
adapter is `20/5 = 4`, which brings `rho ~ 0.44` -- explaining why composition
"recovers" performance. This is not composition helping; it is dilution reducing the
damage.

**Prediction P4:** The optimal scale for composition is NOT the optimal scale for
single adapters. At `scale = 1-2`, single adapters work well but composed
adapters have very small perturbation (`rho ~ 0.04-0.09`), potentially too weak
to have any measurable effect.

**Prediction P5:** There should be a monotonic relationship between `rho` and
benchmark degradation. Plotting (scale, degradation_count) should show a step
function near `rho = 1`.

## E. Assumptions and Breaking Conditions

**Assumption 1:** Ternary weight norms are `O(sqrt(d_out * d_in * (1-p_0)) / sqrt(d_in))`.
This assumes the per-channel scale is `O(1/sqrt(d_in))`. If Falcon-E uses a different
normalization, the critical scale shifts proportionally.
*If violated:* `scale_critical` changes but the framework remains valid. We measure
actual norms to calibrate.

**Assumption 2:** LoRA update norms after 300 steps are `O(1-10)`. This is an empirical
estimate.
*If violated:* The `rho` values shift. We measure actual `||B^T @ A^T||_F` to calibrate.

**Assumption 3:** Isotropic input distribution. Real activations are not isotropic, so
the ratio `E[||Delta*x||^2]/E[||Wx||^2]` may differ from `rho^2` by a factor depending
on the alignment of `x` with the singular vectors of `W` and `Delta`.
*If violated:* The predictions are qualitatively correct but quantitatively approximate.

**Assumption 4:** SFT and NTP produce LoRA updates of comparable magnitude. If SFT
(response-only masking) produces smaller gradients (fewer tokens contribute), B may
grow less, reducing `||B^T @ A^T||_F`.
*If violated:* SFT adapters are more conservative by construction, strengthening the
case for lower scale.

## F. Worked Example (d=16)

Consider a toy ternary layer: `d_in = d_out = 16`, `r = 4`.

```
W_ternary = random {-1, 0, 1}^{16x16}, sparsity p_0 = 0.33
||W_ternary||_F ~ sqrt(16 * 16 * 0.67) = sqrt(171.5) ~ 13.1
```

With per-channel scale `s = 1/sqrt(16) = 0.25`:
```
||W||_F ~ 0.25 * 13.1 = 3.28
```

After training, suppose:
```
A: (16, 4), entries ~ 0.1
B: (4, 16), entries ~ 0.05
||B^T @ A^T||_F ~ sqrt(16 * 16) * 0.1 * 0.05 * sqrt(4) = 16 * 0.01 = 0.16
```

(This is rough; actual norm depends on structure.)

| Scale | `||Delta||_F` | `rho` |
|-------|--------------|-------|
| 1.0   | 0.16         | 0.049 |
| 2.0   | 0.32         | 0.098 |
| 4.0   | 0.64         | 0.195 |
| 8.0   | 1.28         | 0.390 |
| 20.0  | 3.20         | 0.976 |

At `scale=20`, `rho ~ 1` even in this toy example -- right at the critical threshold.
With larger trained B (more steps, higher LR), `rho > 1` easily.

## G. Complexity and Architecture Connection

**FLOPs:** No additional FLOPs. This experiment only varies a scalar multiplier.

**Memory:** Same for all scales. The LoRA parameters are identical across conditions.

**Architecture note:** MLX's default `LoRALinear` uses `scale=20.0` (see source code).
This is an MLX-specific default, not a LoRA standard. The original LoRA paper uses
`alpha/r` which defaults to 1.0. The MLX default appears to be designed for quantized
models where the base weight norm is already reduced by quantization, but for ternary
models the base weight norm is even more constrained, making 20x even more dangerous.

**Connection to LoTA-QAF (arXiv:2407.11024):** Low-bit quantization reduces the
effective expressivity of the base model. Large LoRA perturbations can push activations
outside the representable range of the quantized weights, causing unpredictable behavior.
Conservative scaling is critical for low-bit models.

---

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Keeping `rho = scale * ||B^T @ A^T||_F / ||W||_F < 1` ensures the LoRA perturbation
   cannot dominate the base weight contribution, making overwrite impossible by
   construction.

2. **Which existing theorem(s) does the proof build on?**
   Cauchy-Schwarz inequality for the cosine bound; isotropic energy ratio from random
   matrix theory (E[||Mx||^2] proportional to ||M||_F^2 for isotropic x); Hu et al.
   2022 (arXiv:2106.09685) scaling convention.

3. **What specific numbers does the proof predict?**
   P1: scale<=2 degrades <=1/6 benchmarks. P2: scale=20 degrades most benchmarks.
   P3: 1/N composition at scale=20 recovers because effective rho drops by 1/N.
   P5: Monotonic degradation vs scale.

4. **What would FALSIFY the proof?**
   If scale=20 individual adapters do NOT degrade benchmarks (rho>1 but no damage),
   the isotropic assumption is wrong and activations are aligned to avoid the
   perturbation. Or the ternary weight norms are much larger than estimated.

5. **How many hyperparameters does this approach add?**
   Count: 0. This experiment discovers the RIGHT value for an existing hyperparameter
   (lora_scale) by measuring where rho crosses 1.

6. **Hack check:** No -- this is removing a confound, not adding a fix.
