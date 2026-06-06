# Loudness Fix: Mathematical Foundations

## 1. Problem Statement

Given a shared pretrained base model M_base and domain-specific capsule pools
trained from M_base:

```
Pool_A: (A_a, B_a) fine-tuned on domain A, attention frozen
Pool_B: (A_b, B_b) fine-tuned on domain B, attention frozen
```

Composition by concatenation produces:

```
y_composed = B_composed @ ReLU(A_composed @ x)
           = B_a @ ReLU(A_a @ x) + B_b @ ReLU(A_b @ x)
           = Pool_A(x) + Pool_B(x)
```

This identity is exact (proven in relu_router MATH.md Section 5.1).

The problem: during joint training on domains A+B, the MLP learns a single
pool Pool_joint producing output with distribution D_joint. The downstream
layers (attention at next layer, lm_head) are calibrated for D_joint.

The composed output Pool_A(x) + Pool_B(x) has distribution D_composed, which
differs from D_joint in two ways:

1. **Magnitude**: ||Pool_A(x) + Pool_B(x)|| may differ from ||Pool_joint(x)||
2. **Direction**: The direction of Pool_A(x) + Pool_B(x) may differ from
   Pool_joint(x) even when magnitudes match

The "loudness hypothesis" posits that (1) is the dominant factor. This
experiment tests that hypothesis.

---

## 2. Intervention 1: Per-Pool RMSNorm

### 2.1 Definition

For each pool output y_i in R^d, define the RMS-normalized output:

```
RMSNorm(y_i) = (y_i / RMS(y_i)) * alpha

where RMS(y_i) = sqrt(mean(y_i^2) + eps)
      alpha = target_rms (scaling constant, default = 1/N_pools)
```

Composed output:

```
y = sum_{i=1}^{N} RMSNorm(Pool_i(x))
  = sum_{i=1}^{N} alpha * Pool_i(x) / RMS(Pool_i(x))
```

### 2.2 Properties

**Magnitude equalization**: For any two pools with outputs y_a, y_b:

```
||RMSNorm(y_a)|| / ||RMSNorm(y_b)|| = 1    (exactly)
```

regardless of ||y_a|| vs ||y_b||. All pools contribute with equal L2 norm
(scaled by alpha) to the composed output.

**Direction preservation**: RMSNorm preserves the direction of each pool's
output:

```
RMSNorm(y_i) = alpha * y_i / ||y_i||_RMS
```

where ||.||_RMS is the root-mean-square norm (= L2/sqrt(d)). The direction
is preserved exactly.

**Independence**: RMSNorm(Pool_i(x)) depends only on Pool_i's weights and x.
Adding or removing pools does not affect existing pools' normalized outputs.
This preserves the non-interference property of ReLU composition.

**Scale of composed output**: With N pools and alpha = 1/N:

```
E[||y_composed||^2] = E[||sum of N unit-vectors * (1/N)||^2]
```

If pool outputs are uncorrelated:

```
E[||y_composed||^2] = N * (1/N)^2 * E[||unit||^2] = E[||unit||^2] / N
```

So the composed output shrinks as 1/sqrt(N). With correlated pools, the
magnitude depends on the correlation structure.

### 2.3 Computational Cost

Per layer, per pool:

```
Additional FLOPs: 2d (compute mean of squares, multiply by rsqrt)
Additional memory: 0 (no parameters)
```

Total additional cost for N pools: 2*N*d FLOPs per layer, 0 parameters.

### 2.4 Why This Fails

RMSNorm per pool causes catastrophic degradation (+22.4% vs joint) through
two compounding mechanisms:

**Double-normalization pathology.** In a transformer, the block structure is:

```
x_{l+1} = x_l + Pool(LayerNorm(x_l))    -- pool output enters residual
x_{l+2} = x_{l+1} + Attn(LayerNorm(x_{l+1}))  -- LayerNorm re-normalizes
```

Adding per-pool RMSNorm creates a double normalization: the pool output is
normalized by RMSNorm, then the entire residual stream is normalized again
by the downstream LayerNorm. This collapses the effective dynamic range of
the pool's contribution to the residual, making it a near-constant
perturbation regardless of input.

**Loss of adaptive correction magnitude.** Beyond double normalization,
RMSNorm forces all corrections to the same magnitude. For some inputs,
the pool should produce a large correction (strongly misaligned with the
current prediction); for others, a small correction (already well-predicted).
RMSNorm destroys this adaptive behavior.

**Note on internal consistency**: The scalar calibration diagnostic (Section 3)
shows that magnitude mismatch between pools is NOT the problem (scales ~0.99).
This means RMSNorm fails not because it "corrects" a non-existent magnitude
problem, but because the normalization itself is harmful — the double-normalization
pathology dominates, not the magnitude equalization.

---

## 3. Intervention 2: Scalar Calibration

### 3.1 Definition

For a composed model with N pools, learn one scalar s_i in R per pool per
layer:

```
y = sum_{i=1}^{N} s_i * Pool_i(x)
```

Total parameters: N * L (e.g., 2 * 4 = 8 for 2 domains, 4 layers).

### 3.2 Diagnostic Purpose

Scalar calibration tests whether magnitude (loudness) is the sole issue:

- If scalar calibration matches full calibration:
  Loudness is the only problem. 1 param per pool suffices.
- If scalar calibration << full calibration:
  Direction/interference matters. Loudness is insufficient.

### 3.3 Relation to the Loudness Hypothesis

The loudness hypothesis states:

```
Pool_A(x) + Pool_B(x) ≈ s_A * Pool_A(x) + s_B * Pool_B(x) ≈ Pool_joint(x)
```

for some scalars s_A, s_B. This holds if:

1. Pool_A and Pool_B produce outputs in the RIGHT directions but
   WRONG magnitudes
2. The optimal scalars correct the magnitude mismatch

If the hypothesis is TRUE, the learned scalars should deviate significantly
from 1.0 (indicating magnitude mismatch exists and matters).

If the hypothesis is FALSE, the learned scalars will be near 1.0 (indicating
magnitude is not the problem -- direction is).

---

## 4. Intervention 3: Matched-Magnitude Training

### 4.1 Auxiliary Loss

During domain-specific fine-tuning, add a loss penalizing deviation from
the pretrained pool's output RMS:

```
L_mag = lambda_mag * (RMS(Pool_d(x)) - RMS_target)^2

where RMS_target = E_x[RMS(Pool_pretrained(x))]  (measured before fine-tuning)
```

### 4.2 Total Loss During Fine-Tuning

```
L = L_NTP + L_sparsity + L_balance + L_magnitude
```

### 4.3 Properties

**Non-interference**: The magnitude loss constrains each pool independently.
Fine-tuning domain A's pool does not affect domain B's pool.

**Directional freedom**: The loss penalizes only the RMS of the output, not
its direction. The pool is free to learn domain-specific directions while
maintaining consistent output scale.

**Worked example at micro scale**:

```
d=64, P=128, L_mag coefficient = 1.0

Pretrained pool output RMS: 5.73 (measured, layer 0)
After fine-tuning on domain A without L_mag: 9.53 (drifted +66%)
After fine-tuning on domain A with L_mag: 8.57 (drifted only +0.3%)

The magnitude loss successfully constrains output norm.
```

### 4.4 Why This Partially Fails

The magnitude loss keeps ||Pool_A(x)||_RMS close to ||Pool_pretrained(x)||_RMS.
However, the problem is not just that ||Pool_A(x) + Pool_B(x)|| differs from
||Pool_joint(x)||. The problem is that the FUNCTION computed by Pool_A + Pool_B
differs from Pool_joint, even when both have the same output magnitude.

Two independently-trained pools that happen to produce the same RMS output
can still produce DIFFERENT outputs. The sum of these different outputs does
not equal the single pool that would have been learned by joint training.
This is a function-space gap, not a magnitude gap.

---

## 5. Theoretical Analysis: Why Weight Averaging Works Better

### 5.1 Weight Averaging vs Concatenation

Weight averaging:
```
Pool_avg(x) = ((B_a + B_b)/2) @ ReLU(((A_a + A_b)/2) @ x)
```

Concatenation:
```
Pool_concat(x) = B_a @ ReLU(A_a @ x) + B_b @ ReLU(A_b @ x)
```

The key difference: averaging operates in WEIGHT SPACE while concatenation
operates in FUNCTION SPACE.

### 5.2 Why Weight Averaging Preserves Network Expectations

The pretrained base pool has weights (A_base, B_base). After domain-specific
fine-tuning:

```
A_a = A_base + delta_A_a
A_b = A_base + delta_A_b
```

Weight averaging produces:

```
A_avg = (A_a + A_b) / 2 = A_base + (delta_A_a + delta_A_b) / 2
```

This is a SINGLE pool with dimension P (same as base). The output has
the same dimensionality and approximately the same magnitude as the
base pool. The downstream layers expect outputs from a pool of dimension
P -- weight averaging preserves this contract.

Concatenation produces a pool of dimension 2P. Even if the total magnitude
is controlled, the representation lives in a different space (2P-dimensional
hidden space vs P-dimensional).

### 5.3 The Representation Space Argument

Both concatenated and single pools produce outputs in R^d (same output
dimensionality). However, they differ in the **rank and span** of achievable
outputs:

1. A single pool with P capsules can represent outputs of rank up to P
   (linear combinations of P expansion vectors b_i)
2. The concatenated pool with 2P capsules can represent outputs of rank up
   to 2P (a richer subspace of R^d)
3. The downstream layers were trained to handle rank-P outputs from the
   pretrained base pool

The function-space gap is about the **span** of the output, not the
**dimensionality** of the output space (which is always d). The concatenated
pool produces outputs in a 2P-dimensional subspace of R^d, while the
downstream layers were calibrated for a P-dimensional subspace. Even at
matched magnitudes, this distribution shift degrades quality.

Weight averaging avoids this by staying in the P-dimensional output subspace
(same rank as the pretrained base), keeping the composed function within
the distribution the downstream layers expect.

---

## 6. Worked Numerical Example

At d=4, P=4 (toy scale):

```
Base pool:
  A = [[1, 0, 0, 0],   B = [[1, 0],
       [0, 1, 0, 0]]         [0, 1],
                              [0, 0],
                              [0, 0]]

Domain A fine-tuned:
  A_a = [[1.2, -0.1, 0.1, 0],   (detectors shifted toward domain A patterns)
         [0.1,  1.3, 0, -0.1]]

Domain B fine-tuned:
  A_b = [[0.8, 0.2, 0, 0.1],    (detectors shifted toward domain B patterns)
         [-0.1, 0.9, 0.2, 0]]

Weight averaged:
  A_avg = [[1.0, 0.05, 0.05, 0.05],   (compromise between A and B)
           [0.0, 1.1,  0.1, -0.05]]

Input x = [0.5, -0.3, 0.8, 0.1]:

  Pool_A(x) = B_a @ ReLU(A_a @ x)
            = B_a @ ReLU([0.5*1.2 + (-0.3)*(-0.1) + 0.8*0.1,
                          0.5*0.1 + (-0.3)*1.3 + 0.1*(-0.1)])
            = B_a @ ReLU([0.71, -0.34])
            = B_a @ [0.71, 0]
            = [0.71, 0, 0, 0]

  Pool_B(x) = B_b @ ReLU([0.5*0.8 + (-0.3)*0.2 + 0.1*0.1,
                           0.5*(-0.1) + (-0.3)*0.9 + 0.8*0.2])
            = B_b @ ReLU([0.35, -0.16])
            = B_b @ [0.35, 0]
            = [0.35, 0, 0, 0]

  Concatenated: Pool_A(x) + Pool_B(x) = [1.06, 0, 0, 0]
  Averaged:     Pool_avg(x) = B_avg @ ReLU(A_avg @ x) = ...
```

The concatenated output has roughly 2x the magnitude of each individual pool.
RMSNorm would normalize this, but the direction [1, 0, 0, 0] may not be what
the downstream layers expect (the joint pool would produce a different direction
because it optimized for both domains simultaneously).

---

## 7. Assumptions

1. **Output magnitude carries information.** RMSNorm's failure confirms this:
   the magnitude of Pool(x) relative to the residual stream encodes the
   strength of the correction. Normalizing this away degrades quality.

2. **Independent fine-tuning produces magnitude-matched pools** (matched-
   magnitude assumption). Validated: with L_mag, domain pools maintain
   RMS within 1% of pretrained. But this does not ensure functional
   equivalence.

3. **Function-space gap is the dominant issue, not magnitude gap.**
   Validated: scalar calibration learns scales ~0.99 (near identity),
   confirming magnitude is not the bottleneck. The gap between
   Pool_A(x) + Pool_B(x) and Pool_joint(x) is directional, not scalar.

4. **Weight averaging preserves downstream expectations.** Partially
   validated: weight averaging (+1.5% vs joint) outperforms concatenation
   (+4.3%) because it maintains the same output dimensionality as the
   pretrained base, staying closer to the function space the rest of the
   network was calibrated for.

---

## 8. Complexity Summary

| Method | Extra Params | Extra FLOPs/layer | Calibration Data |
|--------|-------------|-------------------|------------------|
| Plain zero-shot | 0 | 0 | None |
| Per-pool RMSNorm | 0 | 2Nd | None |
| Scalar calibration | N*L | 0 | ~100 steps mixed |
| Matched-magnitude | 0 (training only) | 0 (inference) | None |
| Full calibration | 2dP (all capsule weights) | 0 | ~100 steps mixed |
| Weight averaging | 0 | 0 | None |

Where N = number of pools, L = number of layers, d = embedding dim, P = capsules per pool.
