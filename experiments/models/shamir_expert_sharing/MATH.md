# Shamir Expert Sharing: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Description |
|--------|-----------|-------------|
| W | (d_out, d_in) | Expert MLP weight matrix |
| w_ij | scalar | Individual weight element |
| k | integer | Reconstruction threshold (min shares needed) |
| n | integer >= k | Total number of shares |
| P(x) | polynomial | Degree-(k-1) polynomial over the reals |
| a_0, ..., a_{k-1} | scalars | Polynomial coefficients (a_0 = secret) |
| (x_i, y_i) | pair | Share i: evaluation point and polynomial value |
| L_j(x) | polynomial | Lagrange basis polynomial for point j |

## 1. Shamir's Secret Sharing (over the Reals)

### Standard scheme (Shamir, 1979)

The original scheme operates over a finite field GF(p) to provide information-
theoretic security. For our application (fault tolerance, not cryptographic
secrecy), we operate over the real numbers R.

**Secret encoding.** Given a secret s in R, construct a random polynomial of
degree k-1:

    P(x) = a_0 + a_1 x + a_2 x^2 + ... + a_{k-1} x^{k-1}

where a_0 = s and a_1, ..., a_{k-1} ~ N(0, sigma^2) are random coefficients.

**Share generation.** Evaluate P at n distinct non-zero points x_1, ..., x_n:

    share_i = (x_i, P(x_i))    for i = 1, ..., n

We use x_i = i (positive integers) for simplicity.

**Reconstruction.** Given any k shares {(x_{j_1}, y_{j_1}), ..., (x_{j_k}, y_{j_k})},
reconstruct s = P(0) via Lagrange interpolation:

    P(0) = sum_{m=1}^{k} y_{j_m} * L_{j_m}(0)

where the Lagrange basis polynomials evaluated at 0 are:

    L_{j_m}(0) = prod_{l != m} (-x_{j_l}) / (x_{j_m} - x_{j_l})

### Extension to weight tensors

Apply the scheme element-wise: for a weight matrix W of shape (d_out, d_in),
flatten to a vector w of length d_out * d_in, then create shares of w.

Each share is itself a vector of the same length, which can be reshaped back
to (d_out, d_in). Thus each share looks like a "weight matrix" of the same
shape as the original.

**Cost per expert:**
- Storage: n * |W| (n copies of the weight matrix)
- Share creation: O(k * |W|) -- evaluate degree-(k-1) polynomial at n points
- Reconstruction: O(k^2 * |W|) -- k Lagrange basis evaluations, each O(k)

## 2. Numerical Precision Analysis

Over finite fields, reconstruction is exact. Over the reals, floating-point
arithmetic introduces rounding errors.

### Error bound

The Lagrange basis coefficients L_j(0) involve products and quotients of
(x_i - x_j) terms. For our evaluation points x_i = i:

    |x_i - x_j| = |i - j| >= 1

This means the denominators are always >= 1, avoiding catastrophic cancellation.
The condition number of the Vandermonde matrix determines numerical stability.

For x_i = 1, 2, ..., n with k shares:
- k=2: condition number ~ O(1), machine-epsilon reconstruction
- k=3: condition number ~ O(10), reconstruction error ~ 10 * eps
- k=5: condition number ~ O(10^3), reconstruction error ~ 10^3 * eps
- k=10: condition number ~ O(10^8), reconstruction error approaches float32 precision

**Key insight:** Using float64 for intermediate computation and rounding to
float32 at the end gives exact reconstruction for k <= ~7, because the float64
precision (eps ~ 10^-16) times the condition number (~ 10^4 for k=7) is still
well below float32 precision (eps ~ 10^-7).

### Worked example (d=4, k=3, n=5)

Secret weights: w = [0.5, -0.3, 0.1, 0.7]

Polynomial coefficients (per element):
- a_0 = w (the secret)
- a_1 ~ N(0, 0.01) = [0.02, -0.05, 0.03, -0.01]
- a_2 ~ N(0, 0.01) = [-0.03, 0.01, -0.02, 0.04]

P(x) = a_0 + a_1*x + a_2*x^2

Shares:
- P(1) = [0.5+0.02-0.03, ...] = [0.49, -0.34, 0.11, 0.73]
- P(2) = [0.5+0.04-0.12, ...] = [0.42, -0.36, 0.08, 0.84]
- P(3) = [0.5+0.06-0.27, ...] = [0.29, -0.36, 0.01, 1.03]
- P(4) = ...
- P(5) = ...

Reconstruction from shares 1, 3, 5:
- Lagrange basis at x=0:
  L_1(0) = (-3)(-5) / (1-3)(1-5) = 15/8
  L_3(0) = (-1)(-5) / (3-1)(3-5) = 5/(-4) = -5/4
  L_5(0) = (-1)(-3) / (5-1)(5-3) = 3/8

- P(0) = (15/8)*P(1) + (-5/4)*P(3) + (3/8)*P(5) = w (exactly)

## 3. Amortized Overhead Analysis

The kill criterion is "k-of-n reconstruction overhead >10% of forward pass."

**Per-token cost decomposition:**

Let T_fwd = forward pass time, T_recon = reconstruction time.

If reconstruction is done once and then the model serves B tokens:

    Amortized overhead = T_recon / (B * T_fwd)

For B = 1000 tokens (a single generation):
    Amortized overhead = T_recon / (1000 * T_fwd) ~ 0.018% at k=3

The kill criterion is ambiguous: "overhead of forward pass" could mean:
1. Per-reconstruction vs per-forward (measured: 18.4% -- KILLED)
2. Amortized over a generation (measured: ~0.018% -- PASSED trivially)

Interpretation 1 is the strict reading. Interpretation 2 is the practical one.

## 4. Expert Blending via Polynomial Evaluation

### Mathematical structure

Evaluating P(x) at x != 0 gives a point that is NOT the original secret but
IS a smooth function of x. Near x = 0, this gives:

    P(epsilon) ≈ P(0) + epsilon * a_1 + O(epsilon^2)
    = w + epsilon * (random_direction) + O(epsilon^2)

This is equivalent to adding scaled noise to the weights -- not meaningful
interpolation between "experts."

### Why blending fails at large x

For a degree-(k-1) polynomial, |P(x)| grows as |x|^{k-1}. At x = 3.0 with
k=3, the quadratic term dominates, producing weight magnitudes far from the
trained range. This is confirmed empirically: +4.1% at x=0.75, +16.1% at
x=1.0, exploding beyond.

**Conclusion:** Polynomial evaluation does NOT provide meaningful expert
interpolation. The polynomial structure is a sharing artifact, not a
semantic one. True expert blending requires operating in the loss landscape
(e.g., weight averaging, model soups), not along random polynomial curves.

## 5. Assumptions

1. **Real-number arithmetic suffices.** We do not need information-theoretic
   security (GF(p)), only fault tolerance (exact reconstruction up to float
   precision). Justified: neural network weights are not secrets.

2. **Element-wise sharing is valid.** Each weight is shared independently.
   This is correct because reconstruction is linear (Lagrange interpolation),
   so element-wise sharing of a vector is equivalent to sharing the whole vector.

3. **Float64 intermediate precision.** Using float64 for Lagrange interpolation
   and rounding to float32 gives exact reconstruction for k <= ~7. Justified
   by condition number analysis above.

4. **Reconstruction is a one-time cost.** In inference, you reconstruct the
   expert weights once and then serve many tokens. The amortized cost is
   negligible. This is the standard serving model for MoE expert loading.

5. **Share storage cost is acceptable.** n shares of |W| each costs n * |W|
   total storage. For n=5, this is 5x the original model size. Whether this
   is acceptable depends on the deployment scenario (distributed serving,
   CDN-based expert distribution, etc.).
