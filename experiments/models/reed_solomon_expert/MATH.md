# Reed-Solomon Expert Encoding: Mathematical Foundations

## Notation

| Symbol | Meaning | Dimension/Type |
|--------|---------|----------------|
| N | Number of original experts (data symbols) | integer >= 2 |
| k | Number of parity experts (redundancy) | integer >= 1 |
| D | Flattened weight vector dimension per expert | integer |
| w_i | Weight vector of expert i | R^D |
| x_i | Evaluation point for expert i | R |
| P(x) | Lagrange interpolating polynomial | R -> R^D |
| L_j(x) | Lagrange basis polynomial for point j | R -> R |

## Core Construction

### Problem Statement

Given N expert weight vectors w_1, ..., w_N in R^D, construct k additional
"parity" weight vectors such that any N of the N+k total vectors suffice to
reconstruct all N originals.

### Lagrange Interpolation Over the Reals

Assign each expert a distinct evaluation point x_1, ..., x_N in R. The unique
degree-(N-1) polynomial through these points is:

$$P(x) = \sum_{j=1}^{N} w_j \cdot L_j(x)$$

where the Lagrange basis polynomials are:

$$L_j(x) = \prod_{m \neq j} \frac{x - x_m}{x_j - x_m}$$

**Key property**: L_j(x_i) = delta_{ij} (Kronecker delta), so P(x_i) = w_i exactly.

### Parity Expert Generation

Choose k additional distinct evaluation points x_{N+1}, ..., x_{N+k}, disjoint
from {x_1, ..., x_N}. Define parity expert weights:

$$w_{N+j} = P(x_{N+j}) = \sum_{i=1}^{N} w_i \cdot L_i(x_{N+j}), \quad j = 1, ..., k$$

Each parity expert is a specific linear combination of the original experts,
with coefficients determined by the Lagrange basis evaluated at the parity point.

### Reconstruction

Given any N of the N+k total points {(x_i, w_i)}, the degree-(N-1) polynomial
is uniquely determined. Reconstruct any missing original w_j by evaluating:

$$w_j = \sum_{i \in S} w_i \cdot L_i^{(S)}(x_j)$$

where S is the set of N available points and L_i^{(S)} are the Lagrange basis
polynomials defined over the available points.

**Proof**: A polynomial of degree N-1 is uniquely determined by N distinct
point-value pairs (fundamental theorem of algebra). Since all original and
parity points lie on the same degree-(N-1) polynomial, any N points determine
it, and evaluation at any missing point recovers that value.

## Evaluation Point Selection

### Chebyshev Nodes (Recommended)

For N data points on [-1, 1]:

$$x_j = \cos\left(\frac{(2j-1)\pi}{2N}\right), \quad j = 1, ..., N$$

For k parity points on [1.1, 2.0] (outside data interval):

$$x_{N+j} = \cos\left(\frac{(2j-1)\pi}{2k}\right) \cdot \frac{0.45}{1} + 1.55, \quad j = 1, ..., k$$

**Motivation**: Chebyshev nodes minimize the Lebesgue constant Lambda_N, which
bounds interpolation error:

$$||P - f||_\infty \leq (1 + \Lambda_N) \cdot ||P^* - f||_\infty$$

For equispaced nodes, Lambda_N grows exponentially as ~2^N/(eN log N).
For Chebyshev nodes, Lambda_N grows only as ~(2/pi) log N.

At N=4 (micro scale), both choices give Lambda_N < 3, so the difference is
negligible. At N=20+ (macro scale), Chebyshev nodes are essential.

### Uniform Nodes (Simpler, Less Stable)

$$x_j = j, \quad j = 1, ..., N$$
$$x_{N+j} = N + j, \quad j = 1, ..., k$$

Acceptable for N <= 8. Numerically unstable for N >= 16.

## Computational Complexity

### Encoding (One-Time, Offline)

Generating k parity experts from N originals:
- Per parity expert: evaluate P at one point = O(N * D) multiply-adds
- Total: O(k * N * D) multiply-adds
- Memory: O((N + k) * D) to store all experts

### Reconstruction (One-Time, On Expert Loss)

Reconstructing m missing experts from N available:
- Per missing expert: Lagrange interpolation at one point = O(N * D) multiply-adds
- Total: O(m * N * D) multiply-adds

### Runtime (Per-Inference)

**Zero additional cost**. After reconstruction, the model operates with standard
weights. RS encoding is purely an offline resilience mechanism.

## Parameter Overhead

Overhead = k / N * P_expert, where P_expert is the parameter count of one expert.

| N (experts) | k (parity) | Overhead % | Kill KC2 (>20%) |
|-------------|------------|------------|-----------------|
| 4 | 1 | 25.0% | KILLED |
| 4 | 2 | 50.0% | KILLED |
| 8 | 1 | 12.5% | PASSED |
| 8 | 2 | 25.0% | KILLED |
| 16 | 1 | 6.25% | PASSED |
| 16 | 2 | 12.5% | PASSED |
| 20 | 2 | 10.0% | PASSED |
| 64 | 4 | 6.25% | PASSED |

**Conclusion**: KC2 passes for N >= 6 with k=1, or N >= 10 with k=2.
At micro scale (N=4 layers), overhead is inherently high because N is small.
At macro scale (N=20+ domain experts), overhead is modest.

## Worked Example: N=4, k=2, d=64

Configuration: 4 expert MLP weight matrices, each with fc1 shape (256, 64)
and fc2 shape (64, 256). Flattened D = 256 * 64 = 16,384 per param matrix.

**Encoding**:
1. Assign Chebyshev nodes x_1..x_4 on [-1, 1]: [-0.924, -0.383, 0.383, 0.924]
2. Assign parity nodes x_5, x_6 on [1.1, 2.0]: [1.245, 1.855]
3. For each param matrix (fc1, fc2 separately):
   - Flatten: w_i in R^{16384}
   - Compute P(x_5) and P(x_6) via Lagrange interpolation: 2 * 4 * 16384 = 131K MADs
4. Total encoding cost: 2 * 131K = 262K MADs (trivial, ~0.3ms)

**Storage**: 4 original + 2 parity = 6 total expert weight sets. Overhead = 2/4 = 50%.

**Reconstruction (drop experts 1 and 3)**:
1. Available: experts {2, 4, 5, 6} (two originals + two parity)
2. Rebuild Lagrange basis over available x-values
3. Evaluate at x_1 and x_3 to recover w_1 and w_3
4. Cost: 2 * 4 * 16384 = 131K MADs

**Reconstruction error**: Exact to float64 precision (~1e-14).

## Assumptions

1. **Expert weights are arbitrary real vectors**: No structural constraints needed.
   RS encoding works for any collection of real-valued vectors.

2. **Float64 precision is sufficient**: Lagrange interpolation introduces only
   IEEE 754 rounding errors, which are negligible (< 1e-10) for weight vectors
   with typical magnitude ~0.01-1.0 and N <= 20.

3. **Experts are independent**: The encoding treats each expert's weights as
   an opaque D-dimensional vector. No assumption about relationships between
   experts. (This is a feature: works for both related and unrelated experts.)

4. **N is moderate (< 64)**: For very large N, the Lebesgue constant grows
   even with Chebyshev nodes. At N > 64, consider piecewise or barycentric
   interpolation, or switch to finite-field RS codes.

## Connection to Classical Reed-Solomon

Classical RS codes operate over finite fields GF(2^m). Our construction operates
over the reals R, using Lagrange interpolation instead of polynomial evaluation
over GF. The algebraic structure is identical:

| Classical RS (GF) | Real-valued RS (our approach) |
|-------------------|-------------------------------|
| Finite field GF(2^m) | Real numbers R (float64) |
| Polynomial P in GF[x] | Polynomial P in R[x] |
| Evaluation at field elements | Evaluation at Chebyshev nodes |
| Syndrome decoding | Lagrange interpolation at x=x_j |
| Corrects up to t erasures | Corrects up to k erasures |
| Exact (field arithmetic) | Exact to float64 precision |

The key difference: finite field arithmetic is exact but limited to discrete
symbols; real arithmetic is approximate but works directly on continuous weight
vectors without quantization.

## Relationship to Shamir Secret Sharing

The Shamir experiment (already in this project) proved the Lagrange primitive
works for weight reconstruction. Reed-Solomon encoding is the dual:

| Shamir | Reed-Solomon |
|--------|--------------|
| Hide ONE secret in n shares | Protect N experts with k parity |
| k-of-n threshold access | N-of-(N+k) reconstruction |
| Secret at P(0), shares at P(1..n) | Experts at P(x_1..x_N), parity at P(x_{N+1}..x_{N+k}) |
| Application: access control | Application: fault tolerance |
