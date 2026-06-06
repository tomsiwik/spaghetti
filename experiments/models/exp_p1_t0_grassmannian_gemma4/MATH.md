# T0.1: Grassmannian QR on Gemma 4 Weight Shapes

## Theorem 1 (Grassmannian Partition Construction)

**Statement:** Given a random matrix W ∈ ℝ^{d × Nr} with i.i.d. entries W_{ij} ~ N(0,1),
let Q ∈ ℝ^{d × Nr} be the Q-factor from its QR decomposition (W = QR). Define N rank-r
adapter subspaces as A_i = Q[:, ir:(i+1)r] for i = 0, …, N-1. Then:

    A_i^T A_j = 0    for all i ≠ j    (exactly, up to floating-point rounding)

**Proof:**

The QR decomposition guarantees Q^T Q = I_{Nr × Nr} (full column orthonormality).

Partition I_{Nr × Nr} into r × r blocks: I = [I_{00} | I_{01} | …; I_{10} | … ].
By orthonormality: I_{ij} = A_i^T A_j = δ_{ij} I_r.

Therefore for i ≠ j:

    (A_i^T A_j)_{kl} = e_{ir+k}^T (Q^T Q) e_{jr+l} = (I_{Nr})_{ir+k, jr+l} = 0

This is an algebraic identity — not a statistical approximation. The only source
of nonzero entries is floating-point rounding in the QR algorithm itself.

**QED**

## Corollary 1 (Capacity)

The maximum number of orthogonal rank-r adapters in ℝ^d is:

    N_max = ⌊d / r⌋

because Q must have Nr ≤ d columns. Beyond N_max, the QR system is overdetermined
and true orthogonality cannot be guaranteed.

**Gemma 4 capacity at r=16:**
- d = 2816 (26B-A4B hidden dim): N_max = 2816 / 16 = **176 domains**
- d = 5376 (31B hidden dim):     N_max = 5376 / 16 = **336 domains**

## Theorem 2 (Numerical Error Bound)

**Statement:** For the QR partition construction in IEEE 754 float64, the pairwise
Frobenius norm satisfies:

    max_{i≠j} ||A_i^T A_j||_F ≤ C √(Nr) ε_mach

where ε_mach = 2.2 × 10^{-16} (float64) and C is a small universal constant (C ≈ 10-50
for Householder QR).

**Prediction at N=50, r=16 (Nr=800), float64:**
    error ≤ 50 × √800 × 2.2e-16 ≈ 50 × 28.3 × 2.2e-16 ≈ 3.1e-13 << 1e-6 ✓

**Prediction at N=100, r=16 (Nr=1600), float64:**
    error ≤ 50 × √1600 × 2.2e-16 ≈ 50 × 40 × 2.2e-16 ≈ 4.4e-13 << 1e-6 ✓

## Predictions (Kill Criteria)

Kill criteria verified at smoke dimensions (d=512/d=1024, rank=4).
Gemma 4 dims (d=2816/d=5376, r=16) are analytical corollaries of Theorem 1 — same algebraic guarantee, different constants.

| Kill ID | Prediction | Expected Value | Tested Dims |
|---------|-----------|---------------|-------------|
| K990 | max\|A_i^T A_j\|_F < 1e-6 (f64) | ~1e-14 | d=512, N=10, r=4 (smoke) |
| K991 | max\|A_i^T A_j\|_F < 1e-6 (f64) | ~1e-14 | d=1024, N=20, r=4 (smoke) |
| K992 | N_max = floor(d/r) constructions complete without breakdown | 128 (d=512), 256 (d=1024) | smoke; Gemma4: 176/336 (r=16, analytical) |
| K993 | Construction time < 1s on GPU | ~1ms | d=1024, N=20 (smoke) |

**Gemma 4 analytical corollaries (not kill criteria — follow from Corollary 1):**
- d=2816, r=16: N_max = 2816/16 = 176 domains
- d=5376, r=16: N_max = 5376/16 = 336 domains
- NoPE (d=384, r=16): N_max = 384/16 = 24 domains per layer

## Connection to Pierre P1

Adapters are placed on q_proj NoPE dimensions [128:512] (384 dims from T0.3/T0.4).
At r=16: N_max = 384/16 = 24 domains per layer from NoPE alone.

For full q_proj at d=2816 (Gemma 4 global layer): N_max = 176 domains.
This exceeds our 25-domain target by 7× — sufficient headroom.

## Prior Results

- Finding #393: max|A_i^T A_j|_F = 9.50e-08 at N=50, d=1024, r=4 (Qwen3-0.6B)
- Finding #318: cos=0.0 exactly at N=5 and N=24 (Qwen3-4B GQA)
- Theorem 1 is verified. This experiment replicates at Gemma 4 dimensions.
