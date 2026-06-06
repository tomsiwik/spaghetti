# Room Model POC: Mathematical Foundations

## Type: Verification (Type 1)

The proof of pre-summed equivalence is complete from linearity of matrix
multiplication. The experiment confirms quantitative predictions.

---

## A. Failure Mode Identification

**Potential failure modes:**

1. **Interference at full rank.** When N rank-r adapters are summed into one
   d_out x d_in matrix, the resulting matrix has rank <= N*r. If the adapter
   subspaces are not truly orthogonal, cross-terms in the sum could interfere,
   producing different outputs than sequential application.

2. **Soft routing fails.** The Room Model claims that token hidden states
   automatically "route" by projecting onto adapter walls. If hidden states
   do not align with domain-specific A-matrix subspaces, the automatic routing
   produces uniform (useless) weights.

3. **PPL degradation from uniform composition.** Even with correct equivalence,
   applying ALL adapters uniformly (soft routing) may hurt PPL compared to
   hard top-1 routing, because irrelevant adapters add noise.

**Root cause analysis:** Mode 1 is the disease. Modes 2 and 3 are consequences
of how the combined matrix interacts with input geometry. The single mathematical
property that makes Mode 1 impossible is **linearity of matrix multiplication
combined with Grassmannian orthogonality of A-matrices.**

---

## B. The Right Question

**Wrong:** "How do we prevent adapter interference when summing deltas?"

**Right:** "Under what algebraic conditions does summing N adapter deltas produce
output identical to applying each adapter separately and summing the results?"

**Answer:** Linearity of matrix multiplication. This is not a condition to be
achieved -- it is an axiom of linear algebra. The sum IS the application.

---

## C. Derivation from Existing Mathematics

### C.1 Linearity of Matrix Multiplication (Axiom)

For any matrix M = M_1 + M_2 + ... + M_N and vector x:

    x @ M = x @ (M_1 + M_2 + ... + M_N) = x @ M_1 + x @ M_2 + ... + x @ M_N

This is the distributive property of matrix multiplication over addition.
No conditions required. It holds for all matrices over any field.

### C.2 Grassmannian Orthogonality (Finding #126, #3)

The frozen A-matrices satisfy A_i^T A_j = 0 for i != j (proven: |cos| < 0.001,
Finding #3, 50x below theoretical bound).

Each adapter delta is: DeltaW_i = alpha * B_i^T @ A_i^T

The image of DeltaW_i lies in the column space of B_i^T, which is projected
through A_i^T. Since A_i has rank r and A_i^T A_j ~ 0, the range spaces of
DeltaW_i and DeltaW_j are approximately orthogonal.

### C.3 Non-Interference Bound (from Finding #126)

For two adapters i, j with orthogonal A-matrices:

    ||DeltaW_i^T DeltaW_j|| <= (alpha/r)^2 * ||B_i|| * ||A_i^T A_j|| * ||B_j||

Since ||A_i^T A_j|| ~ 0 (Grassmannian), interference -> 0 regardless of B
correlation. Empirically: 17x filter from A orthogonality (Finding #126).

---

## D. Proof of Guarantee

### Theorem 1 (Pre-Summed Equivalence)

**Statement.** Let W_base be the base model weight matrix for a given module.
Let DeltaW_i = alpha * B_i^T @ A_i^T for i = 1, ..., N be adapter deltas.
Define W_combined = sum_{i=1}^{N} DeltaW_i. Then for any input x:

    W_base(x) + x @ W_combined = W_base(x) + sum_{i=1}^{N} x @ DeltaW_i

The outputs are identical (MSE = 0, exact equality in infinite precision).

**Proof.**

    x @ W_combined
  = x @ sum_{i=1}^{N} DeltaW_i           (definition of W_combined)
  = sum_{i=1}^{N} (x @ DeltaW_i)         (distributive property)

Adding W_base(x) to both sides:

    W_base(x) + x @ W_combined = W_base(x) + sum_{i=1}^{N} x @ DeltaW_i

QED.

**Note on floating point.** In finite precision arithmetic (bf16 or float32),
the equality becomes approximate due to floating-point associativity violations.
For bf16: the relative error is bounded by N * eps_bf16 where eps_bf16 = 2^{-8}
= 3.9e-3. For N=5 adapters at d=2560, the expected MSE is O(N * eps^2 * ||output||^2).
With ||output|| ~ O(1) (post-norm), MSE < 1e-4 in bf16, < 1e-12 in float32.

The kill criterion K763 (MSE < 1e-6) is achievable if we compute the sum in
float32, which is what we will do. In bf16, we predict MSE ~ 1e-4 to 1e-5.

### Theorem 2 (Projection Geometry of Soft Routing)

**Statement.** For an input hidden state h in R^d, the contribution of adapter i
to the room model output is:

    c_i = alpha * h @ (B_i^T @ A_i^T) = alpha * (h @ A_i^T)^T @ B_i^T

The effective routing weight for adapter i is proportional to ||h @ A_i^T||_2,
the norm of h's projection onto adapter i's A-subspace.

**Proof.** Expand the room model output:

    h @ W_combined = h @ sum_i DeltaW_i
                   = sum_i h @ (alpha * B_i^T @ A_i^T)
                   = sum_i alpha * (h @ A_i^T) @ B_i^T

Note that h @ A_i^T in R^r is the projection of h onto adapter i's input
subspace. Call this p_i = h @ A_i^T.

    h @ W_combined = sum_i alpha * p_i @ B_i^T

The contribution of adapter i is: c_i = alpha * p_i @ B_i^T.

The magnitude of this contribution is:

    ||c_i|| = alpha * ||p_i @ B_i^T|| <= alpha * ||p_i|| * ||B_i||

If ||B_i|| is roughly constant across adapters (as expected from similar
training), then ||c_i|| is proportional to ||p_i|| = ||h @ A_i^T||.

This IS soft routing: the adapter whose A-subspace aligns most with h
contributes most to the output. QED.

**Soft routing weight definition:**

    w_i(h) = ||h @ A_i^T||_2 / sum_j ||h @ A_j^T||_2

This is computable from the frozen A-matrices alone (no router needed).

### Theorem 3 (Bandwidth-Speed Prediction)

**Statement.** For a model with L layers, K target modules per layer, base
weight size S_base, and combined delta size S_delta per module:

    T_room = (S_base + L * K * S_delta) / BW + T_dispatch * L * K

where BW is memory bandwidth and T_dispatch is per-dispatch overhead.

**Calculation for BitNet-2B-4T at N=5, r=16:**

Base model: 1.18 GB (ternary packed).

Per-module W_combined sizes:
- q_proj, k_proj, v_proj, o_proj: 2560 x 2560 x 2 bytes = 13.1 MB each
- gate_proj, up_proj: 2560 x 6912 x 2 = 35.4 MB each
- down_proj: 6912 x 2560 x 2 = 35.4 MB

Per layer: 4 * 13.1 + 3 * 35.4 = 52.4 + 106.2 = 158.6 MB
30 layers: 30 * 158.6 = 4,758 MB = 4.65 GB

Total bandwidth: 1.18 + 4.65 = 5.83 GB
M5 Pro bandwidth: 273 GB/s
Minimum time: 5.83 / 273 = 21.4 ms per token
Maximum tok/s: 1000 / 21.4 = 46.7 tok/s

**Prediction:** Room model will achieve approximately 40-50 tok/s. This is
SLOWER than v3's 73 tok/s because:
- v3 reads: 1.18 GB base + 0.018 GB adapter = 1.2 GB -> 4.4 ms -> ~230 tok/s theoretical
- Room model reads: 1.18 GB base + 4.65 GB deltas = 5.83 GB -> 21.4 ms -> ~47 tok/s theoretical
- Both are dispatch-limited in practice, but room model's 256x bandwidth penalty dominates

This experiment is NOT about speed. It is about mechanism correctness.

---

## E. Predictions (Behavioral + Quantitative)

| Prediction | Source | Quantitative Bound |
|-----------|--------|-------------------|
| Pre-summed output = sequential output | Theorem 1 | MSE < 1e-6 (float32 computation) |
| Soft routing weights correlate with domain | Theorem 2 | Domain accuracy >= 60% |
| PPL equivalent to uniform composition | Theorem 1 | Within 10% of v3 N=5 baseline |
| Speed bandwidth-limited | Theorem 3 | 40-50 tok/s |

### Behavioral Predictions
1. **Theorem 1 guarantees:** The room model output is mathematically identical to
   applying all N=5 adapters and summing their contributions. Any MSE observed is
   pure floating-point noise.
2. **Theorem 2 predicts:** Tokens from domain-specific text will have larger
   projection norms onto their corresponding A-subspace than onto other A-subspaces.
   This is the "automatic routing" claim.
3. **Speed is honestly predicted:** The room model will NOT be faster than v3.
   It trades dispatch count for bandwidth.

---

## F. Assumptions and Breaking Conditions

### Assumption 1: Linearity holds
Matrix multiplication IS linear. This cannot break. Theorem 1 is unconditional.

### Assumption 2: A-matrices are approximately orthogonal
If A_i^T A_j is not small, Theorem 2's routing interpretation weakens (cross-talk
between adapter subspaces). But Theorem 1 still holds -- the sum is still the sum.
**Measured:** |cos| < 0.001 (Finding #3). Would need |cos| > 0.1 to meaningfully
affect routing (100x degradation from measured).

### Assumption 3: Hidden states have domain-specific direction
Theorem 2 requires that domain-specific tokens produce hidden states that
preferentially project onto the corresponding A-subspace. If hidden states are
isotropic (uniform projection onto all subspaces), routing weights are uniform
and routing accuracy drops to 1/N = 20%.
**Risk:** This is the main experimental question. Prior routing results (99.6%
with ridge regression, Finding #287) suggest hidden states DO have domain structure,
but the projection through A-matrices may not capture it.

### Assumption 4: bf16 precision suffices for W_combined
W_combined is a sum of N=5 rank-16 deltas. Each delta has entries of order
alpha * ||B|| * ||A|| / sqrt(d). For alpha=20, ||B||~1, ||A||~1/sqrt(d),
entries are O(20 / 2560) ~ O(0.008). In bf16 (precision 3.9e-3), these are
at the edge of representable precision. If bf16 quantization noise exceeds
the signal, PPL may degrade.
**Mitigation:** Compute W_combined in float32, store in bf16. Quantization
noise is O(eps_bf16 * max_val) per entry, which is much smaller than entries.

---

## G. Worked Example (d=4, r=2, N=2)

Two adapters with orthogonal A-matrices in R^4:

    A_0 = [[1, 0],     A_1 = [[0, 0],
            [0, 1],            [0, 0],
            [0, 0],            [1, 0],
            [0, 0]]            [0, 1]]

    B_0 = [[0.5, 0.3, 0.1, -0.2],    B_1 = [[0.4, -0.1, 0.6, 0.2],
            [0.1, -0.4, 0.2, 0.3]]            [-0.3, 0.5, -0.1, 0.4]]

alpha = 1.0

DeltaW_0 = alpha * B_0^T @ A_0^T
         = [[0.5, 0.1, 0, 0],
            [0.3, -0.4, 0, 0],
            [0.1, 0.2, 0, 0],
            [-0.2, 0.3, 0, 0]]

DeltaW_1 = alpha * B_1^T @ A_1^T
         = [[0, 0, 0.4, -0.3],
            [0, 0, -0.1, 0.5],
            [0, 0, 0.6, -0.1],
            [0, 0, 0.2, 0.4]]

W_combined = DeltaW_0 + DeltaW_1
           = [[0.5, 0.1, 0.4, -0.3],
              [0.3, -0.4, -0.1, 0.5],
              [0.1, 0.2, 0.6, -0.1],
              [-0.2, 0.3, 0.2, 0.4]]

For input h = [1.0, 0.2, 0.0, 0.0] (aligned with adapter 0's subspace):

Sequential: h @ DeltaW_0 + h @ DeltaW_1
  = [1*0.5+0.2*0.1, 1*0.3+0.2*(-0.4), 1*0.1+0.2*0.2, 1*(-0.2)+0.2*0.3]
    + [0, 0, 0, 0]
  = [0.52, 0.22, 0.14, -0.14] + [0, 0, 0, 0]
  = [0.52, 0.22, 0.14, -0.14]

Pre-summed: h @ W_combined
  = [1*0.5+0.2*0.3, 1*0.1+0.2*(-0.4), 1*0.4+0.2*(-0.1), 1*(-0.3)+0.2*0.5]
  = [0.56, 0.02, 0.38, -0.20]

Wait -- these are NOT equal. Let me recheck. The issue is that DeltaW = B^T @ A^T,
not A @ B as in the LoRA formula. Let me recalculate carefully.

**Correction:** In pierre.py, the LoRA computation is:
    y = base(x) + (x @ A) @ B * alpha

So the adapter delta in weight space is: DeltaW = A @ B (shape: in_features x out_features)
NOT B^T @ A^T. Let me redo:

    A_0 = [[1, 0],     shape: (4, 2) = (in_features, rank)
            [0, 1],
            [0, 0],
            [0, 0]]

    B_0 = [[0.5, 0.3, 0.1, -0.2],    shape: (2, 4) = (rank, out_features)
            [0.1, -0.4, 0.2, 0.3]]

DeltaW_0 = alpha * A_0 @ B_0 (shape: 4 x 4)
         = [[0.5, 0.3, 0.1, -0.2],
            [0.1, -0.4, 0.2, 0.3],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]

    A_1 = [[0, 0],
            [0, 0],
            [1, 0],
            [0, 1]]

    B_1 = [[0.4, -0.1, 0.6, 0.2],
            [-0.3, 0.5, -0.1, 0.4]]

DeltaW_1 = alpha * A_1 @ B_1
         = [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0.4, -0.1, 0.6, 0.2],
            [-0.3, 0.5, -0.1, 0.4]]

W_combined = DeltaW_0 + DeltaW_1
           = [[0.5, 0.3, 0.1, -0.2],
              [0.1, -0.4, 0.2, 0.3],
              [0.4, -0.1, 0.6, 0.2],
              [-0.3, 0.5, -0.1, 0.4]]

For input h = [1.0, 0.2, 0.0, 0.0]:

Sequential: h @ DeltaW_0 + h @ DeltaW_1
  h @ DeltaW_0 = [1*0.5+0.2*0.1, 1*0.3+0.2*(-0.4), 1*0.1+0.2*0.2, 1*(-0.2)+0.2*0.3]
               = [0.52, 0.22, 0.14, -0.14]
  h @ DeltaW_1 = [0, 0, 0, 0]
  Sum = [0.52, 0.22, 0.14, -0.14]

Pre-summed: h @ W_combined
  = [1*0.5+0.2*0.1+0*0.4+0*(-0.3),
     1*0.3+0.2*(-0.4)+0*(-0.1)+0*0.5,
     1*0.1+0.2*0.2+0*0.6+0*(-0.1),
     1*(-0.2)+0.2*0.3+0*0.2+0*0.4]
  = [0.52, 0.22, 0.14, -0.14]

EQUAL. The pre-summed and sequential outputs are identical.

Soft routing weights for h = [1.0, 0.2, 0.0, 0.0]:
  p_0 = h @ A_0 = [1.0, 0.2]    ||p_0|| = sqrt(1.04) = 1.020
  p_1 = h @ A_1 = [0.0, 0.0]    ||p_1|| = 0.0

  w_0 = 1.020 / (1.020 + 0.0) = 1.0
  w_1 = 0.0 / (1.020 + 0.0) = 0.0

Adapter 0 gets all the weight. Correct: h lives entirely in A_0's subspace.

For a mixed input h = [0.5, 0.0, 0.7, 0.0]:
  p_0 = h @ A_0 = [0.5, 0.0]    ||p_0|| = 0.5
  p_1 = h @ A_1 = [0.7, 0.0]    ||p_1|| = 0.7

  w_0 = 0.5 / 1.2 = 0.417
  w_1 = 0.7 / 1.2 = 0.583

Adapter 1 gets more weight because h projects more onto A_1's subspace.

---

## H. Complexity and Architecture Connection

### FLOPs per token
- Base forward pass: O(L * K * d^2) (ternary, fast)
- Room model addition: O(L * K * d^2) (one bf16 matmul per module)
- Total: approximately 2x base FLOPs

### Memory
- Base model: 1.18 GB (ternary packed)
- W_combined (all 210 modules): ~4.65 GB (bf16)
- Total: ~5.83 GB
- Fits in 48 GB with margin

### Dispatches
- v3: 2 * 210 = 420 dispatches (A matmul + B matmul per module)
- Room model: 210 dispatches (one W_combined matmul per module)
- Room model has 50% fewer dispatches but 256x more bandwidth per dispatch

### Architecture Interaction
The room model replaces RuntimeLoRA (factored A@B) with a single dense matrix
injection. It does NOT modify the base model -- the base BitLinear modules are
untouched. W_combined is applied as an additive residual:
    y = base(x) + x @ W_combined

This is compatible with any base architecture that uses linear projections.

---

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Linearity of matrix multiplication: x @ (M1 + M2) = x @ M1 + x @ M2.

2. **Which existing theorem(s) does the proof build on?**
   Distributive property of matrix multiplication over addition (linear algebra axiom).
   Grassmannian orthogonality of A-matrices (Finding #3, #126).

3. **What specific numbers does the proof predict?**
   - MSE between pre-summed and sequential output: exactly 0 in infinite precision, < 1e-6 in float32
   - Speed: 40-50 tok/s (bandwidth limited by 5.83 GB total)
   - Soft routing: domain accuracy >= 60% (requires hidden state domain structure)

4. **What would FALSIFY the proof (not just the experiment)?**
   Theorem 1 cannot be falsified -- it is a direct consequence of linearity.
   The proof would be wrong ONLY if matrix multiplication were not distributive,
   which contradicts the axioms of linear algebra.
   Theorem 2 can be falsified if hidden states show no domain-specific direction
   (isotropic projection). This would not invalidate the math, only the
   assumption about hidden state structure.

5. **How many hyperparameters does this approach add?**
   Count: 0. The room model has no new hyperparameters. W_combined is determined
   entirely by the existing adapters (alpha, A, B) with no new degrees of freedom.

6. **Hack check:** No stack of fixes. One operation: sum the deltas.
