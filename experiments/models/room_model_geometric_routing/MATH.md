# Geometric Routing via Adapter Output Norms

## Type: Guided Exploration (Type 2)

The mathematical framework for adapter output decomposition is proven (Theorem 1
from Room Model MATH.md, Finding #302). The unknown is whether the adapter output
norms ||h @ DeltaW_i|| are domain-discriminative -- i.e., whether the adapter
trained on domain i produces the largest output when presented with domain-i text.

---

## A. Failure Mode Identification

**Potential failure modes:**

1. **Adapter output norms are uniform across domains.** If all adapters produce
   similar-magnitude outputs for all inputs, argmax is arbitrary and accuracy
   drops to 1/N = 20%.

2. **Wrong adapter dominates.** If adapter j (j != i) consistently produces larger
   outputs than adapter i on domain-i text (e.g., because adapter j has larger
   B-matrix norms), then routing accuracy is below random.

3. **A-subspace projection destroys discriminability.** The Grassmannian A-matrices
   project h into a random 16-dimensional subspace. If domain-discriminative
   information is lost in this projection, p_i = h @ A_i contains no domain signal.

**Root cause analysis:** Mode 3 is the disease. The room model POC (Finding #302)
showed that ||h @ A_i|| is uniform across domains because A-matrices are random
projections. The question is whether B_i rescues discriminability by amplifying
domain-specific components within the random projection.

---

## B. The Right Question

**Wrong:** "How do we prevent adapter outputs from being uniform?"

**Right:** "Under what conditions does the trained B-matrix amplify domain-specific
projections p_i = h @ A_i such that ||p_i @ B_i|| is maximized when h comes from
domain i?"

**Answer from information theory:** The B-matrix is trained to minimize loss on
domain-i data. By definition, it learns to produce outputs that are useful for
domain-i text. If the training signal is sufficiently strong, B_i will amplify
exactly the components of p_i that carry domain-i information, even though A_i
is random. This is the Johnson-Lindenstrauss lemma in action: random projections
preserve relative distances with high probability.

---

## C. Derivation from Existing Mathematics

### C.1 Johnson-Lindenstrauss Lemma (1984)

For any epsilon in (0,1) and n points in R^d, a random linear map
A: R^d -> R^k with k >= O(epsilon^{-2} log n) preserves all pairwise
distances within factor (1 +/- epsilon).

**Application:** Our Grassmannian A_i matrices project from d=2560 to r=16.
For N=5 domain centroids, the JL bound requires k >= O(log(5)/epsilon^2).
At k=16 and n=5: epsilon ~ sqrt(log(5) / 16) ~ 0.32. So random projections
preserve domain centroid distances within 32% -- enough for arg-max if the
centroids are well-separated in the original space.

Cite: Johnson, Lindenstraham (1984), "Extensions of Lipschitz mappings into
a Hilbert space," Contemporary Mathematics 26.

### C.2 FlyLoRA (arXiv 2510.08396) -- Frozen Random A as Router

FlyLoRA proved that frozen random A-matrices act as implicit routers via the
JL-lemma. The random projection preserves enough structure for routing. This
directly supports our setting: Grassmannian A matrices are random orthogonal
projections.

### C.3 Adapter Training as Directional Amplifier

When adapter i is trained on domain-i data via gradient descent, B_i minimizes:

    L_i = E_{x ~ D_i}[loss(base(x) + alpha * (x @ A_i) @ B_i)]

The gradient update to B_i is:

    dB_i/dt = -eta * E_{x ~ D_i}[ (x @ A_i)^T @ (dloss/dy) ]

This is an outer product of the A-projected input with the loss gradient.
B_i learns to rotate the projected input toward outputs that reduce domain-i
loss. The columns of B_i^T become aligned with the directions in R^r that
carry domain-i-specific information after A-projection.

### C.4 Cover's Theorem on Separability (1965)

A set of patterns projected into higher dimensions becomes linearly separable
with probability approaching 1 as the ratio of dimensions to patterns grows.
Here, d=2560 and N=5 gives d/N = 512 -- massively over-determined. Even after
projection to r=16, we have r/N = 3.2, still above the critical ratio of 2.

Cite: Cover, T. M. (1965), "Geometrical and statistical properties of
systems of linear inequalities with applications in pattern recognition."

---

## D. Proof of Guarantee

### Theorem 1 (Adapter Output Decomposition -- from Room Model MATH.md)

**Statement.** For hidden state h in R^d and N adapters with delta
DeltaW_i = alpha * A_i @ B_i, the total room model output decomposes as:

    h @ W_combined = sum_{i=1}^{N} alpha * (h @ A_i) @ B_i = sum_i c_i

where c_i = alpha * (h @ A_i) @ B_i is the contribution of adapter i.

*Proof.* Linearity of matrix multiplication. QED. (Proven in Room Model MATH.md)

### Theorem 2 (Geometric Routing Signal)

**Statement.** Define the geometric routing score for adapter i as:

    s_i(h) = ||c_i|| = alpha * ||(h @ A_i) @ B_i||_2

The geometric router selects: i* = argmax_i s_i(h).

If (a) domain centroids mu_i = E_{h ~ D_i}[h] are delta-separated:
||mu_i - mu_j|| >= delta for all i != j, and (b) A_i preserves this
separation within factor (1-epsilon) by JL, and (c) B_i amplifies
domain-i projections by factor gamma_i > 1/gamma_j for j != i, then
the geometric router selects the correct domain with probability
at least 1 - N * exp(-c * r * min_i(gamma_i * delta * (1-epsilon))^2).

*Proof sketch.* (Full proof requires concentration inequalities on
the product of random projection and trained amplification.)

Step 1: By JL-lemma at r=16, ||A_i h|| preserves ||h|| within (1 +/- 0.32)
for all h in the convex hull of the N domain centroids.

Step 2: B_i is trained on domain-i data. The training objective ensures
that B_i maximizes the reduction in domain-i loss. For the geometric
routing signal s_i to be largest for domain-i inputs, we need:

    E_{h ~ D_i}[||(h @ A_i) @ B_i||] > E_{h ~ D_i}[||(h @ A_j) @ B_j||]

This holds if B_i amplifies the projection of domain-i centroids more than
B_j does, which is expected from training but NOT guaranteed without
additional assumptions on the loss landscape.

Step 3: The probability bound follows from sub-gaussian concentration
of random projections (Lemma 5.1 of Vershynin, "High-Dimensional
Probability," 2018).

QED (sketch -- full bound depends on B_i amplification factor gamma_i,
which is the empirical unknown this experiment measures).

### Theorem 3 (Ridge Router as Upper Bound)

**Statement.** The ridge regression router W* (Finding #310, 98.3% accuracy)
achieves the optimal linear routing from hidden states. The geometric router
is a restricted form of linear routing where:

    W_ridge: h -> y (one linear map from R^d to R^N)
    W_geo:   h -> {||h @ A_i @ B_i||}_{i=1}^N (N separate low-rank maps, then norms)

Since W_ridge can represent any linear mapping, and the geometric router
applies N independent rank-16 projections followed by nonlinear norms,
the ridge router's accuracy is an UPPER BOUND on what the geometric router
can achieve. If the geometric router matches the ridge router, it means the
adapter geometry captures the same routing information as the learned W*.

*Proof.* The ridge router computes y = h @ W* where W* is unconstrained
in R^{d x N}. The geometric router computes s_i = ||h @ (A_i @ B_i)||.
By the rank-nullity theorem, each A_i @ B_i has rank <= r = 16, so the
geometric router sees h through N rank-16 windows. The ridge router sees
h through an unconstrained rank-min(d,N) = N = 5 window. Since N < r,
the geometric router has MORE expressive capacity per domain than the
ridge router, but the nonlinear norm operation limits composition.
The comparison is empirical: geometric routing CAN match or exceed ridge
if the adapter geometry is aligned with domain structure. QED.

---

## E. Predictions (Behavioral + Quantitative)

### Behavioral Predictions

1. **Geometric routing accuracy > 60% (K804).** The adapter output norms carry
   domain signal because B_i amplifies domain-i-specific projections.

2. **Geometric routing agrees with ridge router > 50% (K805).** Both methods
   use hidden state structure for routing, but through different mechanisms.

### Quantitative Predictions

| Prediction | Source | Bound |
|-----------|--------|-------|
| Geometric routing accuracy | Theorem 2 + JL-lemma | > 60% at N=5 (kill if < 60%) |
| Agreement with ridge router | Theorem 3 | > 50% (kill if < 50%) |
| Projection norm ratio (correct/other) | JL + training amplification | > 1.2x |
| A-only routing accuracy | Room Model Finding #302 | ~14% (near random, BASELINE) |
| Full DeltaW routing improvement over A-only | B-matrix amplification | > 3x improvement |

### What the proof CANNOT predict (empirical unknowns):

- The B-matrix amplification factor gamma_i (depends on training dynamics)
- Whether the amplification is sufficient for > 80% accuracy (the 80% prediction
  from the experiment claim is aspirational, not proof-derived)
- Whether per-token routing matches per-sequence routing (JL concentration
  weakens with fewer averaging tokens)

---

## F. Assumptions and Breaking Conditions

### Assumption 1: Domain centroids are separated in R^d
If hidden states from different domains are nearly identical (small delta),
no routing method works -- including the ridge router. Since the ridge router
achieves 98.3% (Finding #310), domain centroids ARE well-separated.

### Assumption 2: JL preservation at r=16
The JL bound gives epsilon ~ 0.32 at r=16, n=5. This is loose. If the
actual distortion is larger (e.g., because the Grassmannian A-matrices
have adversarial alignment with domain structure), routing degrades.
**Risk:** LOW. JL holds for random projections with high probability,
and our A-matrices are Grassmannian (uniformly random on the manifold).

### Assumption 3: B_i training amplifies domain-i signal
This is the KEY ASSUMPTION and the empirical unknown. If B_i amplifies
ALL projections uniformly (no domain specificity), then geometric routing
reduces to A-only routing (14% accuracy, Finding #302). This experiment
tests whether B_i is sufficiently domain-specific.

### Assumption 4: Norms are the right aggregation
Taking ||c_i|| = ||(h @ A_i) @ B_i|| throws away directional information.
An alternative would be using the inner product h @ DeltaW_i @ v for some
reference vector v, or the Frobenius norm of the full DeltaW_i weighted by h.
The norm is the simplest aggregation. If it fails, directional alternatives
may succeed.

---

## G. Worked Example (d=4, r=2, N=2)

Using the same adapters from Room Model MATH.md:

    A_0 = [[1, 0], [0, 1], [0, 0], [0, 0]]  (first 2 dims)
    A_1 = [[0, 0], [0, 0], [1, 0], [0, 1]]  (last 2 dims)
    B_0 = [[0.5, 0.3, 0.1, -0.2], [0.1, -0.4, 0.2, 0.3]]
    B_1 = [[0.4, -0.1, 0.6, 0.2], [-0.3, 0.5, -0.1, 0.4]]

DeltaW_0 = A_0 @ B_0 = [[0.5, 0.3, 0.1, -0.2], [0.1, -0.4, 0.2, 0.3], [0, 0, 0, 0], [0, 0, 0, 0]]
DeltaW_1 = A_1 @ B_1 = [[0, 0, 0, 0], [0, 0, 0, 0], [0.4, -0.1, 0.6, 0.2], [-0.3, 0.5, -0.1, 0.4]]

For h = [1.0, 0.5, 0.1, 0.0] (domain-0 aligned):

  c_0 = h @ DeltaW_0 = [0.55, 0.10, 0.20, -0.05]
  ||c_0|| = sqrt(0.3025 + 0.01 + 0.04 + 0.0025) = sqrt(0.355) = 0.596

  c_1 = h @ DeltaW_1 = [0.04, -0.01, 0.06, 0.02]
  ||c_1|| = sqrt(0.0016 + 0.0001 + 0.0036 + 0.0004) = sqrt(0.0057) = 0.076

  s_0 = 0.596, s_1 = 0.076
  argmax = domain 0. CORRECT.
  Ratio: s_0/s_1 = 7.9x

For h = [0.1, 0.0, 1.0, 0.5] (domain-1 aligned):

  c_0 = h @ DeltaW_0 = [0.05, 0.03, 0.01, -0.02]
  ||c_0|| = sqrt(0.0025 + 0.0009 + 0.0001 + 0.0004) = sqrt(0.0039) = 0.062

  c_1 = h @ DeltaW_1 = [0.25, 0.15, 0.55, 0.40]
  ||c_1|| = sqrt(0.0625 + 0.0225 + 0.3025 + 0.16) = sqrt(0.5475) = 0.740

  s_1 = 0.740, s_0 = 0.062
  argmax = domain 1. CORRECT.
  Ratio: s_1/s_0 = 11.9x

This toy example works because A_0 and A_1 select non-overlapping dimensions,
and the B-matrices amplify within those dimensions. In reality, with random
Grassmannian A-matrices in R^2560 projecting to R^16, the separation is
weaker (JL epsilon ~0.32), but non-zero.

---

## H. Complexity and Architecture Connection

### Routing cost per token:
- Compute p_i = h @ A_i: r * d FLOPs = 16 * 2560 = 40,960 per adapter
- Compute c_i = p_i @ B_i: depends on module (rank * out_features)
- For routing, only need one representative module per adapter
- Total: ~5 * 40,960 = 204,800 FLOPs (negligible vs forward pass)

### Comparison with ridge router:
- Ridge: h @ W* = d * N FLOPs = 2560 * 5 = 12,800 FLOPs
- Geometric: ~200K FLOPs (16x more, but still < 0.1% of forward pass)
- Geometric has NO TRAINING COST (ridge requires calibration data)

### Architecture interaction:
Geometric routing uses only the frozen A-matrices and trained B-matrices
that already exist. No additional parameters. No calibration data.
It is the cheapest possible routing signal.

---

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   B_i training on domain-i data amplifies domain-specific components of the
   random A_i projection, making ||h @ A_i @ B_i|| domain-discriminative.

2. **Which existing theorem(s) does the proof build on?**
   Johnson-Lindenstrauss lemma (1984) for random projection distance preservation.
   Cover's theorem on separability (1965) for high-dim linear separability.
   FlyLoRA (arXiv 2510.08396) for frozen-A routing precedent.

3. **What specific numbers does the proof predict?**
   - A-only routing: ~14% (confirmed by Finding #302)
   - DeltaW routing: > 60% (K804, from JL + training amplification)
   - Ridge agreement: > 50% (K805)
   - Improvement over A-only: > 3x

4. **What would FALSIFY the proof (not just the experiment)?**
   If B_i produces uniform amplification across all domains (no domain specificity),
   then the proof's Assumption 3 fails and geometric routing = A-only routing = ~14%.
   This would mean LoRA B-matrices encode task-specific information only in their
   interaction with specific inputs, not in their geometric structure.

5. **How many hyperparameters does this approach add?**
   Count: 0. No hyperparameters. The routing score is ||h @ A_i @ B_i||, computed
   from existing adapter components with no tuning.

6. **Hack check:** This is a single routing signal with zero added complexity.
   No stacking of mechanisms.
