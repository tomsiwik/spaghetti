# Minimax Grassmannian Packing: Mathematical Foundations

## 1. Setup and Notation

Inherits all notation from `micro/models/grassmannian_expert_init/MATH.md`.
Additional symbols:

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| mu_max(S) | Max coherence of configuration S | max_{i!=j} \|\|U_i^T U_j\|\|_F |
| mu_mean(S) | Mean coherence of configuration S | mean_{i!=j} \|\|U_i^T U_j\|\|_F |
| mu_W | Welch bound (lower bound on mu_max) | sqrt(r(Nr-d)/(d(Nr-r))) |

## 2. The Problem: Mean vs Worst-Case Optimization

### 2.1 What AP Optimizes (Pre-Experiment Hypothesis)

Standard AP's structural projection clips all off-diagonal blocks to a fixed
threshold mu_target:

    For each (i,j) with i != j:
      if ||G_{ij}||_F > mu_target:
        G_{ij} <- G_{ij} * (mu_target / ||G_{ij}||_F)

The hypothesis was that this uniform clipping minimizes sum-of-squares (mean)
but allows outlier pairs. The d=256 tail anomaly (max/mean = 9.4x in
post-training delta vectors) appeared to support this.

### 2.2 What AP Actually Does (Experimental Finding)

**AP converges to an equidistributed configuration where ALL pairwise
coherences are identical** (max/mean = 1.00x to float32 precision).

Empirical pre-training coherence (500 iterations):

| d   | N  | AP max | AP mean | max/mean |
|-----|-----|--------|---------|----------|
| 64  | 12  | 0.6030 | 0.6030  | 1.00x    |
| 128 | 20  | 0.3244 | 0.3244  | 1.00x    |
| 256 | 40  | 0.2265 | 0.2265  | 1.00x    |

**Why this happens:** The structural projection clips all blocks to the SAME
threshold mu_target. If any block exceeds the threshold, it gets clipped to
exactly mu_target. The spectral projection (eigendecomposition, rank truncation,
trace rescaling) redistributes energy equally because the constraint set is
symmetric under permutation of subspace indices. After sufficient iterations,
this converges to the unique permutation-symmetric fixed point where all
off-diagonal blocks have equal norm.

This is a known property of the AP algorithm for symmetric constraint sets
(Tropp et al. 2005). The AP fixed point for Grassmannian packing IS an
equidistributed frame (a generalization of a tight frame).

### 2.3 The Minimax Hypothesis Was Wrong

The hypothesis assumed mean and max optimization would produce different
configurations. In fact, for equidistributed configurations:

    mu_max = mu_mean = constant

So minimizing mean IS minimizing max, and vice versa. There is no mean-max
tradeoff at the AP fixed point because the fixed point is equidistributed.

## 3. The True Source of the Tail Anomaly

### 3.1 Pre-Training vs Post-Training Coherence

The d=256 tail anomaly occurs in post-training delta vectors, NOT in the
pre-training skeleton:

| Stage | Metric | max/mean ratio |
|-------|--------|---------------|
| Pre-training (skeleton, Gram matrix) | \|\|U_i^T U_j\|\|_F | 1.00x |
| Post-training (delta vectors) | \|cos(delta_i, delta_j)\| | 9.36x |

### 3.2 Why Post-Training Develops Tails

The post-training delta vector for expert i is:

    delta_i = vec(A_i @ B_i)     (across all layers)

The pairwise cosine is:

    cos(delta_i, delta_j) = <A_i @ B_i, A_j @ B_j>_F / (||A_i @ B_i|| * ||A_j @ B_j||)

With frozen A, this becomes:

    cos(delta_i, delta_j) = tr(B_i^T A_i^T A_j B_j) / (||A_i B_i|| * ||A_j B_j||)

The skeleton controls A_i^T A_j (bounded by mu_target). But the B matrices
are trained independently on different domain data. The product B_i^T (...) B_j
introduces domain-dependent correlation that the skeleton cannot control.

**Example:** If domains i and j share similar linguistic patterns, their
B matrices will point in similar directions, amplifying the A_i^T A_j overlap.
If domains are very different, B matrices will be uncorrelated, suppressing
the overlap. This creates the distribution tail.

### 3.3 Implication for Interference Control

The interference bound from the parent MATH.md:

    ||delta_W_i^T delta_W_j||_F <= (alpha/r)^2 * ||B_i|| * mu * ||B_j||

This bound uses max(mu) from the skeleton. But the ACTUAL interference is:

    ||delta_W_i^T delta_W_j||_F = (alpha/r)^2 * ||B_i^T (A_i^T A_j) B_j||_F

which depends on the ALIGNMENT between B_i, B_j and the principal vectors
of A_i^T A_j. The skeleton controls ||A_i^T A_j||_F but not the alignment.

To control worst-case interference, one must control either:
1. The B-matrix alignment (training-time regularization)
2. The principal angle structure of A_i^T A_j (more than just Frobenius norm)

## 4. Stochastic Refinement: Why It Failed

### 4.1 Algorithm

Post-AP stochastic local search on the Grassmannian:

    for iter = 1 to n_refine:
      (i*, j*) = argmax_{i!=j} ||U_i^T U_j||_F
      U_{i*}' = Retract(U_{i*} + step * TangentVector)
      if max_coherence(U') < max_coherence(U):
        accept U_{i*}' (greedy descent on L_inf)
      else:
        reject

### 4.2 Why 0% Acceptance

At the AP fixed point, all pairs have coherence mu_AP. Moving ANY frame
changes its coherence with ALL other N-1 frames. A random perturbation:
- Decreases coherence with some neighbors
- Increases coherence with others
- The NEW max is likely HIGHER than the old max (which equaled the mean)

This is because the equidistributed configuration is a SADDLE-FREE LOCAL
MINIMUM of the L_inf objective. Any perturbation of one frame creates at
least one new pair with coherence > mu_AP, increasing the max.

Formally: at the equidistributed point, the gradient of max coherence
with respect to any frame perturbation has rank >= 1, meaning there exists
no descent direction for the max. The Hessian is positive definite in the
tangent space at this point.

## 5. Revised Understanding of SOLE Interference Guarantees

### 5.1 What the Skeleton Guarantees (Proven)

The AP skeleton provides:
- **Equidistributed coherence:** All pairs have the same subspace overlap
- **Bounded coherence:** mu_AP ~ 2.8 * mu_W (Welch bound)
- **Zero drift:** Frozen-A locks experts to slots permanently

### 5.2 What the Skeleton Does NOT Guarantee (Open)

The skeleton does NOT control:
- **B-matrix alignment:** Which direction in the r-dimensional subspace
  the expert's B matrix points
- **Post-training tail behavior:** The max/mean ratio of delta vector
  cosines (observed 9.4x at d=256)
- **Domain-dependent amplification:** Similar domains amplify subspace
  overlap through aligned B matrices

### 5.3 Candidate Solutions (Not Tested)

To address post-training tails, possible approaches:
1. **B-matrix orthogonality regularization:** Add a term to the training
   loss that penalizes ||B_i^T B_j|| for expert pairs
2. **Norm-constrained B training:** Constrain ||B_i|| to prevent
   amplification of skeleton overlap
3. **Domain-aware slot assignment:** Assign similar domains to maximally
   distant skeleton slots (but skeleton is already equidistributed, so
   this gives no benefit)

## 6. Worked Numerical Example (Updated with Actual Results)

d=128, r=8, N=20:

    Welch bound: mu_W = 0.1147
    AP target: mu_target = 1.2 * 0.1147 = 0.1376

    Standard AP (500 iterations):
    - Max coherence: 0.3244 (equidistributed)
    - Mean coherence: 0.3244 (= max, ratio 1.00x)
    - Ratio max/Welch: 0.3244/0.1147 = 2.83x

    Minimax refinement (500 iterations):
    - Accepted moves: 0/500
    - Final max: 0.3244 (unchanged)
    - Conclusion: AP fixed point is locally minimax-optimal

    Post-training (8 experts, B-only training, frozen A):
    - delta vector mean |cos|: 0.00322 (both AP and minimax, identical)
    - delta vector max |cos|: 0.01232 (both AP and minimax, identical)
    - max/mean ratio: 3.83x (tail from B-matrix alignment, not skeleton)

## 7. Assumptions and Limitations

1. **AP equidistribution is a local property.** At larger N or different
   Nr/d ratios, AP may NOT produce perfectly equidistributed configurations,
   and minimax refinement could find improvements.

2. **Float32 precision.** The max/mean ratio of 1.00x is within float32
   precision. In exact arithmetic, small variations may exist.

3. **Greedy local search only.** The stochastic refinement is a greedy
   hill-climber. Global optimization methods (simulated annealing, basin
   hopping) might escape the local minimum. However, the equidistributed
   configuration is the expected global optimum for symmetric constraint sets.

4. **Micro scale.** N=12-40 is small. At N=500+, the AP fixed point may
   not be equidistributed, and the gap between mean and max may be larger.
