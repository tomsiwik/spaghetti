# Structural Orthogonality of LoRA Experts: Mathematical Foundations

**Note:** This document presents an *empirical characterization* of LoRA expert
orthogonality scaling with dimension, supported by standard results from
high-dimensional probability. It does not contain new proofs. The theoretical
results cited (Sections 3.1--3.3) are standard; the empirical contribution is
measuring how gradient-trained LoRA adapters compare to these baselines.

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 to 1024 (micro) |
| d_ff | MLP intermediate dimension | 4d |
| L | Number of MLP layers | 4 |
| r | LoRA rank | 8 |
| alpha | LoRA scaling factor | r (= rank) |
| N | Number of experts | 2+ |
| V | Vocabulary size | 32 |
| T | Context length | 16 |
| D | Flattened delta dimension | L * 2 * d * d_ff |
| tau | Orthogonality threshold | 0.01 |
| theta | Principal angle between subspaces | [0, pi/2] |

## 2. LoRA Delta Structure

For each expert i and MLP layer l, two LoRA-adapted linear maps produce weight deltas:

    dW1_l^(i) = (alpha/r) * A1_l^(i) @ B1_l^(i)    shape: (d, d_ff)
    dW2_l^(i) = (alpha/r) * A2_l^(i) @ B2_l^(i)    shape: (d_ff, d)

where A matrices are random (Kaiming init, frozen) and B matrices are trained.

The full delta vector for expert i is the flattened concatenation:

    v_i = concat[vec(dW1_0), vec(dW2_0), ..., vec(dW1_{L-1}), vec(dW2_{L-1})]

    dim(v_i) = D = L * 2 * d * d_ff

Each delta has rank at most r (as a matrix), but the flattened vector v_i lives in R^D.

## 3. Random Subspace Geometry (Baseline)

*All results in this section are standard. Citations are provided for reference.*

### 3.1 Random Unit Vectors in R^D

For two independent uniform random unit vectors u, v in R^D
(Vershynin, "High-Dimensional Probability", Prop 3.4.6):

    E[<u,v>^2] = 1/D

    E[|<u,v>|] = sqrt(2 / (pi * D))

Concentration (sub-gaussian on the sphere):

    P[|<u,v>| >= t] <= 4 * exp(-t^2 * D / 16)

For D = 131,072 (d=64): E[|cos|] = 0.00220
For D = 8,388,608 (d=1024): E[|cos|] = 0.000138

### 3.2 Random Rank-r Matrices

For two random rank-r subspaces in R^d, the principal angles theta_1,...,theta_r
between them satisfy (for the maximum overlap):

    E[cos^2(theta_max)] ~ r/d

This gives the **random subspace bound** for LoRA cosines:

    E[|cos(v_i, v_j)|] <= sqrt(r/d)

This bound follows from the geometry of the Grassmannian manifold G(d, r).
Two rank-r matrices in R^{d x d_ff} can overlap in at most r dimensions
out of d, giving the r/d scaling.

### 3.3 Grassmann Packing Bound

The maximum number of r-dimensional subspaces in R^d with pairwise
cos(theta_max) < tau follows from dimension counting on the Grassmannian
manifold G(d, r) (dimension = r(d-r)):

    N_max ~ (d/r)^2

This is a scaling argument, not a tight bound. It captures the correct
dependence on d and r.

| d | r | N_max | Note |
|---|---|-------|------|
| 64 | 8 | 64 | Micro scale |
| 256 | 8 | 1,024 | |
| 896 | 16 | 3,136 | Qwen-0.5B |
| 4096 | 16 | 65,536 | Qwen-7B |

## 4. Gradient-Trained Adapters vs Random Baseline

### 4.1 Decomposition

Let the delta vector for expert i be decomposed as:

    v_i = v_base + v_domain_i + epsilon_i

where:
- v_base is the component from the shared base model dynamics (common to all)
- v_domain_i is the domain-specific gradient direction
- epsilon_i is training noise (SGD stochasticity)

This decomposition is conceptual (not a formal orthogonal projection).

### 4.2 Empirical Finding: Gradient Alignment Does NOT Help

Gradient-trained adapters are approximately 3-5x LESS orthogonal than random
subspaces (separation ratio = 0.18-0.36x). This means gradient training introduces
a small POSITIVE correlation between adapters through the shared v_base component.

However, this does NOT undermine the orthogonality guarantee because:

1. The shared component is small in absolute terms (cos < 0.006 at d=64)
2. Both gradient and random cosines are far below sqrt(r/d) (17-40x margin)
3. The dimensional scaling dominates the gradient correlation

**Corrected claim**: SOLE's orthogonality is DIMENSIONAL, not gradient-driven.

## 5. Empirical Scaling Laws

### 5.1 Cosine vs Dimension

Both fit power laws cos ~ a * d^beta:

    E[|cos|]_random ~ 0.103 * d^{-0.936}    (R^2 = 0.997)

    E[|cos|]_gradient ~ 0.118 * d^{-0.722}  (R^2 = 0.950)

Bootstrap 95% CI on gradient exponent:
    beta_gradient in [-0.939, -0.512]

The CI does NOT include -0.5 (the subspace bound slope), indicating that
gradient cosines decay faster than the worst-case subspace bound.

Random cosines decay as ~d^{-0.94}, close to the theoretical D^{-0.5}
expectation (since D ~ d^2, so d^{-1.0}). Gradient cosines decay
more slowly (~d^{-0.72}), reflecting the shared v_base component.

### 5.2 Critical Dimension

The critical dimension d_crit where E[|cos|] < tau:

Random bound:  d_crit = r / tau^2 = 80,000 (for r=8, tau=0.01)
Empirical:     d_crit = 64 (gradient cos already < tau at smallest tested d)

### 5.3 Note on the Tail Bound

The original version presented a modified tail bound with D_eff = D/c^2
where c = gradient_mean/random_mean. This is an ad hoc scaling argument,
not a rigorous derivation. At production scales the tail probability is
negligible regardless of the c correction factor.

## 6. N_max Scaling (Orthogonality Capacity)

The maximum number of nearly orthogonal experts scales as:

    N_max = (d / r)^2

At macro scale (d=4096, r=16): N_max = 65,536.

Since gradient-trained adapters are slightly less orthogonal than random
(by factor c ~ 3-5x), a conservative correction gives:

    N_max_effective ~ (d / r)^2 / c^2

Even conservatively (c=5): N_max_eff = 65,536 / 25 = 2,621 at d=4096.

## 7. Worked Example

d=256, d_ff=1024, L=4, r=8:

    D = 4 * 2 * 256 * 1024 = 2,097,152
    E[|cos|]_random_vector = sqrt(2/(pi*2,097,152)) = 0.000550
    sqrt(r/d) = sqrt(8/256) = 0.177
    N_max = (256/8)^2 = 1,024
    d_crit(random) = 8/0.01^2 = 80,000

Empirical (from experiment, 3 seeds, 5 pairs):
    E[|cos|]_gradient = 0.00167 (median: 0.00091)
    E[|cos|]_random_rank_r = 0.00054
    Separation ratio = 0.32x (gradient LESS orthogonal than random)

## 8. Assumptions

1. **B-only training with random frozen A**: Standard LoRA practice. A provides
   a random projection; B captures domain-specific information.

2. **Flattened vector cosine as interference metric**: Measures overlap in
   parameter space. A necessary condition for functional interference.

3. **Power law extrapolation**: Fitting cos ~ a*d^beta to 5 data points
   with bootstrap CI. The macro measurement at d=896 (cos=0.0002) is
   consistent with the fitted power law.

4. **Convergence quality**: All adapters train to virtually identical loss
   (~3.466 = log(V) = log(32)), indicating minimal domain specialization
   at this micro scale. The gradient cosine measurements therefore reflect
   early gradient directions rather than converged domain-specific features.
   This is a limitation: at macro scale with genuine specialization, the
   shared v_base component may be relatively smaller or larger.
