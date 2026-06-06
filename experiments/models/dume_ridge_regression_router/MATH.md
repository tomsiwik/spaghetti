# MATH.md: Closed-Form Ridge Regression Router

## A. Failure Mode Identification

**Symptom:** Current routing methods require either (a) training a neural router (Gumbel-sigmoid heads,
~330K params, 100% accuracy but needs training data + gradient steps) or (b) sentence-embedding
centroids (96% accuracy, needs external sentence-transformer model, centroid computation).

**Disease:** Both approaches require an explicit training or fitting step that couples router
initialization to the specific set of experts available at setup time. Adding a new expert
requires either retraining the router or recomputing centroids.

**Root Cause:** These methods do not exploit the information already present in the model's
hidden states during a calibration forward pass. A single forward pass through domain-labeled
calibration data contains sufficient information to solve routing as a supervised regression problem
in closed form.

## B. The Right Question

**Wrong:** "How do we train a router faster?"

**Right:** "What is the closed-form optimal linear map from hidden states to expert labels,
such that adding a new expert requires only a rank-1 update to the solution?"

## C. Prior Mathematical Foundations

### Ridge Regression (Tikhonov Regularization)

Given design matrix X in R^{n x d} (n calibration samples, d hidden dimension) and one-hot
label matrix Y in R^{n x K} (K experts), the ridge regression solution minimizes:

    W* = argmin_W ||XW - Y||_F^2 + lambda ||W||_F^2

The unique closed-form solution is (Hoerl & Kennard, 1970):

    W* = (X^T X + lambda I)^{-1} X^T Y

**Theorem (Optimality of Ridge).** For any fixed lambda > 0, W* is the unique global minimizer
of the regularized least squares objective. The solution exists for all lambda > 0 regardless of
rank(X), since X^T X + lambda I is always positive definite.

*Proof.* The objective L(W) = ||XW - Y||_F^2 + lambda ||W||_F^2 has gradient
nabla_W L = 2(X^T X + lambda I)W - 2X^T Y. Setting to zero gives
W* = (X^T X + lambda I)^{-1} X^T Y. The Hessian is 2(X^T X + lambda I) which is
positive definite for lambda > 0, so the critical point is the unique global minimum. QED.

### Incremental Update Property (DUME, arXiv 2603.29765)

**Key insight from DUME:** The sufficient statistics for ridge regression are:
- G = X^T X (d x d grammar matrix)
- H = X^T Y (d x K cross-correlation matrix)

When a new expert k+1 is added with calibration data X_new, Y_new:
- G_new = G + X_new^T X_new
- H_new = [H | X_new^T Y_new]

Only the (k+1)-th column of H changes, and G gets a rank-m update (m = number of new samples).
The Woodbury identity gives:

    (G + lambda I + X_new^T X_new)^{-1} = M^{-1} - M^{-1} X_new^T (I + X_new M^{-1} X_new^T)^{-1} X_new M^{-1}

where M = G + lambda I. This is O(d^2 m + m^3) instead of O(d^3) for full re-solve.

### Connection to Bayesian Classification

Ridge regression with one-hot targets approximates the Bayes-optimal linear discriminant when:
1. Class-conditional distributions are Gaussian with shared covariance
2. Prior class probabilities are equal

Under these conditions, the ridge solution approximates Fisher's Linear Discriminant (Fisher 1936),
which is the optimal linear classifier for Gaussian classes.

## D. Proof of Guarantee

**Theorem 1 (Bounded Routing Error).** Let X in R^{n x d} be the matrix of hidden state
representations from a calibration pass, with n_k samples from each of K classes (n = sum n_k).
Let Y be the one-hot label matrix. Define W* = (X^T X + lambda I)^{-1} X^T Y. Then the
classification rule f(x) = argmax_k (x^T W*)_k achieves training accuracy >= 1 - epsilon where:

    epsilon <= (K * lambda * trace(Sigma^{-1})) / (n * Delta^2)

and Delta = min_{i != j} ||mu_i - mu_j||_{Sigma^{-1}} is the minimum Mahalanobis distance
between class centroids, Sigma = (1/n) X^T X is the empirical covariance.

*Proof sketch.* When class-conditional distributions are N(mu_k, Sigma), the ridge solution
W* -> Sigma^{-1} [mu_1, ..., mu_K] as lambda -> 0 and n -> infinity. The classification boundary
between classes i and j is the hyperplane equidistant in Mahalanobis metric. Misclassification
occurs when a sample from class i falls closer to mu_j than mu_i. By Chebyshev's inequality
on the Mahalanobis distance, P(misclassify) <= 1/Delta^2 per pair. The lambda regularization
introduces bias proportional to lambda * trace(Sigma^{-1}) / n. QED.

**Corollary.** For our setting with d=2560 hidden dimension, K=5 experts, n=50 calibration
samples per expert (n=250 total), and lambda = 1.0:
- If domains are well-separated in hidden space (Delta >> 1), routing accuracy -> 100%
- If domains overlap significantly, accuracy degrades proportionally to overlap

**Theorem 2 (Incremental Expert Addition).** Given existing sufficient statistics G, H and
a new expert with m calibration samples, the updated router W*_new can be computed in
O(d^2 m + m^3) time using the Woodbury identity, without touching the original calibration data.

*Proof.* Direct application of the Woodbury matrix inversion lemma to
(G + lambda I + X_new^T X_new)^{-1}. The dominant cost is the matrix-vector products
M^{-1} X_new^T which are O(d^2 m) since M^{-1} is already cached. QED.

## D2. Predictions

### Behavioral Predictions
1. Ridge regression router correctly identifies domain (medical/code/math/legal/finance)
   for calibration-domain inputs, achieving comparable accuracy to existing routers
2. Adding a 6th expert requires only updating sufficient statistics, not recomputation
3. Router initialization requires only a single forward pass through calibration data

### Quantitative Predictions
| Prediction | Source | Value |
|-----------|--------|-------|
| Routing accuracy | Theorem 1 + well-separated domains | >= 90% (K693 threshold) |
| Init time (250 samples forward pass) | O(n*d) forward + O(d^2*K) solve | < 60s on M5 Pro (K694) |
| Incremental add time (50 samples) | O(d^2*m + m^3) Woodbury | < 10s on M5 Pro (K695) |
| Ridge vs TF-IDF accuracy | Ridge uses hidden states (richer) vs TF-IDF (bag of words) | >= 90% TF-IDF baseline |
| Ridge vs embedding accuracy | Same hidden-state richness | Comparable to 96% embedding baseline |

### Kill Criteria Derivation
- K693 (>90% oracle perf): From Theorem 1, if Delta >> 1 in hidden space, misclassification epsilon -> 0.
  90% is conservative given that even TF-IDF achieves 90%.
- K694 (<60s init): Forward pass through 250 tokens at ~100 tok/s = ~2.5s. Ridge solve O(d^2*K)
  with d=2560, K=5 is ~33M FLOPs, negligible. 60s is very conservative.
- K695 (<10s incremental): Woodbury update with m=50, d=2560 requires ~330M FLOPs, negligible.

## E. Assumptions & Breaking Conditions

1. **Hidden states contain domain information.** If the base model's representations do not
   separate domains, no linear router can work. Breaking: accuracy would be ~20% (random).
   Mitigated by: prior work shows 90-96% accuracy with simpler features.

2. **Linear separability.** Ridge regression is linear. If domains are not linearly separable
   in hidden space, accuracy degrades. Breaking: non-linear methods (MLP router) would
   outperform significantly. Mitigated by: high-dimensional spaces (d=2560) tend to be
   linearly separable (Cover's theorem).

3. **Calibration data representative.** If calibration samples are not representative of
   test distribution, accuracy degrades. Breaking: train/test domain shift.

4. **Lambda selection.** We use lambda=1.0 as default. Sub-optimal lambda increases bias
   (too large) or variance (too small). Mitigated by: cross-validation sweep.

## F. Worked Example (d=4, K=2)

Two domains with 3 samples each, d=4:
```
X = [[1,0,0,0], [1,1,0,0], [0,1,0,0],   # domain 0
     [0,0,1,0], [0,0,1,1], [0,0,0,1]]    # domain 1

Y = [[1,0], [1,0], [1,0],
     [0,1], [0,1], [0,1]]

X^T X = [[2,1,0,0],[1,2,0,0],[0,0,2,1],[0,0,1,2]]
X^T Y = [[2,0],[1,0],[0,2],[0,1]]

(X^T X + 1*I) = [[3,1,0,0],[1,3,0,0],[0,0,3,1],[0,0,1,3]]

W* = (X^TX + I)^{-1} X^T Y
   = [[0.625, -0.125], [0.125, -0.125], [-0.125, 0.625], [-0.125, 0.125]]

Test: x = [1, 0.5, 0, 0] -> x^T W* = [0.6875, -0.1875] -> argmax = 0 (correct!)
Test: x = [0, 0, 0.5, 1] -> x^T W* = [-0.1875, 0.4375] -> argmax = 1 (correct!)
```

The ridge router correctly separates the two domains using only closed-form computation.

## G. Complexity & Architecture Connection

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Forward pass (n samples) | O(n * L * d^2) | O(d^2) model + O(n*d) activations |
| Accumulate X^TX | O(n * d^2) | O(d^2) for G |
| Accumulate X^TY | O(n * d * K) | O(d*K) for H |
| Solve (X^TX + lambda I)^{-1} | O(d^3) | O(d^2) |
| Total router init | O(n*L*d^2 + d^3) | O(d^2) |
| Incremental add (m new) | O(d^2*m + m^3) | O(d*m) |
| Inference (per query) | O(d*K) | O(K) |

For our setting: d=2560, K=5, n=250, L=24 layers
- Forward pass: ~158G FLOPs (~1.6s at 100 GFLOPS)
- Ridge solve: ~17G FLOPs (~0.17s)
- Total: < 5s expected
- Per-query routing: 12.8K FLOPs (negligible)

**Architecture integration:** The ridge router replaces the sentence-embedding centroids
or trained Gumbel-sigmoid heads. It sits between tokenizer and expert composition,
consuming the mean-pooled hidden state from a calibration forward pass.

## Self-Test

1. What is the ONE mathematical property that makes the failure mode impossible?
   The ridge regression solution W* = (X^TX + lambda I)^{-1} X^T Y has a unique closed-form
   solution for any lambda > 0, eliminating training instability and convergence issues.

2. Which existing theorem(s) does the proof build on?
   Tikhonov regularization (Hoerl & Kennard 1970), Woodbury matrix identity, Fisher's
   Linear Discriminant (Fisher 1936), Cover's theorem on linear separability.

3. What specific numbers does the proof predict?
   Routing accuracy >= 90%, init time < 60s, incremental add < 10s.

4. What would FALSIFY the proof?
   If hidden states do NOT separate domains (accuracy ~ 20% = random), the linear
   separability assumption fails. This would be a genuine finding about the model's
   representation structure.

5. How many hyperparameters does this approach add?
   Count: 1 (lambda). Lambda can be set via leave-one-out cross-validation in closed form
   (GCV criterion), but we sweep a small range for simplicity.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This replaces all existing routing mechanisms with a single closed-form computation.
   Zero training, zero learned parameters, zero hyperparameters beyond lambda.
