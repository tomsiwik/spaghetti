# Persistence-Guided Bridge Extraction: Sparse Low-Rank Correction

## Experiment Type: Guided Exploration (Type 2)

The mathematical framework (stability theorem, Corollary 2 vulnerability window) is
proven. The unknowns are empirical: (1) how many H1 features are actually lost (not
just vulnerable), and (2) can a low-rank bridge matrix targeting those directions
restore them.

## Step A: Diagnose the Disease

**Problem:** The dependency experiment (exp_persistence_diagram_diff) reported
"0/17,223 features lost" -- but this only checked H0 (connected components). For H1
(loops), the situation is dramatically different:

- 13,885/15,246 (91%) of H1 features are in modules where the vulnerability bound
  2*max||delta_i|| EXCEEDS median H1 persistence
- Example: layer_29 o_proj has median H1 persistence = 0.350 but vulnerability
  bound = 3.987 (11x ratio)
- The stability theorem guarantees features with persistence > 2*delta survive,
  but says nothing about features with persistence < 2*delta

**Root cause:** H0 features (connected components) have persistence ~30-58 and are
robust to perturbations of norm ~0.3-2.0. But H1 features (loops in the weight row
point cloud) have persistence ~0.2-0.8 and ARE in the vulnerability window at current
adapter scale. The previous experiment's "near-lossless" conclusion was an artifact of
only examining H0.

**The disease:** Low-rank composition can destroy H1 topological structure (loop
features in weight geometry) even at modest adapter scales. If these loops encode
functional relationships between weight directions, their destruction degrades the
model.

## Step B: The Right Question

NOT: "How do we prevent H1 feature loss from composition?"
RIGHT: "Given the set of H1 features destroyed by composition, what is the minimum-rank
correction matrix that restores them, and does this correction improve model quality?"

The answer involves two classical results:
1. The stability theorem tells us WHICH features are vulnerable
2. The Eckart-Young-Mirsky theorem tells us the optimal low-rank approximation to
   the correction needed

## Step C: Prior Mathematical Foundations

### Stability Theorem (Cohen-Steiner et al., 2007, Theorem 5.2)
Already proven in dependency experiment. Key corollary: features with persistence
in [0, 2*max||delta_i||] can be destroyed.

### Eckart-Young-Mirsky Theorem (1936)
**Theorem.** For any matrix M and target rank k, the best rank-k approximation in
Frobenius or operator norm is given by truncated SVD:

  M_k = U_k Sigma_k V_k^T = argmin_{rank(X)<=k} ||M - X||_F

This gives us the optimal low-rank bridge matrix.

### Persistence Feature Localization
From computational topology (Edelsbrunner & Harer, 2010, "Computational Topology"):
each feature in a Rips persistence diagram corresponds to a pair of simplices in the
filtration. For H1 features (loops), the birth simplex is a 2-simplex (triangle) at
some scale, and the death simplex is a 1-simplex (edge) at a larger scale. The VERTICES
of these simplices are specific rows of the weight matrix.

By identifying which rows participate in the birth/death simplices of lost H1 features,
we can target the bridge correction to only those rows.

### Partial Information Decomposition (Williams & Beer, 2010; arXiv:2411.07483)
The cited reference uses PID to decompose knowledge transfer into redundant, unique,
and synergistic components. Lost H1 features represent synergistic information -- they
encode relationships (loops) between multiple weight directions that no single adapter
captures.

## Step D: Proof of Guarantee (Bounded Restoration)

**Theorem 1 (Bridge Restoration Bound).**
Let W be a weight matrix with row point cloud P, and let W' = W + Delta be the
composed weight matrix with point cloud P'. Let S = {f_1, ..., f_m} be the set of
H1 features in Dgm(P) that are absent from Dgm(P') (destroyed by composition).

Define the residual R = W - W' = -Delta. Let R_k = U_k Sigma_k V_k^T be the rank-k
truncated SVD of R.

Then for the bridge-corrected weight matrix W'' = W' + R_k:
  d_B(Dgm(P), Dgm(P'')) <= max_i ||(R - R_k)_i||_2

where (R - R_k)_i is the i-th row of the rank-(n-k) residual.

*Proof.* The bridge-corrected point cloud is P'' = {w'_i + (R_k)_i} = {w_i - (R-R_k)_i}.
The identity correspondence between P and P'' has distortion:

  delta'' = max_i ||w_i - w''_i||_2 = max_i ||(R - R_k)_i||_2

By the Stability Theorem (Cohen-Steiner et al., 2007):
  d_B(Dgm(P), Dgm(P'')) <= delta''

By Eckart-Young-Mirsky, ||R - R_k||_F is minimized by the SVD truncation, so
delta'' is the best achievable per-row residual at rank k.  QED.

**Corollary 1.** The bridge matrix B = R_k = -Delta_k (rank-k truncated SVD of the
negative perturbation) is the optimal rank-k correction for topological restoration.

**Corollary 2 (Rank Budget).** To restore all features with persistence > p_min,
we need rank k such that max_i ||(R - R_k)_i||_2 < p_min/2. This determines the
minimum rank of the bridge matrix.

**Note:** This proof only guarantees that the BOTTLENECK DISTANCE to the original
is reduced. It does not guarantee that the specific destroyed features are restored
(the matched features may be different). However, for features well above the new
vulnerability threshold, the matching is stable.

## Step D (cont.): Predictions

### Quantitative predictions:

1. **P1: H1 features ARE lost at current scale.** For modules where
   2*max||delta_i|| > median H1 persistence (19/35 modules), the bottleneck
   distance for H1 should be non-trivial (comparable to feature persistence).
   We predict loss of H1 features in at least 10/35 modules.

2. **P2: Bridge at rank k reduces bottleneck distance.** For a bridge matrix
   of rank k, the new bottleneck distance satisfies:
   d_B(original, bridge-corrected) <= max_i ||(Delta - Delta_k)_i||_2
   At rank k = r = 16 (same as one adapter), this should reduce d_B by >= 50%.
   (K628 threshold)

3. **P3: Bridge rank < r suffices.** Since the perturbation Delta has rank
   <= 5*16 = 80 but its singular values decay rapidly (adapters are structured),
   a rank-k bridge with k < 16 should capture most of the correction.
   (K629 threshold)

4. **P4: PPL improvement on cross-domain inputs.** If H1 features encode
   functional relationships, restoring them should improve cross-domain PPL
   by >= 5%. (K630 threshold -- this is the weakest prediction since we have
   no proof that H1 features drive PPL.)

### What we do NOT predict (guided exploration unknowns):
- The exact number of H1 features lost (only that the vulnerability window
  contains them)
- Whether lost H1 features are functionally important (only that they are
  topologically destroyed)
- The optimal rank k for the bridge (only that k < r should suffice)

## Step E: Assumptions & Breaking Conditions

**A1: H1 features in the vulnerability window are actually destroyed (not just
relocated).** The stability theorem says they CAN be destroyed, not that they ARE.
If violated: bridge extraction targets non-existent damage, K628 fails because
d_B is already small for H1.

**A2: The SVD of Delta captures the directions of topological damage.** The
perturbation's top singular vectors may not align with the directions that cause
H1 feature loss.
If violated: bridge matrix has high rank but low topological restoration.

**A3: H1 features correlate with model quality.** This is unproven. If H1 loops
in weight geometry are noise, restoring them has no quality impact.
If violated: K628/K629 may pass but K630 fails.

**A4: 500-row subsample captures H1 structure.** H1 features require at least 3
points (triangle). With 500 rows, we sample ~20% of the full matrix.
If violated: H1 counts are underestimates.

## Step F: Worked Example (d=16, n=8)

Consider W in R^{8 x 4} with rows forming a figure-eight pattern in R^4:
```
W = [[1, 0, 0, 0],    # point A
     [0, 1, 0, 0],    # point B
     [-1, 0, 0, 0],   # point C
     [0, -1, 0, 0],   # point D (first loop A-B-C-D)
     [1, 0, 0.1, 0],  # point E (near A, second loop)
     [0, 0, 1, 0],    # point F
     [-1, 0, 0.1, 0], # point G (near C)
     [0, 0, -1, 0]]   # point H (second loop E-F-G-H)
```

This has 1 H0 feature (one component) and 2 H1 features (two loops).

Add rank-1 perturbation Delta = [[0.5, 0, 0, 0]] repeated for all rows.
This shifts all points by (0.5, 0, 0, 0). Pairwise distances unchanged.
d_B = 0. No features lost.

Now add non-uniform rank-1 Delta that collapses points A and E:
```
Delta = [[0, 0, 0.05, 0],   # A moves toward E
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, -0.05, 0],  # E moves toward A
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
```

max||delta_i|| = 0.05. The vulnerability bound = 0.1. If the H1 feature
encoding the second loop has persistence < 0.1, it can be destroyed.

Bridge correction: B = -Delta (rank 1, trivially). After correction,
W'' = W' + B = W. Perfect restoration. rank(B) = 1 < r.

## Step G: Complexity & Architecture Connection

**Computation:**
- SVD of Delta per module: O(n * d * min(n,d)) for full SVD, but we only need
  top-k singular vectors: O(n * d * k) via randomized SVD
- PH computation: O(n^3) per module (unchanged from dependency experiment)
- Bridge correction + re-PH: O(n * d * k) + O(n^3) per module
- Total: ~2x the dependency experiment cost (compute PH 3 times instead of 2)

**Memory:**
- Base weights: same as before (~1.7GB for 5 layers)
- Perturbation Delta: ~50MB per module at most
- Bridge matrix: rank-k, stored as U(n,k) * S(k) * V(k,d) ~ 2*n*k*4 bytes per module
- Total: well within 48GB

**Architecture connection:**
- At inference time, the bridge matrix B would be applied as an additional correction
  term: W'' = W + Delta + B = W + Delta + U_k S_k V_k^T
- This is equivalent to adding another LoRA adapter of rank k
- If k < r, it is strictly cheaper than adding another domain adapter
- The bridge adapter has no domain -- it captures cross-domain synergistic structure

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   The Eckart-Young-Mirsky theorem guarantees the rank-k SVD of the perturbation
   residual is the optimal low-rank correction, making the remaining topological
   distortion provably minimal for any rank budget.

2. **Which existing theorem(s) does the proof build on?**
   Algebraic Stability Theorem (Cohen-Steiner et al., 2007, Thm 5.2);
   Eckart-Young-Mirsky theorem (1936); Rips filtration from Rieck et al. (2018).

3. **What specific numbers does the proof predict?**
   - H1 features lost in >= 10/35 modules (P1)
   - Bridge at rank r=16 reduces d_B by >= 50% (P2, from Thm 1)
   - Bridge effective rank < 16 (P3, from adapter singular value decay)
   - PPL improvement >= 5% on cross-domain inputs (P4, weakest prediction)

4. **What would FALSIFY the proof (not just the experiment)?**
   The proof is correct (it follows from stability + Eckart-Young). It can be
   VACUOUS if: (a) no H1 features are actually lost (A1 fails), or (b) the
   SVD directions don't align with topological damage (A2 fails). The proof
   says the correction is optimal, but if no correction is needed, it's moot.

5. **How many hyperparameters does this approach add?**
   Count: 1 (bridge rank k). Derived from the proof: k is the minimum rank such
   that max_i||(R-R_k)_i|| < p_target/2, where p_target is the minimum persistence
   we want to protect. In practice, we sweep k and measure the restoration curve.

6. **Hack check:** This is a single mechanism (truncated SVD of the perturbation
   residual). No stacking of fixes.
