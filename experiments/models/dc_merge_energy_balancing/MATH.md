# DC-Merge SVD Energy Smoothing for Balanced Adapter Composition

## Type: Guided Exploration (Type 2)

**Paper:** DC-Merge (arXiv:2603.06242, Zhang et al., CVPR 2026)
**Prior findings:**
- Finding #270 -- Flat ternary spectra: individual B-matrix Gini 0.20-0.31, gap 1.003-1.018
- Brainstacks null-space validation: K687 PASS (cosine=0.026), K688 FAIL (legal forgetting 0.025)
- Finding #225 -- Near-lossless composition at N=5

**Proven framework:** DC-Merge's energy smoothing + cover space projection (Algorithm 1)
**Unknown parameter:** Whether smoothing helps for already-flat ternary spectra, and which
smoothing strategy (average vs linear) is optimal for this spectral profile.

## A. Failure Mode: Energy Imbalance in Composed Deltas

### Individual Adapter Spectra

Each ternary adapter has B-matrix singular values with moderate spread:
- Mean Gini coefficient: 0.27-0.29 (across 210 B-matrices per domain)
- Mean max/min singular value ratio: 5.8-6.4
- This is already relatively flat (Finding #270: gap 1.003-1.018 at rank-16 after scale)

### Composed Spectrum Amplification

When composing N=5 adapters via summation (delta_composed = sum_i scale_i * B_i^T @ A_i^T),
the composed delta has dramatically worse spectral properties:
- Composed Gini: 0.47-0.53 (vs 0.20-0.31 individual)
- Composed max/min ratio: 84-211 (vs 5.8-6.4 individual)
- Top-1 singular value captures 10-24% of total energy

**The disease:** Composition via summation amplifies energy concentration. Even though
individual adapters have flat spectra, their sum does not. A few singular directions
dominate the composed delta, meaning during model evaluation, those directions overwhelm
the weaker but potentially important knowledge encoded in smaller singular values.

## B. The Right Question

Not: "How do we prevent energy imbalance in composition?"
But: "What is the optimal energy distribution for the composed task vector such that
no domain's knowledge is overwhelmed by another domain's dominant directions?"

DC-Merge's answer (Eq. 12, arXiv:2603.06242): Replace the singular values of each
individual task vector with their mean before composing. This ensures each task vector
contributes equal energy per retained direction, preventing dominant task vectors from
drowning out smaller ones.

## C. Mathematical Foundation

### Theorem 1 (Energy Equalization -- from DC-Merge Eq. 12)

**Setup.** Let tau_i = U_i S_i V_i^T be the rank-r truncated SVD of task vector i
(i = 1, ..., T). Define the Gini coefficient of singular values:

G(S) = (sum_{j<k} |S_j - S_k|) / (r * sum_j S_j)

**Definition (Average Smoothing).** Replace S_i with S_bar_i where:

S_bar_i = (1/r * sum_j S_{i,j}) * 1_r  (all singular values set to their mean)

**Theorem.** Average smoothing gives G(S_bar) = 0 (perfect equality).

**Proof.** After smoothing, all r singular values equal the mean mu = (1/r) sum_j S_j.
The Gini numerator is sum_{j<k} |mu - mu| = 0. Hence G = 0. QED.

### Theorem 2 (Linear Smoothing Bound -- from DC-Merge Appendix E.4)

**Definition (Linear Smoothing).** Given singular values S_1 >= ... >= S_r and
parameter rho >= 1, define:

S_bar_j = total_energy * w_j,  where w_j = (rho - (rho-1)(j-1)/(r-1)) / sum_k w_k

This creates a linearly decreasing distribution with S_bar_1/S_bar_r = min(S_1/S_r, rho).

**Theorem.** Linear smoothing with parameter rho constrains:

G(S_bar) <= (rho - 1) / (rho + 1)

**Proof.** With linearly decreasing values a_j = rho - (rho-1)(j-1)/(r-1), the Gini
coefficient of a linear distribution on [1, rho] equals (rho-1)/(3(rho+1)) for continuous
distributions; for discrete samples it is bounded by (rho-1)/(rho+1).

For rho = 5: G <= 4/6 = 0.667. For rho = 2: G <= 1/3 = 0.333. QED.

### Theorem 3 (Energy Conservation)

**Theorem.** Both smoothing strategies preserve total singular value energy:
sum_j S_bar_j = sum_j S_j.

**Proof.** For average smoothing: sum_j (mean) = r * (1/r * sum_j S_j) = sum_j S_j.
For linear smoothing: S_bar_j = total_energy * w_j where sum w_j = 1, so
sum_j S_bar_j = total_energy * 1 = sum_j S_j. QED.

### Theorem 4 (Cover Space Directional Alignment)

**Setup.** After energy smoothing, DC-Merge constructs a shared orthonormal cover basis:
- U* = whiten([U_1, ..., U_T]), V* = whiten([V_1, ..., V_T])
- Project: M_i = U*^T @ delta_smooth_i @ V*
- Merge in cover space, project back: Delta_merged = U* @ M_merged @ V*^T

**Theorem (DC-Merge Eq. 9-11).** The cover space projection preserves all directional
information from all task vectors while the block-diagonal structural mask
M_mask = diag(1_{rxr}, ..., 1_{rxr}) eliminates cross-task directional interference.

**Proof sketch.** The whitened basis U* spans colspan(U_1, ..., U_T). Therefore
U*^T @ U_i has no information loss for any i. The block-diagonal mask zeros out
cross-block entries M_merged[i_block, j_block] for i != j, preventing task i's left
singular vectors from mixing with task j's right singular vectors. This is the
"directional consistency" guarantee. (Full proof: DC-Merge Section 3.2.)

## D. Quantitative Predictions

### Measured Baseline (Pre-smoothing)

From our profiling:
- Individual B-matrix Gini: 0.27-0.29 (mean across 210 B-matrices per domain)
- Composed (sum of 5 deltas) Gini: 0.47-0.53
- Composed max/min ratio: 84-211

### Predictions

**P1: Average smoothing reduces individual delta Gini to 0.**
Each task vector gets perfectly equalized singular values. G(S_bar) = 0 exactly (Theorem 1).

**P2: Average smoothing reduces composed Gini by >30%.**
The composed Gini depends on the interaction of equalized task vectors in their
respective subspaces. Since our Grassmannian A-matrices are orthogonal (mean pairwise
cos = 0.026, Finding K687), the composed sum of equalized deltas will have much more
uniform energy distribution. Predicted composed Gini: 0.15-0.30 (30-60% reduction
from 0.47-0.53 baseline).

**P3: Linear smoothing (rho=5) reduces composed Gini by >15%.**
With rho=5, individual Gini is bounded at 0.667 (Theorem 2) but our actual values
are already below that. The constraint primarily affects the composed spectrum.
Predicted composed Gini: 0.35-0.45.

**P4: Energy-balanced composition improves perplexity over raw sum.**
DC-Merge paper shows average smoothing gives 1-3% improvement on LoRA merging
benchmarks (Table 3). For ternary adapters with flatter individual spectra but
amplified composed imbalance, we predict similar or better improvement.
Predicted: 1-5% PPL improvement (conservative -- ternary spectra are already
relatively flat individually, so the gain comes from fixing *composed* imbalance).

**P5: DirSim between domains preserved or improved.**
Energy smoothing should not affect directional similarity since it only modifies
singular value magnitudes, not the singular vectors U_i, V_i.

## E. Assumptions and Breaking Conditions

1. **SVD truncation captures sufficient information.** At rank r=16 (matching LoRA rank),
   truncation is lossless since our deltas already have rank exactly 16 (product of
   rank-16 matrices B^T @ A^T). If violated: impossible in this setting.

2. **Grassmannian A-matrices ensure near-orthogonal subspaces.** Verified: mean pairwise
   cos = 0.026 (K687). If violated: cover space construction would be ill-conditioned.

3. **Composition via summation is the correct baseline.** We compare DC-Merge against
   raw sum. If a better non-SVD composition method exists, this comparison is incomplete.

4. **Perplexity reflects composition quality.** Caveat: r=0.08 correlation between PPL
   and task performance (project finding). We add generation quality evaluation as
   secondary measure.

## F. Worked Example (r=4, T=2)

Two task vectors with rank-4 SVD:
- Task 1: S_1 = [10, 3, 1, 0.5], Gini = 0.464, top-1 fraction = 0.904
- Task 2: S_2 = [8, 4, 2, 0.3], Gini = 0.363, top-1 fraction = 0.774

**Average smoothing:**
- S_bar_1 = [3.625, 3.625, 3.625, 3.625], Gini = 0
- S_bar_2 = [3.575, 3.575, 3.575, 3.575], Gini = 0
- Total energy preserved: sum(S_1) = 14.5, sum(S_bar_1) = 14.5. Check.

**Linear smoothing (rho=5):**
- Task 1: actual ratio = 10/0.5 = 20, clamped to rho=5
  - Weights: w = [5.0, 3.67, 2.33, 1.0], normalized: [0.417, 0.306, 0.194, 0.083]
  - S_bar_1 = 14.5 * w = [6.04, 4.43, 2.82, 1.21], Gini = 0.221
- Task 2: actual ratio = 8/0.3 = 26.7, clamped to rho=5
  - Same weights. S_bar_2 = 14.3 * w = [5.96, 4.37, 2.78, 1.19], Gini = 0.221

**Gini reduction (individual):**
- Task 1: 0.464 -> 0 (average) or 0.221 (linear). 100% and 52% reduction.
- Task 2: 0.363 -> 0 (average) or 0.221 (linear). 100% and 39% reduction.

## G. Complexity and Architecture Connection

**FLOPs per layer (DC-Merge):**
- SVD of (m, n) matrix: O(mn * min(m,n)), but we only need rank-r, so O(mnr)
- Energy smoothing: O(r) per task vector
- Cover basis construction: O((m + n) * r * T + (rT)^3) for whitening SVD
- Projection: O(m*n*rT) per task vector
- Total: O(T * m * n * r) dominated by projection

For our setting (m=n=2560, r=16, T=5):
- Per layer: ~524M FLOPs, but only done once at composition time
- Total for 30 layers x 7 keys: ~110B FLOPs = ~1 second on M5 Pro

**Memory:** O(m*n*T) for holding all task vectors simultaneously = 5 * 2560^2 * 4B = 131 MB
per weight key. Manageable for per-layer processing.

**Integration:** DC-Merge produces a single merged delta. This can be added to base weights
as a pre-merge (like current bf16 merge path) or factored back into LoRA form via SVD.

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Energy equalization via singular value smoothing ensures no task vector's singular
   directions dominate the composed result, making energy imbalance impossible by
   construction.

2. **Which existing theorem(s) does the proof build on?**
   DC-Merge Eq. 12 (average smoothing), Appendix E.4 (linear smoothing),
   SVD energy decomposition, Gini coefficient properties.

3. **What specific numbers does the proof predict?**
   P1: Individual Gini -> 0 (average). P2: Composed Gini reduction >30%.
   P3: Linear Gini reduction >15%. P4: PPL improvement 1-5%.

4. **What would FALSIFY the proof (not just the experiment)?**
   If average smoothing produces Gini != 0 for individual deltas -- this would
   violate basic arithmetic. This cannot happen.
   The experiment can fail if: composed Gini does not decrease (possible if orthogonal
   subspaces cause energy to redistribute unexpectedly), or PPL worsens (possible if
   energy equalization destroys task-specific knowledge encoded in the spectral shape).

5. **How many hyperparameters does this approach add?**
   1 (smoothing strategy choice: average/linear/none). For linear: +1 (rho).
   Average has 0 hyperparameters. The strategy choice is the Type 2 unknown.

6. **Hack check: Am I adding fix #N to an existing stack?**
   No. This is a standalone composition method that replaces raw summation.
   One mechanism (SVD energy smoothing + cover space projection), not stacked fixes.
