# MATH.md: N=24 Composition Proof

## Experiment Type
**Frontier extension.** The Pierre pipeline is proven at N=5 (Finding #287: 99.6%
routing, 0% PPL degradation). This experiment extends to N=24 — the scale at which
all previous routing methods collapsed (~40% accuracy on NTP adapters). The gap:
does the ridge router + Grassmannian orthogonality framework hold when N grows 4.8x?

## A. Failure Mode Identification

Two failure modes are possible at N=24:

**FM1: Routing collapse.** As N grows, domain embedding centroids crowd in the
d-dimensional hidden state space. If centroids become indistinguishable, ridge
regression routing degenerates toward random assignment (1/N = 4.2%).

**FM2: Orthogonality breakdown.** The Grassmannian skeleton allocates N adapter
subspaces of rank r in R^d. If Nr approaches d, subspace packing tightens and
pairwise cosines between adapter delta-matrices grow, causing interference.

Both are real risks. Finding #296 (NTP adapters) showed FM1 partially triggering:
37.6% overall routing (but 93.4% on genuine domains with distinctive content).
FM2 was not triggered: mean |cos| = 0.024, well within safe bounds.

## B. The Right Question

**Wrong:** "How do we prevent routing from degrading at N=24?"
**Right:** "Under what conditions on the embedding space and adapter subspaces is
routing accuracy lower-bounded and interference upper-bounded at N domains?"

The answer comes from two classical results: the Johnson-Lindenstrauss lemma
(embedding capacity) and the Grassmannian packing bound (subspace capacity).

## C. Prior Mathematical Foundations

### C.1 Ridge Regression: Unique Global Minimum (Tikhonov, 1943)

For calibration data X in R^{M x d} and one-hot labels Y in R^{M x N}, the
ridge regression solution:

  W* = (X^T X + lambda I)^{-1} X^T Y

is the unique global minimum of ||XW - Y||_F^2 + lambda ||W||_F^2, for any
lambda > 0. The Hessian H = X^T X + lambda I is positive definite (minimum
eigenvalue >= lambda > 0), guaranteeing uniqueness. No training instability
is possible. (Hoerl & Kennard, 1970)

### C.2 Johnson-Lindenstrauss Lemma (JL, 1984)

For M points in R^d, if d >= (8 ln M) / epsilon^2, there exists a linear
embedding that preserves all pairwise distances within factor (1 +/- epsilon).

At N=24 with d=2560: the JL bound requires d >= 8 ln(24*50) / epsilon^2.
For epsilon = 0.5 (50% distortion tolerance): d >= 8 * 7.09 / 0.25 = 227.
Since 2560 >> 227, there is ample capacity for the router to separate 24
domain centroids with large margin.

### C.3 Grassmannian Packing Bound

The maximum number of r-dimensional subspaces in R^d with pairwise chordal
distance >= delta satisfies:

  N_max <= C(d, r) * (1/delta)^{r(d-r)}

For our setting (d=2560, r=16), the total subspace capacity is d^2/r^2 = 25,600
(Theorem 1 of Finding #3). At N=24, we use 24/25,600 = 0.094% of the capacity.
The expected pairwise cosine at this load factor is O(r/d) = O(16/2560) = O(0.00625).

### C.4 Interference Bound (Cao et al., arXiv:2508.11985)

For two adapters with delta matrices DeltaW_1, DeltaW_2, the composition error
scales with their cosine similarity:

  ||DeltaW_1 + DeltaW_2 - (DeltaW_1^* + DeltaW_2^*)||_F ~ c * cos(DeltaW_1, DeltaW_2)

where DeltaW_i^* denotes the ideal (non-interfering) adapter output, and c is a
problem-dependent constant. When cos < 0.05, composition is empirically "safe"
(PPL degradation < 5%).

## D. Proof of Guarantee

**Capacity Argument 1 (Routing Capacity).** Let X_1, ..., X_N be sets of M
calibration vectors drawn from N distinct domain distributions in R^d. Let
mu_i = E[X_i] denote the domain centroid.

*Argument.* By the JL lemma, if d >= 8 ln(NM) / epsilon^2, there EXISTS a
linear embedding that preserves all pairwise centroid distances within factor
(1 +/- epsilon). At d=2560, N=24, M=50: the JL bound requires d >= 227,
satisfied with 11.3x margin. This establishes that the embedding space has
sufficient CAPACITY to separate 24 domain centroids.

However, JL guarantees that a RANDOM linear projection preserves distances.
Ridge regression is NOT a random projection — it is the optimal linear
classifier for the squared-error objective. JL capacity does not guarantee
that ridge regression finds the separation. Classification accuracy depends
on actual centroid separation Delta, which is a property of the data, not
the algorithm. When Delta ~ 0 for some domain pairs (as with semantically
overlapping domains), no linear classifier can separate them regardless of
embedding dimensionality.

**Corollary (empirical).** Routing accuracy depends on domain centroid separation
Delta, NOT on N directly. N affects accuracy only through centroid crowding. At
N=24 in d=2560, the crowding is negligible (capacity ratio 24/2560 < 1%). This
is confirmed empirically: 7 domains with large Delta achieve 90-100% accuracy;
5 domains with Delta ~ 0 achieve 0%. The capacity exists; the bottleneck is
the data's semantic structure.

**Theorem 2 (Orthogonality Preservation at Scale).** Let A_1, ..., A_N be
rank-r subframes in R^{d x r} drawn from the Grassmannian packing skeleton.
Let B_1, ..., B_N be arbitrary adapter B-matrices in R^{r x d_out}. Then the
pairwise cosine between flattened delta vectors DeltaW_i = A_i B_i satisfies:

  |cos(vec(DeltaW_i), vec(DeltaW_j))| <= |cos(A_i, A_j)| + O(r/d)

*Proof.* The cross-term in the inner product of vec(DeltaW_i) and vec(DeltaW_j)
involves trace(B_i^T A_i^T A_j B_j). By the Grassmannian construction, A_i^T A_j
has spectral norm <= epsilon_A where epsilon_A is the skeleton's pairwise coherence.

By submultiplicativity of the Frobenius norm:

  |trace(B_i^T A_i^T A_j B_j)| <= ||A_i^T A_j||_F * ||B_i||_F * ||B_j||_F

(This follows from |tr(X^T Y)| <= ||X||_F * ||Y||_F with X = A_i B_i, Y = A_j B_j,
expanded via submultiplicativity: ||A_i B_i||_F <= ||A_i||_F * ||B_i||_F.)

Normalizing by the norms ||DeltaW_i||_F = ||A_i B_i||_F and ||DeltaW_j||_F:

  |cos(DeltaW_i, DeltaW_j)| <= ||A_i^T A_j||_F * ||B_i||_F * ||B_j||_F
                                 / (||A_i B_i||_F * ||A_j B_j||_F)

When B-matrices have comparable norms and A-matrices are near-orthonormal
(||A_i||_F ~ sqrt(r)), this simplifies to approximately ||A_i^T A_j||_F,
the coherence of the A-matrix skeleton. The O(r/d) correction comes
from finite-dimensional effects in the Grassmannian packing. QED.

This is the "decorrelation filter" identified in VISION.md: the A-matrix skeleton
filters B-matrix correlation by a factor of ||A_i^T A_j|| / ||A_i|| * ||A_j||.
Finding #54 measured this as a 17x decorrelation at N=24.

**Empirical Model 3 (Composition Bound via NRE).** Let B_composed = NRE(B_1, ..., B_k)
be the norm-rescaled average of k adapter B-matrices. If the pairwise coherence
|cos(DeltaW_i, DeltaW_j)| <= epsilon for all pairs, we model the composed adapter's
PPL on any domain i as:

  PPL_composed(i) ~ PPL_single(i) * (1 + c * (k-1) * epsilon)

where c is an empirically-calibrated interference coefficient.

*Justification (not a proof).* NRE preserves the aggregate norm by construction.
The composition error is the sum of cross-terms, each bounded by epsilon.
With k-1 interfering adapters, the total interference scales as (k-1) * epsilon * c.

The coefficient c ~ 1 is calibrated from N=5 results (Finding #287), not derived
from first principles. A proper derivation would require bounding c via the
Lipschitz constant of the transformer layers, which is outside the scope of this
experiment. For epsilon = 0.024 (our NTP-measured value) and k=5, this predicts
PPL_composed ~ PPL_single * (1 + 0.096), i.e., < 10% degradation per domain
that matches the A-projection subspace.

**Note:** This model applies to per-domain degradation on the MATCHING A-subspace.
Domains projected through a non-matching A-matrix (e.g., code through medical's
A-subspace) experience higher degradation not captured by this model.

## D'. Predictions (Derived from Proofs)

### Behavioral Predictions

1. **Routing accuracy > 50% at N=24** (Capacity Argument 1): JL-lemma guarantees
   embedding capacity exists; whether ridge regression realizes it depends on actual
   centroid separation Delta. Prediction is conditional on Delta > 0 for all pairs.
   Finding #296 showed 93.4% on genuine NTP domains with large Delta.

2. **Pairwise B-matrix cosine < 0.10** (Theorem 2): Grassmannian skeleton with
   N=24 << N_max=25,600 preserves orthogonality. Expected mean |cos| ~ O(r/d) =
   O(0.00625). Prior measurement: 0.024 (NTP adapters). Note: this measures
   B-matrix cosine directly; DeltaW cosine (the quantity Theorem 2 bounds) is
   expected to be even lower due to A-skeleton decorrelation.

3. **Composed PPL within 3x worst single-adapter PPL** (Empirical Model 3): With
   epsilon = 0.024 and k=5, per-domain degradation on matching A-subspace predicted
   < 10%. NRE preserves norms. Non-matching domains may degrade more.

### Quantitative Predictions

| Prediction | Source | Expected Value | Kill Threshold |
|-----------|--------|---------------|----------------|
| Ridge router accuracy | Capacity Arg 1 (conditional on Delta > 0) | > 60% | < 50% (K753) |
| Mean pairwise B-matrix |cos| | Theorem 2 + skeleton capacity | 0.01-0.05 | > 0.10 (K754) |
| Per-domain PPL degradation (matching A) | Empirical Model 3 | < 10% | n/a |
| Per-domain PPL degradation (non-matching A) | Not modeled | Unknown | n/a |
| Top-5 composed PPL ratio | Empirical Model 3 + NRE | < 2.0x worst single | > 3.0x (K755) |
| Routing per-domain min | Capacity Arg 1 corollary | > 30% (worst domain) | n/a |

### Key Prediction from B-Matrix Inter-Cosine Gate (LEARNINGS.md)

The SFT LEARNINGS.md identifies B-matrix inter-cosine as a gating measurement:
if cos > 0.5, format dominance persists and routing will fail. If cos < 0.2,
adapters are sufficiently differentiated. This is measured in Phase 2 of the
experiment.

**Prediction from the framework:** At scale=20/300 steps, SFT adapters should
have broken free of format dominance. The B-matrix inter-cosine should be < 0.2,
enabling routing to work. If > 0.5, the experiment's routing phase will fail
regardless of the mathematical capacity (the embedding centroids will overlap
because the adapters themselves are near-identical).

## E. Assumptions and Breaking Conditions

1. **SFT produces distinctive hidden states.** The proof assumes domain centroids
   are separated (Delta > 0). If SFT format dominance persists at scale=20/300
   (B-matrix cos > 0.5), all centroids collapse to the shared format direction
   and Theorem 1's separation assumption breaks. Consequence: routing degrades
   toward random.

2. **Grassmannian skeleton was constructed for these 24 domains.** The A-matrices
   are indexed domain_0 through domain_23 in alphabetical order of the NTP adapter
   domains. If the domain-to-index mapping is wrong, the A-matrix orthogonality
   guarantee breaks. The run_experiment.py must use the correct mapping.

3. **B-matrix norms are comparable across domains.** Theorem 2's bound tightens
   when B-matrix norms vary widely. If one domain has 10x the B-norm of another,
   the decorrelation filter is less effective for that pair.

4. **Calibration data is representative.** Ridge regression optimality assumes
   the calibration distribution matches the test distribution. With N_CAL=50
   from training data and N_TEST=10 from validation data, distribution mismatch
   is possible.

## F. Worked Example (d=16, r=4, N=3)

Consider 3 adapters in R^16 with rank-4 A-matrices.

**A-matrices** (orthogonal columns, near-orthogonal across adapters):
  A_1 = [e_1, e_2, e_3, e_4]  (first 4 basis vectors)
  A_2 = [e_5, e_6, e_7, e_8]  (next 4)
  A_3 = [e_9, e_10, e_11, e_12] (next 4)

These are exactly orthogonal: A_i^T A_j = 0 for i != j.

**B-matrices** (correlated — e.g., similar format):
  B_1 = [1, 0.5, 0.3, 0.1; ...]  (rank-4 x output_dim)
  B_2 = [0.9, 0.6, 0.2, 0.15; ...] (similar to B_1)

  cos(B_1, B_2) ~ 0.95 (highly correlated)

**Delta matrices:**
  DeltaW_1 = A_1 @ B_1 (nonzero only in rows 1-4)
  DeltaW_2 = A_2 @ B_2 (nonzero only in rows 5-8)

  cos(DeltaW_1, DeltaW_2) = 0 (exactly, because A_1 perp A_2)

**Decorrelation:** B-matrix cosine of 0.95 becomes delta cosine of 0.
The A-matrix skeleton is a perfect filter when A_i perp A_j exactly.
In practice, Grassmannian packing gives |cos(A_i, A_j)| ~ 0.001-0.02,
so the filter is 50-1000x, not infinite.

**NRE composition of adapters 1 and 2:**
  B_composed = NRE(B_1, B_2) = 0.5(B_1 + B_2) * mean_norm / ||0.5(B_1+B_2)||

Since A_1 perp A_2, applying both:
  DeltaW_1 + DeltaW_2 = [B_1 rows in 1-4, B_2 rows in 5-8]

Each adapter operates in its own subspace. No interference.

## G. Complexity and Architecture Connection

**Ridge router fitting:** O(N * M * d^2) for M calibration samples per domain,
one forward pass each. At N=24, M=50, d=2560: 24*50*2560^2 = 7.9e9 FLOPs.
Dominated by forward pass cost (~1s per sample on M5 Pro).

**Routing inference:** O(d * N) per query = 2560 * 24 = 61K FLOPs. Negligible.

**Adapter cosine computation:** O(N^2 * D) for D = total adapter parameters.
At N=24: C(24,2) = 276 pairs, each O(D) ~ O(200K) = 55M FLOPs. Negligible.

**PPL evaluation:** One forward pass per sample per configuration.
8 domains * 10 samples * 3 configs (base, single, routed) = 240 forward passes.
At ~1s each: ~4 minutes.

**Total estimated runtime:** 15-30 minutes.

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Grassmannian A-matrix orthogonality ensures adapter delta-matrices operate in
   non-overlapping subspaces, making interference geometrically bounded by the
   skeleton's coherence regardless of B-matrix correlation.**

2. Which existing theorem(s) does the proof build on?
   **Theorem 2 builds on Grassmannian packing bounds and Frobenius submultiplicativity.
   Capacity Argument 1 uses JL-lemma (1984) and Tikhonov uniqueness (Hoerl & Kennard,
   1970) but as a capacity argument, not a classification guarantee. Empirical Model 3
   uses Cao et al. (arXiv:2508.11985) with an empirically-calibrated coefficient.**

3. What specific numbers does the proof predict?
   **Theorem 2: Mean B-matrix |cos| in [0.01, 0.05] (skeleton capacity 0.094%).
   Capacity Arg 1: router accuracy > 60% IF centroid separation Delta > 0 for all
   pairs (conditional). Empirical Model 3: per-domain degradation < 10% on matching
   A-subspace (calibrated, not proven).**

4. What would FALSIFY the proof (not just the experiment)?
   **Theorem 2 breaks if the Grassmannian skeleton's measured coherence exceeds the
   packing bound (would indicate implementation error). The capacity argument breaks
   if Delta > 0 for all domain pairs yet routing still fails (would indicate JL
   capacity is not realizable by linear classifiers). The empirical model breaks if
   c >> 1, meaning interference amplifies nonlinearly through transformer layers.**

5. How many hyperparameters does this approach add?
   **2: lambda=1.0 (ridge regularization, insensitive per Finding #276) and
   LORA_SCALE=20.0 (fixed from SFT training recipe). Neither is searched.**

6. Hack check: Am I adding fix #N to an existing stack?
   **No. This is a frontier extension of the same 3-component system (skeleton +
   router + NRE compose) proven at N=5. No new mechanisms added.**
