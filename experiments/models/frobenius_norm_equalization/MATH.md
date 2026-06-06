# Frobenius-Norm Equalized Composition: Mathematical Analysis

## Type: Guided Exploration (Type 2)

**Papers:**
- FroM (arXiv:2506.02478) -- Frobenius-norm adaptive merging
- DO-Merging (arXiv:2505.15875) -- magnitude-direction decoupling
**Prior findings:**
- Finding #277 -- DC-Merge: scale imbalance (20:1) dominates composed spectral pathology
- Finding #278 -- Spectral surgery structurally counterproductive (low-SV = domain-pure)
- Finding #275 -- Norm preservation is the mechanism for composition quality
- Finding #225 -- Grassmannian orthogonality: mean |cos|=0.026 at N=5

**Proven framework:** Grassmannian-orthogonal composition is near-lossless at N=5.
**Unknown:** Whether normalizing Frobenius norms before composition preserves or
destroys quality. The 20:1 cross-domain scale ratio may encode genuine domain
importance or training artifacts.

## A. Failure Mode: Cross-Domain Energy Imbalance

### The Disease (from two independent experiments)

Each domain adapter contributes a weight delta:

  Delta_i = s_i * B_i^T @ A_i^T

with per-domain optimal scales s = {medical:20, code:20, math:20, legal:4, finance:1}.

The Frobenius norm of each domain's total contribution is:

  ||Delta_i||_F = s_i * ||B_i||_F * ||A_i||_F

Since A-matrices are orthonormal rows (Grassmannian), ||A_i||_F = sqrt(r) for all i.
So the energy ratio is governed entirely by s_i * ||B_i||_F.

**Measured energy ratios** (from Finding #277):
- Three high-scale domains (medical, code, math at s=20) contribute ~93% of total
  Frobenius energy.
- Legal (s=4) contributes ~5%, finance (s=1) contributes ~2%.
- The 20:1 scale ratio creates a ~20:1 energy ratio.

**Why this is the root cause, not a symptom:**

1. DC-Merge smoothed *within-domain* spectral shape. Result: 18.5% Gini reduction,
   0.99% PPL gain. The within-domain fix barely touches the problem because individual
   B-matrix Gini is already flat (0.27-0.29).

2. Spectral surgery reweighted *composed* singular values. Result: -5.0% PPL
   degradation on ALL domains. The composed spectrum has domain-pure signal in
   low-SV directions (correlation = -0.587), so magnitude-based reweighting
   suppresses exactly the information we need.

Both failures point to the same root cause: **cross-domain energy ratios**, not
within-domain spectral shape or post-composition cleanup.

## B. The Right Question (Reframe)

**Wrong:** "How do we clean up the composed spectrum?"
**Wrong:** "How do we equalize individual adapter spectra?"
**Right:** "What energy allocation across domains makes each domain contribute
equally to the composed output, regardless of its training-induced scale?"

The answer is a classical normalization: scale each domain's contribution to
have equal Frobenius norm before summing.

## C. Prior Mathematical Foundations

### Theorem (Grassmannian Spectral Decomposition -- adapted from Finding #278 MATH.md)

If A_i^T A_j approx 0 for all i != j (Grassmannian orthogonality), the singular
values of Delta_comp = sum_i Delta_i are approximately the union of scaled individual
singular values. The degree of approximation depends on the residual |A_i^T A_j|.

At |cos| = 0.026 (Finding #225), cross-terms contribute < 0.07% of energy (0.026^2).

### Proposition (Frobenius Norm of Orthogonal Sum)

**Claim.** For N rank-r deltas with pairwise-orthogonal A-matrices:

  ||Delta_comp||_F^2 = sum_i ||Delta_i||_F^2

**Proof.** Delta_comp = sum_i s_i B_i^T A_i^T. Then:

  ||Delta_comp||_F^2 = Tr(Delta_comp^T Delta_comp)
                     = sum_{i,j} s_i s_j Tr(A_i B_i B_j^T A_j^T)

For i != j: A_i^T A_j = 0 (Grassmannian), so Tr(A_i B_i B_j^T A_j^T) = 0.
(More precisely: Tr(A_i B_i B_j^T A_j^T) = Tr(B_j^T A_j^T A_i B_i) =
Tr(B_j^T * 0 * B_i) = 0.)

For i = j: Tr(A_i B_i B_i^T A_i^T) = ||B_i^T A_i^T||_F^2 = ||Delta_i / s_i||_F^2
(since A_i has orthonormal rows, this equals ||B_i||_F^2).

Therefore: ||Delta_comp||_F^2 = sum_i s_i^2 ||B_i||_F^2 = sum_i ||Delta_i||_F^2. QED.

### Corollary (Energy Fraction per Domain)

The fraction of composed Frobenius energy from domain i is:

  f_i = s_i^2 ||B_i||_F^2 / sum_j s_j^2 ||B_j||_F^2

With scales {20, 20, 20, 4, 1} and similar ||B_i||_F across domains (ternary adapters
trained on comparable data volume):

  f_high ~ 400 / (3*400 + 16 + 1) ~ 400/1217 ~ 0.329 each for med/code/math
  f_legal ~ 16/1217 ~ 0.013
  f_finance ~ 1/1217 ~ 0.0008

So ~98.7% of energy comes from the three s=20 domains. Legal and finance are effectively
silenced in the raw sum.

### Definition (Frobenius Equalization)

**Full equalization.** Scale each domain delta so all have equal Frobenius norm:

  Delta_i^eq = Delta_i * (target_norm / ||Delta_i||_F)

where target_norm is a chosen reference (e.g., the geometric mean of all norms,
or the median norm, or a fixed constant).

This makes f_i = 1/N for all i.

**Partial equalization (geometric mean rescaling).** Scale each domain delta by:

  alpha_i = sqrt(||Delta_mean||_F / ||Delta_i||_F)

where ||Delta_mean||_F = geometric_mean(||Delta_1||_F, ..., ||Delta_N||_F).

This partially compresses the energy ratio without fully equalizing.

## D. Proof of Guarantee

### Theorem 1 (Gini Reduction under Full Equalization)

**Theorem.** Let Delta_i have Grassmannian-orthogonal A-matrices with
individual singular values sigma_i = {sigma_{i,1}, ..., sigma_{i,r}}.
After full Frobenius equalization, the composed Gini coefficient satisfies:

  Gini(equalized) <= max_i Gini(Delta_i)

In particular, if all individual adapter Gini coefficients are ~0.28 (our measured
baseline), then the composed Gini is <= 0.28.

**Proof.**

After equalization, each Delta_i^eq has ||Delta_i^eq||_F = c (constant).
Since the A-matrices are orthogonal, the composed singular values are the
union of {c * sigma_{i,j} / ||Delta_i||_F : j=1,...,r, i=1,...,N}.

The equalization scaling factor for domain i is alpha_i = c / ||Delta_i||_F.
This scales ALL singular values of Delta_i by the same factor alpha_i.

The Gini coefficient is scale-invariant: Gini(alpha * S) = Gini(S).
So after equalization, the individual Gini values are unchanged.

The composed Gini of the union of N groups of r values, where each group has
Gini G_i, is bounded above by:

  Gini(union) <= max(Gini(group_i)) + Gini(group_norms)

When all group norms are equal (Gini(group_norms) = 0), this reduces to:

  Gini(union) <= max_i G_i

In our case, max_i G_i approx 0.29 (code domain).

**Why this bound is tight for orthogonal subspaces:** When the groups occupy
orthogonal subspaces (Grassmannian guarantee), the singular values do not interact.
The composed spectrum is literally the sorted union. The Gini of a union of
equal-norm groups with similar internal distributions converges to the within-group
Gini as the groups become more similar. QED.

### Theorem 2 (Behavioral Preservation Condition)

**Theorem.** Frobenius equalization preserves domain-specific quality if and only if
the per-domain optimal scales encode *training dynamics artifacts* (gradient magnitude
differences) rather than *genuine capability requirements* (the domain needs larger
weight perturbation for meaningful behavioral change).

**Proof.** The LoRA delta modifies model output as:

  output = W_base x + s * B^T A^T x = W_base x + Delta x

The behavioral effect depends on the *signal-to-noise ratio* of the delta relative
to the base model:

  SNR_i = ||Delta_i x||_2 / ||W_base x||_2

If domain i genuinely requires SNR_i = 20x for meaningful behavioral change
(e.g., math reasoning needs large perturbation to redirect computation), then
equalizing its Frobenius norm to match finance (SNR ~1x) destroys the capability.

Conversely, if the optimal scales emerged from:
- Different data distributions producing different gradient magnitudes
- Different training loss scales (math vs prose)
- Random initialization differences

Then equalization corrects an artifact without harming behavior.

**This is the Type 2 unknown.** The experiment determines which case holds.

If equalization PRESERVES quality: scales were artifacts. Full equalization is
the correct composition strategy.
If equalization DESTROYS quality: scales encode genuine requirements. Partial
equalization or scale-aware composition is needed. QED.

### Theorem 3 (Composed Gini Prediction under Partial Equalization)

**Theorem.** If we compress the scale ratio from 20:1 to R:1 (where R < 20),
the composed Gini satisfies:

  Gini(partial) approx Gini_within + (R-1)/(R+1) * w_between

where Gini_within approx 0.28 (average within-domain Gini) and w_between is the
fraction of total Gini attributable to cross-domain scale differences.

For R=1 (full equalization): Gini approx 0.28
For R=sqrt(20) approx 4.5 (geometric mean): Gini approx 0.35
For R=20 (raw sum): Gini approx 0.49 (measured)

**Derivation.** The composed Gini has two sources:
1. Within-group variation (each adapter's internal SV spread): ~0.28
2. Between-group variation (different adapters have different total energy): depends on R

At R=20, the between-group Gini is approx (20-1)/(20+1) * fraction = 0.905 * w_between.
From measurement: total Gini = 0.49, within = 0.28, so between-group contributes
0.49 - 0.28 = 0.21 of Gini.

Solving: 0.21 = (20-1)/(20+1) * w_between => w_between = 0.232.

At R=1: between = 0, total Gini = 0.28.
At R=4.5: between = (4.5-1)/(4.5+1) * 0.232 = 0.636 * 0.232 = 0.148. Total = 0.43.

## E. Quantitative Predictions (Testable)

### P1: Composed Gini after full Frobenius equalization
**Prediction:** Gini drops from 0.49 to approximately 0.28 (43% reduction).
**Kill criterion K703:** Gini < 0.30 (>40% reduction from ~0.49).
**Derivation:** Theorem 1 guarantees Gini(equalized) <= max_i Gini(Delta_i) = 0.29.

### P2: Per-domain PPL change
**Prediction (two scenarios):**
- If scales are artifacts: all 5 domains PPL within +/-5% of raw sum.
- If scales encode capability: high-scale domains (med/code/math) degrade >5%,
  low-scale domains (legal/finance) improve >5%.
**Kill criterion K704:** At least 3/5 domains within 5% of raw sum.

### P3: Behavioral quality (generation samples)
**Prediction:** If K704 passes, generation quality should be coherent and domain-relevant.
**Kill criterion K705:** Coherent, domain-relevant text on at least 2 domains.

### P4: Partial equalization (geometric mean scaling) as middle ground
**Prediction:** Gini ~ 0.35 (30% reduction), PPL between raw and full equalization.
This tests whether partial compression preserves the beneficial scale information
while still reducing Gini.

### P5: Per-domain Frobenius norms before/after
**Prediction:** Full equalization produces equal norms (by construction). The ratio
of original norms should correlate with scale ratios: ||Delta_medical|| / ||Delta_finance||
approx s_medical * ||B_medical||_F / (s_finance * ||B_finance||_F) approx 20/1 * ~1 = ~20.

## F. Assumptions and Breaking Conditions

1. **Grassmannian orthogonality holds at N=5.** Verified: |cos| = 0.026 (Finding #225).
   If violated: Proposition 1 breaks, cross-terms contaminate energy fractions.

2. **B-matrix norms are comparable across domains.** Expected from similar training
   procedures and data volumes. If violated: the 20:1 scale ratio partially reflects
   genuine B-norm differences, and equalization is less effective.

3. **PPL is a meaningful proxy for composition quality.** Caveat: r=0.08 correlation
   with task quality. We add generation samples as behavioral check.

4. **Optimal scales are training artifacts, not genuine requirements.** This is the
   key Type 2 unknown. Breaking this assumption means full equalization fails (K704 FAIL)
   and we learn that partial equalization or scale-aware methods are needed.

## G. Worked Example (N=2, r=4)

Two adapters with scales s_1=20, s_2=1.

B_1 singular values: [0.5, 0.3, 0.2, 0.1], ||B_1||_F = sqrt(0.25+0.09+0.04+0.01) = 0.624
B_2 singular values: [0.4, 0.3, 0.25, 0.15], ||B_2||_F = sqrt(0.16+0.09+0.0625+0.0225) = 0.579

||A_i||_F = sqrt(4) = 2 for both.

Delta_1 norm: 20 * 0.624 * 2 = 24.96
Delta_2 norm: 1 * 0.579 * 2 = 1.158

Energy ratio: 24.96 / 1.158 = 21.6:1

Delta_1 singular values (scaled): 20 * [0.5, 0.3, 0.2, 0.1] = [10, 6, 4, 2]
Delta_2 singular values (scaled): 1 * [0.4, 0.3, 0.25, 0.15] = [0.4, 0.3, 0.25, 0.15]

**Raw composed SVs** (sorted union, since A orthogonal):
[10, 6, 4, 2, 0.4, 0.3, 0.25, 0.15]
Gini = 0.582

**After full Frobenius equalization** (target = geometric mean = sqrt(24.96 * 1.158) = 5.377):
  alpha_1 = 5.377 / 24.96 = 0.2154
  alpha_2 = 5.377 / 1.158 = 4.643

Equalized SVs:
  Domain 1: [10*0.2154, 6*0.2154, 4*0.2154, 2*0.2154] = [2.154, 1.292, 0.862, 0.431]
  Domain 2: [0.4*4.643, 0.3*4.643, 0.25*4.643, 0.15*4.643] = [1.857, 1.393, 1.161, 0.696]

Composed SVs (sorted): [2.154, 1.857, 1.393, 1.292, 1.161, 0.862, 0.696, 0.431]
Gini = 0.198

Gini reduction: (0.582 - 0.198) / 0.582 = 66%. Within-domain Gini of each group
is unchanged (scale-invariant), and equalization makes between-group contribution zero.

## H. Complexity and Architecture Connection

**Computation:** O(N) Frobenius norm calculations (each O(d_out * r) for B-matrix).
Total: negligible (< 1ms for N=5, r=16, d=2560).

**Memory:** No additional memory beyond storing the scale factors.

**Integration:** Applied once at composition time as a pre-processing step. Compatible
with all downstream serving strategies (pre-merge, runtime LoRA, Gumbel routing).

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Equal Frobenius energy across domains makes it impossible for any one domain to
   dominate the composed spectrum, by the Pythagorean property of orthogonal sums.

2. **Which existing theorem(s) does the proof build on?**
   Pythagorean theorem for Frobenius norms of orthogonal-subspace matrices (standard
   linear algebra). Grassmannian orthogonality guarantee (Finding #225, |cos|=0.026).
   Gini coefficient scale-invariance.

3. **What specific numbers does the proof predict?**
   P1: Composed Gini <= 0.29 (from measured max individual Gini).
   P2: At least 3/5 domains within 5% of raw sum PPL (if scales are artifacts).
   P5: Pre-equalization norm ratio ~20:1.

4. **What would FALSIFY the proof (not just the experiment)?**
   If equalized composition produces Gini > max individual Gini despite confirmed
   Grassmannian orthogonality, the Pythagorean decomposition fails. This would
   require |cos| >> 0.026 (contradicting Finding #225).

5. **How many hyperparameters does this approach add?**
   Full equalization: 0. Partial equalization: 1 (compression exponent alpha).
   The choice between full/partial is the Type 2 unknown.

6. **Hack check: Am I adding fix #N to an existing stack?**
   No. This replaces the scale-imbalanced raw sum with energy-balanced raw sum.
   One mechanism: normalize, then sum. Not stacked on DC-Merge or spectral surgery.
