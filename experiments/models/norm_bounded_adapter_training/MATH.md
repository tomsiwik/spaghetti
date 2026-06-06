# Norm-Bounded Adapter Training: Mathematical Analysis

## Type: Guided Exploration (Type 2)

**Papers:**
- NB-LoRA (arXiv:2501.19050) -- singular value bounding during LoRA training
- DeLoRA (arXiv:2503.18225) -- magnitude-direction decoupling for LoRA
- DO-Merging (arXiv:2505.15875) -- direction-only merging, Frobenius normalization

**Prior findings:**
- Finding #279 -- Frobenius equalization: 50% log-compression ceiling. Full
  equalization kills high-scale domains (+18.5% med, +16.2% math).
- Finding #281 -- Fisher importance reduces to rescaled Frobenius norms (rho=1.0).
- Finding #277 -- DC-Merge: within-domain SV smoothing is wrong variable.
- Finding #278 -- Spectral surgery: structurally inverted for Grassmannian composition.

**Proven framework:** Frobenius norm governs composed energy allocation.
Grassmannian orthogonality decouples domains (|cos|=0.026).

**Unknown:** Can training-time norm constraints produce scale-balanced adapters
that retain domain capability? What norm bound preserves convergence while
equalizing cross-domain Frobenius energy?

---

## A. Failure Mode: Training-Induced Scale Imbalance

### The Disease

Current adapter training uses a uniform scale factor s (set to 20 for
medical/code/math, 4 for legal, 1 for finance) with unconstrained B-matrix
growth. The LoRA delta is:

    Delta_i = s_i * B_i^T @ A_i^T

The Frobenius norms of B-matrices converge to similar values across domains
(||B_i||_F in [29.1, 31.5], 8% spread, Finding #279). The entire 400x energy
ratio between domains comes from s_i^2:

    ||Delta_med||_F^2 / ||Delta_fin||_F^2 = (20^2 * 31.3^2) / (1^2 * 29.1^2)
                                           = 400 * 1.158 = 463

This scale imbalance is a TRAINING ARTIFACT:
1. B-norms converge to ~30 regardless of domain or scale (driven by optimizer
   dynamics, learning rate, STE gradient magnitude)
2. The per-domain scale s_i was hand-tuned for individual adapter quality
3. No constraint ensures that s_i * ||B_i||_F is comparable across domains

Post-hoc equalization (Finding #279) can compress the ratio but has a ceiling:
50% log-compression yields Gini 0.267 (from 0.490) because scales encode a
DUAL SIGNAL -- part artifact, part genuine capability requirement. Post-hoc
methods cannot distinguish these components.

### Why the fix must be at training time

Post-hoc methods operate on the product s_i * ||B_i||_F after training.
They see a single number and cannot determine how much is artifact vs signal.

At training time, we can impose: "learn B_i such that ||B_i||_F = tau for
all domains." This forces the optimizer to encode ALL domain-specific
information in the DIRECTION of B, not the magnitude. The training loss
provides the feedback signal to distribute information between magnitude
and direction optimally under the constraint.

---

## B. The Right Question (Reframe)

**Wrong:** "How do we equalize adapter norms after training?"
**Wrong:** "How do we find better post-hoc scaling factors?"
**Right:** "What constraint on B during training makes cross-domain scale
imbalance IMPOSSIBLE while allowing the optimizer to encode full domain
expertise in B's direction?"

The answer draws from constrained optimization theory: project onto the
feasible set after each gradient step (projected gradient descent), or add
a penalty that makes leaving the feasible set expensive (weight decay).

---

## C. Prior Mathematical Foundations

### Theorem (Projected Gradient Descent Convergence, Bertsekas 1999)

For a differentiable function f on a closed convex set C, the projected
gradient descent iteration:

    theta_{t+1} = Proj_C(theta_t - eta * grad f(theta_t))

converges to a stationary point of the constrained problem min_{theta in C} f(theta),
provided eta is sufficiently small and f satisfies standard smoothness conditions.

The Frobenius ball C = {B : ||B||_F <= tau} is closed and convex.
The projection is:

    Proj_C(B) = B * min(1, tau / ||B||_F)

This is simply clipping the norm.

### Proposition (Weight Decay as Lagrangian Relaxation)

Adding lambda * ||B||_F^2 to the training loss is the Lagrangian relaxation
of the constrained problem:

    min L(B)  subject to  ||B||_F^2 <= tau^2

The Lagrange multiplier lambda implicitly sets the effective norm constraint.
Unlike projection, this is a soft constraint: it pushes B toward smaller
norms without hard clipping.

### Theorem (Equivalence of Scale and Norm, for Grassmannian LoRA)

For LoRA with frozen orthonormal A and trainable B:

    ||Delta||_F = s * ||B||_F * ||A||_F = s * ||B||_F * sqrt(r)

where r is the rank. Therefore, constraining ||B||_F to tau_B is
equivalent to constraining ||Delta||_F to s * tau_B * sqrt(r).

**Key insight:** If we set s = 1 (uniform scale) and constrain
||B_i||_F = tau for all domains, then:

    ||Delta_i||_F = tau * sqrt(r)  for all i

This makes the energy ratio EXACTLY 1:1 by construction.

### NB-LoRA Connection (arXiv:2501.19050)

NB-LoRA bounds individual singular values of the LoRA adaptation matrix.
For rank-r LoRA with B in R^{r x d_out}, bounding each sigma_k(B) <= sigma_max
implies:

    ||B||_F = sqrt(sum sigma_k^2) <= sqrt(r) * sigma_max

This gives a Frobenius bound as a consequence of per-SV bounds. NB-LoRA's
approach is stricter than Frobenius bounding (it controls spectral shape),
but Frobenius bounding is simpler and sufficient for our goal (cross-domain
energy equalization).

---

## D. Proof of Guarantee

### Theorem 1 (Norm-Bounded Composition Energy Equalization)

**Theorem.** Let N domain adapters be trained with frozen orthonormal
A-matrices from a Grassmannian skeleton, uniform scale s=s_0, and the
norm constraint ||B_i||_F <= tau for all i. Then the composed delta
Delta_comp = sum_i s_0 * B_i^T A_i^T satisfies:

(a) Each domain's energy fraction is bounded:
    f_i = ||Delta_i||_F^2 / ||Delta_comp||_F^2 in [1/(N*R^2), R^2/N]

    where R = max_j ||B_j||_F / min_j ||B_j||_F is the realized norm ratio.

(b) If all adapters saturate the constraint (||B_i||_F = tau), then
    f_i = 1/N exactly.

(c) **[CONJECTURE -- EMPIRICALLY FALSIFIED]** The Gini coefficient of composed
    singular values was conjectured to satisfy:
    Gini(composed) <= max_i Gini(B_i) + Gini_between(||B_1||_F,...,||B_N||_F)

    where Gini_between = 0 when all norms are equal.

    **Status:** Falsified by Strategy C measurement (see Addendum). The standard
    Gini decomposition (Pyatt 1976) includes an overlap/interaction term that
    this bound omits. The union of N groups of SVs with different spectral shapes
    produces cross-group interaction Gini not captured by within-group max + between-group Gini.

*Proof.*

(a) With Grassmannian orthogonality (A_i^T A_j approx 0):

    ||Delta_comp||_F^2 = sum_i s_0^2 ||B_i||_F^2 * r    (Pythagorean, proven in #279 MATH.md)

    So f_i = ||B_i||_F^2 / sum_j ||B_j||_F^2.

    With ||B_i||_F in [tau/R_0, tau] (where R_0 = max/min ratio under the constraint):

    f_i >= (tau/R_0)^2 / (N * tau^2) = 1/(N*R_0^2)
    f_i <= tau^2 / (N * (tau/R_0)^2) = R_0^2 / N

    Under the hard constraint with target tau, R_0 is bounded by the
    convergence properties of the domain -- easy domains may converge with
    smaller ||B||_F, but the constraint prevents any domain from exceeding tau.

(b) If all ||B_i||_F = tau exactly, then f_i = tau^2 / (N*tau^2) = 1/N. QED.

(c) **[WITHDRAWN -- bound falsified experimentally.]**
    The original argument was: composed SVs are the union of scaled individual SVs
    (Grassmannian orthogonality), Gini of union = within-group Gini + between-group
    Gini, and between-group Gini is zero when group norms are equal.

    This omits the overlap/interaction term from the full Gini decomposition
    (Pyatt 1976). The standard decomposition is:

        Gini(total) = Gini_within + Gini_between + Gini_overlap

    The overlap term captures cross-group SV interleaving: when SVs from different
    groups have different distributions, their union produces additional inequality
    not accounted for by within-group or between-group components alone.

    **Empirical falsification:** Strategy C achieves norm Gini = 0.036
    (Gini_between approx 0), with max within-domain Gini(B_i) approx 0.28.
    The bound predicts Gini(composed) <= 0.28 + 0.036 = 0.316.
    Measured: 0.456. Exceeds bound by 44%.

    Note: Finding #279 (full equalization on the same baseline adapters) measured
    Gini = 0.267, which happens to fall below 0.316, but this does not validate
    the bound -- the bound was derived without the overlap term and its agreement
    with Finding #279 is coincidental. The bound is structurally incomplete.

### Theorem 2 (Training-Time vs Post-Hoc: Information Preservation)

**Theorem.** Training with norm constraint ||B||_F <= tau allows the
optimizer to find the minimum-loss solution within the constraint set.
Post-hoc rescaling to the same norm target tau may produce a DIFFERENT
(higher-loss) solution.

*Proof.*

Let B* = argmin_{||B||_F <= tau} L(B) be the constrained optimum.
Let B_free = argmin L(B) be the unconstrained optimum.
Let B_post = B_free * (tau / ||B_free||_F) be the post-hoc projected point.

In general, B* != B_post because:
1. B* optimizes direction and magnitude jointly under the constraint
2. B_post optimizes freely then clips to the constraint surface

The difference is exactly the gap between projected gradient descent
(which converges to B*) and project-after-train (which gives B_post).

For a convex loss: L(B*) <= L(B_post) by optimality of B*.
For non-convex loss (our setting): B* is a local minimum of the
constrained problem, while B_post need not be stationary for the
constrained problem.

The practical consequence: training-time constraints let the optimizer
learn different B-directions that compensate for the reduced magnitude.
Post-hoc rescaling keeps the unconstrained directions, which may not be
optimal at the constrained magnitude. QED.

---

## E. Quantitative Predictions

### P1: Composed Gini under norm-bounded training (uniform scale)
**Prediction:** With hard norm projection and s=1.0 uniform scale,
the composed Gini will be <= 0.15 (assuming all B_i saturate the
constraint, Gini_between approaches 0, and within-domain Gini is ~0.28
but domains converge to similar directions reducing cross-domain spread).

**Note:** The within-domain Gini of ~0.28 from current training may change
under norm-bounded training. The prediction of 0.15 assumes the norm
constraint compresses the SV spread within each domain as the optimizer
allocates signal more efficiently.

**Kill criterion K709:** Gini > 0.15 means training did not equalize scales.

### P2: Composition PPL vs partial equalization baseline
**Prediction:** Norm-bounded composition PPL <= 6.508 (partial equalization
from Finding #279). If training-time constraints let the optimizer find
better B-directions, the PPL may improve further.

**Kill criterion K710:** PPL worse than 6.508 means norm constraint hurts quality.

### P3: Training convergence
**Prediction:** All 5 domains converge (loss reduction > 5% from start).
The norm constraint may slow convergence but should not prevent it
(projected gradient descent converges for convex sets, Bertsekas 1999).

**Kill criterion K711:** >= 2/5 domains fail to converge means norm bound
is too tight and prevents learning.

### P4: B-matrix norm ratio at convergence
**Prediction:** Under hard projection, all ||B_i||_F <= tau. The realized
ratio R = max/min should be < 2.0 (vs current 1.08 for B-norms alone,
but the point is that WITH uniform scale, the Delta norms are equalized).

### P5: Per-domain PPL vs individual adapter PPL
**Prediction:** Per-domain PPL should be similar to or slightly worse than
individual adapters trained with optimal per-domain scales, because the
norm constraint limits the adapter's perturbation magnitude. The key test
is whether the COMPOSED PPL improves enough to compensate.

---

## F. Assumptions and Breaking Conditions

1. **Grassmannian orthogonality holds at N=5.** Verified: |cos| = 0.026.
   If violated: energy fractions are not additive (Theorem 1 breaks).

2. **Domain expertise can be encoded in B-direction without large magnitude.**
   This is the key unknown. If math/code genuinely need ||Delta|| >> 1 to
   redirect base model computation, then norm-bounding will destroy capability.
   Finding #251 showed math has a scale phase transition at s* in [4,6] --
   this is evidence that math needs minimum perturbation magnitude.

3. **STE ternary training is compatible with norm projection.** The STE
   passes gradients through ternary quantization; adding norm projection
   after the optimizer step should not break the STE mechanism. But
   quantization noise may interact with the projection in unpredictable ways.

4. **Convergence under projected gradient descent.** Standard theory
   guarantees convergence for convex feasible sets and smooth losses.
   Our loss is non-convex (neural network), but projected SGD works
   empirically for non-convex problems as well.

5. **The training data and procedure are identical to the baseline.**
   We use the same data, learning rate, and number of steps as
   real_data_domain_experts. The only change is the norm constraint.

---

## G. Worked Example (N=2, r=4, d=16)

### Setup
Two adapters (medical at s=20, finance at s=1 in current training).
Under norm-bounded training: both use s=1, ||B_i||_F <= tau = 30.

### Current training (unconstrained B, different scales)
B_med SVs: [8.0, 6.0, 4.0, 2.0], ||B_med||_F = sqrt(64+36+16+4) = 10.95
B_fin SVs: [7.5, 5.5, 3.5, 1.5], ||B_fin||_F = sqrt(56.25+30.25+12.25+2.25) = 10.05

Delta_med norm: 20 * 10.95 * 2 = 438.0  (sqrt(r)=2)
Delta_fin norm:  1 * 10.05 * 2 =  20.1

Energy ratio: 438^2 / 20.1^2 = 475:1
Composed Gini dominated by medical, finance silenced.

### Norm-bounded training (constrained B, uniform s=1)
All adapters train with s=1 and ||B||_F <= 30.
Optimizer finds B-directions that minimize loss under the constraint.

B_med_nb SVs: [10, 8, 7, 5], ||B_med_nb||_F = sqrt(100+64+49+25) = 15.43
B_fin_nb SVs: [9, 7, 6, 4], ||B_fin_nb||_F = sqrt(81+49+36+16) = 13.49

(The optimizer may not saturate the constraint; norms are illustrative.)

Delta_med_nb norm: 1 * 15.43 * 2 = 30.86
Delta_fin_nb norm: 1 * 13.49 * 2 = 26.98

Energy ratio: 30.86^2 / 26.98^2 = 1.31:1
Composed Gini: dominated by within-domain spread (both groups similar).
Gini ~ 0.15 (within-domain only, no between-domain imbalance).

---

## H. Complexity and Architecture Connection

**Training cost:** Same as current training (200 steps per domain) plus:
- Weight decay: zero extra cost (standard optimizer feature)
- Norm projection: O(r * d_out) per parameter per step (negligible, just
  compute norm and multiply)
- Both are cheaper than the forward pass itself.

**Memory:** No additional memory beyond the standard training state.

**Integration:** Norm-bounded adapters slot directly into the existing
composition pipeline. The key architectural change is using uniform scale
s=1 instead of per-domain optimal scales, because the B-norms now carry
the information that scales previously encoded.

**Production implication:** If this works, per-domain scale tuning (currently
a manual/heuristic process) becomes unnecessary. Contributors simply train
with the norm constraint and adapters compose at equal weight.

---

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Uniform Frobenius norm constraint on B during training makes cross-domain
   energy imbalance impossible because ||Delta_i||_F = s * ||B_i||_F * sqrt(r)
   with s=1 and ||B_i||_F = tau for all i.

2. **Which existing theorem(s) does the proof build on?**
   Projected gradient descent convergence (Bertsekas, Nonlinear Programming,
   2nd ed., 1999). Pythagorean theorem for Frobenius norms of orthogonal sums
   (standard linear algebra). Gini scale-invariance.

3. **What specific numbers does the proof predict?**
   P1: Composed Gini <= 0.15. P2: Mixed PPL <= 6.508. P3: >= 3/5 converge.
   P4: B-norm ratio < 2.0.

4. **What would FALSIFY the proof (not just the experiment)?**
   If norm-bounded training produces Gini > 0.15 with all B-norms equal,
   the Pythagorean decomposition of Gini is wrong.
   **POST-EXPERIMENT: This happened.** Strategy C achieves near-equal B-norms
   (ratio 1.2:1) but Gini = 0.456, exceeding the Theorem 1c bound of 0.316
   by 44%. The Gini decomposition used (Theorem 1c) was structurally incomplete --
   it omitted the overlap term from Pyatt (1976). The Pythagorean energy
   decomposition (Theorem 1a,b) remains valid; only the Gini bound (1c) is falsified.

5. **How many hyperparameters does this approach add?**
   2: norm constraint tau, and weight decay lambda. tau can be derived from
   the median observed B-norm (~30). lambda is standard and interacts with
   learning rate (explored as Type 2 parameter).

6. **Hack check: Am I adding fix #N to an existing stack?**
   No. This REPLACES the per-domain scale + post-hoc equalization stack with
   a single training-time constraint. The number of mechanisms goes from 3
   (scale tuning + partial equalization + compression factor) to 1 (norm bound).

---

## POST-EXPERIMENT ADDENDUM: Proof Error Analysis

### Theorem 1c FALSIFIED: Gini union bound is structurally incomplete

The proof (Theorem 1c) claimed:
  Gini(composed) <= max_i Gini(B_i) + Gini_between

Strategy C achieves norm Gini = 0.036 (Gini_between approx 0), with
max within-domain Gini(B_i) approx 0.28. The bound predicts:
  Gini(composed) <= 0.28 + 0.036 = 0.316

**Measured: 0.456. Exceeds bound by 44%.**

The bound omits the overlap/interaction term from the standard Gini
decomposition (Pyatt 1976). When SVs from different domain groups have
different spectral shapes, their interleaved union produces inequality
beyond what within-group and between-group components capture.

Theorem 1c has been withdrawn. This also retroactively affects Finding #279
which uses the same bound structure, though that experiment's measurement
(0.267) happened to fall below the bound by coincidence.

### P1 Prediction FAILED: Gini <= 0.15 was wrong

The prediction of 0.15 assumed two things:
1. Theorem 1c's bound was correct (it is not -- see above)
2. The norm constraint would compress within-domain SV spread

Both assumptions were wrong. STE ternary quantization creates B-matrices
with inherent SV spread that the norm constraint does not change (it
scales all SVs uniformly).

### Theorem 2 partially falsified for non-convex case

Theorem 2 predicted that training-time constraints should produce
lower-loss solutions than post-hoc projection (for convex loss, this
is guaranteed by optimality of projected gradient descent). For our
non-convex setting:

- Strategy C (training-time equalization): Gini 0.456, PPL 7.129
- Finding #279 full equalization (post-hoc on same baseline adapters): Gini 0.267

Training-time constraints produced WORSE results than post-hoc correction.
The non-convex caveat in Theorem 2's proof ("B* need not be stationary for
the constrained problem") is the operative case here: the constrained
optimizer landscape is sufficiently different that 200 training steps do
not find directions as effective as the unconstrained optimizer finds.

### Corrected Gini Decomposition

The original analysis compared baseline Gini (0.490, adapters trained at
per-domain optimal scales) vs Strategy C Gini (0.456, DIFFERENT adapters
trained at uniform s=10). This comparison is CONFOUNDED because Strategy C
uses different adapters with different within-domain SV structures.

The correct decomposition uses Finding #279, which applied full equalization
to the SAME baseline adapters:

| Comparison | Gini | Adapters |
|-----------|------|----------|
| Baseline (raw sum) | 0.490 | Baseline, per-domain scales |
| Finding #279 (full eq, SAME adapters) | 0.267 | Baseline, equalized scales |
| Strategy C (uniform s=10) | 0.456 | DIFFERENT adapters, uniform scale |

Correct between-domain contribution (from Finding #279):
  0.490 - 0.267 = 0.223 (~45%)

Correct within-domain contribution:
  0.267 (~55%)

The original claim of "7% between-domain, 93% within-domain" was wrong
because it compared different adapter populations. The actual split is
approximately 45% between-domain / 55% within-domain.

Strategy C's higher Gini (0.456 > 0.267) despite near-perfect energy
equalization proves that adapter quality (within-domain SV structure)
matters as much as energy balance. Training at s=10 instead of per-domain
optimal scales produces adapters with WORSE within-domain spectral
structure, negating the equalization benefit.

### Actual contribution of this experiment

1. Training-time norm constraints produce WORSE composition quality than
   post-hoc equalization on the same adapters (Strategy C Gini 0.456 vs
   Finding #279 full eq Gini 0.267)
2. Adapter quality (within-domain SV structure) matters as much as energy
   balance -- proven by Strategy C having near-perfect energy equalization
   yet worse Gini than Finding #279
3. The Gini union bound (Theorem 1c) is empirically falsified
4. The practical ceiling remains 50% log-compression from Finding #279
   (Gini 0.267)
