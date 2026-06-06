# Rank Sensitivity Per Domain: Mathematical Foundations

## Type: Guided Exploration (Type 2)

The Eckart-Young theorem (proven) provides the framework. The unknowns are:
1. Whether low-rank PPL improvement is from directional selection or magnitude reduction
2. Behavioral quality at each rank (generation, not just PPL)
3. The fine-grained rank-quality curve at ranks {1, 2, 4, 8, 16}

---

## A. Failure Mode Identification

**The disease:** Finding #325 showed that SVD truncation at rank=4 IMPROVES
domain PPL by 23% over raw LoRA. This is surprising: Eckart-Young guarantees
rank=4 has LARGER reconstruction error than rank=16. Two hypotheses explain this:

**H1 (Directional regularization):** The small singular values encode noise or
interference directions. Removing them focuses the adapter on the top domain-signal
directions. The improvement comes from WHICH directions are kept.

**H2 (Magnitude reduction):** Truncation reduces ||delta||_F. The improvement
comes purely from lower perturbation magnitude (Davis-Kahan bound tightens),
not from the specific directions retained. Scaling down the full-rank delta
to the same Frobenius norm would achieve the same effect.

**Why this matters:** If H1 is true, SVD rank is a meaningful design parameter
that differs per domain. If H2 is true, we should just reduce adapter scale
instead (simpler, no SVD needed).

---

## B. The Right Question (Reframe)

**Wrong:** "What is the optimal SVD rank per domain?"
(This treats rank as a hyperparameter to tune.)

**Right:** "Is the PPL improvement from rank truncation attributable to
directional selection (keeping top SVD components) or magnitude reduction
(smaller ||delta||_F)? What is the decomposition of the improvement?"

This decomposes into a testable prediction via an isotropic scaling control.

---

## C. Prior Mathematical Foundations

### Theorem (Eckart-Young-Mirsky, 1936)

For A with SVD U Sigma V^T, the truncated SVD at rank r is the unique
minimizer of ||A - B||_F over rank-r matrices B.

**Relevant property:** Truncation at rank r preserves energy
E(r) = sum_{i=1}^r sigma_i^2 / sum_{i=1}^p sigma_i^2.

From Finding #325, our adapters have:
- E(4) in [0.459, 0.556] across domains (46-56% energy)
- E(8) in [0.685, 0.771] across domains (69-77% energy)

### Frobenius Norm Decomposition

For delta with SVD, ||delta||_F^2 = sum sigma_i^2.

If we scale the full-rank delta by c to match the truncated norm:
||c * delta||_F = ||delta_r||_F implies c = sqrt(E(r)).

For rank=4: c = sqrt(0.46-0.56) = 0.68-0.75.

**Key test:** If PPL(c * delta_16) = PPL(delta_4), then H2 (magnitude reduction).
If PPL(c * delta_16) > PPL(delta_4), then H1 (directional selection).

### Theorem (Davis-Kahan sin-theta, 1970)

sin(theta) <= ||E||_op / delta_gap, where theta is the subspace rotation angle.

**Connection to magnitude reduction (H2):** Reducing ||delta||_F by factor c
reduces ||delta||_op by at most factor c. If the spectral gap of the base model
is the binding constraint, this uniformly tightens the bound.

**Connection to directional selection (H1):** The operator norm ||delta||_op
equals sigma_1 regardless of rank truncation (as long as rank >= 1). So
directional selection does NOT reduce operator norm. It reduces the
SECOND-ORDER perturbation effects (off-principal-subspace components).

---

## D. Predictions

### Quantitative (from Eckart-Young + Finding #325 interpolation)

| Prediction | Source | Expected |
|------------|--------|----------|
| P1: rank=2 PPL ratio < rank=4 ratio (0.77) | Monotonic violation continues | ratio in [0.65, 0.75] |
| P2: rank=1 PPL ratio < rank=2 ratio | Monotonic violation continues | ratio in [0.55, 0.70] |
| P3: Scale-control at c=sqrt(E(4)) matches rank=4 PPL within 10% if H2 | H2 test | H2: within 10%. H1: >10% gap |
| P4: All domains peak at same rank (no domain differentiation) | Finding #325: all domains had identical rank ordering | Same rank is best for all 5 |
| P5: Generation quality tracks PPL at all ranks | PPL as proxy assumption | Correlation > 0.8 |

### Behavioral

| Prediction | Source | Note |
|------------|--------|------|
| B1: rank=4 generations are more focused/on-topic than rank=16 | H1 directional regularization | Type 2 |
| B2: rank=1 generations may be degenerate (too little information) | Rank-1 retains only 17-23% energy | Failure expected at very low rank |
| B3: Scale-control generations differ qualitatively from rank=4 | H1 vs H2 discriminator | Type 2 |

---

## E. Assumptions & Breaking Conditions

1. **PPL as proxy for quality.** Finding: this project proved PPL-task correlation
   is r=0.08. However, within a single adapter and domain, the relative PPL ordering
   may still be informative. Breaking: PPL ordering disagrees with generation quality
   ordering. This experiment TESTS this assumption with behavioral evaluation.

2. **Singular value spectrum is representative.** Finding #325 reported SVs from the
   first module only. If the spectrum varies wildly across layers, the first-module
   spectrum is misleading. Breaking: per-layer spectral analysis shows heterogeneity.

3. **Scale-control is achievable.** We assume we can apply a scalar to the full-rank
   delta to match rank-4 Frobenius norm. This is trivially true for the factored
   form: multiply B by c.

---

## F. Worked Example (rank=4, medical domain)

From Finding #325 data:
- Medical SVs: [10.82, 8.07, 7.64, 7.18, 6.81, 6.52, 6.29, 5.58, 5.26, 4.98, 4.79, 4.50, 4.37, 4.13, 3.66, 3.43]
- ||delta||_F^2 = sum(s_i^2) = 608.1 (all 16 SVs squared and summed)
- E(4) = (10.82^2 + 8.07^2 + 7.64^2 + 7.18^2) / 608.1 = 292.0 / 608.1 = 0.480
- ||delta_4||_F = sqrt(292.0) = 17.09
- ||delta_16||_F = sqrt(608.1) = 24.66
- Scale factor c = sqrt(0.480) = 0.693

Scale control: multiply adapter scale by 0.693.
- Original scale: 20.0
- Controlled scale: 20.0 * 0.693 = 13.86

If PPL(scaled_full_rank) = PPL(rank_4), the improvement is from magnitude.
If PPL(scaled_full_rank) > PPL(rank_4), the improvement is from direction selection.

---

## G. Complexity & Architecture Connection

**Runtime cost of this experiment:**
- 5 domains x 5 ranks {1,2,4,8,16} = 25 SVD extractions + PPL evals
- 5 domains x 1 scale control = 5 additional evals
- 5 domains x 5 ranks x 1 generation = 25 behavioral evals
- Previous experiment: 5 domains x 6 ranks took ~119s. This should be ~90s for PPL.
- Generation: ~5-10s per generation x 25 = ~125-250s
- Total estimate: ~5-8 minutes

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   Eckart-Young guarantees the truncated SVD is optimal. The experiment tests
   whether the PPL improvement is from this optimality (directional) or from
   incidental magnitude reduction.

2. Which existing theorem(s) does the proof build on?
   Eckart-Young-Mirsky (1936), Davis-Kahan sin-theta (1970).

3. What specific numbers does the proof predict?
   Scale factor c = sqrt(E(4)) in [0.68, 0.75] per domain. If H2 holds,
   scaled full-rank PPL matches rank-4 PPL within 10%.

4. What would FALSIFY the proof (not just the experiment)?
   Nothing falsifies Eckart-Young (it is a theorem). What would falsify H1/H2:
   If scale-control matches rank-4 exactly, H1 is false (H2 confirmed).
   If scale-control is much worse than rank-4, H2 is false (H1 confirmed).

5. How many hyperparameters does this approach add?
   Count: 1 (SVD rank). This experiment determines whether it matters as a
   per-domain parameter or whether simple scaling suffices.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is an analysis experiment to understand WHY rank=4 SVD helps,
   not a new mechanism.
