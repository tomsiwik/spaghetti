# MATH.md: Solidified Expert Composition MMLU

## Type: Frontier Extension (Type 3)

Extending Finding #326's single-adapter magnitude-reduction result to the
multi-adapter composition setting.

---

## A. Failure Mode: Knowledge Destruction Under Composition

**The disease:** LoRA adapters trained at scale=20 produce weight perturbations
whose Frobenius norm exceeds the base model's spectral gap, causing uncontrolled
rotation of knowledge-encoding subspaces. Under composition (N=5 via NRE averaging),
the perturbation norms partially cancel (NRE rescaling preserves average norm, not
sum), but the remaining perturbation still exceeds the knowledge-preservation threshold.

**Evidence it is a real risk:**
- Finding #320: Single adapter at scale=20 destroys MMLU by -60pp on Qwen3-4B-4bit.
- Finding #320: N=5 composed at scale=20 destroys MMLU by -44pp.
- Finding #326: The root cause is magnitude (||delta||_F), not direction.

**Why this is the ROOT CAUSE, not a symptom:**
Previous hypothesis (H1) was that SVD selects "good" vs "bad" singular directions.
Finding #326 disproved this: full-rank at reduced scale beats SVD rank=4 in 4/5 domains.
The disease is OVER-PERTURBATION, not WRONG-DIRECTION perturbation.

---

## B. The Right Question

**Wrong question:** "Does SVD composition fix the scale=20 catastrophe?"

**Right question:** "What is the minimum norm reduction factor c such that
composed perturbation ||delta_composed||_op < delta_gap, ensuring knowledge
subspace rotation is bounded below the MMLU-critical threshold?"

The answer should come from the Davis-Kahan bound applied to composition.

---

## C. Prior Mathematical Foundations

**Davis-Kahan sin-theta theorem** (Davis & Kahan, 1970):

For symmetric matrices A, A+E with eigenspaces V, V' separated by spectral gap delta:

    sin(theta(V, V')) <= ||E||_op / delta

where theta is the canonical angle between the eigenspaces.

**Eckart-Young-Mirsky theorem** (Eckart & Young, 1936):

The best rank-r approximation of matrix M (in Frobenius or operator norm) is the
truncated SVD. For M with singular values s_1 >= ... >= s_n:

    ||M - M_r||_F = sqrt(sum_{i=r+1}^n s_i^2)

**NRE (Norm-Rescaled Ensemble) composition:**

For N adapters with B-matrices {B_1, ..., B_N}, NRE computes:

    B_composed = (1/N) * sum(B_i) * (mean(||B_i||) / ||(1/N) sum(B_i)||)

The norm-rescaling ensures ||B_composed|| ~= mean(||B_i||), preventing 1/sqrt(N)
shrinkage from naive averaging while preserving the averaging direction.

---

## D. Proof of Bounded Degradation

We prove that magnitude reduction (via SVD truncation or scale reduction) tightens
the Davis-Kahan bound under composition, and derive quantitative predictions.

**Setup.** Let W_0 be the base model weight matrix at a given layer. The LoRA
adapter adds perturbation delta = scale * B^T @ A^T. For N composed adapters
via NRE: delta_composed = scale * B_composed^T @ A^T (using domain_0's A-matrix).

**Theorem 1 (Composition magnitude bound).**
Let {delta_1, ..., delta_N} be N adapter perturbations, each with
||delta_i||_F = sigma (approximately equal, as they share the same training regime).
Under NRE composition, the composed perturbation satisfies:

    ||delta_composed||_F <= sigma  (with equality when all adapters are identical)

*Proof sketch.* NRE averaging: B_composed = mean(B_i) * (mean(||B_i||)/||mean(B_i)||).
By triangle inequality, ||mean(B_i)|| <= mean(||B_i||), so the rescaling factor >= 1.
However, the rescaling targets mean(||B_i||), which equals sigma (the individual norm).
When adapters are orthogonal in B-space, ||mean(B_i)|| = sigma/sqrt(N), and the
rescaling factor = sqrt(N). The composed delta then has ||delta_composed||_F = sigma.
When adapters are correlated, ||mean(B_i)|| > sigma/sqrt(N), rescaling < sqrt(N),
and ||delta_composed||_F <= sigma still holds because rescaling targets mean norm.

In practice, Grassmannian A-matrices ensure A-orthogonality, but B-matrices are
trained independently and have moderate correlation (cos ~0.03 from Finding #225).
The NRE rescaling normalizes the composed adapter to have approximately the same
norm as individual adapters.

**Key implication:** NRE composition does NOT amplify perturbation beyond individual
adapter magnitude. The MMLU destruction is already present in single adapters at
scale=20 (-60pp), and composition makes it somewhat BETTER (-44pp) because directional
averaging partially cancels destructive components.

**Theorem 2 (SVD truncation tightens Davis-Kahan).**
Let delta = U S V^T be the SVD of the adapter perturbation with singular values
s_1 >= ... >= s_r (r=16). SVD truncation at rank k retains energy fraction:

    E(k) = sum_{i=1}^k s_i^2 / sum_{i=1}^r s_i^2

The truncated perturbation has:

    ||delta_k||_F = sqrt(E(k)) * ||delta||_F

By Davis-Kahan:

    sin(theta_k) <= ||delta_k||_op / delta_gap = s_1 / delta_gap

Note: SVD truncation does NOT reduce ||.||_op (the largest singular value s_1 is
always kept). The Frobenius norm reduction tightens the AVERAGE subspace rotation
across all knowledge directions, but the WORST-CASE rotation remains the same.

This is why scale reduction is more effective than SVD truncation (Finding #326):
scale reduction by factor c reduces BOTH ||delta||_F and ||delta||_op by factor c.

*Proof.* Immediate from Eckart-Young and Davis-Kahan. QED.

**Theorem 3 (Scale-equivalent SVD truncation under composition).**
SVD rank=4 retains E(4) fraction of energy. A full-rank adapter at scale
s_eff = scale * sqrt(E(4)) has the same Frobenius norm:

    ||delta_{full, s_eff}||_F = s_eff/scale * ||delta||_F = sqrt(E(4)) * ||delta||_F = ||delta_4||_F

Under NRE composition of N such reduced-scale adapters:
    ||delta_composed_{full, s_eff}||_F ~= s_eff/scale * ||delta||_F = sqrt(E(4)) * sigma

Under NRE composition of N SVD-truncated adapters:
    ||delta_composed_{SVD}||_F ~= sqrt(E(4)) * sigma

These are approximately equal. Therefore:
**SVD-truncated composition ~= scale-reduced composition in MMLU impact.**

---

## D. Predictions (Derived from the proof)

### Behavioral Predictions

1. **SVD rank=4 composition preserves MMLU better than raw composition at scale=20.**
   - Raw N=5 at scale=20: -44pp (Finding #320)
   - SVD rank=4 single: -30pp (Finding #325)
   - SVD rank=4 composed N=5: predicted -25 to -35pp (better than raw, but NOT solved)
   
   Rationale: SVD truncation reduces ||delta||_F by sqrt(E(4)) ~= 0.72 (E(4) ~= 0.52
   from Finding #327). NRE averaging does not amplify beyond individual adapter norm.
   So composed SVD should be comparable to single SVD (-30pp) with possible improvement
   from directional averaging.

2. **Scale-reduced composition matches or beats SVD composition.**
   - Full-rank at scale ~13 (energy-matched to SVD rank=4): predicted -25 to -35pp
   - This should be within 5pp of SVD rank=4 composition (Theorem 3)

3. **Full-rank at scale=5 composition preserves MMLU (~0pp).**
   - Finding #320 confirmed: 0pp degradation at scale=5.
   - This is the CONTROL: proves composition itself is sound.

### Quantitative Predictions

| Configuration | Predicted MMLU | Predicted degradation vs base |
|---|---|---|
| Base Qwen3-4B | 92% | 0pp |
| Raw LoRA N=5 scale=20 | ~48% | -44pp (replication of Finding #320) |
| SVD rank=4 N=5 scale=20 | 57-67% | -25 to -35pp |
| SVD rank=1 N=5 scale=20 | 65-75% | -17 to -27pp (most aggressive truncation) |
| Full-rank N=5 scale=13 | 57-67% | -25 to -35pp |
| Full-rank N=5 scale=5 | 90-92% | 0 to -2pp |

### Kill Criteria (Derived)

- **K837: MMLU degradation > 15pp** -- If SVD composition WORSENS MMLU beyond what
  single SVD adapter does (-30pp), then composition amplifies rather than averages
  destructive components. This would falsify Theorem 1.
  
- **K838: Domain quality < 50% of raw LoRA** -- If SVD truncation destroys domain
  expertise under composition more than expected, the truncation loses critical
  directions specific to each domain.

### Success Criterion

- **S83: MMLU degradation < 5pp** -- This would require that SVD truncation +
  composition brings degradation from -44pp to under -5pp. Based on the math,
  this is UNLIKELY for SVD rank=4 (predicted -25 to -35pp). It may be achievable
  only at very aggressive truncation (rank=1) or low scale (<=5).

---

## E. Assumptions & Breaking Conditions

1. **NRE preserves average adapter norm.** If NRE rescaling amplifies beyond
   individual adapter norms, composition could be WORSE than single adapter.
   Breaking condition: ||delta_composed||_F > 1.5 * mean(||delta_i||_F).

2. **SVD truncation removes noise, not signal.** If domain-specific signal is
   concentrated in small singular values for some domains, truncation destroys
   useful information. Breaking condition: SVD composition domain PPL > 2x raw
   composition domain PPL. (Finding #326 suggests this is unlikely -- scale control
   beats SVD, meaning the signal IS in the top directions.)

3. **MMLU questions are answerable from base model knowledge subspace.** If some
   MMLU questions require adapter knowledge, then any magnitude reduction hurts.
   Breaking condition: adapted model at low scale scores HIGHER than base on MMLU.
   (Finding #320 shows this is false: scale<=5 gives exactly base accuracy.)

4. **Grassmannian A-orthogonality enables clean NRE averaging.** If A-matrices
   are not sufficiently orthogonal, NRE averaging of B-matrices does not
   approximate independent perturbation averaging.

---

## F. Worked Example (Toy)

Consider d=16, r=4, scale=20, N=2 adapters.

Adapter 1: delta_1 = 20 * B_1^T @ A_1^T, ||delta_1||_F = 5.0
Adapter 2: delta_2 = 20 * B_2^T @ A_2^T, ||delta_2||_F = 5.0

SVD of delta_1: singular values [3.0, 2.5, 2.0, 1.0]
E(2) = (9 + 6.25)/(9 + 6.25 + 4 + 1) = 15.25/20.25 = 0.753
||delta_1,r=2||_F = sqrt(0.753) * 5.0 = 4.34

NRE composition (raw): B_composed = mean(B_1, B_2) * rescale
||delta_composed||_F ~= 5.0 (norm-preserved)
Davis-Kahan: sin(theta) <= 5.0/delta_gap

NRE composition (SVD r=2): 
||delta_composed_SVD||_F ~= 4.34 (norm of truncated adapters, preserved by NRE)
Davis-Kahan: sin(theta) <= 4.34/delta_gap (13% tighter bound)

Scale-matched full-rank: scale_eff = 20 * sqrt(0.753) = 17.36
||delta_eff||_F = 17.36/20 * 5.0 = 4.34 (same as SVD r=2)

---

## G. Complexity & Architecture Connection

**SVD extraction:** One-time O(d * r^2) per module per adapter. Already computed
and saved at micro/models/svd_extraction_quality/svd_experts/.

**Composition:** O(N * modules * r * d) for NRE averaging. Same as raw composition.

**Inference:** SVD experts are stored as (A_svd, B_svd) pairs. Runtime cost is
identical to standard LoRA: x @ A_svd @ B_svd per module.

**Memory:** SVD rank=4 experts use 4/16 = 25% of full LoRA parameters. Under
composition of N=5, the composed adapter has the same size as one adapter.

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   Magnitude reduction via SVD truncation or scale reduction tightens the Davis-Kahan
   bound, limiting knowledge subspace rotation to below the MMLU-critical threshold.

2. Which existing theorem(s) does the proof build on?
   Davis-Kahan sin-theta theorem (1970), Eckart-Young-Mirsky theorem (1936).

3. What specific numbers does the proof predict?
   SVD rank=4 composition: -25 to -35pp MMLU degradation (vs -44pp raw composition).
   Scale-reduced to ~13 composition: within 5pp of SVD rank=4 composition.
   Scale=5 composition: 0 to -2pp degradation (control).

4. What would FALSIFY the proof (not just the experiment)?
   If SVD composition is WORSE than raw composition (would mean NRE amplifies
   truncated perturbations, contradicting Theorem 1).
   If scale-reduced composition is >10pp different from SVD composition at
   energy-matched scale (would mean direction matters, contradicting Theorem 3).

5. How many hyperparameters does this approach add?
   Count: 2 (SVD rank, scale). SVD rank is determined by energy fraction (derivable
   from spectral profile). Scale is the training hyperparameter (not added by us).

6. Hack check: Am I adding fix #N to an existing stack?
   No. This experiment TESTS whether a single operation (magnitude reduction) solves
   the composition MMLU catastrophe. It does not add fixes.
