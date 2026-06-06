# Peer Review: notebooklm_composition_theory

## NotebookLM Findings

Skipped. This is a theoretical survey experiment; the source material IS the
MATH.md and PAPER.md themselves. NotebookLM review would be circular.

## Mathematical Soundness

### A. Proof Completeness (BLOCKING)

MATH.md contains one labeled theorem: "Theorem (Concentration Bound on Adapter
Interference)" in Section 0. It has a statement and a bound but **no proof**.
The text states:

```
P(|cos(delta_i, delta_j)| > epsilon) <= 2 exp(-c * epsilon^2 * d_eff)
```

There is no derivation. No QED. No citation to a specific theorem that yields
this exact bound. The text says "Johnson-Lindenstrauss effect" informally but
JL is about distance preservation under random projection, not about cosine
concentration of independently-trained neural network weight vectors. The
actual relevant result would be something like Levy's lemma on concentration
of Lipschitz functions on the sphere, or Vershynin's sub-Gaussian concentration
results.

**Verdict on proof completeness: FAIL.** The central "theorem" is an assertion,
not a proof. The remaining document is mechanism description with equations,
which the reviewer instructions explicitly say "is NOT a proof."

However: this experiment is declared as a P3 theoretical survey / literature
synthesis, not a proof-driven micro-experiment. I will evaluate it on those
terms while noting the gap.

### B. Specific Mathematical Issues

**Issue 1: The d_eff definition is circular.**

The theorem states d_eff = min(d_out, d_in) * r / r_eff, then says
"empirically E[|cos|] ~ 0.001, which is consistent with d_eff ~ 10^6."
This is curve-fitting: d_eff is defined to make the bound match the data.
There is no independent prediction of d_eff.

**Issue 2: Concentration for trained (non-random) vectors is unproven.**

The JL/concentration arguments apply to random vectors on a sphere.
LoRA adapters are NOT random -- they are trained by gradient descent on
(possibly overlapping) data distributions. The prior adversarial review
(SOLE_ADVERSARIAL_REVIEW.md) documented this exactly: at macro scale,
converged adapters on semantically related domains hit cos=0.703
(math-medical pair), which is 562x higher than the predicted 0.00125.

The MATH.md acknowledges this only obliquely in the caveat about "domain
overlap creates non-random correlations" (Section 4.2) but does not
incorporate it into the bound. This is the most serious flaw: the central
concentration argument has a known empirical counterexample at production
scale.

**Issue 3: The E[|cos|] ~ 1/sqrt(d*r) formula is wrong for flattened
weight matrices.**

For two random unit vectors in R^n, E[|cos|] ~ sqrt(2/(pi*n)). For
flattened Delta_i = B_i A_i in R^{d_out x d_in} with B in R^{d_out x r}
and A in R^{r x d_in}, the flattened vector lives in R^{d_out * d_in}
but has rank at most r. The effective dimensionality of the subspace
explored is at most r * (d_out + d_in), not d_out * d_in. So the
concentration should scale as 1/sqrt(r*(d_out + d_in)), not
1/sqrt(d_out * d_in * r). At d=2560, r=16: sqrt(16*5120) = 286,
giving E[|cos|] ~ 0.0035, not 0.005. The numerical difference is small
here but the derivation is sloppy -- mixing up ambient dimension,
intrinsic dimension, and parameter count.

**Issue 4: The N^2 interference bound (Section 1.3) conflates two
different quantities.**

The bound states:
```
||f_composed(x) - (f_base(x) + sum_i delta_f_i(x))|| <=
    C * N^2 * max_i ||Delta_i||^2 * ||H||_op * ||x||^2
```

This bounds the deviation of the composed model from the sum of individual
first-order perturbations. It does NOT bound the deviation from the oracle
(best single adapter). gamma(N) = PPL_composed / PPL_oracle mixes these
two very different notions. The paper then fits gamma(N) ~ 1 - alpha/sqrt(N)
+ beta/N to two data points (N=5, N=25), which is not a prediction -- it is
an interpolation of a 3-parameter curve through 2 points.

**Issue 5: The gamma(N=5) = 3.45x value contradicts the "improvement with N" narrative.**

The paper claims gamma improves with N (Section 4.1: "Why gamma IMPROVES
with N"). But gamma(5) = 3.45x means composed PPL is 3.45x WORSE than oracle
at N=5. At N=25, gamma = 0.982, meaning composed is 1.8% BETTER than oracle.
This is a 3.5x discontinuous jump. The explanation ("more adapters means more
constructive transfer") is hand-waving -- why would going from 5 to 25
adapters cause a qualitative phase transition from 3.45x worse to 0.982x?

The most likely explanation is that the N=5 and N=25 experiments used different
methodologies, evaluation protocols, or definitions of "oracle." This should
have been flagged and reconciled, not fitted to a scaling curve.

**Issue 6: The Grassmannian cross-talk derivation (Section 1.1) is wrong.**

The paper claims that with Grassmannian skeleton (A_i^T A_j = 0):
```
<h_i(x), h_j(x)> = x^T * 0 * B_j x = 0
```

This is algebraically incorrect. h_i(x) = B_i A_i x, h_j(x) = B_j A_j x.
The inner product is:
```
<h_i(x), h_j(x)> = (B_i A_i x)^T (B_j A_j x) = x^T A_i^T B_i^T B_j A_j x
```

Even when A_i^T A_j = 0, this does NOT simplify to zero unless B_i^T B_j = 0
AS WELL (or more precisely, B_i^T B_j has zero projection onto the relevant
subspace). The paper acknowledges B-matrix correlation elsewhere (|cos| = 0.0298)
but claims the Grassmannian A_i^T A_j = 0 makes cross-talk vanish. It does not.
The cross-talk depends on B_i^T B_j projected through A_i, A_j, which is a more
complex expression than shown.

**Issue 7: The optimal scaling derivation is internally inconsistent.**

Section 2.1 derives that optimal per-adapter scaling is lambda* = delta_L / kappa,
independent of N. But Section 1.2 derives the quadratic model as:

```
L(lambda) = L_base - N * lambda * delta_L + (N * lambda^2 / 2) * kappa
```

Setting dL/dlambda = 0: -N * delta_L + N * lambda * kappa = 0,
so lambda* = delta_L / kappa.

But this assumes all N adapters use the SAME lambda, and the loss function has
N adapters each contributing -lambda * delta_L (linear in N) and each contributing
(lambda^2 / 2) * kappa (also linear in N since adapters are orthogonal and their
Hessian cross-terms vanish). This is correct.

However, the claim "lambda_total = N * delta_L / kappa" (for total perturbation)
later in the same section contradicts the per-adapter lambda = delta_L / kappa
claim. If per-adapter lambda is independent of N, and total perturbation is
N * lambda, then total perturbation grows linearly with N, which means the
composed model moves further from the base with every adapter added. The paper
never checks whether this exits the quadratic basin. At N=25 with lambda=0.5,
the total perturbation norm is 12.5x a single adapter -- the quadratic
approximation may not hold.

### C. Prediction Verification (BLOCKING)

**This is a theoretical survey with no new empirical data.** All "predictions"
are post-hoc explanations of existing findings. The document does make
forward-looking predictions (Section 9.1) but these are untested:

1. Lambda ~ 0.3-0.5 per adapter at N=25 should beat 1/N
2. gamma(100) ~ 0.94
3. DARE + composition improves quality
4. Composition degrades at d < 256

None of these have been verified. The document is honest about this (Limitations
section). For a survey, this is acceptable -- the predictions exist for future
experiments.

## Novelty Assessment

**Low novelty.** The document synthesizes known results:

- Concentration of measure in high dimensions: textbook result
- Model Soups / loss basin theory: Wortsman et al. 2022
- Perturbation theory for neural networks: standard (e.g., Fort & Jastrzebski 2019)
- DARE sparsification: Yu et al. 2023
- Task Arithmetic: Ilharco et al. 2022

The "novel synthesis" (Section 1.3, Perturbation Theory Framework) combines
these into a unified narrative. This has value as a design document but does
not constitute new mathematical results.

The most valuable contribution is the explicit connection between three
independent frameworks (superposition, loss basins, perturbation theory) and
the argument that they all predict the same thing. This is useful for the
project even if not publishable.

## Experimental Design

Not applicable -- no experiment was run. The document is a survey.

The seven recommendations are reasonable and actionable. The kill criteria for
each recommendation are specific and testable. This is the most useful part
of the document.

## Macro-Scale Risks (advisory)

**Risk 1: The concentration argument breaks for semantically related domains.**
This is documented in the prior adversarial review (cos=0.703 for math-medical
at macro scale) but not addressed in this document. The entire framework
assumes independence of adapter training, which fails when domains share
significant knowledge structure.

**Risk 2: The quadratic basin assumption is unvalidated beyond N=25.**
At N=100 with lambda=0.5, the total perturbation is 50x a single adapter.
The quadratic approximation almost certainly fails at this scale.

**Risk 3: The "constructive transfer" mechanism (Section 3.1) is unfalsifiable
as stated.** The shared/specific decomposition Delta_i = Delta_shared + Delta_specific
is not measured -- it is assumed. Finding #68's "4/5 pairs composed > individual"
could be explained by other mechanisms (regularization effect of averaging,
evaluation noise, etc.). The prior adversarial review noted that
"exp_cross_adapter_knowledge_transfer KILLED this interpretation; 0/20 pairwise
transfers >2%."

This is important: Section 3.1 proposes "constructive transfer" as the mechanism
for why composition beats individuals, but Finding KILLED the knowledge transfer
hypothesis. The MATH.md paper does not cite or address this contradictory finding.

## Verdict

**REVISE**

This is a useful internal design document that synthesizes the project's
theoretical understanding of additive LoRA composition. The recommendations
are actionable and well-structured. However, it has specific issues that
should be fixed before being treated as a reliable theoretical foundation:

### Required Revisions

1. **Add an actual proof for the central theorem, or cite one.** The
   "Concentration Bound on Adapter Interference" needs either a derivation
   (from Levy's lemma or sub-Gaussian concentration) or an explicit citation
   to a published proof with page number. State the precise conditions under
   which it holds (i.i.d. training runs, specific distributional assumptions
   on the trained parameters).

2. **Address the cos=0.703 counterexample.** The prior adversarial review
   documented that semantically related domains (math-medical) produce
   cos=0.703 at macro scale. The concentration bound predicts this is a
   measure-zero event. Either the bound does not apply to trained (non-random)
   adapters, or there is a domain-overlap correction term. The paper must
   address this or explicitly restrict its claims to "independently-trained
   on dissimilar domains."

3. **Reconcile constructive transfer with the killed knowledge transfer finding.**
   Section 3.1 claims shared beneficial structure reinforces across adapters.
   But the project's own finding says "0/20 pairwise transfers >2%, the benefit
   is 1/N regularization not knowledge sharing." Either the mechanisms are
   different (constructive transfer != knowledge transfer) and this should be
   explained, or Section 3.1 contradicts the project's own evidence.

4. **Fix the gamma(N=5)=3.45x vs gamma(N=25)=0.982 discontinuity.** Explain
   whether these use the same methodology and definition of "oracle." A 3.5x
   qualitative change needs more than "more adapters = more transfer."

5. **Fix the Grassmannian cross-talk algebra (Section 1.1).** A_i^T A_j = 0
   does not make <h_i(x), h_j(x)> = 0. Write out the full expression and
   show what it actually equals.

6. **Downgrade status from SUPPORTED to PROVISIONAL.** Per the project's own
   definitions, "supported = proof exists, predictions mostly match." This
   document contains no proof (the theorem is unproven) and no new predictions
   that have been verified. "Provisional = no proof, empirical observation only"
   is the correct status for a literature survey that synthesizes prior findings
   without formal proofs.

### Advisory (non-blocking)

7. Make d_eff an independent prediction rather than a curve-fit to data.
8. Check whether the total perturbation at N=25 with lambda=0.5 stays within
   the quadratic basin (measure higher-order terms, not just assume they are
   small).
9. The gamma(N) scaling law fitted through 2 points with 3 parameters is
   not predictive. Either get more data points or simplify the model.
