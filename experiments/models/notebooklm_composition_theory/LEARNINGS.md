# LEARNINGS: Theoretical Foundations of Additive LoRA Composition

## Status: PROVISIONAL (downgraded from SUPPORTED per reviewer)

The researcher produced a unified perturbation-theory framework explaining
why additive LoRA composition works. The adversarial review correctly identified
that the central "theorem" is unproven (assertion, not derivation), has a known
counterexample (cos=0.703 for math-medical at macro scale), and the constructive
transfer mechanism contradicts the project's own killed knowledge transfer finding.

## What We Learned

### 1. Three frameworks converge on the same explanation (useful, not proven)

Superposition (Cao et al.), loss basin / Model Soups (Wortsman et al.), and
perturbation theory all predict that additive composition works when:
- Perturbations are small (||Delta||/||W|| ~ 0.006)
- Cross-terms are negligible (dimensional concentration at d=2560)
- Shared signal reinforces, noise cancels

This convergence is informative as a design heuristic but none of the three
frameworks has been rigorously proven for TRAINED (non-random) adapter weights.

### 2. The concentration bound is CLAIMED, not PROVEN

The central theorem P(|cos| > eps) <= 2 exp(-c * eps^2 * d_eff) has:
- No derivation (JL-lemma is about random projection, not trained weights)
- A circular d_eff definition (curve-fit to match observed |cos| ~ 0.001)
- A known counterexample: cos=0.703 for semantically related domains at macro scale

The reviewer is correct: this should cite Levy's lemma or sub-Gaussian concentration
and explicitly restrict to dissimilar domains. Until proven, treat as a useful
approximation, not a theorem.

### 3. "Constructive transfer" contradicts killed knowledge transfer finding

Section 3.1 proposes shared beneficial structure reinforces across adapters.
But the project's own exp_cross_adapter_knowledge_transfer found 0/20 pairwise
transfers > 2% — the benefit is 1/N regularization, not knowledge sharing.

These may be different mechanisms (transfer = one adapter helping another's domain;
constructive = overlapping general improvements). But the paper doesn't distinguish
them. Until clarified, the mechanism for "composed > individual" remains unexplained.

### 4. gamma(N) data is inconsistent and unfittable

gamma(5) = 3.45x (composed 3.45x WORSE than oracle) vs gamma(25) = 0.982
(composed 1.8% BETTER). This 3.5x discontinuity likely reflects different
experimental methodologies, not a smooth scaling law. Fitting a 3-parameter
curve through 2 inconsistent points is not prediction.

### 5. The Grassmannian cross-talk algebra is wrong

A_i^T A_j = 0 does NOT make output cross-talk vanish. The full inner product
x^T A_i^T B_i^T B_j A_j x depends on B_i^T B_j, which is NOT zero even with
Grassmannian A matrices. The Grassmannian skeleton reduces interference via the
A-space separation but does not eliminate it.

### 6. Seven recommendations are the valuable output

Despite mathematical issues, the recommendations are well-structured:
- R1 (increase lambda from 1/N to 0.5-1.0): Highest expected impact, testable
- R2 (DARE sparsification): Medium priority, validated at p=0.9
- R4 (runtime LoRA as default): Validates existing architecture
- R5 (stop orthogonality engineering): Correct conclusion from 3 killed experiments
- R7 (DARE + Task Arithmetic for static deploy): Practical, actionable

### 7. The orthogonality closure IS solid

Across three independent experiments (#68 weight-space, #169 data-space, #164
scaling > weighting), the orthogonality hypothesis is dead. This conclusion
does not require the unproven concentration theorem — it follows directly from
the empirical results showing identical outcomes across all init methods.

## Confirming Evidence

- Cao et al. (arXiv:2508.11985): RMS cosine correlates with PPL change
- Wortsman et al. (arXiv:2212.04089): same-basin averaging theory
- Yu et al. (arXiv:2311.03099): DARE works at 90-99% drop rate
- Finding #164: lambda=0.5 beats 1/N by 8.1%
- Finding #169: OSRM = random = Grassmannian

## Contradicting Evidence

- cos=0.703 for math-medical pair at macro scale (breaks concentration assumption)
- exp_cross_adapter_knowledge_transfer: 0/20 pairwise transfers > 2% (kills
  constructive transfer as knowledge sharing)
- gamma(5)=3.45x vs gamma(25)=0.982 inconsistency (undermines scaling law)
- Pause Recycling LoRAs (arXiv:2506.13479): weight composition != semantic composition

## Follow-up Recommendations

No new experiment from this survey. The recommendations become relevant when:
1. Deployment track proves generation quality (exp_generation_quality_test)
2. Lambda scaling sweep at N=25 (validates R1, highest expected impact)
3. DARE + composition at N=25 (validates R2)

Priority remains: deployment track > mechanism questions.
