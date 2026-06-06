# Peer Review: Spectral Surgery Post-Composition

## Experiment Type

Guided exploration (Type 2). The Grassmannian orthogonality framework is the
proven foundation; the unknown being explored is whether post-composition SVD
reweighting can exploit residual B-matrix overlap to improve quality. The
experiment narrows the unknown by measuring the spectral structure of composed
deltas and testing surgery's empirical effect.

Type 2 requirements are met: MATH.md cites the proven Grassmannian framework
(Theorem 1 with A_i A_j^T = 0), identifies the unknown precisely (can
B-matrix overlap create surgically exploitable artifacts?), and the experiment
narrows the unknown (answer: no -- overlap concentrates in high-SV directions,
structurally inverting surgery's premise).

## Hack Detector

- Fix count: 1 (spectral surgery is the single technique under test). No flags.
- Is MATH.md a proof or a description? **Proof with QED** for Theorem 1a and 1b.
  Corollaries 1-2 follow validly from the theorem. The worked example in section
  F is a genuine calculation, not hand-waving. The derivation in section C is
  unusually honest -- it shows the author catching and correcting their own
  dimensional errors mid-derivation.
- Metric used as evidence: PPL improvement (K696), timing (K697), correlation
  between SV magnitude and domain purity (K698). PPL is a standard proxy for
  language model quality. The correlation metric (K698) is custom but directly
  tests the structural prediction from the proof.
- Kill criteria source: K696 is partially derived from the proof (Corollary 1
  predicts no improvement). K697 is a practical engineering constraint. K698 is
  derived from the proof (Theorem 1a predicts no cross-domain interference
  artifacts, therefore no correlation). Mixed: 2/3 proof-derived, 1/3
  engineering.

## Self-Test Audit

1. **One-sentence impossibility property:** "Grassmannian orthogonality
   eliminates cross-terms in the LEFT Gram matrix, so the composed delta's
   spectral structure is determined entirely by sum_i s_i^2 B_i B_i^T with no
   interference artifacts." This is one property (Gram matrix cross-term
   cancellation) with one consequence. PASS.

2. **Cited theorems:** Weyl's inequality (1912) -- real, applicable. Orthogonal
   projector properties -- standard linear algebra. SVD structure of rank-r
   sums -- standard. All conditions apply to the setting. PASS.

3. **Predicted numbers:** Four specific predictions: (i) spectral deviation
   < 5%, (ii) surgery improvement < 0.5%, (iii) calibration > 30s, (iv) no
   harmful-interference correlation. These are quantitative and falsifiable.
   PASS.

4. **Falsification condition:** "If composed singular values deviate > 10% from
   sorted union." This targets the proof's core prediction. PASS.

5. **Hyperparameter count:** 3 (eta_sup, eta_amp, n_calibration). Acknowledged
   as coming from the paper, not the proof. Noted as moot since the proof
   predicts surgery is unnecessary. PASS.

6. **Hack check:** Correctly identifies this as testing whether a fix is
   necessary, not adding a fix. PASS.

All 6 self-test items present and reasonable. No blanks or evasions.

## Mathematical Soundness

### Theorem 1a: Cross-term cancellation in left Gram matrix

**Claim:** Delta_comp @ Delta_comp^T = sum_i s_i^2 B_i B_i^T.

**Verification:** Step by step:
- Delta_comp = sum_i s_i B_i A_i where B_i in R^{d_out x r}, A_i in R^{r x d_in}
- Delta_comp @ Delta_comp^T = sum_{i,j} s_i s_j B_i (A_i A_j^T) B_j^T
- By Grassmannian orthogonality: A_i A_j^T = 0 for i != j
- By construction: A_i A_i^T = I_r (orthonormal rows)
- Therefore: sum reduces to sum_i s_i^2 B_i I_r B_i^T = sum_i s_i^2 B_i B_i^T

**Verdict: Correct.** Each step follows from the prior. The key insight that
cross-terms vanish in the LEFT Gram matrix (but not necessarily the right) is
correctly identified.

### Theorem 1b: Union of spectra under B-orthogonality

**Claim:** If additionally B_i^T B_j = 0, singular values are the union of
individual scaled SVs.

**Verification:** If col(B_i) perp col(B_j), then sum_i s_i^2 B_i B_i^T is
block-diagonal in the union of column spaces. Eigenvalues of a block-diagonal
matrix are the union of block eigenvalues. Each block s_i^2 B_i B_i^T has
eigenvalues s_i^2 sigma_k(B_i)^2. Taking square roots gives singular values
{s_i sigma_k(B_i)}.

**Verdict: Correct.**

### Theorem 1c: Weyl interlacing for general B

**Claim:** In general, SVs satisfy Weyl interlacing but are not the union.

**Verdict: Correct** but imprecise. Weyl's inequality applies to the
eigenvalues of sum_i s_i^2 B_i B_i^T as a sum of PSD matrices. The statement
is standard.

### Corollary 1: Surgery cannot help under perfect orthogonality

**Claim:** Under both A and B orthogonality, removing any singular component
can only hurt.

**Verdict: Correct** as a logical consequence of Theorem 1b. If every SV maps
to exactly one adapter, removing it removes that adapter's signal.

### Corollary 2: Surgery target is B-matrix overlap

**Claim:** The only potential target for surgery is B_i^T B_j != 0.

**Verdict: Correct** but the claim should be stated more carefully. B-matrix
overlap creates eigenvalue mixing in sum_i s_i^2 B_i B_i^T (as shown in the
worked example where rank drops from 4 to 2 because B matrices share the same
2D output space). The proof correctly identifies this as the ONLY source of
spectral "mixing" given Grassmannian A-matrix orthogonality. The question is
whether this mixing is surgically exploitable.

### Worked Example (Section F)

The d=8, r=2, N=2 example is manually verified:
- B_1 B_1^T = diag(9, 1). Correct.
- B_2 B_2^T = [[4.25, 2], [2, 4.25]]. Let me check: B_2 = [[2, 0.5], [0.5, 2]],
  B_2 B_2^T = [[2*2+0.5*0.5, 2*0.5+0.5*2], [0.5*2+2*0.5, 0.5*0.5+2*2]] =
  [[4.25, 2], [2, 4.25]]. Correct.
- Sum = [[13.25, 2], [2, 5.25]]. Correct.
- Eigenvalues: trace = 18.5, det = 13.25*5.25 - 4 = 69.5625 - 4 = 65.5625.
  lambda = (18.5 +/- sqrt(18.5^2 - 4*65.5625))/2 = (18.5 +/- sqrt(342.25 - 262.25))/2
  = (18.5 +/- sqrt(80))/2 = (18.5 +/- 8.944)/2 = {13.72, 4.78}. Correct.
- Singular values: {3.70, 2.19}. Correct.

**Key insight from the example is correctly drawn:** when B matrices share
output dimensions, the composed delta has LOWER effective rank than the sum of
individual ranks, and constructive interference boosts the top SVs.

### Hidden Assumptions

1. **A_i A_i^T = I_r (orthonormal rows).** Stated and justified by the
   Grassmannian construction. Empirically confirmed.

2. **B-matrix cosine ~0.03.** Measured empirically as 0.028. This is the
   critical assumption: if B-cosine were much higher, the spectral deviation
   from "union of individuals" would be larger, and surgery might find genuine
   targets.

3. **SV magnitude as sensitivity proxy.** MATH.md does not prove this is a
   valid proxy. The paper (arXiv 2603.03995) uses gradient-based sensitivity.
   PAPER.md correctly flags this as a limitation. However, the negative
   correlation finding (low-SV = domain-pure) is a structural property that
   holds regardless of sensitivity metric. This is because the structural claim
   follows from the proof (overlap concentrates in top eigenvalues of the sum
   of PSD matrices, a standard result), not from the sensitivity metric choice.

4. **Scale factors create spectral dominance.** The 20:1 scale ratio
   (medical/code/math at 20.0 vs finance at 1.0) means medical/code/math
   adapters contribute 400x more spectral energy than finance. This is not
   explicitly modeled in the proof's predictions but is noted in PAPER.md's
   reference to Finding #277. It makes the "surgery has no target" conclusion
   even stronger: surgery's median-based threshold will classify finance
   components as "harmful" simply because they are small-scale, not because
   they are interference.

### Dimensional Error in MATH.md Section C

The derivation openly shows the author catching a dimensional error in the
initial formulation (lines 57-76: "Wait -- let me be precise about
dimensions"), correcting it, and proceeding. While unconventional for a formal
proof, this is honest and the final derivation is correct. The initial
formulation had Delta_i = s_i B_i^T A_i^T which does not contract correctly;
the corrected formulation Delta_i = s_i B_i A_i (with B_i in R^{d_out x r},
A_i in R^{r x d_in}) is dimensionally sound.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table with 6 entries.

| # | Prediction | Measured | Match? | Assessment |
|---|-----------|----------|--------|------------|
| P1 | Spectral deviation < 5% | 8.35% | NO | Prediction too tight. B-overlap effect underestimated. |
| P2 | Surgery < 0.5% improvement | Mean -5.0% (all worse) | YES | Direction correct, magnitude underestimated by 10x. |
| P3 | Calibration > 30s | 600.2s | YES | Correct direction, magnitude underestimated by 20x. |
| P4 | No harmful-interference correlation | -0.587 (strong negative) | PARTIAL | Correlation exists but inverted. |
| P5 | Gram error < 1% | 7.5% | NO | Substantially exceeded. |
| P6 | B-cosine ~0.03 | 0.028 | YES | Match. |

**Assessment:** 2 clean matches (P2, P6), 1 correct direction (P3), 1 partial
(P4 -- correlation exists but is inverted, which is MORE informative than no
correlation), 2 misses (P1, P5).

The misses (P1, P5) both concern the same thing: the proof assumed B-matrix
overlap of 0.03 would create negligible deviation, but 8.35% spectral
deviation and 7.5% Gram error are non-negligible. However, PAPER.md provides a
reasonable explanation: numerical precision in float32 with scale factors up to
20.0 on 2560-dimensional matrices accumulates rounding error. This is plausible
but not proven. An alternative explanation is that the bound on off-diagonal
contribution was too loose (MATH.md section D actually gives up on computing
the bound: "this needs more careful calculation with actual norms").

**Critical note on P5:** The 7.5% Gram error is between Delta_comp @
Delta_comp^T (computed directly from the composed delta) and sum_i Delta_i @
Delta_i^T (computed from individual deltas). Theorem 1a says these should be
EQUAL. The 7.5% gap could be:
(a) Floating-point arithmetic differences (different summation orders), or
(b) The A-matrices not being perfectly orthogonal (|cos| = 0.00125, not exactly 0).

With |cos(A_i, A_j)| = 0.00125, the cross-term A_i A_j^T is not exactly zero.
Its contribution to the Gram matrix is s_i s_j B_i (A_i A_j^T) B_j^T. With
s_max = 20, and the Frobenius norm of the cross-term being O(0.00125 * 20^2)
= O(0.5) relative to the main diagonal terms, this could explain several
percent of Gram error. PAPER.md does not perform this calculation. This is a
minor gap: the finding "surgery does not help" does not depend on whether the
Gram error is from numerical precision or from imperfect A-orthogonality.

## NotebookLM Findings

Skipped. The experiment is already killed with clear negative results. The
mathematical analysis above covers the necessary rigor for a killed experiment
with well-documented results.

## Novelty Assessment

### Prior Art

This is the SECOND spectral surgery experiment in the project. The first
(exp_bitnet_spectral_surgery, KILLED) applied surgery to individual adapters
pre-composition and found that short-trained adapters have efficient spectra --
nothing to fix. This experiment tests a different target: the COMPOSED delta
post-summation, where B-matrix overlap could create new artifacts.

The distinction is meaningful: individual adapters can have clean spectra while
the composition introduces new spectral structure via B-matrix interaction. The
mathematical analysis of this interaction (Theorem 1 and its corollaries) is
the novel contribution.

### Delta Over Prior Experiment

The first spectral surgery experiment killed the technique because adapters are
spectrally efficient (no noise to remove). This experiment provides a
STRUCTURAL explanation for why post-composition surgery also fails: under
Grassmannian orthogonality, B-matrix overlap concentrates in TOP singular
directions (constructive interference), while BOTTOM SVs are clean,
domain-pure signals. This inverts surgery's premise. This is a genuinely new
finding not present in the first experiment.

### External Prior Art

arXiv 2603.03995 (Spectral Surgery) is the source technique. No prior work
testing spectral surgery on Grassmannian-orthogonal adapter compositions was
identified. The negative result and the structural explanation (spectral
inversion under Grassmannian orthogonality) are novel.

## Macro-Scale Risks (advisory)

1. **B-matrix overlap may increase at higher rank.** At r=16 with d_out=2560,
   each B_i spans at most 16/2560 = 0.625% of the output space. At r=128
   (8x), each spans 5%, and N=25 adapters collectively span 125% -- overlap
   becomes mandatory. The "surgery has no target" finding may not hold at
   higher rank.

2. **The spectral inversion finding may not hold with non-uniform scales.**
   The current scale distribution (20:1) creates extreme spectral dominance.
   With more balanced scales, the mixing pattern could change, and low-SV
   components might no longer be domain-pure.

3. **Domain purity decreasing in later layers** (0.711 to 0.675) suggests that
   at greater depth (larger models), B-matrix overlap could become significant
   enough to create genuine interference artifacts. This is worth monitoring
   but does not invalidate the micro-scale finding.

## Additional Observations

### Gram Error Deserves More Analysis

The 7.5% Gram error is the most interesting loose end. Theorem 1a predicts
exact equality. The experiment should have decomposed the error into:
(a) contribution from A-matrix imperfect orthogonality (|cos| = 0.00125), and
(b) floating-point precision.

A simple test: compute A_i A_j^T for the sampled layers and check if
||A_i A_j^T||_F * s_i * s_j accounts for the Gram error. This was not done.
However, since the experiment is killed regardless, this is not blocking.

### The -0.587 Correlation Is the Key Finding

The experiment's most valuable output is the strong negative correlation
between SV magnitude and domain purity. This is not just "surgery did not
help" -- it reveals a structural property of Grassmannian-orthogonal
compositions: constructive interference (B-matrix overlap) concentrates in the
TOP singular directions, while domain-pure signals live in the bottom. This
inverts the assumptions of any magnitude-based spectral cleanup method.

This finding should be recorded as a standalone result because it has
implications beyond spectral surgery: any method that assumes "large SVs =
useful, small SVs = noise" will fail on Grassmannian-orthogonal compositions.
This includes truncated SVD compression, low-rank approximation of the composed
delta, and similar techniques.

### SV Magnitude as Proxy: Limitation Is Real But Not Fatal

The code (lines 556-565) uses SV magnitude as the sensitivity proxy instead of
gradient-based sensitivity from the paper. This is a genuine limitation: the
paper's gradient-based approach might identify different components as
"harmful." However, the structural finding (negative correlation between
magnitude and domain purity) is independent of which sensitivity metric is used
because it describes the SPECTRAL STRUCTURE, not the sensitivity ordering. The
claim "surgery would fail even with gradient-based sensitivity" is supported by
the structural argument but not directly tested.

## Verdict

**KILL confirmed.**

The kill is justified on all fronts:

1. **Mathematical framework is sound.** Theorem 1 is correctly proved. The
   dimensional stumbles in the derivation are corrected. The worked example
   verifies the algebra. The predictions follow logically from the theorem.

2. **Predictions are partially verified.** 2/6 exact matches, 1 correct
   direction, 1 informatively wrong (inverted correlation is MORE informative
   than no correlation). 2 misses (P1, P5) concern the same gap (B-matrix
   overlap creates more spectral deviation than predicted), but this gap
   actually STRENGTHENS the kill: more mixing means surgery's targets are even
   more entangled with useful signal.

3. **Kill criteria are unambiguous.** K696 FAIL (surgery hurts -1.4% to
   -7.8%), K697 FAIL (600s vs 30s target), K698 technically PASS but
   structurally inverted (correlation is negative, meaning surgery's premise
   is wrong).

4. **The impossibility structure is well-characterized.** Under Grassmannian
   orthogonality, B-matrix overlap concentrates in top SVs (constructive
   interference). Bottom SVs are domain-pure. Spectral surgery targeting low
   SVs as "interference" is structurally inverted. This is a genuine
   impossibility insight, not just "it did not work."

5. **Relationship to prior experiment is clear.** First experiment killed
   per-adapter surgery (efficient spectra). This experiment kills
   post-composition surgery (structural inversion). Together they close the
   spectral surgery research direction for Grassmannian-orthogonal compositions.

The experiment is a well-executed negative result with a structurally
informative failure mode. No revisions needed. The kill is clean.
