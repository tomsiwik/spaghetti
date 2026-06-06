# Peer Review: Spectral Surgery on BitNet-2B LoRA Adapters

## NotebookLM Findings

Skipped. The experiment is already killed with triple-FAIL on all kill criteria. A deep NotebookLM review is not warranted for a completed negative result -- the analysis below covers the necessary rigor.

## Mathematical Soundness

### Derivations

The MATH.md derivation of gradient sensitivity reconstruction is correct in principle:

1. **SVD of delta_W = A @ B * alpha**: Valid. Rank at most r=16, so thin SVD gives exactly r nonzero singular values. Correct.

2. **Gradient reconstruction via pseudo-inverse**: The formula G = (1/scale) * dL/dA @ pinv(B^T) where pinv(B^T) = (B @ B^T)^{-1} @ B is dimensionally correct: dL/dA is (d_in, r), pinv(B^T) is (r, d_out), giving G as (d_in, d_out). Sound.

3. **Sensitivity g_k = diag(U_r^T @ G @ V_r)_k**: Correct. This extracts the projection of the gradient onto each rank-1 component u_k * v_k^T.

4. **Signed update with exp(-g_eff)**: This is the paper's proposed mechanism. Mathematically well-defined.

5. **Nuclear-norm renormalization**: Preserves L1 mass of singular values. This is the correct constraint from the paper.

### Implementation vs. Math

**One concern in the code (lines 530-547)**: The code reconstructs g_k using the *original* B matrix from the LoRA factorization rather than the SVD factors. This is correct but introduces a subtle numerical coupling: since A and B are the *trained* factorization (not the SVD factorization), the pseudo-inverse path through B may amplify noise differently than going through the SVD factors directly. In practice, at r=16 with regularization (1e-6 * I added to BBt at line 532), this is unlikely to cause issues. Not a blocking concern.

**Reconstruction (lines 582-584)**: The sqrt split A_new = U_r * sqrt(S'/alpha), B_new = sqrt(S'/alpha) * Vt_r is correct: A_new @ B_new * alpha = U_r @ diag(S') @ Vt_r = delta_W'. Sound.

### Hidden Assumptions

1. **Assumption that B @ B^T is well-conditioned**: Regularized with 1e-6 * I, which is appropriate for r=16 at float32 precision. Not problematic.

2. **Independence of gradient estimates across calibration examples**: Standard assumption. Fine for 128 examples.

3. **The paper's core assumption -- inefficient spectra**: Correctly identified as violated in MATH.md. This is the root cause of failure and the paper honestly acknowledges it.

**Verdict on math: Sound.** No errors found. The failure is not mathematical but empirical (wrong precondition).

## Novelty Assessment

### Prior Art

The experiment straightforwardly applies arXiv 2603.03995 (Spectral Surgery) to the project's ternary LoRA adapters. There is no novelty claim -- this is an application experiment testing whether an existing technique works in a new setting. That is an appropriate use of a micro-experiment.

### Delta Over Existing Work

The negative result itself is a useful data point: Spectral Surgery does not help short-trained rank-16 adapters on ternary base models. The paper correctly identifies three root causes (efficient spectra, zero-sum nuclear norm constraint, gradient noise), which constitute genuine insight about when this technique does and does not apply.

No prior work testing Spectral Surgery on ternary (BitNet) base models was identified. The negative result is novel in that narrow sense.

## Experimental Design

### Does This Test What It Claims?

Yes. The experiment claims to test whether Spectral Surgery improves adapter quality as a training-free post-hoc refinement. It measures PPL, KR-Test, speed, and composition impact before and after surgery. All kill criteria are directly tested.

### Controls

**Adequate.** The experiment uses the same adapters, same eval data, same KR-Test pairs for before/after comparison. The base model KR-Test is also reported for context. The composition impact (cosine and composed PPL) is measured as a secondary safety check.

### Could a Positive Result Be Explained More Simply?

Not applicable (negative result). But worth noting: even the marginal python improvement (+0.4% PPL) is within noise given 25 validation batches.

### Potential Confounds

1. **Calibration data = training data**: The paper uses calibration data from the training domain. This is standard practice per the source paper, but if surgery were tested with cross-domain calibration, results might differ. The PAPER.md Limitations section acknowledges this.

2. **Hyperparameters eta_sup=1.0, eta_amp=0.5**: These are reasonable defaults per the source paper's description, but the paper might use different values. However, given the zero-effect on KR-Test and sub-1% PPL changes, hyperparameter tuning is extremely unlikely to change the verdict.

### Kill Criteria Consistency

The HYPOTHESES.yml kill criteria match exactly what was tested:
- K1: PPL not better (tested: 1/5 improved, threshold majority)
- K2: KR-Test not better (tested: 0/5 improved, threshold majority)
- K3: >5 min per adapter (tested: 420-468s, threshold 300s)

All three FAIL. The kill is clean and unambiguous.

### Minor Issue: KR-Test Sensitivity

The KR-Test shows *exactly* 0.000 delta across all 5 domains. This could mean either (a) surgery has no effect on discrimination, or (b) the KR-Test at n=35-50 pairs lacks sensitivity to detect sub-1% PPL changes. Given that the PPL changes themselves are sub-1% and inconsistent in direction, interpretation (a) is more parsimonious. The PAPER.md correctly notes this.

## Hypothesis Graph Consistency

The experiment matches `exp_bitnet_spectral_surgery_quality_gate` in HYPOTHESES.yml. Status is `killed`. Kill criteria match. Evidence entry accurately summarizes the results. No inconsistency.

The experiment correctly references the dependency on `exp_bitnet_kr_test_evaluation` (KR-Test infrastructure). It blocks nothing (blocks: []).

## Integration Risk

None. The experiment is killed. It does not produce any artifact that enters the architecture.

The one actionable finding -- SVD re-factorization increases inter-adapter cosine by 3.2x -- is correctly flagged. If any future work applies per-adapter SVD transformations, a composition-aware constraint would be needed to avoid rotating adapter subspaces toward shared directions.

## Macro-Scale Risks (advisory)

Not applicable. The mechanism is killed. However, for the record:

1. If adapters were trained for many more steps (thousands of iterations with learning rate scheduling), the "inefficient spectrum" precondition might be met, and surgery could become viable. This is a conditional that applies only if the project ever moves to longer training regimes.

2. The speed bottleneck (128 backprop passes through 2.4B) would be proportionally worse on larger base models and would not improve at scale without algorithmic changes (e.g., Hessian-free sensitivity estimation).

## Verdict

**KILL confirmed.**

The experiment is a clean, well-executed negative result. All three kill criteria fail unambiguously. The mathematical implementation is correct -- the failure is that the technique's precondition (inefficient adapter spectra) is not met for short-trained rank-16 LoRA on BitNet-2B-4T. The root cause analysis is thorough and actionable: spectral surgery addresses a problem these adapters do not have, and the nuclear-norm constraint makes the intervention zero-sum when spectra are already efficient.

The 3.2x cosine increase is a useful secondary finding: SVD re-factorization harms composition orthogonality, which should be noted in FINDINGS.md if any future experiment proposes adapter-level SVD transformations.

No revisions needed. The kill is justified. Move on.
