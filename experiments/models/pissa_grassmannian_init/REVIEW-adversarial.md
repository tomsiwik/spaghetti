# Peer Review: PiSSA SVD-Init vs Grassmannian Init

## NotebookLM Findings

Skipped (local review sufficient -- the mathematical claims are straightforward
and the code is self-contained).

## Mathematical Soundness

### What holds

1. **PiSSA gives the same A to all adapters per weight matrix.** This is
   mathematically trivial: SVD of a matrix W is unique (up to sign/rotation
   of degenerate eigenspaces). If all adapters target the same W, they get
   the same V[:, :r]. The cos(A_i, A_j) = 1.0 claim is correct by construction.

2. **Ternary SVD spectrum flatness.** The analysis that ternary matrices
   {-1, 0, +1} have flatter spectra than float weights is correct. The
   measured 32.8% variance at rank-8 out of effective rank ~58 is consistent
   with the prediction r/rank(W) ~ 8/58 ~ 13.8%. The actual 32.8% is higher
   than the uniform prediction, suggesting the trained ternary weights do
   develop SOME spectral structure (not purely random), but far less than
   float weights (40-60%). Sound analysis.

3. **Interference bound derivation.** The bound
   `||dW_i^T dW_j|| = (scale/r)^2 * ||B_i|| * ||A_i^T A_j|| * ||B_j||`
   is correct for the factored form delta_W = A @ B. The conclusion that
   A_i = A_j => ||A_i^T A_j|| = ||I_r||_F = sqrt(r) is correct.

### What does not hold

4. **The "Grassmannian" init is NOT Grassmannian.** The function
   `grassmannian_ap_init` (line 331) generates N independent random QR
   frames with NO Alternating Projection and NO cross-frame orthogonality
   enforcement. The code:

   ```python
   for i in range(N):
       M = rng.randn(d, r).astype(np.float32)
       Q, _ = np.linalg.qr(M)
       frames[i] = Q[:, :r]
   ```

   This produces N frames where each A_i has orthonormal columns, but
   A_i^T A_j != 0 in general. At N=5, r=8, d=64 (Nr/d = 0.625), random
   frames will have small but non-zero cross-coherence (~0.1-0.2 expected
   by concentration of measure). The paper claims "A_i^T A_j = 0 for all
   i != j" and "interference bound = 0" for the Grassmannian condition --
   this is FALSE for the code as written.

   **Impact on conclusions:** This weakens but does not kill the main finding.
   The PiSSA-frozen condition has cos(A_i, A_j) = 1.0 (maximum), while
   the "Grassmannian" condition has small but non-zero coherence. The
   comparison is still directionally valid (near-orthogonal vs identical),
   but the claim of "guaranteed zero interference" is overstated for this
   implementation. The experiment's own prior work (bitnet_grassmannian_init)
   uses proper Alternating Projection -- this experiment should have reused
   that code.

5. **Cosine measurement for frozen-A conditions is broken.** The paper
   acknowledges this (Limitations section, line 155-156), but the data
   still reports Grassmannian cosine = 0.0 and PiSSA-frozen cosine = 0.0.
   This is because `get_lora_params` only extracts trainable parameters,
   and frozen A is not trainable. The `get_adapter_delta_vector` function
   computes `A @ B` using whatever A it finds in saved params -- but for
   frozen A, it finds nothing, so the delta is computed from B alone.

   For PiSSA-frozen: the TRUE delta cosine should be HIGH (all adapters
   share identical A, so delta = A @ B_i and cos(delta_i, delta_j) depends
   on cos(B_i, B_j) amplified by shared A). The reported 0.0 is wrong.

   For Grassmannian: the TRUE delta cosine should be NEAR-zero (different A
   matrices filter B correlations). The reported 0.0 happens to be correct
   for the wrong reason.

   **Impact:** K2 is assessed on PiSSA-unfrozen (0.784), which IS correctly
   measured (A is trainable and saved). But the paper's comparison of
   frozen-condition cosines is invalid. This matters for the composition
   ratio analysis -- we cannot attribute the Grassmannian composition
   advantage to orthogonality when the cosine measurement is broken.

6. **Composition methodology is inconsistent across conditions.** For
   Grassmannian (lines 801-814), the code applies adapter 0's A matrices
   and then... does nothing (the `pass` statement). It falls through to
   the B-averaging code below. But Grassmannian adapters have DIFFERENT A
   matrices per adapter -- averaging B matrices while using adapter 0's A
   is not a valid composition method for Grassmannian. The correct composition
   would be sum(A_i @ B_i) / N.

   For PiSSA-frozen: averaging B with shared A is valid since all A are
   identical.

   For PiSSA-unfrozen: averaging both A and B (lines 831-834) is a
   reasonable approximation.

   **Impact:** The Grassmannian composition PPL (1.8850) is computed using
   an incorrect composition method (adapter-0's A with averaged B). This
   means the Grassmannian composition advantage may be overstated or
   understated -- we cannot tell.

## Novelty Assessment

**Prior art:** PiSSA (Fang et al., 2404.02948) is well-cited. The specific
question "does PiSSA work for multi-adapter composition?" is novel and
relevant. No published work tests PiSSA in a composition setting.

**Delta:** The insight that PiSSA gives identical A matrices to all adapters
targeting the same weight is trivially obvious from the PiSSA definition,
but no one has stated it as a composition blocker before. The ternary SVD
flatness analysis is a genuinely useful finding.

**Prior work in this codebase:** The `bitnet_grassmannian_init` experiment
has a proper Alternating Projection implementation that should have been
reused. The `mixed_rank_grassmannian_capacity` experiment has an even more
sophisticated version. Reinventing the init as a simplified (broken)
version is a miss.

## Experimental Design

### Does this test what it claims?

Partially. The experiment correctly identifies the fundamental PiSSA
limitation (same A for all adapters). The SVD analysis is well-designed.
The single-adapter PPL comparison is valid.

### Controls adequate?

The 3-seed design is good. The 5-domain structure is adequate. However:

- **Confound in PiSSA-unfrozen:** 1.77x more parameters (both A and B
  trainable). The paper acknowledges this but the 8.7% quality gain cannot
  be attributed to SVD init vs. simply having more parameters. A proper
  control would be: random init with unfrozen A (same parameter count,
  different init).

- **No direct A-matrix cosine measurement:** The experiment measures
  delta cosine (vec(A@B)) but never directly measures A_i^T A_j. For the
  frozen conditions this is known by construction, but for PiSSA-unfrozen
  it would be informative to measure how much A matrices actually diverge
  from the shared SVD init.

### Could a positive result be explained by a simpler mechanism?

N/A -- the result is negative (kill). The kill is well-justified by the
mathematical argument alone; the experiment is confirmatory.

## Macro-Scale Risks (advisory)

1. **Ternary SVD flatness may not scale.** At d=2560, the trained ternary
   weights may develop more spectral structure than at d=64. The 32.8%
   variance capture could increase at larger d if the model develops
   meaningful low-rank structure. This would make PiSSA more attractive
   for single-adapter use (but still kills composition).

2. **The fundamental PiSSA composition blocker is scale-invariant.** SVD
   is unique regardless of d. PiSSA-frozen will always give identical A
   to same-weight adapters. This kill is permanent for the frozen-A case.

3. **Domain-specific SVD (different A per domain per weight) was mentioned
   but not tested.** This would be O(N * L * d^3) computation but would
   avoid the shared-A problem. Worth noting but probably not worth pursuing
   given Grassmannian works.

## Verdict

**PROCEED** (as a kill)

The experiment correctly kills PiSSA for composable ternary experts. The
core finding -- PiSSA gives identical A matrices to all adapters, destroying
composition -- is mathematically certain and does not depend on scale.

**Caveats to record in FINDINGS.md:**

1. The "Grassmannian" baseline uses independent random QR, not proper
   Alternating Projection. Cross-adapter orthogonality is approximate (by
   concentration of measure at Nr/d = 0.625), not exact. The comparison
   is still directionally valid.

2. Cosine measurements for frozen-A conditions are broken (only B saved).
   The reported 0.0 for Grassmannian and PiSSA-frozen are artifacts. Only
   the PiSSA-unfrozen cosine (0.784) is correctly measured.

3. Composition PPL for Grassmannian uses an incorrect method (adapter-0's
   A with averaged B instead of sum of A_i @ B_i). The composition ratio
   comparison across conditions is unreliable.

4. PiSSA-unfrozen's 8.7% quality gain is confounded by 1.77x more
   trainable parameters. No random-init-unfrozen-A control exists.

None of these caveats change the kill verdict. The mathematical argument
(same A for all adapters => cos = 1.0 => composition destroyed) stands
independently of the experimental implementation.
