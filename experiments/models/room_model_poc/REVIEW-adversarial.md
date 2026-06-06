# Peer Review: Room Model POC

## Experiment Type
Verification (Type 1) -- MATH.md declares this explicitly.

## Hack Detector
- Fix count: 1 (single operation: sum the deltas). No stacking of fixes. CLEAN.
- Is MATH.md a proof or a description? **Mixed.** Theorem 1 is a genuine proof (trivially correct -- distributive property). Theorem 2 is a DESCRIPTION dressed as a proof. Theorem 3 is a bandwidth calculation, not a theorem. See detailed analysis below.
- Metric used as evidence: Logit MSE, routing accuracy, PPL ratio, tok/s. The first three are reasonable for the claims being tested. PPL ratio is a proxy but appropriate for a kill-or-not decision.
- Kill criteria source: K763 derived from Theorem 1 (but incorrectly -- see below). K764 derived from Theorem 2 (but Theorem 2 does not actually predict >= 60%). K765 is an arbitrary threshold dressed as a proof consequence.

## Self-Test Audit

1. **One-sentence impossibility property:** "Linearity of matrix multiplication: x @ (M1 + M2) = x @ M1 + x @ M2." -- Correct and genuinely one property. PASS.

2. **Cited theorems:** Distributive property of matrix multiplication (real, unconditional). Grassmannian orthogonality from Finding #3, #126 (internal findings, accepted). PASS.

3. **Predicted numbers:** MSE < 1e-6, speed 40-50 tok/s, routing >= 60%. The first two are specific and falsifiable. The routing prediction is asserted without derivation from Theorem 2 -- Theorem 2 says routing weight is *proportional to* projection norm, but provides no bound on discrimination quality. The 60% threshold appears to have been chosen because it is "well above random" (20%), not because the proof predicts it. PARTIAL PASS -- the routing prediction is not derived from the proof.

4. **Falsification condition:** "Theorem 1 cannot be falsified -- it is a direct consequence of linearity." This is technically correct for Theorem 1 as stated (per-module), but the experiment's K763 tests full-model logit MSE, which Theorem 1 does NOT predict. This is a critical mismatch between what was proved and what was tested. The MATH.md says the kill criterion K763 is "achievable if we compute the sum in float32" but this claim conflates per-module and full-model equivalence. FLAG.

5. **Hyperparameter count:** 0. Correct -- no new hyperparameters introduced. PASS.

6. **Hack check:** One operation, no stacking. PASS.

## Mathematical Soundness

### Theorem 1 (Pre-Summed Equivalence): CORRECT but SCOPE IS WRONG

The proof is trivially correct. The distributive property of matrix multiplication is an axiom. x @ sum(DeltaW_i) = sum(x @ DeltaW_i). QED. No issue.

**However:** The theorem proves per-module equivalence only. The statement says "for any input x" and proves that the output of a single linear operation with the summed matrix equals the sum of individual linear operations. This is true and verified experimentally (MSE 5.6e-7).

The critical gap: **K763 tests full-model logit MSE, not per-module MSE.** Theorem 1 does not predict full-model equivalence because the full model contains nonlinear operations (LayerNorm, SiLU, softmax) between linear modules. The MATH.md *acknowledges* this in Section F ("Assumption 1: Linearity holds -- Matrix multiplication IS linear. This cannot break.") but fails to note that the full model is NOT a linear function.

This is an honest error, not a dishonest one. The MATH.md's Worked Example (Section G) correctly shows per-module equivalence. The problem is that K763 was set to test something stronger than what was proved, and the "Note on floating point" in the theorem section discusses float32 vs bf16 precision as if that is the only source of error -- implicitly assuming full-model linearity.

**Verdict on Theorem 1:** Proof correct. Prediction scope wrong. The theorem predicts per-module MSE ~ 0 (verified). It does NOT predict full-model logit MSE ~ 0 (falsified, MSE = 53.9). The kill criterion K763 tests a claim the theorem does not make.

### Theorem 2 (Projection Geometry): DESCRIPTION, NOT PROOF

Theorem 2 "proves" that the contribution of adapter i is proportional to ||h @ A_i^T||. This is algebraically trivially true -- it is just expanding the matrix multiplication. But the *claim* that this constitutes "soft routing" requires the additional assumption that ||h @ A_i^T|| discriminates between domains. The "proof" includes the hedge "If ||B_i|| is roughly constant across adapters (as expected from similar training)..." This is an assumption, not a proved condition.

More critically, Theorem 2's statement that "the adapter whose A-subspace aligns most with h contributes most to the output" is true in the trivial algebraic sense but says nothing about whether domain-specific tokens actually align with their corresponding A-subspaces. The A-matrices are random orthonormal frames on the Grassmannian -- they were chosen for geometric packing, not for semantic alignment.

**The 60% accuracy threshold has no derivation from Theorem 2.** Theorem 2 tells you HOW routing weights are computed, not how ACCURATE they are. Accuracy depends entirely on whether hidden states have domain-specific projections onto the A-subspaces. The MATH.md acknowledges this in Assumption 3 ("This is the main experimental question") but still lists "Domain accuracy >= 60%" as a prediction of the theorem.

**Verdict on Theorem 2:** Description of a mechanism, not a proof of a guarantee. The "QED" is misleading -- what was proved is that the algebra works out to a weighted sum, not that the weights are semantically meaningful. The 60% prediction is aspirational, not derived.

### Theorem 3 (Bandwidth-Speed): CALCULATION, NOT THEOREM

This is a straightforward bandwidth calculation: total bytes / bandwidth = minimum time. The calculation is correct and the prediction (40-50 tok/s) matches the measurement (39.2 tok/s). The slight miss is expected -- 39.2 is at the edge of the predicted range, plausibly from dispatch overhead not fully modeled.

**Verdict on Theorem 3:** Correct calculation, correctly verified. Calling it a "theorem" is generous but the content is sound.

### The Worked Example Correction (Section G)

MATH.md contains a notable moment of honesty: the worked example initially computes DeltaW = B^T @ A^T, gets the wrong answer, catches the error, corrects to DeltaW = A @ B, and gets the right answer. This is good -- it shows the derivation was done live and checked. However, leaving the error and correction in the final document is unusual and suggests the MATH.md was written in a single pass without cleanup.

### Nonlinearity Gap -- The Central Missing Piece

The fundamental issue is that MATH.md does not contain a theorem about full-model behavior. It proves per-module linearity (trivial) and then tests full-model equivalence (which requires a different theorem that was never stated). The gap between these two levels is not a minor technicality -- it is the entire reason the experiment failed.

A complete MATH.md would have stated:

> **Non-Theorem:** Full-model equivalence does NOT follow from per-module equivalence because LayerNorm(h + delta) != LayerNorm(h) + LayerNorm(delta). The full-model output is a composition of linear and nonlinear operations, and the distributive property does not distribute over nonlinear functions.

This should have been apparent before running the experiment. LayerNorm is layer 1 of any transformer block. The fact that f(a+b) != f(a) + f(b) for nonlinear f is undergraduate mathematics.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Evaluation:

| Prediction | Source | Predicted | Measured | Match? | Assessment |
|---|---|---|---|---|---|
| Per-module MSE | Theorem 1 | ~0 | 5.6e-7 | YES | Theorem verified |
| Full-model logit MSE | Theorem 1 (overclaimed) | < 1e-6 | 53.9 | NO | Theorem does not predict this |
| Routing accuracy | Theorem 2 (overclaimed) | >= 60% | 14% | NO | Threshold not derived |
| PPL ratio | Theorem 1 (overclaimed) | <= 1.10 | 1.29 | NO | Theorem does not predict this |
| Speed | Theorem 3 | 40-50 tok/s | 39.2 | YES | Close, within range |
| Memory | Theorem 3 | ~5.8 GB | 5.69 GB | YES | Verified |

Three of six predictions match. The three that fail are the ones that were overclaimed relative to what the proofs actually guarantee.

## NotebookLM Findings

NotebookLM was not run for this review. Manual analysis was sufficient given the scope.

## Novelty Assessment

The Room Model concept combines two known ideas:
1. **LoRA merging/summing** -- well-studied in LoRA Soups (2410.13025), TIES-Merging, Task Arithmetic, etc. The "sum the deltas" operation is standard.
2. **Grassmannian orthogonality for interference control** -- the project's own prior work (Finding #3, #126).

The novel claim is "routing IS the matmul" -- that A-subspace projection norms constitute implicit routing. This is falsified (14% accuracy). The implicit routing claim is dead.

The pre-summing itself is not novel. Finding #300 already tested "Full precompute bandwidth-killed (42.1 tok/s)" and Finding #301 tested "Hybrid additive cost model." The Room Model POC appears to be a re-test of the full-precompute concept with the additional (falsified) claim about implicit routing.

## Experimental Design Assessment

**Phase 1 (Equivalence):** The per-module test is well-designed -- random inputs, float32 computation, correct dimensionality. The full-model test has a SUBTLE METHODOLOGICAL ISSUE: the "sequential" baseline computes base_logits + sum(logits_with_adapter_i - base_logits). This is NOT the same as running all 5 adapters simultaneously. It linearizes the adapter contributions around the base model's hidden states. The room model, by contrast, runs all 5 adapter deltas as a single dense perturbation, so nonlinear interactions compound through all 30 layers.

This means the full-model test is not comparing the room model against "all adapters applied simultaneously" but against "a linearized approximation of all adapters." The MSE of 53.9 captures both:
(a) the nonlinear compounding of simultaneous adapter application, AND
(b) the difference between the linearized sequential sum and the true joint application.

The PAPER.md correctly identifies this issue in the root cause analysis ("This is a general problem with 'all adapters active simultaneously'").

**Phase 2 (Routing):** Methodologically sound. Uses hidden states from the model, projects through A-matrices, measures domain classification accuracy. The choice to use layer 14 (middle) is reasonable. The 14% result is convincing -- with 100 samples across 5 domains, the confidence interval is well below 60%.

**Phase 3 (PPL):** The comparison to v3-composed is informative but, as the PAPER.md honestly notes, the two are fundamentally different operations. The room model applies all 5 deltas at full alpha simultaneously without norm control. This is not a fair comparison -- it is comparing an uncontrolled sum against a norm-rescaled average. The 29% degradation is likely mostly from the Nx alpha amplification, not from the pre-summing per se.

**Phase 4 (Speed):** Clean measurement, prediction verified.

## Quality of Post-Mortem

The PAPER.md's analysis of WHY the experiment failed is excellent. Specifically:

1. The identification of LayerNorm nonlinearity as the root cause for K763 is correct.
2. The observation that A-matrices are "geometrically orthogonal but semantically arbitrary" for K764 is incisive.
3. The diagnosis of Nx alpha amplification without norm control for K765 is correct.
4. The "What Would Fix This" section is honest about what cannot be fixed.

This post-mortem is more valuable than many "passing" experiment reports.

## Critical Issues (Summary)

1. **K763 tests a claim the proof does not make.** Theorem 1 proves per-module equivalence. K763 tests full-model equivalence. The gap (nonlinear operations between modules) should have been identified in MATH.md before the experiment was run.

2. **Theorem 2 is a description, not a proof.** The 60% routing accuracy threshold is not derived from any mathematical guarantee. The theorem shows that routing weights are computed by projection norms, but provides no bound on accuracy.

3. **K765 threshold is arbitrary.** "Within 10% of v3 composed" is not derived from any theorem. Theorem 1 predicts exact per-module equivalence, not approximate full-model quality equivalence.

4. **The full-model nonlinearity should have been a kill-before-running insight.** The fact that LayerNorm(a+b) != LayerNorm(a) + LayerNorm(b) is textbook nonlinearity. This could have been identified from MATH.md alone, saving the compute.

## Macro-Scale Risks (advisory)

Not applicable -- the experiment is killed. For reference: even if the per-module equivalence were exploited in some narrower form, the 4.65 GB bandwidth cost at N=5 would scale linearly with dimension, making it even more expensive at macro scale.

## Verdict

**KILL** -- Confirmed.

The experiment is correctly killed. All three kill criteria failed. The post-mortem analysis is honest and thorough. The finding ("per-module linearity is exact but full-model nonlinearities make multi-adapter pre-summing structurally impossible") is valuable negative knowledge.

However, the MATH.md has a structural defect that should be recorded: **it proved the wrong thing.** Theorem 1 guarantees per-module equivalence (trivially), but the experiment tested full-model equivalence (which requires a different, unstated theorem that is actually false). The kill criteria were not derived from the proofs -- they were derived from hopes about what the proofs might imply at the full-model level.

For the finding record:
- **Finding status:** killed (correct)
- **Failure mode:** Full-model nonlinearity (LayerNorm, SiLU, softmax) breaks per-module additivity. A-subspace projection is semantically arbitrary (14% routing accuracy). Nx alpha amplification without norm control.
- **Impossibility structure:** For any neural network containing at least one nonlinear layer between two linear modules, per-module additivity of adapter deltas does NOT imply full-model additivity. Specifically: LayerNorm(h + delta) = (h + delta - mu) / sigma != LayerNorm(h) + LayerNorm(delta). The cross-term -(h * delta) / sigma^3 compounds multiplicatively through L layers.
- **Valuable insight preserved:** Per-module pre-summing IS exact (MSE 5.6e-7). This means pre-merge strategies that operate within a single module (e.g., bf16 merge for always-on adapters) are mathematically sound. The failure is in attempting to pre-sum across the full model with all adapters active simultaneously.

### What would have prevented the wasted compute

A 5-minute thought experiment before coding: "Does x @ (A+B) = x @ A + x @ B hold when there is a LayerNorm between x and the next layer?" Answer: no. This would have immediately narrowed the scope to per-module verification only, killing K763/K764/K765 as stated before any code was written.

The speed measurement (Theorem 3) and per-module equivalence (Theorem 1, properly scoped) are the two genuinely verified results and could have been the entire experiment, positioned as "guided exploration of bandwidth costs for full-precompute approaches."
