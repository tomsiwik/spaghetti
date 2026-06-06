# Adversarial Review: exp_m2p_composition_n5_qwen3

## Verdict: REVISE

## Experiment Type
Claimed: verification (Type 1).
Actual: The experiment claims to verify Theorem 1 (Grassmannian orthogonal composition), but the code does not implement the mechanism Theorem 1 describes. This makes the experiment a mismatch between proof and implementation.

## Hack Detector
- Fix count: 3 mechanisms (Grassmannian A, TF-IDF routing, 1/N scaling). Each addresses a different failure mode. Not a hack stack -- each is independently motivated. However, only TF-IDF routing was actually tested as designed.
- Is MATH.md a proof or a description? Theorem 1 is a correct proof. Theorem 2 is a trivially true statement (routing on input is invariant to model state -- this is a definition, not a theorem). Theorem 3 is a proof sketch with an explicit caveat that undermines its own bound.
- Metric used as evidence: quality_ratio (composed_acc / single_acc in code; (composed - base) / (single - base) in MATH.md). The two formulas are inconsistent.
- Kill criteria source: K925 derived from Theorem 5 (v3). K926 derived from Theorem 2. K927 threshold of 0.75 is reasonable but the formula mismatch between MATH.md and code is a problem.

## Self-Test Audit

1. **One-sentence impossibility property:** Lists TWO properties (interference and routing). Should be one. Minor issue.

2. **Cited theorems:** QR decomposition, Frobenius identity, LoraRetriever, Finding #50, Theorem 5 (v3) -- all real and correctly cited. However, Finding #50 verified multi-domain Grassmannian with SEPARATE A-matrices per domain. This experiment does NOT use separate A-matrices.

3. **Predicted numbers:** Specific and falsifiable. K925: grad_norm 1.5-6.3. K926: >= 95%. K927: >= 0.75 with lower bound 1.15. All good in principle, though the 1.15 bound assumes the wrong formula.

4. **Falsification condition:** "Theorem 1 is falsified if grad_norm=0" -- this is wrong. K925 tests gradient flow (Theorem 5), not Grassmannian orthogonality (Theorem 1). Nothing in the experiment actually measures A_math^T A_sort.

5. **Hyperparameter count:** Claims 0 new hyperparameters. Correct.

6. **Hack check:** Claims no new tricks. Fair -- these are reused mechanisms.

## Mathematical Soundness

### Theorem 1 (Grassmannian Orthogonality) -- CORRECT proof, WRONG experiment

The proof itself is mathematically correct:
- QR decomposition gives orthonormal columns (standard linear algebra).
- Partitioning Q into blocks gives A_i^T A_j = 0 for i != j (correct block extraction from identity).
- tr(B_math^T * 0 * B_sort) = 0 (correct).

**BLOCKING ISSUE:** The proof requires A_math != A_sort, drawn from different columns of Q. The code uses the SAME A-matrices for both M2P networks. Both m2p_math and m2p_sort call `load_lora_a_matrices_v2()` and use identical A_q_layers, A_v_layers. The composition is:

```
DW_composed = A * (0.5 * B_math + 0.5 * B_sort)
```

The Frobenius inner product between the two adapter contributions is:

```
<A * B_math, A * B_sort>_F = tr(B_math^T * A^T * A * B_sort)
```

Since A has orthonormal columns (from QR), A^T * A = I_r, so:

```
<DW_math, DW_sort>_F = tr(B_math^T * B_sort)
```

This is NOT zero in general. It is zero only if B_math and B_sort happen to be orthogonal, which is not guaranteed and not measured. Theorem 1's guarantee of zero interference for ANY B-matrices requires DIFFERENT A-slots.

**The entire orthogonality claim is vacuous for this experiment.** The experiment tests additive B-matrix blending within a single A-slot, not Grassmannian multi-slot composition.

### Theorem 2 (TF-IDF Routing Invariance) -- TRIVIALLY TRUE but IRRELEVANT to composition eval

The statement "Route(x) depends only on input text x, not on model parameters theta" is a definition of the routing function, not a theorem. It is trivially true because TF-IDF never calls the model.

More importantly: the routing is measured (100% accuracy) but NEVER USED for adapter selection. The composed evaluation always applies `0.5 * B_math + 0.5 * B_sort` regardless of routing decision. The routing accuracy is a diagnostic metric, not part of the composition mechanism.

### Theorem 3 (Quality Lower Bound) -- DOES NOT APPLY

Theorem 3 models composition as: with probability rho, apply the correct single adapter; with probability (1-rho), apply the wrong adapter. This assumes ROUTING-BASED SELECTION.

The actual experiment applies ADDITIVE BLENDING (both adapters always active at 0.5 weight). The quality_ratio formula in MATH.md is `(composed - base) / (single - base)`, but the code uses `composed / single`. These give different values:

- MATH.md formula: (0.23 - 0.20) / (0.24 - 0.20) = 0.75
- Code formula: 0.23 / 0.24 = 0.958

The reported quality_ratio of 0.958 uses the code formula, not the MATH.md formula. Under the MATH.md formula, quality_ratio_math = 0.75, which is exactly at the kill threshold.

### Summary of Mathematical Issues

| Theorem | Proof correct? | Experiment implements it? |
|---------|---------------|--------------------------|
| Theorem 1 (Grassmannian orthogonality) | Yes | NO -- same A-matrices for both domains |
| Theorem 2 (TF-IDF invariance) | Trivially true | Measured but not used in composition |
| Theorem 3 (Quality lower bound) | Correct given assumptions | NO -- additive blending, not routed selection |

## Prediction vs Measurement

The PAPER.md prediction-vs-measurement table exists and is well-formatted. However:

| Kill Criterion | Issue |
|---|---|
| K925 (grad_norm > 0) | PASS is real -- gradient flow works. But this verifies Theorem 5 (functional forward), NOT Theorem 1 (orthogonality). |
| K926 (routing >= 80%) | PASS is real -- TF-IDF separates the two task formats trivially. But routing is never used for adapter selection. |
| K927 (quality_ratio >= 0.75) | MATH.md formula gives 0.75 for math (borderline), code formula gives 0.958. Sort fails because the adapter never learned the task. |

The prediction-vs-measurement table conflates two different quality_ratio formulas without noting the discrepancy.

## Novelty Assessment

The idea of composing M2P-generated adapters on a real LLM is genuinely novel relative to the toy-scale synthetic experiments. The TF-IDF routing fix for the prior covariate-shift failure (Finding #351) is a sound engineering contribution.

However, the Grassmannian orthogonality mechanism -- the central theoretical contribution -- is not actually implemented. The experiment is an additive B-matrix blending test, not a Grassmannian composition test.

## Macro-Scale Risks (advisory)

1. At scale, additive blending of B-matrices within a shared A-slot will have non-zero interference proportional to tr(B_i^T B_j). This grows with adapter norm and is not bounded.
2. The sort task failure at 300 steps suggests M2P training budget may need to be task-dependent, not uniform.
3. The quality_ratio formula discrepancy needs to be resolved before any finding is recorded.

## Finding Status Assessment

The KILL decision is correct but for the wrong reason. The paper says K927 failed because the sort adapter did not learn. The deeper issue is that the experiment does not implement the mechanism it claims to verify (Grassmannian multi-slot composition). The KILL should also cite:

1. Sort adapter did not learn the task (acknowledged)
2. Both adapters share the same A-slot, so Theorem 1 is not tested (not acknowledged)
3. quality_ratio formula mismatch between MATH.md and code (not acknowledged)
4. Routing is measured but not used in composition (not acknowledged)

Status: KILLED is appropriate, but the failure mode analysis in PAPER.md is incomplete.

## Blocking Issues (must fix for PROCEED)

1. **A-matrix architecture mismatch (CRITICAL).** MATH.md proves interference-free composition via separate Grassmannian A-slots (A_math^T A_sort = 0). The code uses a single A for both M2P networks. Either: (a) implement separate A-slots per domain as described in MATH.md, or (b) rewrite MATH.md to describe what the code actually does (additive B-blending within a shared subspace) and derive new guarantees for that setting.

2. **quality_ratio formula mismatch.** MATH.md: `(composed - base) / (single - base)`. Code: `composed / single`. Pick one and be consistent. Under MATH.md formula, math quality_ratio = 0.75 (borderline pass, not 0.958).

3. **Routing not used in composition.** Theorem 3 models routed selection but the code applies additive blending. Either use routing to select which adapter to apply, or rewrite Theorem 3 to model additive blending.

4. **Sort adapter convergence is a precondition, not an independent failure.** The experiment needs a convergence check (e.g., sort_single_acc > some threshold) before evaluating composition. Running K927 on an adapter that produced 0% accuracy is not a composition test.

## Non-Blocking Issues

5. **Theorem 2 is a definition, not a theorem.** "A function of x that does not depend on theta is invariant to theta" is tautological. This is fine as a design observation but should not be called a theorem with QED.

6. **Self-test question 1 lists two properties.** Should be one.

7. **Theorem 3 proof is a "sketch" not a proof.** The QED is marked "modulo the empirical step," which means it is not actually proven. The caveat (acc_wrong_routing could be less than acc_base) undermines the bound.

## Recommendations

1. **Immediate fix:** Implement separate Grassmannian A-slots per domain. Generate A_math = Q[:,0:r] and A_sort = Q[:,r:2r] from the QR construction. Each M2P network uses its own A-slot. This is what MATH.md describes and what Finding #50 verified at toy scale.

2. **Fix sort adapter convergence:** Either increase training steps (1000+), use a simpler evaluation metric (word overlap / F1 instead of exact match), or add a convergence gate before composition testing.

3. **Decide on routing model:** Either implement routed selection (apply only the routed adapter at full weight) which matches Theorem 3, or model additive blending mathematically and derive appropriate kill criteria.

4. **Resolve quality_ratio formula.** The MATH.md relative-to-base formula is more informative. Use it consistently.

5. **Record the TF-IDF routing fix as a separate finding.** K926 PASS (100% routing) with TF-IDF vs the prior 36.6% with hidden-state routing is a genuine result that should not be lost in the KILL.
