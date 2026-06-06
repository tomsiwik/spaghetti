# Peer Review: Brainstacks Null-Space Validation

## Experiment Type
Verification (Type 1) -- verifying Brainstacks null-space SVD projection (arXiv:2604.01152) on ternary adapters.

## Hack Detector
- Fix count: 1 (null-space SVD projection only). Clean.
- Is MATH.md a proof or a description? Theorem 1 is a valid proof with QED. Theorem 2 is a proof with QED but has issues (see below).
- Metric used as evidence: val loss delta (forgetting), cosine similarity, gradient norm ratio. Val loss is a reasonable proxy for forgetting. Cosine and gradient norm are structural.
- Kill criteria source: K687 and K689 derived from proof predictions. K688 threshold (0.01) is NOT derived from the proof -- the proof predicts leakage norm of 0.08, but the kill threshold is on val loss, which has no proven relationship to leakage norm. This is a disconnect.

## Self-Test Audit

1. **Failure mode:** "Ternary quantization noise leaks into prior domains' subspaces" -- clear and singular. PASS.
2. **Cited theorems:** SVD projection theory (standard), JL-style concentration (invoked loosely -- see below), STE quantization bounds (standard). The JL citation is hand-wavy -- the experiment uses a direct K/d argument, not JL. Minor issue.
3. **Predicted numbers:** Cosine < 0.1, leakage norm <= 0.08, gradient preservation >= 90%. Specific and falsifiable. PASS.
4. **Falsification condition:** "Cross-domain cosine > 0.2, forgetting > 0.01, gradient preservation < 95%." The forgetting threshold 0.01 does not target the proof (proof says 0.08 leakage norm). The gradient threshold 95% contradicts the proof's prediction of 90%. These are stricter than the proof predicts, which means the experiment can fail even when the proof holds. This is backwards -- kill criteria should test the proof, not set arbitrary tighter thresholds. FLAG.
5. **Hyperparameter count:** 2 (NS_TOP_K=64, NS_N_SAMPLES=200). Acknowledged in MATH.md context. PASS.
6. **Hack check:** First experiment on this mechanism. PASS.

## Mathematical Soundness

### Theorem 1: Correct
The proof that (I - P_i) projects out the component in colspan(V_i^K) is standard linear algebra. The key condition -- rank(D_i) <= K -- is satisfied trivially because n_samples (200) < K (64) is false... wait. Actually n_samples=200 > K=64, so rank(D_i) could be up to 200, and K=64 would NOT capture the full column space. The theorem's precondition "rank(D_i) <= K" is NOT guaranteed.

However, the energy_captured computation in the code is broken:
```python
energy_captured = mx.sum(S_k**2) / mx.sum(S_k**2)  # Always 1.0!
```
This divides S_k by itself (both numerator and denominator use S_k, the truncated singular values), so it always returns 1.0. The denominator should use S (all singular values), not S_k. This means the reported "100% energy captured" is a bug, and we have NO idea how much energy the top-64 directions actually capture. This is a BLOCKING issue -- Theorem 1's precondition is unverified.

Note: the `compute_principal_directions` function computes the correct energy ratio on line 327 and logs it. But the results.json stores the buggy version from lines 614-616. We cannot tell from PAPER.md whether the logged values or the stored values were used for analysis.

### Theorem 2: Partially correct with a loose step

The derivation of P_i epsilon(x) is correct up to the final bound. The claim:
```
||P_i epsilon(x)|| <= min(1, sqrt(K/d)) * alpha_max * sqrt(d) = alpha_max * sqrt(K)
```
The min(1, sqrt(K/d)) factor comes from the rank-K projector concentrating a random vector, but epsilon is NOT a random vector -- it is structured quantization noise. The bound alpha_max * sqrt(K) is still valid as a worst case (since ||P_i|| = 1 and ||epsilon|| <= alpha_max * sqrt(d), and then the K-dimensional projection reduces it), but the tightness claim is suspect. For structured noise, the actual leakage could be much larger or smaller than alpha_max * sqrt(K).

More importantly: the proof bounds leakage NORM, but K688 measures val LOSS delta. There is no theorem connecting leakage norm to val loss increase. The MATH.md states "val loss impact depends on downstream sensitivity" -- acknowledging the gap but not resolving it. This means K688 is not actually testing the proof.

### The K/d cosine prediction

MATH.md predicts E[|cos(v_i, v_j)|] ~ K/d = 0.025. This is approximately correct for random subspaces (the expected squared cosine of a random unit vector projected through a K-dimensional projector in d dimensions is K/d). The measurement of 0.026 matches well. However, the max cosine of 0.977 is extremely high -- PAPER.md attributes this to shared instruction-following template directions, which is plausible. But this means the subspaces are NOT independent/random as the proof assumes. They share 1-2 nearly identical directions. The null-space projection would remove these shared directions from later domains, which could explain the forgetting in legal (a low-scale domain loses its shared instruction-following direction).

### Gradient preservation prediction

MATH.md predicts (d - K*N_prior) / d for domain 5: (2560 - 256)/2560 = 90%. Measured: 95.2%. The prediction was for norm-squared ratio; the code computes norm ratio (sqrt). Let's check: if norm-squared preservation = 0.90, then norm preservation = sqrt(0.90) = 0.949. Measured: 0.952. This is an excellent match -- the MATH.md prediction was stated as "90%" without specifying norm vs norm-squared, but the measurement matches the sqrt interpretation. This is actually a nice verification.

## Prediction vs Measurement

PAPER.md contains the table. Assessment:

| Prediction | Predicted | Measured | Match? | Review Comment |
|---|---|---|---|---|
| Cross-domain cosine | < 0.1 | 0.026 | YES | But max 0.977 shows shared structure |
| Leakage norm | <= 0.08 | NOT MEASURED | N/A | K688 measures val loss, not leakage norm |
| Gradient preservation | ~90% | 95.2% | YES | Matches sqrt interpretation |
| Energy captured | meaningful % | 1.0 (BUG) | INVALID | Code bug: S_k/S_k = 1 always |

The critical gap: the proof's core quantitative prediction (Theorem 2: leakage norm <= alpha*sqrt(K)) was never measured. K688 measures val loss, which is a different quantity. PAPER.md claims "YES -- within proof bound" for forgetting 0.025 < 0.05, but 0.05 is the proof's leakage NORM bound, not a val loss bound. These are incommensurable quantities.

## NotebookLM Findings

Skipping -- proceeding with manual deep review given the issues found above are sufficient for verdict.

## Novelty Assessment

The Brainstacks paper (arXiv:2604.01152) provides the null-space SVD approach. This experiment's contribution is testing it on ternary adapters. This is a reasonable verification experiment -- the novelty is narrow but appropriate for Type 1 verification. The connection to Findings #270 and #271 is well-motivated.

## Macro-Scale Risks (advisory)

1. The max cosine of 0.977 between all domain pairs suggests a dominant shared direction (likely the instruction template). At scale with 25+ domains, this shared direction will be projected out for all but the first domain, potentially degrading instruction following. The fix would be to identify and exclude shared directions before computing domain-specific projectors.

2. The (hidden_dim, hidden_dim) projector matrices P_i are dense and O(d^2) per domain. For d=2560 this is 26MB per domain in float32. At d=4096 (larger models) with 25 domains, this is 25 * 67MB = 1.7GB of projectors alone.

3. The forgetting measurement applies ALL other domains' projectors to each domain. In sequential training (the actual use case), domain k only gets projected through domains 1..k-1. The current test is more aggressive than the real scenario.

## Code Issues

1. **Energy computation bug (BLOCKING):** Lines 614-616 compute `sum(S_k^2) / sum(S_k^2)` which is always 1.0. Should be `sum(S_k^2) / sum(S_full^2)`. This means Theorem 1's precondition (rank <= K captures full space) is unverified.

2. **NS_N_SAMPLES vs PAPER.md claim:** Code uses NS_N_SAMPLES=200, but PAPER.md Section 4 says "50 samples" and discusses rank(D) <= 50 < 64. The results.json SVD values suggest the actual run used 200 samples (singular value magnitudes are consistent with 200-sample matrices). But PAPER.md's analysis about rank <= 50 is either from an earlier run or incorrect for the reported results.

   Wait -- re-reading PAPER.md: "All 5 domains have 100% of their output delta variance captured in K=64 directions (out of 50 samples)." This references "50 samples" but the code uses NS_N_SAMPLES=200. Either PAPER.md was written from an earlier run with n=50, or the energy ratio is always 1.0 due to the bug regardless. Given the bug, we cannot distinguish these cases.

3. **Forgetting measurement design:** The experiment projects out ALL other domains' subspaces from domain d's adapter, then measures val loss on domain d. This is measuring self-interference (does projecting out other domains' directions hurt domain d?), which is the correct test. However, the real forgetting test should be: train domains sequentially, project each new domain through prior domains' null space, then evaluate ALL prior domains. The current test is a proxy.

## Verdict

**REVISE**

### Required fixes (BLOCKING):

1. **Fix the energy computation bug.** Lines 614-616: denominator must use the FULL singular value vector S (all singular values from SVD), not S_k (truncated). This is needed to verify Theorem 1's precondition. Re-run and report actual energy ratios.

2. **Measure the actual leakage norm predicted by Theorem 2.** Currently K688 measures val loss delta, but the proof predicts leakage NORM (||P_i * epsilon||). Add a direct measurement: compute ||P_i * delta_q^proj|| for each domain pair and compare against the bound alpha*sqrt(K). This is the proof's core quantitative prediction and it was never tested.

3. **Reconcile K688 kill threshold with the proof.** The proof predicts leakage norm <= 0.08. The kill threshold is val loss < 0.01. These measure different things. Either: (a) derive a val-loss bound from the leakage norm bound (requires a sensitivity analysis), or (b) change K688 to test the leakage norm directly, or (c) acknowledge that K688 tests a behavioral outcome separate from the proof and add a separate kill criterion for the proof's prediction.

4. **Reconcile K689 kill threshold with the proof.** The proof predicts 90% gradient preservation for domain 5. The kill threshold requires >95%. The proof PREDICTS the kill criterion will fail. Either update the threshold to match the proof (>85%, giving margin), or update the proof if the 90% prediction is conservative.

5. **Reconcile PAPER.md with actual run parameters.** PAPER.md discusses "50 samples" and rank <= 50, but the code uses NS_N_SAMPLES=200. Clarify which run produced the reported results.

### Advisory (not blocking):

6. Consider computing the energy ratio correctly and, if it is significantly below 1.0, increasing K or decreasing NS_N_SAMPLES to satisfy Theorem 1's precondition.

7. The max cosine ~0.977 in every domain pair deserves more investigation. If all pairs share the same dominant direction, that direction should be identified and potentially excluded from the projectors (it is shared structure, not domain-specific signal).
