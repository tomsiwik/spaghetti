# Peer Review: Composition Weight Sensitivity (v2, post-REVISE)

## NotebookLM Findings

Skipped. The documents are small enough for direct line-by-line analysis. The critical issues are mathematical, not conceptual.

## Assessment of REVISE Fixes

All six fixes from the v1 review were applied:

1. **SNR relabeled as prediction**: Thoroughly applied in both MATH.md and PAPER.md. The +0.57 vs -1.0 contradiction is prominently discussed in multiple sections.
2. **Noise samples 3 to 20**: Applied (code line 377).
3. **Signal/noise symmetry**: Applied -- both signal and noise now use LOO reweighting (code lines 363-390).
4. **Orthogonality scope qualified**: Applied in PAPER.md and FINDINGS.md (20 of 100 experts).
5. **Status to supported (K2 only)**: Applied in HYPOTHESES.yml and PAPER.md.
6. **Trivial K2 caveat**: Applied in FINDINGS.md with strong language.

The fixes are well-executed. However, they resolved the *identified* issues while leaving a more fundamental problem untouched.

## Mathematical Soundness

### CRITICAL: Internal contradiction between MATH.md and PAPER.md on noise scaling

MATH.md Section 2.5 derives:

    sigma_noise ~ O(1/N) * ||dW_i||_F / sqrt(D)

and concludes:

    "Since both signal and noise scale as O(1/N), the SNR should be
    approximately N-independent (constant) under orthogonality."

PAPER.md structural finding #3 states:

    "noise floor drops as O(1/N^2) while signal drops as O(1/N),
    predicting SNR ~ O(N)"

These are contradictory. MATH.md predicts constant SNR. PAPER.md predicts SNR ~ O(N). The paper labels "SNR ~ O(N)" as the theoretical prediction, contradicting its own mathematical derivation.

### Root cause: noise norm double-division by N

The `make_noise_delta` function (code line 252) sets the noise delta norm to `avg_norm / N`:

    scale = (avg_norm / N) / max(np.sqrt(noise_sq), 1e-12)

This noise delta then enters `compose_avg`, which divides the sum of all N deltas by N. The effective contribution of the noise expert to the composed model is therefore:

    ||noise_effective|| = (avg_norm / N) / N = avg_norm / N^2

Meanwhile, a real expert's delta has raw norm ~avg_norm, and its effective contribution after compose_avg is:

    ||expert_effective|| = avg_norm / N

The noise is N times smaller than a real expert's effective contribution. This is by design -- MATH.md Section 2.3 says "The noise delta has norm matched to (1/N) * ||dW_i||." But this pre-scaling followed by compose_avg's 1/N division creates a double-counting: the noise is scaled to match the *effective* expert contribution, then scaled again by compose_avg.

The consequence: the noise LOO marginal varies as O(avg_norm/N^2) while the signal LOO marginal is O(avg_norm/N). This produces SNR ~ O(N) as a norm-matching artifact, not a physical property.

If the noise norm were set to `avg_norm` (matching the raw expert delta norm, so that compose_avg produces the correct effective norm of avg_norm/N), the noise floor would scale as O(1/N) -- matching the signal -- and SNR would be O(1) constant, consistent with MATH.md Section 2.5's own derivation.

This means:
- The measured SNR values (2.2, 13.6, 16.0 at N=20,50,100) are inflated by the noise being N times too small
- The "SNR ~ O(N)" prediction is not supported by the MATH.md derivation -- it contradicts it
- The MATH.md derivation of constant SNR is actually correct under its stated assumptions

### The +0.57 vs -1.0 exponent contradiction

The paper handles this honestly: it labels +0.57 as "meaningless" due to the zero-signal regime and acknowledges the prediction is untested. This is adequate given the micro-scale constraints. The contradiction cannot be resolved at this scale.

However, the +0.57 exponent is *also* consistent with the noise norm bug above. If the marginal contributions are dominated by the reweighting effect (O(avg_norm/N)) rather than the expert-removal effect, and if the reweighting effect has some N-dependent numerical structure, one could observe any power law exponent. This is another reason the power law analysis is meaningless at micro scale, and the paper correctly says so.

### Section 2.6: Signal/noise symmetry analysis

The v2 symmetry fix is correct in structure: both signal and noise use the same LOO operation (remove one element from an N-element compose_avg, see the 1/N to 1/(N-1) reweighting). The analysis in Section 2.6 is clear and well-reasoned.

However, the analysis does not catch the noise norm double-division issue. Section 2.6 says "The noise floor then measures the variance of LOO marginal contributions when the removed expert is random noise rather than a trained expert." This is true, but the noise expert has norm avg_norm/N while the real expert has norm avg_norm, so "random noise" is N times weaker than "random expert of matched norm." The comparison is not like-for-like in magnitude.

### Section 2.2-2.4: Marginal contribution analysis

Correct. The leave-one-out definition, reweighting analysis, and first-order Taylor expansion are all sound. The reweighting effect at 1/(N(N-1)) ~ 0.01% per expert at N=100 is correctly identified as subdominant.

### Section 4: Worked example

Correct arithmetic. The random baseline cosine sqrt(r/d) = 0.354 at d=64 vs measured 0.002 is a factor of 177x, not 167x as stated (minor).

## Novelty Assessment

The experiment tests a legitimate concern (pre-merge dilution at scale) and the theoretical framing (SNR under orthogonal composition) appears novel. However, the noise norm bug undermines the SNR prediction. AdapterSoup and TIES-Merging do not study this specific question.

The contribution after fixing: "averaged composition is numerically stable" is modest but genuine. The theoretical prediction of constant SNR (from MATH.md Section 2.5, which is the correct derivation) is a reasonable prediction for macro validation.

## Experimental Design

### Positive: honesty about limitations

The paper is remarkably honest. Every claim is qualified. The trivial K2 PASS is explicitly called out. The zero specialization is discussed at length. This is high-quality self-assessment.

### Negative: no positive control

There is no condition where experts carry real signal. A positive control (e.g., 10x training steps, or training on truly disjoint data subsets) would distinguish "composition is stable" from "there is nothing to compose." This was not flagged in v1 review but is worth noting as a macro design consideration.

### The 3-seed design is adequate for micro

Three seeds with consistent results (all showing the same patterns) is sufficient for a micro-scale structural test.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry status is "supported" with qualifier "K2 only." This accurately reflects the evidence: K2 passes trivially, K1 is inconclusive. The kill criteria are the ones actually tested. The evidence section in HYPOTHESES.yml is detailed and honest.

The results.json overall_kill = true (because K1 formally triggers) while status = supported creates a minor discrepancy, but the paper explains this clearly as a false positive.

## Macro-Scale Risks (advisory)

1. **Fix the noise norm before macro validation.** The SNR analysis must use noise at the correct scale (raw norm = avg_norm, effective norm = avg_norm/N after compose_avg). With correct noise, the theoretical prediction is constant SNR independent of N, which is still a positive result (detectability does not degrade).

2. **SUM vs AVG decision.** At macro scale with 42% PPL improvement per expert, SUM composition at N=50 would produce a massive perturbation (50 * delta). AVG is the only viable option. This is implicit but should be explicit.

3. **Cross-expert interactions.** The first-order Taylor analysis assumes independence. At macro scale with real gradients, second-order cross-terms may matter, especially for within-cluster experts (cos = 0.85 in attention layers).

4. **Shadow scoring sensitivity.** The answer-conditioned PPL was validated at ~42% improvement level. At N=500 with 0.084% per-expert contribution, the scoring noise floor may swamp the signal.

## Verdict

**PROCEED**

The experiment is honest, the REVISE fixes are well-applied, and the "supported (K2 only)" status is appropriately conservative. The noise norm bug does not change the core finding (composition is numerically stable) because the core finding is K2, not the SNR analysis.

However, two non-blocking fixes should be applied before the SNR prediction is used in any downstream reasoning:

1. **Resolve the MATH.md vs PAPER.md noise scaling contradiction.** MATH.md Section 2.5 derives constant SNR (noise and signal both O(1/N)). PAPER.md claims SNR ~ O(N) (noise O(1/N^2), signal O(1/N)). The code implements the PAPER.md version, but the MATH.md derivation is correct for the intended noise model (noise matching per-expert effective contribution). Either: (a) fix the noise norm in the code to avg_norm (not avg_norm/N) so compose_avg produces the correct effective norm, and update the SNR prediction to "constant" (consistent with MATH.md), or (b) if the avg_norm/N noise is intentional, update MATH.md Section 2.5 to derive O(1/N^2) noise scaling and explain why the noise should be smaller than a real expert's effective contribution.

2. **Clarify which prediction is "the prediction."** The paper labels "SNR ~ O(N)" as the theoretical prediction while MATH.md Section 2.5 predicts constant SNR. These cannot both be the prediction. After resolving fix #1, update all references consistently.

These are non-blocking because the experiment's status rests on K2 (trivial PASS), not on the SNR analysis. The SNR analysis is labeled as untested prediction, and corrections to the prediction do not change the experiment's status.
