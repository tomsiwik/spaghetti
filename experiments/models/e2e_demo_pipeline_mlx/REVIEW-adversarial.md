# Peer Re-Review: E2E Demo Pipeline (BitNet-SOLE on M5 Pro)

**Re-review of revised version.** The original review issued a REVISE verdict with 6 required fixes (R1-R6). This re-review verifies whether each was addressed and checks for new issues.

## Revision Status

### R1: K1 must be acknowledged as FAIL (not "MARGINAL FAIL"). Remove post-hoc threshold relaxation.

**ADDRESSED.** PAPER.md now states "K1 FAIL" unambiguously in the header (line 9), the kill criteria table (line 119), and throughout the analysis. The phrase "MARGINAL FAIL" has been removed entirely. There is no post-hoc suggestion to relax the threshold to 2.5x. The paper instead proposes a follow-up experiment with a newly pre-registered threshold of 1.5x (line 166), which is the correct approach. The code's verdict and the paper's verdict are now consistent.

### R2: Report worst-case latency (merged-only 2.33x), not just average that depends on skip rate.

**ADDRESSED.** PAPER.md now leads with the 2.33x worst-case number prominently (lines 52-57). The per-query breakdown table (lines 49-53) clearly separates entropy-skip queries (1.01x) from merged queries (2.33x). The text explicitly states: "The architecturally meaningful number is 2.33x: the latency penalty for any query that goes through the merge path." The data-dependent nature of the 2.012x average is called out (line 54). This is exactly what was requested.

### R3: Add PPL confidence intervals (single seed was insufficient).

**ADDRESSED.** A dedicated script `compute_ppls_ci.py` was written and executed, computing per-sample PPL on N=25 validation samples per domain with t-distribution 95% CIs. Results are saved in `results_ppls_ci.json` and incorporated into PAPER.md Table (lines 74-81). The CI computation is methodologically correct: uses scipy.stats.t.ppf(0.975, df=24), reports mean +/- SE, and correctly identifies that no domain's composed 95% CI upper bound reaches the base mean (i.e., all improvements are statistically significant).

One note: the CIs for code and finance show overlap between composed CI and base CI (`cis_overlap: true` in results_ppls_ci.json), but this is the weaker test (CI overlap). The stronger test -- composed CI upper bound vs base mean -- passes for all domains. The paper correctly reports the stronger test result. This is acceptable.

### R4: Remove hidden 5% K2 tolerance that was not pre-registered.

**ADDRESSED.** The code now uses strict comparison (line 877: `e2e1_worse = e2e1_ppl > base_ppl`) without the 1.05 multiplier. The comment on line 876 explicitly says "strict, as pre-registered." PAPER.md line 86 confirms: "0/5 domains worse (strict comparison, no tolerance)." Code and paper are now consistent with the pre-registered kill criterion.

### R5: Reconcile MATH.md prediction (<5% overhead) with actual result (101% overhead / 2x).

**ADDRESSED.** MATH.md Section 3 has been split into two subsections: 3a "Pipeline Machinery Overhead (prediction: CORRECT)" and 3b "Weight Structure Effect (prediction: MISSED -- FALSIFIED)." Section 3a correctly notes that the FLOP analysis for pipeline machinery was accurate. Section 3b explicitly states: "The original prediction of '<5% total overhead' was FALSIFIED. Actual overhead: 101%." It then provides a detailed mechanistic explanation of why merged weights degrade generation speed (sparsity loss, value distribution changes) and why this was missed (prior single-pass measurement did not capture sustained autoregressive behavior). This is honest and thorough.

### R6: Clearly separate contribution: quality works, latency is the open problem.

**ADDRESSED.** PAPER.md Section "What This Means for the Architecture" (lines 171-185) clearly states: "The composition mechanism works: all components integrate cleanly and produce excellent quality. However, pre-merging LoRA adapters into ternary base weights introduces a fundamental tension." The header result line (line 9) leads with "K1 FAIL" before "K2 STRONG PASS." The analysis section (lines 135-152) is titled "Why K1 Fails" and identifies the ternary-to-dense conversion as the specific, solvable problem. The paper no longer presents K1 failure as a qualified success.

## New Issues Introduced by Revisions

### Minor: PPL CI script uses single seed for model loading (not blocking)

The `compute_ppls_ci.py` script computes per-sample PPL variance (N=25 samples), which addresses the original concern about single-seed PPL estimates. However, the PPL computation is deterministic given the same model weights and validation data (cross-entropy loss has no stochastic component during evaluation), so multiple seeds would not change these numbers. The CI here captures sample-to-sample variance, not run-to-run variance. This is the correct variance to report for PPL.

### Minor: Code syntax decrease still underexplored (carried over, not new)

PAPER.md notes code syntax drops from 50% to 40% with the code adapter (top-1) while PPL improves 46.7%. The paper correctly attributes this to N=10 noise, which is reasonable. However, this tension between PPL improvement and task metric regression on the same domain is worth flagging for the follow-up. It could indicate that PPL improvement does not always translate to task performance, especially for structured outputs like code. This was noted in the original review and remains advisory.

### Minor: Entropy gating confound still present (carried over, not new)

The original review noted that PPL is evaluated on all validation data with the composed model, while latency includes queries that used base weights (entropy-skip). This means quality and latency are measured on different effective model configurations. This is inherent to the pipeline design and not a flaw in the revision -- it is a limitation of any entropy-gated architecture. The paper acknowledges this in the limitations section (line 200: "Entropy skip rate is data-dependent").

### No new critical issues found.

The revisions are clean. No new hidden tolerances, no new post-hoc threshold changes, no new misleading framing.

## Mathematical Soundness (unchanged from original review)

- Pre-merge algebra: CORRECT
- Entropy gating: CORRECT (threshold calibration acknowledged as limitation)
- PPL computation: CORRECT (apples-to-apples comparison confirmed)
- Latency analysis: Now CORRECTLY reconciled (machinery prediction correct, weight-structure effect honestly marked as falsified)

## Experimental Design (re-assessed)

The experiment now honestly reports its results:
- K1 FAIL at 2.33x worst-case (the architecturally meaningful number)
- K2 PASS with statistical significance (95% CIs, N=25, all domains)
- S1 PASS at 38ms/tok (interactive threshold)
- Strict K2 comparison (no hidden tolerance)
- MATH.md predictions reconciled with actuals

The single-seed limitation for latency measurements remains (acknowledged in Limitations, line 192). Given that K1 is now reported as a clear FAIL (not borderline), the single-seed issue is less critical -- the question is no longer whether K1 passes, but rather the magnitude of the penalty (2.33x). Single-seed is adequate to establish this order of magnitude.

## Macro-Scale Risks (advisory, unchanged)

1. Ternary-to-dense slowdown may worsen with larger models
2. Entropy skip rate is distribution-dependent
3. Oracle routing masks real integration issues
4. Memory scaling with more adapters

## Verdict

**PROCEED**

All 6 required revisions have been addressed substantively, not just cosmetically. The paper now:

1. Reports K1 as an unambiguous FAIL
2. Leads with the worst-case 2.33x latency ratio
3. Provides statistically rigorous PPL CIs (N=25, t-distribution, all significant)
4. Uses strict K2 comparison matching the pre-registration
5. Honestly reconciles MATH.md predictions with falsified results
6. Clearly separates the quality contribution from the latency open problem

The quality result (34-61% PPL improvement across all 5 domains, all statistically significant) is the strongest in the project and validates the core composition mechanism. The latency result (2.33x for merged queries) is honestly reported as a K1 FAIL and correctly identified as a ternary-specific problem requiring a dedicated follow-up experiment. The proposed follow-up with a newly pre-registered 1.5x threshold and specific mitigations (always-on merge, re-quantization) is the right next step.

### Advisory findings (non-blocking)

1. The code syntax regression (50% to 40% with code adapter) should be investigated at larger N in follow-up work. PPL improvement without task metric improvement would weaken the quality narrative.
2. Latency variance across runs should be measured in the follow-up experiment (multi-seed or repeated runs) to establish whether 2.33x is stable or variable.
3. The entropy threshold calibration (24% skip rate vs prior experiment's 63%) suggests the Otsu threshold may need per-distribution recalibration. This should be tested with the follow-up's prompt distribution.
