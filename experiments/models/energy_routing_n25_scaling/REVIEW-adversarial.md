# Peer Review: Energy Gap Routing at N=25 Scaling

## Experiment Type
Frontier extension -- extending N=5 energy gap routing (Finding #185) to N=24.

## Hack Detector
- Fix count: 0 mechanisms/losses/tricks. This is a pure scaling measurement. CLEAN.
- Is MATH.md a proof or a description? **Description dressed in analytical framework.** The Gumbel EVT analysis is invoked but never reaches Theorem/Proof/QED. It provides approximate reasoning about scaling, which is appropriate for frontier extension type but should not be mistaken for a proof.
- Metric used as evidence: Routing accuracy (top-1 match). This is a direct behavioral measure -- the adapter selected is either the correct domain or not. CLEAN.
- Kill criteria source: K581 (>=60%) is loosely derived from the Gumbel prediction of 60-75%. K582 (>=50% math correctness) is from the N=5 baseline (70%) with expected degradation. K583 (<120s) from linear scaling estimate. Partially derived from the analysis, partially from heuristic rounding.

## Self-Test Audit

1. **One-sentence impossibility property:** "The argmin selector's error probability increases logarithmically with N via extreme value theory (Gumbel), not linearly." -- This is a scaling claim, not an impossibility property. It says error grows slowly, but it does not state what structural property would make failure impossible. However, for a frontier extension probing whether a mechanism breaks, this is acceptable. MINOR FLAG.

2. **Cited theorems:** Fisher-Tippett (1928) / Gumbel extreme value theorem. Real theorem, correctly cited. However, the application has a critical precondition violation: the theorem requires i.i.d. random variables. The energy gaps across adapters are NOT i.i.d. -- different adapters have different mean strengths (the very thing that caused the failure). The MATH.md acknowledges "common variance sigma^2" as an assumption but does not flag that heterogeneous adapter strengths violate the i.i.d. assumption. SIGNIFICANT FLAG.

3. **Specific predictions:** "Accuracy 60-75% at N=24" -- specific and falsifiable. "Block-diagonal confusion matrix" -- qualitative but testable. "Math/code accuracy >80%" -- specific. "Overhead <120s" -- specific. PASS.

4. **Falsification condition:** "The frontier extension is wrong if accuracy drops BELOW 60%." This targets the Gumbel prediction, not just the experiment. PASS.

5. **Hyperparameter count:** 0. Energy gap routing is hyperparameter-free. PASS.

6. **Hack check:** "No. This is measuring scaling of an existing zero-parameter method." Correct. PASS.

## Mathematical Soundness

The Gumbel analysis in MATH.md Section C is the core analytical contribution. Step-by-step:

**Step 1: Order statistics setup (lines 38-55).** Correct framing. The integral formulation for P(correct) is standard.

**Step 2: Gumbel approximation (lines 57-66).** The formula E[min] ~ mu - sigma * sqrt(2 ln(N-1)) is the standard result for the minimum of N-1 i.i.d. Gaussian variables. Correctly stated.

**Step 3: Effective margin formula (lines 68-79).** This is where the analysis breaks down in retrospect:
- The formula assumes all "other" adapters share a common mean mu_other and common variance sigma^2.
- In reality, adapter strengths vary by >10x (health_fitness gap = +2.95, math gap = -0.002).
- The i.i.d. assumption is catastrophically violated. The "competitors" are not drawn from a common distribution; some have enormous mean gaps that dominate the argmin regardless of the query.

**Step 4: Worked example (lines 149-161).** Uses sigma=0.3 and delta=0.5 as "typical." The actual data shows adapter mean gaps ranging from ~0 to ~3 nats -- a 10x range. The "typical" parameters were not grounded in any measurement from the N=5 experiment.

**Verdict on math:** The analytical framework correctly identifies the right direction (accuracy degrades with N) but makes a quantitatively wrong prediction (60-75% vs actual 8.3%) because it assumes i.i.d. competitor gaps. The actual failure mode -- adapter strength disparity, not domain similarity confusion -- was outside the model's assumptions. This is a legitimate failure of a frontier extension; the gap was found.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table (lines 11-18). This is well-structured.

| Prediction | Measured | Match |
|------------|----------|-------|
| 60-75% accuracy | 8.3% | NO -- off by 7-9x |
| Block-diagonal confusion | Two-attractor collapse | PARTIAL |
| Math/code >80% routing | Code 100%, Math 0% | PARTIAL |
| Math correctness >=50% | 20% | NO |
| Overhead <120s | 3.2s | YES |
| Degradation from confusable pairs | Degradation from magnitude disparity | NO -- wrong mechanism |

The table is honest and the mismatch analysis is thorough. The root cause identification (adapter strength disparity, not domain confusion) is the key finding.

## NotebookLM Findings
Skipped -- the experiment is already killed with clear root cause analysis. NotebookLM review would not change the verdict.

## Novelty Assessment

**Prior art concern:** The "loudest voice" / calibration problem in ensemble methods and mixture-of-experts is extremely well-known. The realization that argmin over uncalibrated scores is biased toward highest-magnitude experts is not novel -- it is a standard failure mode documented in:
- Jacobs et al. (1991), "Adaptive Mixtures of Local Experts" -- gating networks learn calibrated weights precisely to avoid this
- Any MoE paper that uses softmax gating instead of raw scores

The PAPER.md does cite LoRAuter and MoLoRA as alternatives with learned routers, which is appropriate. The contribution is confirming this known failure mode in the specific context of energy-gap adapter routing.

**Delta over existing work:** The value is in the specific diagnosis: energy gap routing's implicit calibration assumption, and that it held at N=5 by accident (matched adapter strengths) not by design. This is a useful negative result.

## Macro-Scale Risks (advisory)

1. Even with z-score normalization (the proposed fix), the approach requires N forward passes per query. At N=50 or N=100 adapters, this is O(N) latency per routing decision -- fundamentally unscalable without approximation (e.g., pre-computed statistics, learned routers, or hierarchical routing).

2. The proposed z-score fix (DeltaE_i / E[DeltaE_i | domain_i]) requires maintaining per-adapter calibration statistics, which introduces a maintenance burden and drift risk when adapters are updated.

3. The real lesson for macro: routing mechanisms need to be calibrated by design (learned gating), not by coincidence (matched adapter strengths).

## Verdict

**KILL**

The experiment correctly identifies itself as killed. The review confirms the kill is justified and the root cause analysis is sound. Specific assessment:

1. **The kill is legitimate.** 8.3% accuracy vs 60% threshold is unambiguous. The mechanism fails catastrophically at N=24.

2. **The root cause analysis is the real finding.** The identification of adapter strength disparity (not domain confusion) as the failure mode is valuable. The Gumbel analysis predicted the wrong failure mechanism, which is itself informative.

3. **The MATH.md Gumbel analysis had a known-bad assumption.** The i.i.d. assumption on competitor gaps was violated by construction (heterogeneous adapters trained on different data quality). This should have been flagged as a risk in the predictions, not just in the assumptions section. The worked example used made-up parameters (sigma=0.3, delta=0.5) rather than measured values from the N=5 experiment.

4. **Finding status should be "killed" with the impossibility structure recorded:** "Argmin routing over uncalibrated energy gaps is dominated by adapter magnitude, not domain relevance. When adapter NLL reduction magnitudes vary by >2x, routing collapses to the highest-magnitude adapter regardless of query domain."

5. **Next step is clear:** Test z-score normalized energy gaps or learned routing heads. The experiment correctly identifies this.
