# Peer Review: Segment Adapter Scale Sweep (Re-Review)

Post-revision re-review. The initial review identified 6 blocking issues.
This review verifies that all 6 fixes were applied correctly.

## Experiment Type
Frontier extension (Type 3)

## Hack Detector
- Fix count: 1 (single parameter sweep). No stacking.
- Is MATH.md a proof or a description? Description dressed in equations, now correctly labeled as Type 3 (frontier extension). The PBR identity and IVT argument are acknowledged as insufficient for a proof.
- Metric used as evidence: PPL delta (-0.06%) correctly identified as noise. Behavioral metrics measured but showed zero signal.
- Kill criteria source: Derived from framework predictions. Both K787 and K788 now correctly FAIL.

## Self-Test Audit
1. One-sentence impossibility property: Still uses the IVT existence argument, but the POST-EXPERIMENT annotation (lines 258-266) correctly acknowledges this was falsified. Acceptable for a killed experiment.
2. Cited theorems: Hu et al. 2022 (real), IVT (real but preconditions not met -- now acknowledged).
3. Predicted numbers: s* in [5,15], U-shaped curve. Both wrong. Honestly reported.
4. Falsification condition: "If PPL(s) is monotonically increasing from s=0." POST-EXPERIMENT block clearly states this was met. Good.
5. Hyperparameter count: 1. Fine.
6. Hack check: Clean.

## Fix Verification

### Fix 1: Finding #311 downgraded from "supported" to "killed"
**APPLIED.** PAPER.md header reads "Verdict: KILLED." results.json reads `"verdict": "KILLED"`. The explanation is thorough and honest: "The hypothesis that there exists an optimal s* making segment-isolated adapters effective is refuted."

### Fix 2: K787 changed from PASS to FAIL
**APPLIED.** results.json: `"K787": {"pass": false, ...}`. PAPER.md kill criteria table: "K787: Best PPL < base | FAIL | 7.988 vs 7.993 (-0.06%) is ~5 nats over 6350 tokens. No CI, no bootstrap, no multi-seed. This is noise, not evidence of improvement." The reasoning is correct and the statistical caveat is explicit.

### Fix 3: K788 changed from PASS to FAIL on U-shape prediction
**APPLIED.** results.json: `"K788": {"pass": false, ...}`. PAPER.md: "K788: Non-monotonic curve (U-shape) | FAIL | Curve is monotonically increasing (7.988, 8.007, 8.051, 8.084, 8.130). The U-shape prediction -- the actual prediction being tested -- is refuted." The distinction between the weak sub-condition (s* != s_train) and the actual prediction (non-monotonicity) is clearly drawn.

### Fix 4: Self-test falsification condition acknowledged as met
**APPLIED.** MATH.md Self-Test item 4 (lines 258-266) now contains a POST-EXPERIMENT block:
"THIS FALSIFICATION CONDITION WAS MET. The measured curve is monotonically increasing from s=2 through s=20... The framework is therefore falsified by its own stated conditions: even near-infinitesimal adapter perturbation does not help on isolated segments."

### Fix 5: Per-sequence PPL contradiction addressed in MATH.md
**APPLIED.** MATH.md Section C now contains a "Framework Contradiction (post-experiment)" subsection (lines 94-115) that:
- Notes per-sequence PPL = 10.48, dramatically worse than base 7.99
- Explains the discrepancy with Finding #310 (different evaluation protocol)
- Concludes "the premise 'adapter content is correct, only scale is wrong' is not verified in this experiment's own data"
- Notes the IVT argument requires PPL(s*) < PPL(0) for some s*, which may not hold

### Fix 6: Type downgraded from Type 2 to Type 3
**APPLIED.** MATH.md header (line 3): "Type: Frontier Extension (Type 3)". Lines 5-9 explain why: "The PBR definition (linear in s) is a tautological identity, not a proven theorem with predictive power. The IVT argument for existence of s* had unsatisfied preconditions."

## Remaining Minor Issues (Non-Blocking)

1. **Internal type inconsistency (cosmetic).** MATH.md Section D line 121 still reads: "This is a Type 2 experiment: the framework (PBR scaling) is proven, but the optimal s* for L=128 is unknown." This contradicts the header (Type 3) and the detailed explanation (lines 5-9). This is a leftover from pre-revision text. Since the header, the explanation, and the post-experiment annotations are all correct, this is cosmetic -- but should be cleaned up if the file is edited again.

2. **P1 "WEAK YES" in prediction table.** PAPER.md prediction table still lists P1 as "WEAK YES" while K787 is FAIL. These are slightly different things (raw measurement vs. statistical significance judgment), and the surrounding text correctly explains the delta is noise. Not blocking, but the prediction table could be updated to "NO (within noise)" for full consistency with the kill criteria assessment.

3. **Code not updated.** The run_experiment.py still contains the original logic at lines 564-603 where K787 and K788 would compute as PASS under the original criteria. The results.json was manually corrected rather than the code being re-run with updated criteria. This is acceptable for a killed experiment that will not be re-run, but it means the code and the results.json are out of sync.

## Mathematical Soundness

The experiment is killed. The post-experiment annotations honestly acknowledge:
- The IVT argument's preconditions were not met
- The PBR identity has no predictive power for PPL
- The "adapter content is correct" premise is contradicted by the data
- The falsification condition was met

There is no proof to verify because this is correctly labeled as Type 3 (frontier extension) with a killed hypothesis. The mathematical framework was a set of heuristic predictions, not a proof, and the predictions were wrong. This is fine -- the value is in the negative result and its interpretation.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Score: 0 confirmed, 2 clear failures, 1 noise-level marginal, 1 trivially true (adapter has no effect). The framework's predictions were wrong, the hypothesis is killed. The table is present and honest.

## NotebookLM Findings

Not consulted. The issues are straightforward and the fixes are mechanical. A deep review is not warranted for a re-review of a killed experiment.

## Novelty Assessment

The negative result -- that LoRA adapters trained on full sequences provide zero useful domain signal when applied to isolated 128-token segments at any scale -- is a genuine and useful finding. It is specific to the SOLE architecture's segment-isolation approach and not previously published. The interpretation (context dependency, not scale, is the bottleneck) is actionable: it correctly redirects future work toward context-preserving adapter application rather than scale tuning.

## Macro-Scale Risks (advisory)

The finding likely generalizes: at macro scale, the gap between training context length and segment length would be even larger, making the problem worse. Per-sequence or overlapping-segment adapter application is the correct direction.

## Verdict

**PROCEED**

All 6 required fixes from the initial review were applied correctly:

| Fix | Status | Quality |
|-----|--------|---------|
| 1. Finding #311 -> killed | Applied | Thorough explanation |
| 2. K787 -> FAIL | Applied | Statistical reasoning included |
| 3. K788 -> FAIL (U-shape refuted) | Applied | Weak vs strong condition distinguished |
| 4. Self-test falsification acknowledged | Applied | Detailed POST-EXPERIMENT block |
| 5. Per-sequence PPL contradiction in MATH.md | Applied | New subsection with clear analysis |
| 6. Type 2 -> Type 3 | Applied | With explanation of why |

The experiment is a clean negative result, honestly reported as KILLED. The three remaining issues (stale Type 2 reference in Section D, P1 "WEAK YES" wording, code not updated) are cosmetic and non-blocking for a killed experiment. The interpretation (context dependency is the real bottleneck, not scale) is the experiment's genuine contribution.

As a Type 3 frontier extension with a killed hypothesis, finding #311 status should be "killed" (which it is in the results.json). No further action required.
