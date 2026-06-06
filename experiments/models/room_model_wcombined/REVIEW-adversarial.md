# Peer Review: Room Model W_combined Retest

## Experiment Type
Verification (retest of Finding #303 with lenient thresholds)

## Hack Detector
- Fix count: 0. No new mechanisms. Pure retest with relaxed acceptance threshold.
- Is MATH.md a proof or a description? **Description dressed in equations.** MATH.md explicitly acknowledges this: "This is a utilitarian question, not a mathematical one" (Section B). There is no Theorem/Proof/QED block. The "Error Bound" section (C) is labeled "Informal, Based on Prior Measurement" and makes no formal claim. This is honest -- the authors do not pretend to have a proof.
- Metric used as evidence: PPL ratio (room/single-adapter) and tok/s. PPL is acknowledged as a weak proxy (Limitation 3 cites Finding #246, r=0.08).
- Kill criteria source: K802 threshold (2.0x) is a relaxation of the prior 1.10x from Finding #303, chosen to test whether the mechanism is "acceptable" rather than "equivalent." K803 threshold (90 tok/s) comes from ROOM_MODEL.md's prediction of 100+ tok/s. Neither is derived from a proof. Both are policy decisions, not mathematical predictions.

## Self-Test Audit

1. **One-sentence impossibility property:** "There is no such property. The failure mode (nonlinear compounding) is a real structural feature of deep networks." -- Honest and correct. The experiment is testing whether a known degradation is within tolerance, not claiming it is absent.

2. **Cited theorems:** Matrix distributivity (axiom, trivially correct). Finding #303 (empirical, cited correctly). Perturbation theory / first-order Taylor (informal, not a theorem with stated preconditions). Raghu et al. 2017 and Veit et al. 2016 are cited as context for nonlinear compounding -- appropriate references. No false citations.

3. **Predicted numbers:** PPL ratio 1.25-1.35x, worst domain 1.5-2.0x, speed 35-50 tok/s. These are extrapolations from Finding #303, not derived from theory. Honest framing: "from Finding #303 extrapolation."

4. **Falsification condition:** "If PPL ratio is significantly BETTER than 1.29x (< 1.10x), the POC measurement was wrong. If significantly WORSE (> 2.5x), the perturbation analysis is too optimistic." This targets the reproducibility of Finding #303's measurement, not a proof. Acceptable for a retest experiment.

5. **Hyperparameter count:** 0. Correct. W_combined is deterministic given the adapters. Alpha is inherited.

6. **Hack check:** "No. This is a retest of a killed experiment with a different acceptance threshold." Correct.

**Self-Test verdict: Complete, honest, no blanks or evasions.**

## Mathematical Soundness

### BLOCKING: No Theorem/Proof/QED

This is labeled as TYPE=verification, which requires MATH.md to contain at least one Theorem/Proof/QED block. MATH.md contains none. The document describes the mechanism (per-module linearity + cross-layer nonlinear compounding), makes informal predictions from prior measurements, and provides a worked example. This is a well-organized description, not a proof.

However, this is a nuanced case. The experiment is a **retest** of an already-killed finding, not a test of a new mechanism. The "theorem" being verified is really "Finding #303's measurements are reproducible" -- an empirical claim, not a mathematical one. Requiring a formal proof of reproducibility would be absurd. MATH.md is honest about this: "No new mathematical mechanism is proposed."

**Assessment:** The TYPE=verification label is technically wrong. This is closer to TYPE=guided-exploration within the proven framework of Finding #303 (the unknown being "does the measurement reproduce under lenient thresholds?"). The absence of a formal proof is acknowledged, not hidden. The experiment is not pretending to verify a theorem.

### What holds

1. **Per-module linearity (Section A, equation 1):** Trivially correct by matrix distributivity. The worked example (Section F) is correct -- verified the matrix algebra.

2. **Orthogonal norm scaling (Section C):** The claim that ||Sum_i DW_i||_F^2 = Sum_i ||DW_i||_F^2 when A_i are orthogonal is correct. This follows from the fact that B_i^T @ A_i^T and B_j^T @ A_j^T have disjoint column supports when A_i perp A_j (as shown in the worked example). The sqrt(N) norm scaling is correctly derived.

3. **Bandwidth analysis (Section G):** The calculation that 210 modules x d_out x d_in x 2 bytes = ~2.7 GB is correct for d=2560 (2560 x 2560 x 2 x 210 = 2.75 GB). The speed prediction of ~40-67 tok/s from bandwidth division is reasonable. The measured 41.9 tok/s is consistent.

### What does not hold

4. **The "sublinear in norm growth" claim (Section C) is hand-wavy.** The statement "1.29 < 2.24, consistent with nonlinearities being approximately linear for small perturbations" does not follow from first-order Taylor expansion in any rigorous sense. The Taylor expansion applies locally to each nonlinear function, but the full-network PPL ratio depends on the composition of L=30 such expansions. Without bounding the higher-order terms or the Jacobian chain, the "approximately linear" argument is not quantitative. This is clearly labeled "Informal" so it is not misleading, but it is also not a proof.

5. **The W_combined memory size differs between MATH.md and PAPER.md.** MATH.md Section G states "~2.7 GB" (from 210 x 2560 x 2560 x 2 bytes). PAPER.md line 94 and results.json report "4.17 GB." The discrepancy is because MATH.md assumes all modules are d x d, but ADAPTER_TARGETS includes modules of varying dimensions (self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj). The actual total is 4.17 GB, not 2.7 GB. **MATH.md's bandwidth analysis underestimates by 1.5x.** This does not change the conclusion (4.17 GB is even worse than 2.7 GB for speed), but it is an error in the analysis.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table (lines 42-49). Assessment:

| Prediction | Predicted | Measured | Paper's Verdict | My Verdict |
|------------|-----------|----------|-----------------|------------|
| Mean PPL ratio | 1.25-1.35x | 1.447x | PARTIAL | **MISS** -- 1.447 is outside the predicted 1.25-1.35 range by 7% |
| Worst domain ratio | 1.5-2.0x | 1.905x | YES | YES -- within range |
| Speed | 35-50 tok/s | 41.9 tok/s | YES | YES |
| K802 (worst < 2.0x) | Marginal | 1.905x PASS | YES | YES |
| K803 (speed >= 90) | Likely fail | 41.9 FAIL | YES (predicted failure) | YES |
| Room PPL < base for all | Yes for adapted domains | 3/5 YES, 2/5 NO | PARTIAL | **WRONG -- see below** |

### REPORTING INTEGRITY ISSUE: Room vs Base comparison

**PAPER.md line 49 reports "3/5 YES, 2/5 NO" for "Room PPL < base PPL for all domains."**

**PAPER.md lines 53-59 show Room > Base for ALL 5 domains:**
- medical: 10.918 > 6.412
- code: 6.994 > 4.752
- math: 4.954 > 3.734
- legal: 24.838 > 22.813
- finance: 22.586 > 19.990

The "Room < Base?" column correctly says NO for all 5 domains (lines 55-59). But the summary table (line 49) says "3/5 YES, 2/5 NO." This is an internal contradiction. The per-domain table is correct; the summary line is wrong. It should read "0/5 YES, 5/5 NO."

PAPER.md lines 63-66 then correctly state: "Room model PPL is WORSE than base for ALL 5 domains. The adapter sum doesn't just degrade relative to single-adapter -- it actively hurts performance relative to having no adapters at all." This paragraph contradicts the summary table entry.

**This is a significant reporting error that understates the severity of the failure.** The correct finding is that W_combined makes things worse than having no adapters at all -- a much stronger negative result than "3 out of 5 domains benefit." It appears to be a copy-paste or editing error rather than intentional misrepresentation, given that the same document correctly identifies the all-5-domains failure in the paragraph below and in the conclusion. Nevertheless, the summary table is the first thing readers check, and it is wrong.

### Mean PPL ratio miss

MATH.md predicted 1.25-1.35x based on Finding #303's 1.29x. The measured mean is 1.447x. The paper calls this "PARTIAL" but the value is outside the predicted range. Computing the actual mean from the per-domain ratios: (1.905 + 1.776 + 1.331 + 1.114 + 1.110) / 5 = 1.447x. Finding #303's comparison table (PAPER.md lines 69-76) shows the ratios are nearly identical between POC and this retest (e.g., medical 1.92x vs 1.91x), so the discrepancy is in how "mean ratio" was computed in the two experiments. If Finding #303 reported a mean of 1.29x, it may have used a different averaging method (geometric mean, or weighted by token count). This is a minor discrepancy but should be noted.

## NotebookLM Findings

Skipped (authentication not configured for this session). Review conducted via direct close reading of MATH.md, PAPER.md, results.json, run_experiment.py, and VISION.md.

## Novelty Assessment

This experiment claims no novelty. It is explicitly a retest of Finding #303 with a relaxed threshold. The contribution is confirming that the measurements are reproducible (which they are -- the per-domain ratios match within 0.02-0.09x). No prior art search is needed because no new mechanism is proposed.

The PAPER.md conclusion that "the room model concept is dead at N>1" is correctly grounded in both the quality degradation (all 5 domains worse than base) and the bandwidth cost (4.17 GB dense vs 18 MB factored). This conclusion is consistent with VISION.md, which lists factored LoRA as the serving architecture (97 tok/s, 1.22 GB).

## Macro-Scale Risks (advisory)

Not applicable. The architecture is killed. No macro transition planned.

The one useful macro insight is the bandwidth analysis: at any scale, pre-summing N adapters into a dense matrix costs O(d^2) bandwidth per module vs O(d*r) for factored form. Since r << d (16 vs 2560), factored form wins by d/r = 160x per module. This ratio is scale-invariant and would hold at any model size.

## Kill Criteria Assessment

**K802 (worst domain PPL ratio < 2.0x): PASS at 1.905x.**
Honestly evaluated. The paper notes the 5% margin and correctly predicts this would fail at N>5.

**K803 (speed >= 90 tok/s): FAIL at 41.9 tok/s.**
Honestly evaluated. The paper correctly identifies that the 90 tok/s threshold was based on a flawed dispatch-counting argument in ROOM_MODEL.md, and that the actual bottleneck is bandwidth.

**Kill criteria quality:** K802 is a reasonable quality gate. K803's 90 tok/s threshold is poorly motivated (derived from a prediction now known to be wrong), but the failure is so decisive (41.9 vs 90) that the threshold choice is irrelevant. Even at 50 tok/s, factored LoRA at 97 tok/s would still dominate.

**The kill is justified.** Even setting K803 aside, the finding that Room PPL > Base PPL for all 5 domains means the architecture is net-negative -- it would be better to use no adapters at all than to pre-sum them. This is a stronger kill than either kill criterion captures.

## Failure Mode and Impossibility Structure

PAPER.md correctly identifies the failure mode: nonlinear compounding through L layers makes pre-summed adapter deltas interfere at the network level despite per-module orthogonality. This is a structural property, not a bug.

The impossibility structure is sound: for any deep nonlinear network with L > 1 layers, activating N > 1 adapter deltas simultaneously at all layers produces cross-layer interference that grows with both N and L. The only regime where pre-summing is exact is N=1 (single adapter merge, used in Pierre v6).

The paper's framing that "pre-summing works for N=1, fails for N>1" correctly identifies the boundary. The sqrt(N) norm scaling provides a quantitative handle on how degradation grows, though the mapping from norm to PPL is not formally bounded.

## Verdict

**PROCEED** (as a killed experiment -- the kill is justified and properly documented)

The experiment successfully confirms Finding #303's measurements, correctly identifies the kill, and provides a clear impossibility argument for why pre-summing N>1 adapters fails in deep nonlinear networks. The conclusion that factored LoRA dominates W_combined in both quality and speed is well-supported.

### Required fixes before archival (minor, non-blocking for the kill decision):

1. **Fix the prediction table on PAPER.md line 49.** Change "3/5 YES, 2/5 NO" to "0/5 YES, 5/5 NO" for "Room PPL < base PPL for all domains." The per-domain table and the text paragraph both correctly state all 5 domains are worse, but the summary table is wrong.

2. **Fix the W_combined memory estimate in MATH.md Section G.** Change "~2.7 GB" to "~4.2 GB" to match the actual measured size (4.17 GB from results.json). The bandwidth analysis underestimates by 1.5x because it assumes uniform d x d modules.

3. **Acknowledge that the TYPE=verification label is technically incorrect.** This is a retest with relaxed thresholds, not a verification of a formal theorem. The experiment is more accurately TYPE=guided-exploration within Finding #303's empirical framework. Alternatively, add a note that the "theorem" being verified is the reproducibility of Finding #303's measurements.

4. **Clarify the mean PPL ratio discrepancy.** MATH.md predicts 1.25-1.35x (from Finding #303's "1.29x"), but this experiment measures 1.447x. The per-domain ratios are nearly identical to the POC. The discrepancy is likely in how "mean ratio" was computed (arithmetic mean of ratios vs ratio of means, or different domain weighting). Document which averaging method was used in each case.

None of these issues affect the kill decision. The experiment is dead for the right reasons, documented honestly (with the exception of the line-49 error), and the architectural conclusion is sound.
