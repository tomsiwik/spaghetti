# Peer Review: Behavioral Eval Routed (Re-Review)

## Prior Review Required 5 Fixes -- Verification

| # | Required Fix | Status | Detail |
|---|-------------|--------|--------|
| 1 | Reclassify as guided exploration | DONE (MATH.md, PAPER.md) / PARTIAL (code) | MATH.md line 3 and PAPER.md line 5 correctly say "guided exploration." run_experiment.py line 12 still says "Type: verification." Minor residual inconsistency. |
| 2 | Fix K2 kill criterion | MOSTLY DONE | MATH.md, PAPER.md, and code logic (line 874: `k2_pass = behavioral_contradicts_mmlu >= 1`) all agree: kill if 0 contradictions. But code docstring (line 10) still says "contradicts MMLU direction on >=2 domains -> KILL" which is the opposite. The logic is correct; only the stale docstring remains wrong. |
| 3 | Acknowledge max_tokens=128 truncation confound | DONE | PAPER.md line 76 explicitly notes the truncation artifact. Limitation #3 provides thorough discussion. Interpretation reframed as "format efficiency under token constraints." |
| 4 | Report statistical significance | DONE | PAPER.md lines 87-95 contain Fisher's exact p-values per domain. Honest summary: only math is statistically significant. |
| 5 | Downgrade claims | DONE | PAPER.md honest summary (line 95) is properly calibrated. No overclaim of "3/5 domains conclusively better." |

## Experiment Type

Guided exploration. The proven framework (Finding #217 per-domain scales, Finding #210 behavioral eval with kappa=0.800) is clearly stated. The unknown being narrowed (does routed composition improve behavioral quality?) is precisely identified. The experiment narrows this unknown rather than just measuring an outcome -- it establishes that the math domain robustly improves and that format-metric dissociation is real for at least one domain.

## Hack Detector

- Fix count: 0. This is a measurement experiment, not an intervention. No new mechanisms, losses, or tricks.
- Is MATH.md a proof or a description? Description -- explicitly acknowledged as "not a deep theorem." Acceptable for guided exploration, which does not require Theorem/Proof/QED.
- Metric used as evidence: Behavioral scores validated by Cohen's kappa = 0.800 (Finding #210). Math uses numerical answer correctness (verifiable from raw data). Code uses ast.parse (deterministic). These are reasonable proxies.
- Kill criteria source: Derived from the format-confound hypothesis. K1 tests overall behavioral quality. K2 tests whether the dissociation exists. Both are hypothesis-driven, not arbitrary.

## Self-Test Audit

1. **One property:** "Format-sensitive metrics conflate format compliance with knowledge." Single property, clearly stated. PASS.
2. **Cited theorems:** HELM (Liang et al., 2022) is real. Finding #236, #210, #217 are internal. No mathematical theorem claimed because none needed for guided exploration. PASS.
3. **Specific predictions:** Math >= 0.20, Code >= 0.50, routed >= base on >= 3/5 domains. Specific and falsifiable. PASS.
4. **Falsification:** K1 kills if >= 3/5 domains worse. K2 kills if zero contradictions. Both target the hypothesis. PASS.
5. **Hyperparameters:** 0 new. Scales from Finding #217, oracle routing deterministic. PASS.
6. **Hack check:** "Measurement experiment, not a fix." Correct. PASS.

## Mathematical Soundness

Not applicable in the traditional sense -- this is guided exploration, not verification. The mathematical framework is the prior proven results (Finding #217 scales, Finding #210 eval validation). MATH.md Section C's "Observation" about M(f+P) < M(f) not implying B(f+P) < B(f) is trivially correct and honestly described as such. No unsound claims.

## Prediction vs Measurement

PAPER.md contains a clear table (lines 13-19). All 5 predictions confirmed. The measurements are internally consistent with the raw data in results.json.

The strongest result (math: 1/10 to 8/10, p<0.005) is properly qualified with the truncation caveat. The weakest results (medical, legal, finance) are honestly reported as inconclusive at n=10. This is well-calibrated.

## Remaining Issues (Non-Blocking)

1. **Stale code docstring (run_experiment.py line 10):** Still says K2 kills on >=2 contradictions. The code logic is correct (kills on 0 contradictions), so this is cosmetic. Should be fixed for hygiene but does not affect results.

2. **Stale type label (run_experiment.py line 12):** Still says "Type: verification." MATH.md and PAPER.md are correct. Cosmetic.

3. **Legal partial contradiction unaddressed:** The prior review noted that legal shows MMLU degraded (-10pp) but behavioral neutral (-2.1%), which is arguably a partial contradiction. PAPER.md does not explicitly address why this is counted as "no gap" rather than a partial gap. This is a judgment call, not a flaw -- neutral is not positive -- but the categorization threshold could be stated explicitly.

## Novelty Assessment

The observation that MMLU conflates format compliance with knowledge is well-established (HELM, etc.). The contribution here is applying this insight to the SOLE architecture to demonstrate that prior MMLU-based kills were false negatives. This is operationally important for the project (it unblocks math adapter development) even though it is not novel research.

## Macro-Scale Risks (advisory)

1. Oracle routing must be replaced with learned routing -- acknowledged in PAPER.md. This is the actual existential test.
2. max_tokens truncation confound diminishes at scale but should be controlled for (run at multiple token limits).
3. n=10 is sufficient for the math signal but prose domains need larger samples at macro scale.

## Verdict

**PROCEED**

All 5 required fixes from the prior review have been substantively addressed. The two remaining issues (stale code docstring, stale type label) are cosmetic and do not affect the experimental results or their interpretation.

The core finding is real and properly qualified: math adapter robustly improves behavioral quality (+700%, p<0.005) despite MMLU degradation, establishing that at least one prior MMLU-based kill was a false negative. The writeup is now honest about what is and is not established:
- Math: robust (p<0.005), with truncation caveat acknowledged
- Code: suggestive (p~0.16), needs larger n
- Prose: inconclusive at n=10

Finding status should be "supported" (not "conclusive"), consistent with guided exploration that narrowed but did not fully resolve the unknown.
