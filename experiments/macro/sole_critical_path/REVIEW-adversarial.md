# Peer Review: SOLE Critical Path (Revision 2)

## Revision Check

The prior review returned REVISE with 5 specific fixes. Assessment of each:

| Fix | Requirement | Status |
|-----|------------|--------|
| 1. Flag bf16-vs-NF4 confound | PAPER.md must note training precision difference; HYPOTHESES.yml SUPPORTED not KILLED | **DONE.** Dedicated "Major Caveat: Training Precision Confound" section (PAPER.md lines 231-241). HYPOTHESES.yml evidence explicitly notes confound and SUPPORTED status. |
| 2. Headline: 1/N scaling resolves catastrophe | Must be the leading finding, connecting to composition_dropout_robustness KILL | **DONE.** Prominent section at PAPER.md top (lines 9-28) before any experiment results. "10^12x improvement" clearly stated. |
| 3. K2 for poisoned detection -> N/A | Must be N/A not FAIL since no poisoned adapter exists under 1/N scaling | **DONE.** PAPER.md line 73 reads "N/A". HYPOTHESES.yml evidence reads "K2 N/A: premise invalid under 1/N scaling." |
| 4. Note calibration set sizes | Cross-experiment reconciliation must note 30/10/25 difference | **DONE.** PAPER.md lines 273-281 explicitly lists per-experiment sample counts and states values are "NOT directly comparable." |
| 5. Per-domain PPL as blocking gap | Must flag K1 unmeasured for Exp 3 | **DONE.** PAPER.md lines 222-229: "Blocking gap: K1 (per-domain) is unmeasured." HYPOTHESES.yml: "K1 UNMEASURED." |

All 5 fixes adequately addressed.

## Mathematical Soundness

### Experiment 1: LOO Detection

The LOO influence score and Shapley approximation bound are standard and correctly stated. The MATH.md properly qualifies the LOO-Shapley gap at production cosines (0.142). No errors found.

**Remaining subtlety (noted in prior review, still present but not blocking):** The LOO subsets use `n_total=len(adapters)=5` for scaling even when composing 4 adapters, meaning each adapter in the 4-adapter subset gets 1/5 weight, not 1/4. This is a defensible choice (keeps total contribution magnitude constant across conditions) and is consistent with the code. The MATH.md does not mention this choice explicitly but the code is transparent. Not a mathematical error -- a documentation gap.

### Experiment 2: PPL-Probe Weighting

Softmax(1/ppl) formulation is correct. Weight computation matches the code. The 0.27% improvement remains within noise for 10 samples and no significance test is provided, but the paper correctly labels this "marginal." Directionally consistent with prior v2 result (+0.36pp). Sound.

### Experiment 3: SOLE vs Monolithic

The bf16-vs-NF4 confound is now properly flagged. The PAPER.md correctly frames the 32.7% gap as an "upper bound on the true architectural penalty."

One remaining issue: the learning rate discrepancy noted in the prior review. MATH.md says lr=1e-4 (line 306). PAPER.md says lr=3e-5 (line 196). The code uses lr=1e-4 (line 468 of run_all_critical.py). The PAPER.md contains an error on the learning rate. This is a minor documentation inconsistency -- the actual experiment used 1e-4, matching MATH.md and code. The PAPER.md line 196 ("lr=3e-5") should be corrected to 1e-4 but this does not affect any conclusion.

## Novelty Assessment

No novelty claims are made. LOO ranking, softmax weighting, and monolithic-vs-composed comparisons are standard techniques applied to SOLE's specific architecture. This is mechanism validation, not novelty research. Appropriate framing.

The 1/N scaling headline -- that equal-weight 1/N scaling reduces PPL from trillions to 2.36 -- is the genuinely useful finding. It is not novel in the literature (scaling adapter contributions is well-known in LoRA merging work, e.g., LoRA Soups, TIES-merging), but it is important for this project because it resolves the composition_dropout_robustness KILL that was threatening the entire SOLE architecture.

## Experimental Design

### Strengths

1. **Function-scoped cleanup** with explicit GPU memory management is solid engineering.
2. **Manual delta addition** correctly identified as superior to PeftModel sequential merge (2.36 vs 3.51 PPL). This is a useful implementation finding.
3. **Honest reporting** of the monolithic win. No spin on the 32.7% gap.
4. **Proper caveat structure** -- confounds flagged, blocking gaps identified, limitations sections present for each experiment.

### Remaining weaknesses (non-blocking)

1. **No confidence intervals or significance tests.** Single run, single seed, single calibration set across all three experiments. This is acceptable at macro cost constraints but should remain flagged. The 0.27% PPL-probe improvement (Exp 2) is indistinguishable from noise without a paired test.

2. **Calibration texts from training data tails.** Both SOLE and union are evaluated on data from their training distribution. Absolute PPL values (2.36, 1.99) are optimistic. Relative comparisons remain valid because both conditions are identically contaminated.

3. **Top-1 beating all compositions (Exp 2) is not explored.** Medical alone (2.96) beats equal-weight (3.51) and PPL-probe (3.50). This suggests 1/N scaling may over-dilute even at N=5. The paper notes this but does not suggest follow-up experiments testing total scaling > 1.0 (e.g., weights summing to 2.0 or 3.0). This is an important direction for the architecture. Advisory, not blocking.

4. **The SPEC.md notes Exp 3 originally used unscaled PeftModel composition, but the actual script and PAPER.md show 1/N scaled manual delta addition was used.** This inconsistency between SPEC.md line 161 ("UNSCALED addition (sequential PeftModel merge at full alpha=1.0)") and the actual results (manual 1/N scaled, PAPER.md line 207) suggests the script was updated after SPEC.md was written. The PAPER.md is correct about what actually ran. Not blocking, but the SPEC.md is stale.

## Hypothesis Graph Consistency

**exp_poisoned_adapter_detection:** K1 PASS (sql least impactful), K2 N/A (no poisoned adapter), K3 PASS. Status SUPPORTED. HYPOTHESES.yml correctly updated with evidence lines. Consistent.

**exp_ppl_probe_macro_composition:** Directional validation. +0.27% consistent with v2. SUPPORTED. No hard kill criteria violated. Consistent.

**exp_sole_vs_monolithic_v2:** K2 TRIGGERED (32.7% > 10%), K1 UNMEASURED. Status SUPPORTED due to bf16 confound. HYPOTHESES.yml evidence correctly reflects all three facts (K2 triggered, confound, K1 unmeasured). This is the right call -- killing on K2 alone when there is a known systematic confound favoring the winner would be premature.

## Macro-Scale Risks (advisory)

1. **1/N dilution at large N.** At N=50, each adapter contributes 2% of its delta. The top-1-beating-composition finding at N=5 already shows dilution effects. Whether PPL-probe weighting or top-k selection can compensate at N=50 is untested.

2. **Per-domain evaluation is the critical next step for Exp 3.** The entire SOLE value proposition is that specialized experts outperform a generalist on their respective domains. The aggregate PPL comparison disadvantages SOLE (which spreads capacity across domains) versus monolithic (which optimizes a single objective). Per-domain results could show SOLE winning on 4/5 domains despite losing aggregate.

3. **The monolithic comparison was inadvertently advantaged.** Beyond the bf16-vs-NF4 confound, the union adapter sees all data jointly (enabling cross-domain transfer during training), while SOLE adapters train in isolation. At N=5 this matters; at N=50 the monolithic approach would need rank >> 16 to absorb 50 domains, narrowing its advantage.

## Verdict

**PROCEED**

All 5 prior revision items have been adequately addressed. The remaining issues are:
- One minor documentation error (PAPER.md states lr=3e-5, should be 1e-4) -- cosmetic
- SPEC.md is stale regarding Exp 3 composition method -- cosmetic
- LOO scaling detail (1/5 for 4-adapter subsets) undocumented -- minor

None of these affect the conclusions. The three core findings are sound:

1. **1/N scaling resolves composition catastrophe** (PPL trillions -> 2.36). This is the headline and it is properly featured.
2. **LOO ranking works** for adapter contribution assessment. SUPPORTED is the correct status.
3. **Monolithic wins on aggregate PPL** (+32.7%), but the comparison is confounded by training precision. SUPPORTED (not KILLED) is the right call pending fair retraining.

The experiment can update HYPOTHESES.yml as written. The per-domain PPL follow-up for Exp 3 K1 should be prioritized in the next GPU session.
