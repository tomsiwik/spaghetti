# Peer Review: BitNet-2B Ternary LoRA Convergence (RE-REVIEW)

## Context

This is a re-review. The prior review returned REVISE with 5 required fixes. The researcher claims all 5 have been addressed. This review verifies each fix and checks for any new issues introduced.

## NotebookLM Findings

NotebookLM authentication not available in this session. Review proceeds with manual analysis against all source documents (MATH.md, PAPER.md, results.json, HYPOTHESES.yml, prior REVIEW-adversarial.md).

## Verification of Required Fixes

### Fix 1: Status Inconsistency -- ADDRESSED (minor residual)

**Original issue**: HYPOTHESES.yml said "supported" but PAPER.md said "KILLED (K2)."

**Current state**: PAPER.md now says "SUPPORTED (K2 inconclusive, multi-seed pending)" (line 8). HYPOTHESES.yml says `status: supported`. These now agree.

**Residual**: results.json still says `"verdict": "KILLED"` (line 300). The HYPOTHESES.yml evidence text still says "K2 MARGINAL KILL" rather than "K2 INCONCLUSIVE." These are minor -- results.json is a code output that was not regenerated, and the evidence text is supplementary to the status field. The primary documents agree. **Non-blocking.**

### Fix 2: Confounded K2 Reframed -- ADDRESSED

**Original issue**: Comparing ternary-400 vs FP16-200 conflates quantization with training duration.

**Current state**: MATH.md now has a dedicated "K2 Analysis (Inconclusive -- Confounded Comparison)" section with the explicit 2x2 factorial table showing the two missing cells (FP16-400 and Ternary-200). PAPER.md K2 assessment is labeled "INCONCLUSIVE due to confounded comparison" with the full factorial table. The paper correctly states "An inconclusive criterion cannot kill a hypothesis."

This is the right framing. The 1.6% gap is real but uninterpretable without the missing controls. The researcher does not overclaim.

### Fix 3: FP16 Latent Composition Documented -- ADDRESSED

**Original issue**: Composition operates on FP16 latent weights, not quantized ternary. The LoTA-QAF path is untested.

**Current state**: MATH.md has a dedicated "Composition Path: FP16 Latent vs Ternary-Native" section explaining the distinction. PAPER.md Limitation #8 documents this clearly: "Composition operates on FP16 latent weights, not quantized ternary weights... True ternary composition requires quantize-then-compose, which is deferred to exp_bitnet_serving_path."

Well handled. The experiment correctly scopes what it tests (QAT training quality for composition) vs what it does not (ternary-native merge).

### Fix 4: K1 Convergence Criterion -- ADDRESSED

**Original issue**: Training loss threshold is unreliable at batch_size=1. Reporting K1 as 3/5 was misleading.

**Current state**: PAPER.md now reports both metrics side-by-side: "3/5 by training loss (unreliable at batch_size=1), 5/5 by val PPL improvement (reliable)." MATH.md has a "Convergence Criterion Reliability" section with the variance analysis showing why batch_size=1 training loss is noisy. The val PPL trajectories are presented as the reliable signal.

This is the right approach. The paper is transparent about the unreliable metric while showing the reliable one passes.

### Fix 5: FP16 Adapter Provenance -- ADDRESSED

**Original issue**: FP16 adapters were "loaded_from_prior_run" with unclear provenance.

**Current state**: PAPER.md Limitation #9 documents filesystem timestamps (adapters created 2026-03-20 23:14-23:30, data splits created 22:17-22:34) confirming the FP16 adapters were trained by a prior run of the same script on the same data splits. The distinction from the contaminated experiment (exp_bitnet_2b_real_composition, which stored data in its own directory) is documented.

Adequate for micro scale. The ideal fix would be to retrain FP16 adapters within the same run, but the timestamp-based provenance is reasonable given the constraints.

## Mathematical Soundness

No changes from prior review. The math remains correct:

- STE implementation is standard and correctly described.
- 1/N scaling formula is correct.
- Convergence criterion variance analysis (new in MATH.md) is sound -- SE of ~0.01-0.02 over 50 batches at batch_size=1 confirms the metric is borderline at best.
- The K2 factorial analysis correctly identifies the confound.

One observation carried forward: the composition ratio metric (avg composed PPL / best individual PPL) uses a cherry-picked denominator. The paper now includes per-domain composition ratios (1.32x avg for ternary vs 1.23x for FP16), which is more informative. This was a non-blocking observation and has been addressed.

## Experimental Design

The core design limitations remain but are now honestly documented:

1. **Single seed** -- acknowledged, deferred to exp_bitnet_multiseed_validation.
2. **400 steps, not 1000** -- acknowledged, runtime constraint documented.
3. **Thin validation** (~3,200 tokens per domain) -- acknowledged as Limitation #7.
4. **Confounded K2** -- now properly labeled INCONCLUSIVE rather than killed or passed.

The experiment tests what it claims to test (convergence of QAT+STE ternary LoRA on BitNet-2B) and is honest about what it does not test (ternary-native composition, matched-step K2, multi-seed reproducibility).

## Hypothesis Graph Consistency

HYPOTHESES.yml status is `supported`, matching PAPER.md verdict. The node correctly blocks exp_bitnet_multiseed_validation, exp_bitnet_task_eval, exp_bitnet_scale_n15, and exp_bitnet_sole_vs_monolithic.

Kill criteria in HYPOTHESES.yml reference "1000 steps" but the experiment ran at 400. The notes explain the discrepancy ("limited by Apple Silicon runtime; original target 1000"). For a re-review, this is acceptable -- the kill criteria were tested at a reduced scale, with all pass/fail assessments honest about the actual conditions.

The evidence text in HYPOTHESES.yml still says "K2 MARGINAL KILL" which is inconsistent with the revised PAPER.md framing of "INCONCLUSIVE." This should be updated for consistency but is non-blocking -- the status field is the authoritative signal.

## Macro-Scale Risks (advisory)

Carried forward from prior review, all remain relevant:

1. **STE oscillation at longer training**: Val PPL still improving at 400 steps. Monitor alpha (quantization scale) stability at 800-1000+ steps.
2. **Composition gap may grow with N**: At N=5, 1.6% is small. At N=15-25 (exp_bitnet_scale_n15), accumulated interference from 1.9x higher cosine could compound.
3. **Ternary-native composition untested**: The LoTA-QAF path (quantize-then-compose) is architecturally different from compose-then-quantize. exp_bitnet_serving_path must test this before production claims.

## Verdict

**PROCEED**

All 5 required fixes from the prior review have been substantively addressed:

1. Status inconsistency resolved (PAPER.md and HYPOTHESES.yml both say SUPPORTED).
2. K2 properly reframed as INCONCLUSIVE with 2x2 factorial table.
3. FP16 latent composition limitation documented with LoTA-QAF path deferred.
4. K1 now reports both metrics with reliability assessment.
5. FP16 adapter provenance verified via timestamps.

The experiment's core contribution -- demonstrating that QAT+STE ternary LoRA converges on BitNet-2B-4T across 5 domains with individual quality exceeding FP16 -- is well-supported. The confounded K2 is honestly labeled inconclusive rather than overclaimed. The remaining open questions (multi-seed, matched-step comparison, ternary-native composition) are correctly deferred to downstream experiments.

### Non-Blocking Items for Future Reference

- Update HYPOTHESES.yml evidence text from "K2 MARGINAL KILL" to "K2 INCONCLUSIVE" for consistency with PAPER.md.
- results.json verdict field ("KILLED") is stale relative to the revised assessment. If the experiment is rerun, the code should emit the revised verdict.
- The creative domain val PPL trajectory (3.13 -> 3.18 -> 3.23 -> 3.12) is non-monotonic. MATH.md correctly labels it "No (overfit at 200-300, recovers)" while PAPER.md says "Yes (recovered)." Both are technically accurate for their respective questions (monotonicity vs final improvement) but the discrepancy could confuse readers.
