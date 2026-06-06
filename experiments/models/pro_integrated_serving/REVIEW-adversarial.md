# Peer Review: pro_integrated_serving

## Experiment Type
Verification (Type 1) -- replicating proven pipeline from tiny (Finding #323) on Qwen3-4B-4bit.

## Hack Detector
- Fix count: 5 components (block-diag, MLP routing, DARE, ridge router, LoRA adapters). Same as tiny. These are independently proven subsystems being composed. **NO FLAG** -- composition verification is the stated purpose.
- Is MATH.md a proof or a description? **Description with conjecture, honestly labeled.** MATH.md labels the core claim as "Conjecture 1 (Additive independence)" throughout. No QED stamp. This is an improvement over the original tiny MATH.md. However, this means the experiment lacks a formal proof -- the framework is conjectural.
- Metric used as evidence: PPL gap (%) for composition quality; factual-recall keyword overlap for behavioral quality. PPL is appropriate for composition verification. Factual recall is a weak behavioral proxy but acknowledged as such.
- Kill criteria source: K821 threshold (behavioral >= 0.3) is reasonable but not mathematically derived from the conjecture. The conjecture predicts a PPL gap < 10%, but the kill criterion measures behavioral score instead. These are different metrics.

## Self-Test Audit

1. **One-sentence impossibility property:** "Conjectured additive independence of perturbation sources in log-probability space." Genuinely one property, clearly stated. Honestly labeled as conjectured, not proven. **PASS.**
2. **Cited theorems:** RoPE invariance (Su et al. 2104.09864), DARE unbiased estimator (Yu et al. 2311.03099), Ridge regression optimality, Davis-Kahan perturbation bound, Findings #322/#313/#320/#330. All real, all cited correctly. **PASS.**
3. **Predicted numbers:** Router accuracy >= 90%, behavioral >= 0.3, integrated vs per-seq < 10%, speed > 30 tok/s. Specific and falsifiable. **PASS.**
4. **Falsification condition:** "If MLP token-independence fails for SiLU (impossible)", "If block-diagonal fails with QK-norm (impossible)", "If additive independence is wrong (would manifest as gap >> 10%)." The first two are structural guarantees (good). The third is testable but vague -- ">> 10%" is not precise. **PARTIAL PASS.**
5. **Hyperparameter count:** Claims 0 new. Accurate -- all inherited from prior experiments. **PASS.**
6. **Hack check:** Claims no fix-on-fix. Accurate -- this is replication, not new mechanisms. **PASS.**

## Mathematical Soundness

### The framework is a conjecture, not a proof -- and this is honestly stated

MATH.md labels the core claim as "Conjecture 1" and does not stamp QED. This is the correct epistemic level given that:
- The independence of perturbation sources through a nonlinear transformer is asserted, not derived
- The tiny experiment's -2.8% improvement contradicted the conjecture's prediction of positive degradation
- No formal bound on cross-derivatives between perturbation sources is established

The cited theorems (RoPE invariance, DARE unbiased estimator, Ridge optimality, MLP token-independence) are individually correct and applicable. The gap is in the composition claim -- that these individual guarantees compose additively. This gap is acknowledged.

### SiLU and QK-norm arguments are sound

MATH.md correctly notes that:
- SiLU(x) = x * sigmoid(x) is elementwise, so MLP token-independence holds for Qwen3's activation (Section E, item 2)
- QK-norm (RMSNorm on query/key heads) is a per-token operation that does not introduce cross-token dependencies (Section E, item 3)

These are structural guarantees, not empirical claims. They hold.

### The Davis-Kahan scale bound is correctly applied

The reference to Finding #320/#330 (scale<=5 preserves MMLU, scale=20 catastrophic) justifies LORA_SCALE=5.0. This is a sound parameter choice derived from prior results.

### Quantitative predictions are loose but falsifiable

The predictions (router >= 90%, behavioral >= 0.3, gap < 10%, speed > 30) are not tight bounds derived from the conjecture -- they are reasonable thresholds from analogical reasoning. A formal conjecture should produce tighter predictions. But for a replication experiment, these are adequate.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table. Verified cell by cell:

| Prediction | Expected | Measured | Match? | Verified vs results.json? |
|-----------|----------|----------|--------|--------------------------|
| Router accuracy >= 90% | >= 90% | 98.0% (49/50) | YES | routing_accuracy: 0.98 -- MATCH |
| Overall behavioral >= 0.3 | >= 0.3 | 0.364 | YES | overall_routed_behavioral: 0.364 -- MATCH |
| Integrated vs per-seq < 10% | < 10% | -6.2% | YES | mean_gap_vs_perseq_pct: -6.168 -- MATCH |
| Integrated vs isolated < 10% | < 10% | -3.4% | YES | mean_gap_vs_iso_pct: -3.403 -- MATCH |
| Speed > 30 tok/s | > 30 | 32.5 tok/s (gen) | YES | generate_tps: 32.5 -- MATCH |

All numbers in PAPER.md match results.json exactly. Manual verification of all 18 sample gaps confirms:
- Sum of gap_int_vs_iso_pct = -61.26, /18 = -3.403 (matches)
- Sum of gap_int_vs_perseq_pct = -111.03, /18 = -6.168 (matches)
- Count of negative gap_int_vs_iso: 14/18 (matches "78%" claim)
- Count of negative gap_int_vs_perseq: 16/18 (matches "89%" claim)
- Max gap_int_vs_iso: +5.340 at code+legal (matches "+5.3% (code+legal pair)" claim)
- Max gap_int_vs_perseq: +2.164 at code+legal (matches "+2.2% (code+legal pair)" claim)

**Data integrity is excellent.** All claims are verifiable and verified.

## Critical Issues

### 1. BLOCKING: Attention LoRA asymmetry invalidates the PPL comparison

This is the most serious issue and it is not discussed in MATH.md or PAPER.md.

The integrated pipeline (`single_pass_mixed_mlp_forward`) applies LoRA ONLY to MLP layers, using base weights for attention (lines 327-354 of run_experiment.py: `attn.q_proj(h_norm)` calls the base projection, not a RuntimeLoRA wrapper).

Both baselines (per-sequence and isolated oracle) use `attach_adapter` which wraps ALL 7 target projections -- including `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj` -- in RuntimeLoRA. So both baselines compute attention with LoRA perturbations.

This means the comparison is:
- **Integrated:** base attention + per-token MLP LoRA
- **Per-sequence baseline:** adapted attention + uniform MLP LoRA
- **Isolated oracle:** adapted attention + correct MLP LoRA

The integrated pipeline has an unfair advantage if attention LoRA at scale=5 is harmful (plausible -- adapters were trained at scale=20, applied at scale=5, so the attention perturbation may be poorly calibrated). Or an unfair disadvantage if attention LoRA is helpful.

The -3.4% improvement over isolated oracle and -6.2% over per-sequence could be ENTIRELY explained by "dropping attention LoRA at scale=5 is beneficial." This alternative hypothesis is simpler and more parsimonious than the "block-diagonal prevents cross-segment interference" explanation offered in PAPER.md.

**This confound cannot be resolved from the current data.** To disentangle:
- Run the isolated oracle WITHOUT attention LoRA (MLP-only adapter, same as integrated)
- If the improvement disappears, the explanation is "attention LoRA at scale=5 is harmful"
- If it persists, the block-diagonal isolation explanation has support

This is the same class of issue as the "code-path confound" identified in the tiny review, but more specific and actionable.

### 2. BLOCKING: The sign flip explanation is inconsistent between PAPER.md sections

PAPER.md offers this explanation for integrated beating isolated: "Block-diagonal masking prevents cross-segment interference that the per-sequence baseline suffers from."

But the isolated oracle has NO cross-segment interference (each segment is evaluated completely separately). So "preventing cross-segment interference" cannot explain why integrated beats isolated. It can only explain why integrated beats per-sequence.

The comparison table (PAPER.md line 179) notes the tiny experiment showed +3.0% vs per-sequence (worse) while pro shows -6.2% (better) -- a directional reversal. PAPER.md attributes this to "scale=5 reduces the adapter perturbation magnitude, making cross-segment interference from wrong-adapter application more damaging relative to the smaller correct adapter signal." This explanation is ad hoc and untested. The attention LoRA asymmetry (Issue #1) is a simpler explanation.

### 3. NON-BLOCKING: Speed measurement mixes prefill and generation inappropriately

PAPER.md reports two speed numbers:
- 1209.5 tok/s (prefill, integrated pipeline)
- 32.5 tok/s (generation, single adapter)

These measure fundamentally different operations. Prefill processes all tokens in parallel; generation processes one token at a time. Reporting them in the same table invites misleading comparisons. The "Speed > 30 tok/s" prediction was intended for generation speed, and the 32.5 tok/s barely passes.

However, PAPER.md Limitation #3 explicitly notes this distinction: "The 1209.5 tok/s measures prefill (batch forward pass), not autoregressive generation." This is honest and adequate.

Credit: the pro experiment DOES measure integrated pipeline prefill speed, which directly addresses the tiny review's criticism (#3) about only measuring single-adapter generation. The prefill measurement is new and useful. But the generation speed measurement still uses single-adapter mlx_generate, not integrated generation.

### 4. NON-BLOCKING: Only 6 of 10 domain pairs tested (same gap as tiny)

The code tests `combinations(range(5), 2)[:6]`, excluding: code+finance, math+legal, math+finance, legal+finance. The worst-performing pair (code+legal, +5.3% gap) involves one weak domain. The untested legal+finance pair (both weak domains) could produce larger gaps. This was flagged in the tiny review and remains unaddressed.

### 5. NON-BLOCKING: Behavioral metric is keyword overlap, not real behavioral assessment

`factual_recall` counts word overlap between generated text and reference text after removing stop words. This has well-known failure modes: paraphrase generates low scores, copying reference text generates high scores, hallucinated content with correct vocabulary generates high scores. The 0.364 average could be noise.

PAPER.md does not overclaim behavioral results. The per-domain breakdown (medical 0.540, code 0.470, math 0.627, legal 0.096, finance 0.086) shows clear differentiation between strong and weak domains, which is signal. But the absolute numbers are hard to interpret without a base model control (no adapter).

### 6. NON-BLOCKING: DARE seeds differ between Phase 3 and Phase 5

Phase 3 uses `seed=SEED + di` per domain. Phase 5 uses `seed=SEED` for all domains. This means the DARE masks are different between phases. Since DARE is unbiased regardless of seed, this does not invalidate results, but it means the specific adapter weights differ between the PPL comparison (Phase 3) and the behavioral evaluation (Phase 5). Acknowledging this would improve reproducibility documentation.

### 7. NON-BLOCKING: Only 5 eval samples per domain, single seed

Acknowledged in PAPER.md Limitation #5. Standard for micro experiments. Not blocking.

## Comparison Table Verification (Pro vs Tiny)

| Metric | Tiny (PAPER.md) | Pro (PAPER.md) | Consistent? |
|--------|-----------------|----------------|-------------|
| Router accuracy | 100% | 98.0% | YES -- slightly lower on larger model, plausible |
| Behavioral (routed) | 0.333 | 0.364 | YES -- slightly better on larger model |
| Integrated vs isolated | -2.8% | -3.4% | YES -- same sign, similar magnitude |
| Integrated vs per-seq | +3.0% | -6.2% | **DIRECTIONAL FLIP** -- see Issue #2 |
| Generate speed | 47.4 tok/s | 32.5 tok/s | YES -- larger model is slower |

The directional flip in "integrated vs per-seq" (+3.0% to -6.2%) is the most notable cross-experiment difference. PAPER.md offers an explanation but it is ad hoc. The attention LoRA asymmetry (Issue #1) is a more parsimonious explanation because:
- At scale=20 (tiny), attention LoRA may be more helpful (trained at that scale)
- At scale=5 (pro), attention LoRA may be harmful (applied at 4x lower scale than training)
- Dropping attention LoRA (as the integrated pipeline does) would show more benefit at scale=5

This is testable and should be tested before making claims about the sign flip.

## Novelty Assessment

This is a replication study, not a novelty claim. The contribution is showing that the tiny pipeline transfers to a different architecture (Qwen3-4B, 4-bit quantized, SiLU, QK-norm, GQA). The pipeline's individual components (Block-Attention, per-token MLP routing, DARE, ridge regression) are established.

The replication across architectures is valuable engineering validation. The novelty is low but appropriate for a verification experiment.

## Macro-Scale Risks (advisory)

1. **Attention LoRA question must be resolved before macro.** If the improvement over baselines is from dropping attention LoRA (a handicap, not a feature), then the pipeline's advantage may evaporate when attention LoRA is properly calibrated.
2. **Boundary detection in production.** Both tiny and pro use oracle domain labels. Production requires automatic boundary detection, which is unsolved.
3. **K > 2 segments.** Only K=2 tested. Block-diagonal mask construction is O(T^2) per forward pass.
4. **Generation speed with integrated pipeline.** Neither tiny nor pro measures autoregressive generation with the integrated forward pass. The actual integrated generation speed is unknown.
5. **Adapter quality at scale=5.** Legal (0.096) and finance (0.086) are near-zero behavioral scores. At scale=5, these adapters may be providing negligible domain specialization. This needs better adapters, not pipeline fixes.

## Verdict

**REVISE**

The data is internally consistent, the numbers check out, and the replication across architectures is a genuine contribution. The code is well-structured and the PAPER.md is mostly honest about limitations. The improvement over the tiny experiment (conjecture labeling, prefill speed measurement, explicit limitation sections) shows the researcher learned from the prior review.

However, two issues must be addressed before this can proceed:

### Blocking Fixes

1. **Acknowledge and test the attention LoRA asymmetry.** The integrated pipeline uses base attention (no LoRA) while both baselines use attention LoRA (via RuntimeLoRA wrapping of q/k/v/o_proj). This is a fundamental confound that could explain the entire -3.4% vs isolated and -6.2% vs per-sequence improvement. Either:
   - (a) Add a control: run the isolated oracle with MLP-only adapters (detach attention LoRA, keep MLP LoRA). If the integrated vs this new baseline is ~0%, the improvement is from dropping attention LoRA, not from pipeline composition. This is the cheap test.
   - (b) At minimum, add a paragraph to PAPER.md and MATH.md Section E explicitly acknowledging this asymmetry and identifying it as an alternative explanation for the sign flip. State that the current data cannot distinguish "pipeline composition benefit" from "attention LoRA at scale=5 is harmful."

2. **Correct the sign flip explanation for the vs-isolated comparison.** PAPER.md's explanation ("block-diagonal prevents cross-segment interference") applies only to the vs per-sequence comparison, not the vs isolated comparison. The isolated oracle already has no cross-segment interference. The explanation must be differentiated:
   - vs per-sequence: "integrated uses correct per-segment adapters instead of one adapter for both segments" (simple, expected)
   - vs isolated: cause unknown, most likely the attention LoRA asymmetry (Issue #1) or code-path confound (as identified in tiny review)

### Non-Blocking (should fix, not required)

3. Test remaining 4 domain pairs (code+finance, math+legal, math+finance, legal+finance).
4. Add a base model control (no adapter) to the behavioral evaluation to establish a floor.
5. Note the DARE seed inconsistency between Phase 3 and Phase 5 in the limitations.
6. The speed prediction "Speed > 30 tok/s" is so low as to be unfalsifiable for a 4B model on M5 Pro. Consider whether this prediction adds value.

---

## Re-Review (Post-Revision)

**Reviewer:** Peer review re-assessment of revisions to MATH.md and PAPER.md.
**Date:** 2026-04-06
**Original blocking issues:** (1) Attention LoRA asymmetry confound unacknowledged, (2) Sign flip explanation inconsistent between vs-isolated and vs-per-sequence comparisons.

### Fix 1: Attention LoRA Asymmetry Confound

**Status: FIXED.**

The revision adds thorough acknowledgment of this confound across multiple locations:

- **MATH.md Section E, item 5 (lines 141-155):** New assumption documenting the
  asymmetry. Explicitly states the integrated pipeline uses base attention (no LoRA
  on q/k/v/o_proj) while both baselines use RuntimeLoRA on all 7 projections.
  Identifies the confound precisely: "the integrated pipeline's improvement over
  baselines could be caused by dropping attention LoRA (which may be harmful at
  scale=5, since adapters were trained at scale=20) rather than by pipeline
  composition quality." Specifies the breaking condition: MLP-only isolated control
  matching integrated PPL. This is exactly the control I requested.

- **MATH.md Self-Test item 4 (lines 233-235):** Adds the attention LoRA asymmetry
  as a falsification condition, correctly noting it would "reattribute the sign
  flip from 'composition benefit' to 'attention LoRA is harmful at scale=5'" --
  this distinguishes between invalidating the pipeline and reattributing the cause,
  which is the correct epistemic framing.

- **PAPER.md "Interpretation (vs isolated)" (lines 129-139):** Calls the -3.4%
  improvement "UNEXPECTED and its cause is not established." Identifies the attention
  LoRA asymmetry as "the most parsimonious explanation." States "The current data
  cannot distinguish" between the two hypotheses. Proposes the follow-up.

- **PAPER.md comparison section (lines 196-203):** Offers both explanations (attention
  LoRA asymmetry at scale=5, and scale reduction effect) as "not mutually exclusive
  and cannot be distinguished from the current data."

- **PAPER.md Limitation 6 (lines 230-238):** Dedicated paragraph. States bluntly:
  "The entire -3.4% vs isolated improvement could be explained by 'dropping attention
  LoRA at scale=5 is beneficial.' This is the most serious unresolved confound."
  Specifies the exact follow-up experiment needed.

The confound is now documented at every relevant location. The claims are properly
scoped: the paper does NOT claim the integrated pipeline is compositionally better
than isolated; it reports the measurement and flags the confound. This is honest science.

### Fix 2: Sign Flip Explanation Differentiation

**Status: FIXED.**

The revision now properly differentiates the two comparisons:

- **vs per-sequence (lines 123-127):** "The per-sequence baseline applies ONE adapter
  uniformly to a mixed-domain input, so wrong-adapter tokens receive incorrect MLP
  perturbations. The integrated pipeline applies the correct adapter per token via
  block-diagonal masking + MLP routing, eliminating cross-segment interference."
  This is expected, correct, and well-explained.

- **vs isolated (lines 129-139):** "The isolated oracle already has zero cross-segment
  interference (each segment runs separately), so 'preventing interference' cannot
  explain this gap." The original review's objection -- that the "block-diagonal
  prevents interference" explanation was being applied to the vs-isolated comparison
  where it makes no sense -- is directly addressed. The cause is now attributed to
  the attention LoRA asymmetry confound, with appropriate uncertainty.

The previous inconsistency where "preventing cross-segment interference" was
invoked for the vs-isolated comparison (where interference is already zero) is
fully resolved.

### Data Integrity Verification

All numbers in PAPER.md were independently verified against results.json:

| Claim in PAPER.md | results.json value | Match? |
|-|-|-|
| Router accuracy 98.0% (49/50) | routing_accuracy: 0.98 | YES |
| Overall behavioral 0.364 | overall_routed_behavioral: 0.364 | YES |
| Mean gap vs iso -3.4% | mean_gap_vs_iso_pct: -3.403 | YES |
| Mean gap vs perseq -6.2% | mean_gap_vs_perseq_pct: -6.168 | YES |
| Max gap vs iso +5.3% (code+legal) | max_gap_vs_iso_pct: 5.34 | YES |
| Max gap vs perseq +2.2% (code+legal) | max_gap_vs_perseq_pct: 2.164 | YES |
| 14/18 integrated beats isolated (78%) | Verified: 4 positive gaps out of 18 | YES |
| 16/18 integrated beats per-seq (89%) | Verified: 2 positive gaps out of 18 | YES |
| Prefill speed 1209.5 tok/s | integrated_tps: 1209.5 | YES |
| Generate speed 32.5 tok/s | generate_tps: 32.5 | YES |

Additionally, I recomputed both means from the 18 individual sample gaps:
- Sum of gap_int_vs_iso_pct = -61.260, /18 = -3.403 (matches)
- Sum of gap_int_vs_perseq_pct = -111.027, /18 = -6.168 (matches)

DARE sparsification is applied consistently to all three conditions (per-sequence,
isolated, integrated) in Phase 3. No DARE confound.

Token counts for fair PPL: integrated excludes the boundary token, yielding T-2
tokens. Isolated oracle computes (boundary-1) + (T-boundary-1) = T-2 tokens.
Counts match. No PPL denominator confound.

### Check for NEW Blocking Issues

**No new blocking issues found.** Specifically checked:

1. **DARE consistency across conditions:** All three conditions in Phase 3 use the
   same DARE-sparsified adapters (loaded at line 506 of run_experiment.py). No confound.

2. **RoPE position offset between isolated and integrated:** In the integrated
   pipeline, segment B tokens occupy RoPE positions boundary..T-1, while in isolated
   evaluation they occupy positions 0..len_B-1. Finding #322 (bd fair gap = 0.244%)
   establishes that this does not matter because RoPE attention scores depend only on
   relative position. Block-diagonal masking ensures within-segment relative positions
   are preserved. Not blocking.

3. **Per-sequence baseline uses min(PPL_A, PPL_B):** This is extremely favorable
   to the baseline (it gets to pick the better adapter after the fact). The integrated
   pipeline still beats this oracle selection, which strengthens the result -- though
   the attention LoRA confound remains the most likely explanation.

4. **Revision did not introduce overclaims:** The verdict remains "SUPPORTED" (not
   "conclusive" or "proven"). The conjecture is still labeled as a conjecture, not a
   theorem. No QED stamps were added. Claims are scoped correctly.

### Remaining Non-Blocking Issues from Original Review

Items 3-6 from the original review were non-blocking and remain unaddressed. This
is acceptable -- they are genuine improvements but not required for the finding:

3. 4 of 10 domain pairs untested (same as original review)
4. No base model control for behavioral floor
5. DARE seed inconsistency between Phase 3 and Phase 5
6. Speed prediction (> 30 tok/s) unfalsifiably low

### Summary

The researcher addressed both blocking issues thoroughly and honestly:
- The attention LoRA asymmetry is now the single most prominently documented
  limitation, appearing in MATH.md (Section E + Self-Test) and PAPER.md
  (interpretation + comparison + dedicated limitation).
- The sign flip explanation is properly differentiated between vs-per-sequence
  (expected: correct routing) and vs-isolated (unknown: most likely attention LoRA
  asymmetry).
- No overclaims were introduced. The paper is appropriately cautious about what
  the data can and cannot show.
- Data integrity is excellent throughout.

The experiment demonstrates that the integrated pipeline (block-diagonal masking +
per-token MLP routing + DARE + ridge router) composes without catastrophic
degradation on Qwen3-4B-4bit, meeting K821 (behavioral >= 0.3). The -3.4%
improvement over isolated remains unexplained (most likely attention LoRA confound)
and should not be cited as evidence of compositional benefit until the MLP-only
isolated control is run.

**Verdict: PROCEED**

The experiment is ready to record as a finding with status=supported. The finding
should note that the attention LoRA asymmetry confound is unresolved and that a
follow-up MLP-only isolated control experiment is needed to determine whether the
vs-isolated improvement comes from pipeline composition or from dropping attention
LoRA at scale=5.
