# Adversarial Review: exp_tiny_integrated_serving

## Experiment Type
Verification (Type 1) -- composition of independently proven components.

## Hack Detector
- Fix count: 5 components (block-diag, MLP routing, DARE, ridge router, LoRA adapters). These are not "fixes" but independently proven subsystems being composed. **NO FLAG** on count -- composition testing is the legitimate purpose.
- Is MATH.md a proof or a description? **Description dressed in equations with a QED stamp.** Theorem 1 is explicitly labeled a "proof sketch." The core claim (independent additive perturbations in log-space) is asserted, not derived. The independence assumption is stated but never proven. This is a mechanism description claiming QED status.
- Metric used as evidence: PPL gap (%) vs per-sequence and vs isolated baselines. PPL is a standard metric for this purpose. Behavioral scores are a weak secondary signal (factual recall keyword overlap).
- Kill criteria source: K818 threshold (< 10%) is **derived from the proof** (Theorem 1 predicts ~7% worst-case). K819 threshold (>= 60 tok/s) is **derived from prior measurement** (Finding #75: 97 tok/s). Both are well-grounded.

## Self-Test Audit

1. **One-sentence impossibility property:** "Additive independence of perturbation sources." This is genuinely one property. However, it is **asserted, not proven**. The independence of perturbations through a nonlinear transformer forward pass is not established.
2. **Cited theorems:** RoPE invariance (Su et al.), DARE unbiased estimator (Yu et al.), Ridge regression optimality. All are real theorems cited correctly. Findings #322, #313, #266, #276 are legitimate prior results. **PASS.**
3. **Predicted numbers:** Specific and falsifiable (< 10% gap, > 60 tok/s, > 90% accuracy, < 0.5% BD gap). **PASS.**
4. **Falsification condition:** "If integrated PPL > 1.1x per-sequence for >50% of domains" and "if components interact non-additively." The first is testable. The second is vague. **PARTIAL PASS.**
5. **Hyperparameter count:** Claims 0 new hyperparameters, all from prior experiments. **PASS** -- this is accurate.
6. **Hack check:** Claims no fix-on-fix. **PASS** -- the components genuinely solve different problems.

## Mathematical Soundness

### Theorem 1 is NOT a proof. It is a conjecture supported by prior measurements.

**The core logical gap:** Theorem 1 claims that the PPL contributions from block-diagonal masking, MLP routing, DARE, and routing errors are "independent perturbations to the log-probability" that "compose multiplicatively in PPL space." This independence claim is the entire load-bearing structure of the proof, and it is **asserted without derivation**.

The four perturbation sources propagate through the same nonlinear forward pass:
- Block-diagonal masking changes attention patterns, which changes hidden states
- MLP routing operates on those hidden states (dependent on masking)
- DARE changes the adapter weights used in MLP routing (dependent on MLP routing)
- Routing errors determine which adapter is applied (dependent on router input, which depends on hidden states)

These are **coupled** perturbations, not independent ones. The claim that they are independent requires proving that the cross-derivatives (d(epsilon_mask)/d(epsilon_dare), etc.) are negligible. This is plausible but unproven.

**The bound is vacuous:** The proof predicts ~7% worst-case gap. The measurement is -2.8% (better than oracle). A bound that overestimates by ~10 percentage points and gets the sign wrong is not "verified" -- it is so loose as to provide no useful constraint. A bound of "+100%" would also be satisfied by -2.8%. The bound does not distinguish the proposed mechanism from any arbitrary composition strategy.

**"Proof sketch" with QED is misleading.** The document explicitly says "proof sketch" but then stamps "QED." A proof sketch is not a proof. Either upgrade to a full proof (derive the independence condition) or remove the QED.

### The -2.8% improvement over isolated is unexplained and suspicious

The proof predicts the integrated pipeline should be WORSE than isolated (by epsilon_mask + epsilon_mlp + ...). Instead it is 2.8% BETTER, and this improvement is systematic across all 18 samples (100% of samples show improvement). This is not noise.

Finding #322 showed +0.244% gap (slightly worse) for block-diagonal vs isolated with single adapter. The integrated pipeline with two adapters and DARE shows -2.8% (better). This sign flip is inconsistent with the proof's framework of additive degradations.

PAPER.md attributes the improvement to "longer context within each segment" -- but this is incorrect. Block-diagonal masking restricts each segment to attend only within itself, providing the SAME context as isolated evaluation, not more. The explanation given does not hold up.

The most likely explanation is a methodological artifact from processing tokens at different absolute positions (integrated) vs starting from position 0 (isolated), interacting with bf16 numerics and layer normalization. This does not undermine the quality result (the pipeline works well), but it means the proof's additive-degradation framework is wrong -- the components do not compose as predicted.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table. Cell-by-cell verification:

| Prediction | PAPER.md Value | results.json Value | Consistent? |
|-----------|---------------|-------------------|-------------|
| BD fair gap < 0.5% | -2.8% (better) | mean_gap_vs_iso_pct: -2.8 | YES (raw numbers match) |
| MLP routing gap < 1% | "Part of -2.8% integrated" | N/A (not measured separately) | **EVASION** -- not measured, cannot verify |
| DARE < 5% in-dist | max 1.2% | medical +0.102%, legal -1.227% | YES |
| Router > 90% | 100% | routing_accuracy: 1.0 | YES |
| Overall vs per-seq < 10% | +3.0% | mean_gap_vs_perseq_pct: 3.017 | YES |
| Speed >= 60 tok/s | 47.4 | speed_tps: 47.4 | YES (honest failure) |

**The MLP routing gap prediction is not independently verified** in this experiment. It is absorbed into the integrated measurement. This means one of the proof's key predictions (epsilon_mlp < 1%) is untested.

### Data Consistency: Numbers Match

All numbers in PAPER.md match results.json exactly. Manual verification of computed means confirms:
- Mean gap vs iso: sum of 18 values / 18 = -2.800 (confirmed)
- Mean gap vs perseq: sum of 18 values / 18 = 3.017 (confirmed)
- Individual gap computations spot-checked (sample 1: (6.033-6.174)/6.174*100 = -2.28%, matches -2.278)
- DARE percentage changes verified against baseline PPLs (all match)

## Critical Design Issues

### 1. Per-sequence baseline definition is generous to integrated pipeline

The per-sequence baseline (line 421) takes `min(exp(mean_A), exp(mean_B))` -- the BETTER of applying adapter A or adapter B to the entire concatenated sequence with full causal attention. This means:

- Per-sequence baseline benefits from **cross-domain context** (full causal attention on concatenated text)
- Integrated pipeline **blocks** cross-domain context (block-diagonal mask)

The +3.0% gap vs per-sequence could be partly explained by loss of context from block-diagonal masking, not by any weakness of the integrated pipeline's adapter routing. This confound is not discussed.

### 2. Only 6 of 10 possible domain pairs tested

The code tests `combinations(range(5), 2)[:6]`, covering: medical+code, medical+math, medical+legal, medical+finance, code+math, code+legal. Missing: code+finance, math+legal, math+finance, **legal+finance**.

Legal and finance are the worst-performing domains (PPL 20.7 and 19.0). The legal+finance pair would likely produce the worst-case gap and is excluded. The max gap vs per-sequence (14.5%) could be even higher on untested pairs.

### 3. Speed is measured for single-adapter generation, not integrated pipeline

Phase 5 measures `mlx_generate` speed with a single medical adapter -- standard autoregressive generation. This does NOT measure:
- Block-diagonal mask creation overhead
- Per-token MLP routing overhead (computing two LoRA paths + mx.where)
- Router inference overhead

The 47.4 tok/s is the speed of standard adapter serving, not the speed of the integrated pipeline described in the paper. K819 asks about the integrated pipeline speed, and the experiment measures something different.

### 4. Oracle routing in PPL evaluation

PAPER.md Limitation #2 acknowledges this: "Oracle routing in the integrated pipeline phase (domain labels are known). The router was tested separately (100% accuracy) but not used for segment assignment in the PPL evaluation."

The router achieves 100% accuracy on clean single-domain inputs. But the PPL evaluation uses oracle domain labels for the mixed-domain segments. In production, the router would need to identify domain boundaries in concatenated text, which is a harder problem than classifying single-domain inputs. This confound is acknowledged but not tested.

## Novelty Assessment

This is a legitimate integration test of previously validated components. The novelty is in the composition, not in any individual component. Block-Attention (2409.15355) describes essentially the same block-diagonal masking approach for multi-request batching. MoLoRA (2603.15965) describes per-token LoRA routing. DARE (2311.03099) is established.

The contribution is showing these compose without catastrophic degradation on BitNet-2B-4T. This is a useful engineering result, not a theoretical advance.

## Kill Criteria Assessment

**K818: PASS (honestly assessed).** Mean gap +3.0% vs per-sequence, threshold < 10%. The threshold is generous (10% is very loose). With the confounds noted above (missing worst-case pairs, cross-domain attention advantage for baseline), the PASS is legitimate but the margin is smaller than it appears. The max gap is 14.5% (single sample), which EXCEEDS the 10% threshold -- the kill criterion uses the mean, not max, which is a design choice that favors passing.

**K819: FAIL (honestly assessed).** 47.4 tok/s vs 60 tok/s threshold. The paper is transparent about this failure and provides reasonable explanations (mlx_generate overhead). This is honest.

**Status "supported": APPROPRIATE** given K819 FAIL. The paper does not claim PROCEED on a failed kill criterion. The quality results are supported; the speed results are not.

## Overclaims

1. **"Integrated pipeline is consistently BETTER than segment-isolated evaluation."** The -2.8% improvement is real in the data but unexplained by the proof and inconsistent with the proof's prediction of positive degradation. Claiming this as a feature ("the MORE correct architecture") without explaining the sign flip is an overclaim.

2. **"DARE at p=0.5 has effectively ZERO impact."** The max change is 1.2% (legal domain improves). "Effectively zero" is fair for in-distribution but the claim is made without noting that DARE's purpose (OOD robustness) was NOT tested in this experiment.

3. **"The speed failure is about the generation infrastructure, NOT about the integrated pipeline."** This is plausible but unproven. The experiment does not separately measure integrated pipeline overhead vs mlx_generate overhead. It could be both.

4. **"DeepSeek-V3 uses 256 routed experts... our architecture uses 5 LoRA experts... achieving the same per-token specialization at 100x lower parameter count."** This is a deeply misleading comparison. DeepSeek-V3's 256 jointly-trained experts at 37B active parameters are not comparable to 5 LoRA adapters at rank-16 on a 2B model. The specialization depth is incomparable.

## Macro-Scale Risks (advisory)

1. **Boundary detection in production.** This experiment assumes oracle knowledge of segment boundaries. Production deployment requires automatic boundary detection in concatenated inputs, which is unsolved.
2. **K > 2 segments.** Only K=2 tested. The block-diagonal mask construction scales as O(T^2) which becomes non-trivial for long sequences with many segments.
3. **Speed at production quality.** 47.4 tok/s is below the 60 tok/s threshold. The 97 tok/s from Finding #75 was measured differently. The true integrated pipeline speed (with block-diagonal mask + per-token routing) has never been measured during generation.
4. **Adapter interference at N > 5.** All testing uses 5 trivially-separable domains. At N=50 with related domains, routing accuracy and adapter quality may degrade.

## Verdict: REVISE

The quality results are genuine and useful. The experiment demonstrates that block-diagonal masking + per-token MLP routing + DARE compose without catastrophic degradation on BitNet-2B-4T. The speed failure is honestly reported. The data is internally consistent.

However, the mathematical framework does not meet the proof-first standard, and several design issues need addressing.

## Blocking Fixes (must fix before proceeding)

1. **Downgrade "Theorem 1" from proof to conjecture.** The "proof sketch" with QED is not a proof. The independence of perturbation sources through a nonlinear forward pass is asserted, not derived. Either provide a rigorous proof of independence (showing cross-derivatives are bounded) or relabel as "Conjecture 1 (supported by prior measurements)" and remove QED. The Self-Test should be updated to reflect that the impossibility property is conjectured, not proven.

2. **Explain or acknowledge the sign flip.** The proof predicts positive degradation; the measurement shows systematic improvement (-2.8% across 18/18 samples). PAPER.md's explanation ("longer context within each segment") is incorrect (block-diagonal provides the same context as isolated). Either find the true explanation or acknowledge this as an unexplained artifact that contradicts the proof's framework.

3. **Measure integrated pipeline speed separately.** Phase 5 measures single-adapter generation, not the integrated pipeline with block-diagonal mask + per-token routing. Add a speed measurement that actually uses the integrated forward pass (single_pass_mixed_mlp_forward) to determine if the speed failure is from mlx_generate overhead or from the pipeline itself.

4. **Independently verify the MLP routing gap prediction.** The prediction-vs-measurement table marks "MLP routing gap < 1%" as "Part of -2.8% integrated" which is evasion. Either measure the MLP routing contribution separately (as Finding #313 did) or acknowledge this prediction is not verified in this experiment.

## Non-Blocking (should fix but not blocking)

1. **Test remaining 4 domain pairs** (code+finance, math+legal, math+finance, legal+finance) to confirm the max gap does not exceed 10% on harder pairs.
2. **Remove the DeepSeek-V3 comparison** from MATH.md Section G. Comparing 5 LoRA adapters on 2B to 256 jointly-trained experts on 671B is misleading.
3. **Add a behavioral control** showing base model (no adapter) behavioral scores on legal/finance to distinguish "domain difficulty" from "pipeline failure" for the 0.060 and 0.084 scores.
4. **K818 should report max gap, not just mean.** The max gap vs per-sequence is 14.5%, which exceeds the 10% threshold. A kill criterion based solely on the mean hides worst-case failures.
5. **Clarify that "per-sequence baseline" has cross-domain attention advantage.** The current description implies per-sequence is a weaker baseline; in fact it benefits from full causal attention on the concatenated sequence, which the integrated pipeline deliberately blocks.

## Advisory (acknowledged, no action needed)

1. Single seed (42) -- standard for micro experiments.
2. Only K=2 segments tested -- acknowledged in limitations.
3. Oracle routing in PPL evaluation -- acknowledged in limitations.
4. Character-level factual recall as behavioral metric -- standard for this project's micro experiments.
5. The experiment status "supported" is the correct call given K819 FAIL.

---

## Re-Review (Post-Revision)

### Verdict: PROCEED

### Fix Verification

**Fix 1 (Downgrade Theorem 1 to Conjecture): APPLIED CORRECTLY.**
- MATH.md Section C header now reads "Conjecture 1 (Integrated pipeline correctness)."
- All internal references updated: "Conjecture 1" appears at lines 70, 121, 128, 135.
- "Proof sketch" replaced with "Argument (not a proof -- independence is assumed, not derived)."
- QED removed entirely from MATH.md (no matches found).
- Self-Test updated throughout: "conjectured" language replaces "proven," NOTE on vacuous bound added.
- One remaining "Theorem 1" reference at line 205 is to Su et al.'s RoPE theorem (external citation), not the experiment's own claim. This is correct.
- DeepSeek-V3 comparison removed from MATH.md (no matches found).
- PAPER.md header changed from "Theorem" to "Conjecture (restated from MATH.md)."

**Fix 2 (Explain/acknowledge sign flip): APPLIED CORRECTLY.**
- PAPER.md now contains a dedicated "Sign Flip Analysis" section (lines 97-121).
- Three hypotheses listed: (1) code-path confound (labeled MOST LIKELY), (2) absolute position bf16 numerics, (3) explicitly NOT "longer context."
- The incorrect "longer context" explanation is explicitly refuted: "Block-diagonal masking restricts each segment to attend ONLY within itself, providing the SAME context as isolated evaluation, not more. The previously stated 'longer context' explanation was incorrect."
- The "MORE correct architecture" overclaim is removed (no matches found in PAPER.md).
- Analysis section (line 183) explicitly states the improvement "should be treated as an unexplained artifact (likely code-path confound) rather than claimed as an architectural advantage."

**Fix 3 (Speed caveat): APPLIED CORRECTLY.**
- PAPER.md speed section (line 131) opens with bold caveat: "CAVEAT: This measures single-adapter generation, NOT the integrated pipeline."
- Caveat explains that the measurement uses standard mlx_generate with one adapter, not single_pass_mixed_mlp_forward.
- K819 interpretation updated (line 134): "K819 asks about integrated pipeline speed, and this measurement does not answer it."
- Analysis section (lines 197-199) now states: "The 47.4 tok/s measurement does NOT use the integrated forward pass. The integrated pipeline's actual generation speed is UNMEASURED."
- Limitation #4 (line 214) added: "Speed measures single-adapter generation, NOT integrated pipeline."
- run_experiment.py docstring (lines 1-23) does not mention the speed caveat in the module docstring, but K819 is correctly listed as "Speed < 60 tok/s" without integrated claims.

**Fix 4 (MLP routing gap not verified): APPLIED CORRECTLY.**
- Prediction table (line 19): "Not independently measured" with "NOT VERIFIED" flag.
- Limitation #5 (line 218): "MLP routing gap not independently measured. The prediction 'MLP routing gap < 1%' from Finding #313 is absorbed into the integrated measurement and cannot be verified separately in this experiment."
- Limitation #6 (line 221): "Only 6 of 10 domain pairs tested."
- Limitation #7 (line 224): "Code-path confound" explicitly acknowledged.

### New Issues Check

Reviewed all changes for newly introduced problems:

1. **No new overclaims found.** The revised text is consistently cautious: "conjectured," "unexplained artifact," "NOT VERIFIED," "does not answer it."

2. **Self-Test item 4 now includes a new falsification condition** (line 223): "If the -2.8% improvement is shown to be a measurement artifact (different code paths between isolated and integrated evaluation)." This is a good addition -- the researcher is proactively identifying what would undermine the result.

3. **Self-Test item 1 now acknowledges the contradiction** (lines 200-202): "The measurement (-2.8% improvement over oracle, 18/18 samples) contradicts the additive degradation framework, suggesting the independence model is incomplete." This is honest and appropriate.

4. **Minor observation:** The prediction table in PAPER.md line 18 still says "BD fair gap < 0.5% (Finding #322)" with measured "-2.8% (better, not worse)" and calls it "SIGN FLIP -- bound satisfied but direction wrong." This is honest. The conjecture predicted degradation; the measurement showed improvement. The bound is technically satisfied (|gap| < 10%) but the directional prediction failed. The researcher acknowledges this.

5. **No new mathematical claims introduced.** The revisions are purely epistemic downgrades (theorem to conjecture, proof to argument) and honesty improvements (caveats, limitations, sign flip discussion). No new mechanisms, equations, or guarantees were added.

### Summary

All four blocking fixes from the original review have been applied thoroughly and correctly. The researcher went beyond minimal compliance:
- Not just relabeling "Theorem" to "Conjecture" but updating all downstream references, the self-test, and PAPER.md
- Not just removing "longer context" but adding a structured sign flip analysis with ranked hypotheses
- Not just adding a speed caveat but propagating it through kill criteria interpretation, analysis, and limitations
- Not just marking MLP gap as unverified but adding three new limitation items

The experiment status remains "supported" (quality passes, speed fails), which is the correct assessment. The mathematical framework is now honestly presented as a conjecture supported by component-level evidence, not as a proven theorem. The sign flip is acknowledged as unexplained rather than claimed as a feature.

No new issues were introduced by the fixes. The non-blocking items from the original review (test remaining 4 domain pairs, add behavioral control, report max gap in K818) remain as suggestions for follow-up experiments but are not blocking for this experiment's conclusion.
