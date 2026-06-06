# Peer Review: Reasoning Expert Distillation

**Reviewer:** Adversarial Peer Review Process
**Date:** 2026-03-13
**Status:** Pre-execution review (no GPU results yet)

## NotebookLM Findings

Skipped -- this is a pre-execution design review. The mathematical foundations are straightforward extensions of proven SOLE results, so the value is in experimental design scrutiny rather than deep mathematical novelty analysis.

## Mathematical Soundness

### What Holds

1. **Orthogonality bound (MATH.md Section 2.1):** The Johnson-Lindenstrauss concentration bound E[|cos|] <= sqrt(2/(pi*D)) is correctly applied. The D_total calculation for Qwen2.5-7B (47.7M params) is correct given the stated architecture. The expected cosine of ~1.15e-4 is a valid upper bound for independently trained adapters. This is a direct scaling of the proven d=896 result (cos=0.0002).

2. **Interference bound (Section 2.3):** The decomposition h_composed = h_base + h_R + h_D is exact for linear layers. The cross-term bound |<h_R, h_D>| <= ||Delta_R||_F * ||Delta_D||_F * ||x||^2 * |cos(Delta_R, Delta_D)| is correct. However, the bound applies per-layer and the stated "domain degradation < 0.1%" does not follow directly from this -- see below.

3. **Cost analysis (Section 4.2):** Correctly computed. $0.07 for training is plausible given the stated parameters.

### Issues Found

**Issue 1: The interference bound is per-layer, not end-to-end.** MATH.md Section 2.3 bounds the cross-term for a single linear layer. A 28-layer transformer applies these perturbations sequentially, and errors can compound through residual connections. The stated "degradation < 0.1%" prediction is not mathematically derived -- it is an extrapolation from the per-layer bound under an implicit assumption that interference does not accumulate across layers. This assumption is plausible (residual connections add, they do not multiply perturbations) but should be stated explicitly. The empirical K2 test (5% threshold) will settle this, so this is a **non-blocking** concern.

**Issue 2: The "orthogonality of reasoning vs domain" argument (Section 2.2) is informal.** The claim that Span(A_R) intersect Span(A_D) is approximately {0} relies on two conditions: (a) capacity (32 << 3584, trivially satisfied) and (b) "training data induces different gradient directions." Condition (b) is the entire hypothesis under test, restated as a prerequisite. This is circular as written. The MATH.md should be clearer that this is the hypothesis to be tested, not a derived result. The random-initialization J-L bound is the actual theoretical grounding; the gradient-direction argument is the empirical claim.

**Issue 3: D_per_layer calculation has a minor inconsistency.** MATH.md Section 2.1 lists 8 projection sizes in the sum (d, d, d_kv, d_kv, d, d_ff, d_ff, d_ff) but Section 1.2 states N_mod = 7 (q/k/v/o/gate/up/down). For Qwen2.5-7B with GQA, k_proj and v_proj have smaller dimensions (d_kv = 512, not d = 3584). The 8-element sum appears to count q, k, v, o, gate, up, down, plus one extra. This should be verified: is it 7 modules or 8? The total D_per_layer = 1,703,936 should be recomputed. If there are only 7 modules, the calculation is slightly off, but the order of magnitude and the final cosine bound remain valid. **Non-blocking** -- the empirical cosine measurement will give the actual value.

**Issue 4: Section 4.3 cites d=3584 for Qwen2.5-7B but VISION.md uses d=4096.** The correct hidden dimension for Qwen2.5-7B is 3584, which the experiment uses. VISION.md's table uses d=4096 for "Qwen 7B" which may refer to a different variant. This inconsistency should be noted but does not affect the experiment.

## Novelty Assessment

### Prior Art

1. **DeepSeek-R1 distillation (arXiv:2501.12948):** The DeepSeek team already distilled R1 reasoning traces into Qwen-7B via SFT, producing DeepSeek-R1-Distill-Qwen-7B. This is exactly the same approach: hard distillation of reasoning traces into the same base model. The novelty here is NOT the distillation itself (which is prior art) but the claim that the resulting adapter is **composable** with domain experts via pre-merge addition.

2. **LoRA Soups (Prabhakar et al., 2024):** Composes task-specific LoRA adapters. The SOLE framing adds orthogonality guarantees. This experiment extends the composition paradigm from domain knowledge to capabilities -- a meaningful novel axis.

3. **Rasbt reasoning-from-scratch ch08:** The reference repo's chapter 8 is incomplete ("In progress"). The experiment correctly uses the rasbt/math_distill dataset but does not rely on ch08 code. The training script is original.

### Delta Over Existing Work

The genuine novelty is narrow but significant: **testing whether a reasoning adapter composed with domain adapters via weight addition preserves both reasoning and domain quality.** No prior work has tested this specific composition. DeepSeek did full-model distillation, not LoRA. LoRA Soups tested skill composition, not reasoning-as-capability.

This is a well-targeted experiment for the SOLE thesis.

## Experimental Design

### Strengths

1. **Four-condition design is correct.** Base, reasoning-only, domain-only, composed. This is the minimal set needed to test all three kill criteria.

2. **Kill criteria are well-specified and measurable.**
   - K1: >10pp improvement on MATH-500 (clear threshold, standard benchmark)
   - K2: <5% domain PPL degradation (conservative, theory predicts <1%)
   - K3: composed > max(single) (clear superiority test)

3. **Smoke test in run_all.sh.** Running 50 examples first before committing to 500 is good engineering practice.

4. **Orthogonality measurement included.** Measuring cosine similarity between reasoning and domain deltas gives a direct diagnostic, connecting to the core SOLE theory.

5. **Training config matches pilot50.** Same rank, alpha, target modules -- this ensures composition compatibility and makes the interference test meaningful.

### Concerns

**Concern 1 (Medium): System prompt mismatch between training and evaluation.**

Training (`train_reasoning_expert.py`, line 131): System prompt says "shows your reasoning step by step inside <think>...</think> tags before giving your final answer."

Evaluation (`eval_math500.py`, line 332): System prompt says "Solve the problem step by step and write your final answer as \boxed{ANSWER}." No mention of <think> tags.

This mismatch means the evaluation does not prompt the model to use its trained reasoning format. The adapter learned to generate <think>...</think> traces when given the training system prompt. Under the evaluation prompt, the model may or may not produce <think> traces, depending on how strongly the LoRA encodes the behavior. If the model does not produce reasoning traces at eval time, K1 could fail not because distillation failed but because the prompt does not elicit the trained behavior.

**Recommendation:** Test both prompts. Run the reasoning condition with the training system prompt AND the evaluation system prompt. Report both. If only the training prompt works, that is still a valid result (the adapter works, just needs the right prompt), but it weakens the universality claim.

**Concern 2 (Low): Dead code in `load_with_merged_adapters`.** Lines 253-279 compute `merged_deltas` from all adapters but this dictionary is never used. The actual merge logic starts at line 283 using a different approach (PeftModel + merge_and_unload + manual delta addition). The dead code is confusing but not functionally harmful.

**Concern 3 (Low): Eval contamination risk for K2 (interference test).** The interference test uses "tail of training data" as eval data (line 119: `eval_lines = lines[-max_eval:]`). These are the same examples used in the pilot50 evaluation. If the domain adapters memorized these examples, the PPL measurements could be artificially low, making degradation percentages unreliable. However, since the SAME contaminated eval is used for both baseline and composed conditions, the relative degradation measurement is still valid. The absolute PPL values are unreliable but the ratio is not affected by contamination.

**Concern 4 (Medium): K3 may be too aggressive for 500 training steps.** K3 requires composed > max(reasoning, domain). If the math domain adapter from pilot50 already captures some reasoning (it was trained on math Q&A which includes implicit reasoning), the composed model needs to exceed this with an under-trained reasoning adapter. Consider: if domain_math already achieves 45% on MATH-500 (plausible -- it was trained on math content), and reasoning_only achieves 45% (500 steps on reasoning traces), the composed model needs >45%. This is not guaranteed -- the reasoning adapter may be too weak to add value on top of a math-specialized adapter. K3 failure would not necessarily mean the hypothesis is wrong, just that 500 steps is insufficient.

**Recommendation:** If K1 passes but K3 fails, declare REVISE (more training steps) rather than KILL. The notes in PAPER.md already hint at this ("If 500 steps is insufficient, the kill criterion will catch it") but the run_all.sh does not have a fallback path.

**Concern 5 (Low): No confidence intervals on MATH-500.** With 500 binary trials, a 40% accuracy has 95% CI of approximately +/-4.3pp (binomial). The 10pp threshold for K1 is robust against this uncertainty, but K3 (composed > best_single) could easily fall within noise. Consider reporting binomial confidence intervals or running with 2-3 seeds.

## Hypothesis Graph Consistency

The experiment matches `exp_reasoning_expert_distillation` in HYPOTHESES.yml. The three kill criteria in the code map directly to the three listed:

| HYPOTHESES.yml | Code Implementation |
|----------------|-------------------|
| "does not improve MATH-500 accuracy >10pp" | K1 in eval_math500.py |
| "degrades domain quality >5%" | K2 in eval_composition_interference.py |
| "does not outperform either alone" | K3 in eval_math500.py (K2 in that file's naming) |

Note the naming inconsistency: MATH.md calls these K1/K2/K3, eval_math500.py calls the MATH-500 tests K1/K2, and eval_composition_interference.py calls PPL degradation K2. This is confusing but the logic is consistent.

Dependencies are correct: `exp_distillation_pilot_50` is listed as a dependency and is in "supported" status. The blocking experiments (`exp_reasoning_domain_composition`, `exp_reasoning_expert_universality`) correctly depend on this experiment.

## Macro-Scale Risks (advisory)

1. **The 10pp threshold may be too easy or too hard depending on base model accuracy.** If Qwen2.5-7B base already achieves 45%+ on MATH-500 (some recent benchmarks show this), then 10pp improvement to 55%+ is quite ambitious for 500 steps of LoRA training. Conversely, if base is 25%, then 35% is modest. Measure base first and calibrate expectations.

2. **<think> token behavior under composition is the real risk.** The reasoning adapter teaches the model to generate <think> traces. When composed with a domain adapter, the domain adapter's influence on attention patterns could disrupt the sequential reasoning trace generation. This is an emergent behavior test that micro-scale cannot predict -- it must be tested empirically.

3. **Generation length explosion.** Reasoning traces can be very long (800+ tokens). The composition might cause the model to generate excessively long or repetitive traces. MAX_NEW_TOKENS = 2048 is a reasonable cap, but monitor for degenerate generation patterns.

4. **Answer format sensitivity.** The boxed-answer parser is brittle. Real MATH-500 answers include complex LaTeX, fractions, intervals, and set notation. The normalize_text function handles common cases but will miss edge cases. This is a known limitation of all MATH-500 evaluations, not specific to this experiment.

## Verdict

**PROCEED**

The experiment is well-designed and ready for GPU execution. The mathematical foundations build correctly on proven SOLE results. The experimental design has appropriate controls and kill criteria. The novelty claim (reasoning as a composable capability adapter) is genuine and significant for the SOLE thesis.

### Non-blocking recommendations (implement if time permits):

1. **Add training system prompt as an evaluation variant.** In `eval_math500.py`, add a `reasoning_prompted` condition that uses the same system prompt from training. Compare accuracy with and without the <think>-eliciting prompt. This is 30 min additional eval time.

2. **Remove dead code in `load_with_merged_adapters`.** Lines 253-279 compute `merged_deltas` but never use it. Delete for clarity.

3. **Add binomial confidence intervals to MATH-500 accuracy.** `scipy.stats.proportion_confint(correct, total, method='wilson')` -- one line, no extra compute.

4. **Clarify MATH.md Section 2.2:** Explicitly state that the gradient-direction argument is the hypothesis under test, not a derived result. The J-L bound is the theoretical grounding; orthogonality beyond random is the empirical claim.

5. **If K1 passes but K3 fails:** Interpret as REVISE (more training steps or different LR), not as a fundamental kill. The experiment design is sound either way.
